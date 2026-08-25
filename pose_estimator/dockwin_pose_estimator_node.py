from typing import Optional

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from yolo_msgs.msg import Detection, DetectionArray

from pose_estimator.config.robotx26_object_points import (
    WINDOW_FRAME_REMAP,
    WINDOW_OBJECT_POINTS_DICT,
)
from pose_estimator.utils.detections import (
    filter_detections_by_num_points,
    get_best_detections_per_class,
    get_detection_obb,
    get_detection_polygon,
    match_polygon_points,
)
from pose_estimator.utils.pose_estimator import (
    get_detection_centroid_position,
    get_object_pose,
    get_object_pose_from_detection_using_best_fit_quad,
    refine_object_pose,
)
from pose_estimator.utils.pose_estimator_node import (
    PoseEstimatorPosePubNode,
    get_translation_quaternion,
)


def _polygon_to_binary_mask(
    polygon: np.ndarray, resolution_wh: tuple[int, int]
) -> np.ndarray:
    """Create a binary uint8 mask (0 or 255) from polygon coordinate points."""
    w, h = int(resolution_wh[0]), int(resolution_wh[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.asarray(polygon, dtype=np.float32)
    if len(poly) >= 3:
        pts = np.round(poly).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], color=255)
    return mask


class DockwinPoseEstimator(PoseEstimatorPosePubNode):
    """Estimate dock-window poses using OBB-assisted planar PnP with optional stereo depth.

    ``best_fit_quad`` keeps the previous generic estimator available for
    comparison and fallback.  The default ``improved`` mode follows the bin
    estimator's angle-assisted point matching, which is less sensitive to the
    cyclic ordering of noisy segmentation corners.

    When ``use_depth_image`` is enabled and ``input_depth_image_topic`` is provided,
    the node extracts the median stereo depth within the eroded segmentation mask
    to anchor the 3D translation distance while retaining PnP orientation.
    """

    def __init__(self):
        super().__init__("dockwin_pose_estimator_node")

        self.object_points = {
            class_name: np.asarray(points, dtype=np.float32)
            for class_name, points in WINDOW_OBJECT_POINTS_DICT.items()
        }
        self.frame_name_remap = WINDOW_FRAME_REMAP.copy()

        if set(self.object_points) != set(self.frame_name_remap):
            raise ValueError(
                "Window object-point and output-frame mappings must have the same keys"
            )
        if len(set(self.frame_name_remap.values())) != len(self.frame_name_remap):
            raise ValueError("Window output-frame names must be unique")

        self.input_detection_classes = list(self.object_points)
        self.estimation_method = (
            self.declare_parameter("estimation_method", "improved")
            .get_parameter_value()
            .string_value
        )
        if self.estimation_method not in {"improved", "best_fit_quad"}:
            raise ValueError(
                "estimation_method must be 'improved' or 'best_fit_quad'"
            )

        self.max_reprojection_error = (
            self.declare_parameter("max_reprojection_error", 2.0)
            .get_parameter_value()
            .double_value
        )
        self.min_inliers = (
            self.declare_parameter("min_inliers", 4)
            .get_parameter_value()
            .integer_value
        )
        if self.max_reprojection_error <= 0.0:
            raise ValueError("max_reprojection_error must be positive")
        if self.min_inliers < 4:
            raise ValueError("min_inliers must be at least 4 for a quadrilateral")
        self.depth_filter_alpha = (
            self.declare_parameter("depth_filter_alpha", 0.2)
            .get_parameter_value()
            .double_value
        )
        if not 0.0 <= self.depth_filter_alpha <= 1.0:
            raise ValueError("depth_filter_alpha must be in the range [0, 1]")

        detections_topic = (
            self.declare_parameter("input_detections_topic", "yolo/detections")
            .get_parameter_value()
            .string_value
        )
        self.input_depth_image_topic = (
            self.declare_parameter("input_depth_image_topic", "")
            .get_parameter_value()
            .string_value
        )
        self.use_depth_image = (
            self.declare_parameter("use_depth_image", True)
            .get_parameter_value()
            .bool_value
        )
        self.depth_sync_slop = (
            self.declare_parameter("depth_sync_slop", 0.1)
            .get_parameter_value()
            .double_value
        )
        self.mask_erosion_iters = (
            self.declare_parameter("mask_erosion_iters", 2)
            .get_parameter_value()
            .integer_value
        )
        self.min_depth = (
            self.declare_parameter("min_depth", 0.2)
            .get_parameter_value()
            .double_value
        )
        self.max_depth = (
            self.declare_parameter("max_depth", 15.0)
            .get_parameter_value()
            .double_value
        )

        self._pose_publishers = {
            frame_name: self.create_publisher(
                PoseStamped, f"{frame_name}/pose", qos_profile_sensor_data
            )
            for frame_name in self.frame_name_remap.values()
        }
        self._filtered_tvecs = {}

        if self.use_depth_image and self.input_depth_image_topic:
            self.cv_bridge = CvBridge()
            self._detections_sub = message_filters.Subscriber(
                self,
                DetectionArray,
                detections_topic,
                qos_profile=qos_profile_sensor_data,
            )
            self._depth_sub = message_filters.Subscriber(
                self,
                Image,
                self.input_depth_image_topic,
                qos_profile=qos_profile_sensor_data,
            )
            self._time_sync = message_filters.ApproximateTimeSynchronizer(
                [self._detections_sub, self._depth_sub],
                queue_size=10,
                slop=self.depth_sync_slop,
            )
            self._time_sync.registerCallback(self.synced_detections_callback)
        else:
            self.detections_sub = self.create_subscription(
                DetectionArray,
                detections_topic,
                self.detections_callback,
                qos_profile_sensor_data,
            )

    @property
    def pose_publishers(self):
        return self._pose_publishers

    def synced_detections_callback(
        self, detections_msg: DetectionArray, depth_msg: Image
    ):
        try:
            depth_img = self.cv_bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
        except Exception as exc:
            self.get_logger().warn(
                f"Failed to convert depth image from {self.input_depth_image_topic}: {exc}"
            )
            depth_img = None

        self._process_detections(detections_msg, depth_img)

    def detections_callback(self, msg: DetectionArray):
        self._process_detections(msg, depth_img=None)

    def _process_detections(
        self, msg: DetectionArray, depth_img: Optional[np.ndarray] = None
    ):
        filtered_detections = filter_detections_by_num_points(msg, 3)
        best_detections = get_best_detections_per_class(
            filtered_detections, self.input_detection_classes
        )

        for class_name, detection in best_detections.items():
            try:
                if self.estimation_method == "best_fit_quad":
                    rvec, tvec = self._estimate_with_best_fit_quad(
                        class_name, detection
                    )
                else:
                    rvec, tvec = self._estimate_with_oriented_quad(
                        class_name, detection
                    )
            except Exception as exc:
                self.get_logger().warn(
                    f"Pose estimation failed for {class_name}: {exc}"
                )
                continue

            if depth_img is not None and self.use_depth_image:
                measured_depth = self._extract_mask_depth(depth_img, detection)
                if np.isfinite(measured_depth) and measured_depth > 0.0:
                    tvec = self._refine_translation_with_depth(
                        detection, tvec, measured_depth
                    )

            depth = float(tvec[2, 0])
            if not np.isfinite(depth) or depth <= 0.0:
                self.get_logger().warn(
                    f"Rejecting {class_name} pose with invalid depth: {depth:.3f} m"
                )
                continue

            tvec = self._filter_translation(class_name, tvec)
            frame_name = self.frame_name_remap[class_name]
            object_points = np.hstack(
                [
                    self.object_points[class_name],
                    np.zeros(
                        (len(self.object_points[class_name]), 1), dtype=np.float32
                    ),
                ]
            )
            self.publish_data(tvec, rvec, object_points, msg.header, frame_name)

    def publish_data(
        self,
        tvec: np.ndarray,
        rvec: np.ndarray,
        object_points: np.ndarray,
        header,
        object_frame_id: str,
    ) -> None:
        try:
            t, q = get_translation_quaternion(tvec, rvec)
        except np.linalg.LinAlgError as e:
            self.get_logger().warn(f"Error in mat2quat, failed to convert R: {e}")
            return
        except Exception as e:
            self.get_logger().warn(f"Rodrigues conversion failed: {e}")
            return

        pose = PoseStamped()
        pose.header = header
        pose.pose.position = Point(
            x=float(t[0]), y=float(t[1]), z=float(t[2])
        )
        pose.pose.orientation = Quaternion(
            x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])
        )
        self.pose_publishers[object_frame_id].publish(pose)

    def _extract_mask_depth(
        self, depth_img: np.ndarray, detection: Detection
    ) -> float:
        """Extract median depth in meters from the YOLO detection mask."""
        if depth_img is None or detection is None:
            return float("nan")

        mask_polygon = get_detection_polygon(detection)
        if len(mask_polygon) < 3:
            return float("nan")

        h, w = depth_img.shape[:2]
        det_w = getattr(detection.mask, "width", w) or w
        det_h = getattr(detection.mask, "height", h) or h

        if (det_w, det_h) != (w, h):
            scale_x = float(w) / float(det_w)
            scale_y = float(h) / float(det_h)
            scaled_polygon = mask_polygon * np.array([scale_x, scale_y], dtype=np.float32)
        else:
            scaled_polygon = mask_polygon

        binary_mask = _polygon_to_binary_mask(scaled_polygon, (w, h))

        if self.mask_erosion_iters > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eroded = cv2.erode(
                binary_mask, kernel, iterations=self.mask_erosion_iters
            )
            if np.any(eroded > 0):
                binary_mask = eroded

        masked_depth = depth_img[binary_mask > 0].astype(np.float32)
        if masked_depth.size == 0:
            return float("nan")

        # Auto-detect millimeter encoding (e.g. 16UC1 from OAK-D depth)
        if np.issubdtype(depth_img.dtype, np.integer) or (
            np.nanmedian(masked_depth) > 50.0
        ):
            masked_depth = masked_depth / 1000.0

        valid_mask = (
            np.isfinite(masked_depth)
            & (masked_depth >= self.min_depth)
            & (masked_depth <= self.max_depth)
        )
        valid_depths = masked_depth[valid_mask]

        if valid_depths.size == 0:
            return float("nan")

        return float(np.median(valid_depths))

    def _refine_translation_with_depth(
        self, detection: Detection, tvec: np.ndarray, measured_depth: float
    ) -> np.ndarray:
        """Refine translation vector using measured stereo depth."""
        pnp_depth = float(tvec[2, 0])
        if pnp_depth > 1e-4:
            scale = float(measured_depth) / pnp_depth
            refined_tvec = (tvec * scale).astype(np.float64)
            refined_tvec[2, 0] = float(measured_depth)
            return refined_tvec
        else:
            centroid_pos = get_detection_centroid_position(
                detection, self.camera, float(measured_depth)
            )
            return centroid_pos.reshape((3, 1)).astype(np.float64)

    def _estimate_with_oriented_quad(self, class_name, detection):
        object_points_2d = self.object_points[class_name]
        angle, image_points = get_detection_obb(detection)
        angle = (angle + 90.0) % 180.0 - 90.0

        # Rotate the known rectangle during matching, as in the bin estimator.
        # This uses the known 0.21 m x 0.29 m aspect ratio from the configuration.
        matched_object_points, image_points = self._match_points(
            object_points_2d, image_points, angle
        )
        object_points = np.hstack(
            [
                matched_object_points,
                np.zeros((len(matched_object_points), 1), dtype=np.float32),
            ]
        )

        rvec, tvec, inliers = get_object_pose(
            self.camera,
            object_points,
            image_points,
            max_reprojection_error=self.max_reprojection_error,
        )
        if inliers is None or len(inliers) < self.min_inliers:
            count = 0 if inliers is None else len(inliers)
            raise ValueError(f"Only {count} PnP inliers found")

        rvec, tvec = refine_object_pose(
            self.camera, object_points, image_points, rvec, tvec
        )
        mean_error, max_error = self._reprojection_errors(
            object_points, image_points, rvec, tvec
        )
        if max_error > self.max_reprojection_error:
            raise ValueError(
                f"Reprojection error too large: mean={mean_error:.2f}, "
                f"max={max_error:.2f} pixels"
            )
        return rvec, tvec

    def _estimate_with_best_fit_quad(self, class_name, detection):
        # Preserve the original generic implementation as an explicit fallback.
        return get_object_pose_from_detection_using_best_fit_quad(
            self.camera, self.object_points[class_name], detection, self.get_logger()
        )

    @staticmethod
    def _match_points(object_points, image_points, angle):
        return match_polygon_points(object_points, image_points, A_angle=angle)

    def _reprojection_errors(self, object_points, image_points, rvec, tvec):
        projected_points, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.camera.camera_matrix(),
            self.camera.dist_coeffs(),
        )
        errors = np.linalg.norm(
            projected_points.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1
        )
        return float(np.mean(errors)), float(np.max(errors))

    def _filter_translation(self, class_name, tvec):
        previous = self._filtered_tvecs.get(class_name)
        if (
            previous is None
            or self.depth_filter_alpha in (0.0, 1.0)
        ):
            filtered = tvec.copy()
        else:
            alpha = self.depth_filter_alpha
            filtered = alpha * tvec + (1.0 - alpha) * previous
        self._filtered_tvecs[class_name] = filtered
        return filtered


def main(args=None):
    rclpy.init(args=args)
    node = DockwinPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
