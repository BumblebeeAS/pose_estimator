import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import qos_profile_sensor_data
from yolo_msgs.msg import DetectionArray

from pose_estimator.config.robotx26_object_points import (
    WINDOW_FRAME_REMAP,
    WINDOW_OBJECT_POINTS_DICT,
)
from pose_estimator.utils.detections import (
    filter_detections_by_num_points,
    get_best_detections_per_class,
    get_detection_obb,
    match_polygon_points,
)
from pose_estimator.utils.pose_estimator import (
    get_object_pose,
    get_object_pose_from_detection_using_best_fit_quad,
    refine_object_pose,
)
from pose_estimator.utils.pose_estimator_node import PoseEstimatorPosePubNode


class DockwinPoseEstimator(PoseEstimatorPosePubNode):
    """Estimate dock-window poses using OBB-assisted planar PnP.

    ``best_fit_quad`` keeps the previous generic estimator available for
    comparison and fallback.  The default ``improved`` mode follows the bin
    estimator's angle-assisted point matching, which is less sensitive to the
    cyclic ordering of noisy segmentation corners.
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

        self._pose_publishers = {
            frame_name: self.create_publisher(
                PoseStamped, f"{frame_name}/pose", qos_profile_sensor_data
            )
            for frame_name in self.frame_name_remap.values()
        }
        self._filtered_tvecs = {}
        self.detections_sub = self.create_subscription(
            DetectionArray,
            detections_topic,
            self.detections_callback,
            qos_profile_sensor_data,
        )

    @property
    def pose_publishers(self):
        return self._pose_publishers

    def detections_callback(self, msg: DetectionArray):
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
