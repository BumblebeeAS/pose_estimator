"""Estimate dock panel x, y, yaw from a YOLO OBB, using stereo depth or PnP.

Pipeline (``use_stereo_depth: true``):
    1. YOLO OBB (image pixels) -> shrunken OBB mask on the depth image.
    2. Depth pixels inside the mask are back-projected with the pinhole model
       into the camera optical frame, then transformed to ``target_frame``.
    3. A vertical plane is fitted with RANSAC + PCA refinement; its horizontal
       normal gives the panel yaw, the inlier centroid gives x and y.
    4. The pose is published with z = 0, roll = 0, pitch = 0.

Pipeline (``use_stereo_depth: false``):
    1. The 4 OBB corners are matched to the panel corners in
       ``WINDOW_WHITE_PLANE_DICT`` and solved with planar PnP (IPPE).
    2. The panel centre and normal are transformed to ``target_frame`` and
       published as in step 4 above.

A single plane fit over the ROI is used instead of per-pixel (k-NN) normals:
stereo depth is too noisy for local normals, while a fit over hundreds of
points averages that noise out. Depth noise still grows ~Z^2, so yaw quality
degrades with range; the RMSE / inlier-ratio gates reject bad fits.

The depth image must be registered to the camera whose intrinsics are given on
``camera_info_topic`` (e.g. OAK-D depth aligned to RGB) and must encode Z depth,
not ray range. Lens distortion is ignored, so the depth image should be
rectified.
"""

import math

import cv2
import message_filters
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf_transformations import quaternion_matrix
from yolo_msgs.msg import DetectionArray

from pose_estimator.config.robotx26_object_points import WINDOW_WHITE_PLANE_DICT


def transform_points(points: np.ndarray, transform) -> np.ndarray:
    """Transform XYZ points using a geometry_msgs TransformStamped."""
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    matrix = quaternion_matrix((rotation.x, rotation.y, rotation.z, rotation.w))
    matrix[:3, 3] = (translation.x, translation.y, translation.z)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def fit_vertical_plane(
    points: np.ndarray,
    distance_threshold: float,
    max_tilt: float,
    iterations: int,
    rng: np.random.Generator,
):
    """Fit a vertical plane and return centroid, horizontal normal, and quality."""
    if len(points) < 3:
        return None

    best_inliers = None
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        if abs(normal[2]) > math.sin(max_tilt):
            continue
        offset = -normal @ sample[0]
        inliers = np.abs(points @ normal + offset) <= distance_threshold
        if best_inliers is None or np.count_nonzero(inliers) > np.count_nonzero(
            best_inliers
        ):
            best_inliers = inliers

    if best_inliers is None or np.count_nonzero(best_inliers) < 3:
        return None

    inlier_points = points[best_inliers]
    centroid = np.mean(inlier_points, axis=0)
    covariance = np.cov((inlier_points - centroid).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    if abs(normal[2]) > math.sin(max_tilt):
        return None

    normal_xy = normal[:2]
    normal_xy_norm = np.linalg.norm(normal_xy)
    if normal_xy_norm < 1e-6:
        return None
    normal_xy /= normal_xy_norm

    residuals = (inlier_points - centroid) @ normal
    rmse = math.sqrt(float(np.mean(residuals**2)))
    inlier_ratio = len(inlier_points) / len(points)
    return centroid, normal_xy, rmse, inlier_ratio


def obb_corners(
    center: tuple[float, float],
    size: tuple[float, float],
    theta: float,
    shrink: float = 1.0,
) -> np.ndarray:
    """Return the 4 corners of an OBB (theta in radians) scaled by ``shrink``."""
    return cv2.boxPoints(
        (
            (float(center[0]), float(center[1])),
            (float(size[0]) * shrink, float(size[1]) * shrink),
            math.degrees(theta),
        )
    )


def obb_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    size: tuple[float, float],
    theta: float,
    shrink: float,
) -> np.ndarray:
    """Rasterise an OBB (theta in radians) scaled by ``shrink`` into a bool mask."""
    corners = obb_corners(center, size, theta, shrink)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(corners).astype(np.int32)], 1)
    return mask.astype(bool)


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Order 4 image corners as top-left, top-right, bottom-right, bottom-left.

    shortcut: sum/diff ordering is only valid for in-image rotations < 45 deg,
    fine for an upright panel; switch to angle sorting if the camera rolls.
    """
    s = corners.sum(axis=1)
    d = corners[:, 1] - corners[:, 0]
    return corners[[np.argmin(s), np.argmin(d), np.argmax(s), np.argmax(d)]]


class DockPlanarPoseEstimator(Node):
    """Estimate the dock panel pose from the YOLO OBB with depth or PnP."""

    def __init__(self):
        super().__init__("dock_planar_pose_estimator_node")

        self.detections_topic = self.declare_parameter(
            "input_detections_topic", "/asv5/dock_planar_window/yolo/detections"
        ).value
        self.depth_image_topic = self.declare_parameter(
            "input_depth_image_topic", "/asv5/oakd/stereo/image_raw"
        ).value
        self.camera_info_topic = self.declare_parameter(
            "camera_info_topic", "/asv5/oakd/rgb/camera_info"
        ).value
        self.use_rectified_intrinsics = bool(
            self.declare_parameter("is_image_rectified", False).value
        )
        self.pose_topic = self.declare_parameter(
            "output_pose_topic", "/asv5/dock/depth_planar/pose"
        ).value
        self.target_frame = self.declare_parameter(
            "target_frame", "asv5/odom"
        ).value
        # Empty -> accept any class and keep the highest-scoring detection.
        self.class_name = self.declare_parameter("class_name", "").value
        self.min_confidence = float(
            self.declare_parameter("min_confidence", 0.3).value
        )
        self.depth_sync_slop = float(
            self.declare_parameter("depth_sync_slop", 0.1).value
        )
        # False -> no depth image; solve planar PnP on the OBB corners instead.
        self.use_stereo_depth = bool(
            self.declare_parameter("use_stereo_depth", True).value
        )
        self.obb_shrink = float(self.declare_parameter("obb_shrink", 0.8).value)
        self.pixel_stride = int(self.declare_parameter("pixel_stride", 2).value)
        self.min_depth = float(self.declare_parameter("min_depth", 0.2).value)
        self.max_depth = float(self.declare_parameter("max_depth", 15.0).value)
        # 0 -> auto: integer depth (16UC1) is mm, float depth (32FC1) is m.
        self.depth_scale = float(self.declare_parameter("depth_scale", 0.0).value)
        self.min_points = int(self.declare_parameter("min_points", 50).value)
        self.plane_distance_threshold = float(
            self.declare_parameter("plane_distance_threshold", 0.05).value
        )
        self.max_plane_tilt = math.radians(
            float(self.declare_parameter("max_plane_tilt_deg", 10.0).value)
        )
        self.min_inlier_ratio = float(
            self.declare_parameter("min_inlier_ratio", 0.6).value
        )
        self.max_plane_rmse = float(
            self.declare_parameter("max_plane_rmse", 0.05).value
        )
        self.ransac_iterations = int(
            self.declare_parameter("ransac_iterations", 200).value
        )
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in the range [0, 1]")
        if not 0.0 < self.obb_shrink <= 1.0:
            raise ValueError("obb_shrink must be in (0, 1]")
        if self.pixel_stride < 1:
            raise ValueError("pixel_stride must be at least 1")
        if not 0.0 < self.min_depth < self.max_depth:
            raise ValueError("require 0 < min_depth < max_depth")

        self.camera_info = None
        self.rng = np.random.default_rng()
        self.cv_bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(node=self)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pose_publisher = self.create_publisher(
            PoseStamped, self.pose_topic, qos_profile_sensor_data
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._camera_info_callback,
            qos_profile_sensor_data,
        )

        if self.use_stereo_depth:
            self._detections_sub = message_filters.Subscriber(
                self,
                DetectionArray,
                self.detections_topic,
                qos_profile=qos_profile_sensor_data,
            )
            self._depth_sub = message_filters.Subscriber(
                self,
                Image,
                self.depth_image_topic,
                qos_profile=qos_profile_sensor_data,
            )
            self._time_sync = message_filters.ApproximateTimeSynchronizer(
                [self._detections_sub, self._depth_sub],
                queue_size=10,
                slop=self.depth_sync_slop,
            )
            self._time_sync.registerCallback(self.synced_detections_callback)
            source = f"{self.detections_topic} + {self.depth_image_topic}"
        else:
            self._detections_sub = self.create_subscription(
                DetectionArray,
                self.detections_topic,
                self.detections_callback,
                qos_profile_sensor_data,
            )
            source = f"{self.detections_topic} (PnP)"

        self.get_logger().info(
            f"Publishing dock planar pose on {self.pose_topic} from {source}"
        )

    def _camera_info_callback(self, message: CameraInfo) -> None:
        self.camera_info = message

    def detections_callback(self, detections_msg: DetectionArray):
        if self.camera_info is None:
            self.get_logger().warn(
                f"No CameraInfo on {self.camera_info_topic}", throttle_duration_sec=5.0
            )
            return
        detection = self._best_detection(detections_msg)
        if detection is None:
            return
        pose = self._estimate_pose(detection, detections_msg.header)
        if pose is not None:
            self.pose_publisher.publish(pose)

    def synced_detections_callback(
        self, detections_msg: DetectionArray, depth_msg: Image
    ):
        if self.camera_info is None:
            return
        detection = self._best_detection(detections_msg)
        if detection is None:
            return
        try:
            depth_img = self.cv_bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
        except Exception as exc:
            self.get_logger().warn(
                f"Failed to convert depth image from {self.depth_image_topic}: {exc}"
            )
            return

        pose = self._estimate_pose(detection, depth_msg.header, depth_img)
        if pose is not None:
            self.pose_publisher.publish(pose)

    def _best_detection(self, msg: DetectionArray):
        candidates = [
            detection
            for detection in msg.detections
            if (not self.class_name or detection.class_name == self.class_name)
            and detection.score >= self.min_confidence
            and detection.bbox.size.x > 0.0
            and detection.bbox.size.y > 0.0
        ]
        return max(candidates, key=lambda d: d.score, default=None)

    def _intrinsics(self, depth_shape: tuple[int, int]):
        """Return (fx, fy, cx, cy, sx, sy) scaled to the depth image resolution."""
        info = self.camera_info
        if self.use_rectified_intrinsics:
            fx, fy, cx, cy = info.p[0], info.p[5], info.p[2], info.p[6]
        else:
            fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        height, width = depth_shape
        sx = width / float(info.width) if info.width else 1.0
        sy = height / float(info.height) if info.height else 1.0
        return fx * sx, fy * sy, cx * sx, cy * sy, sx, sy

    def _camera_points(self, detection, depth_img: np.ndarray) -> np.ndarray:
        """Back-project depth pixels inside the shrunken OBB to the optical frame."""
        fx, fy, cx, cy, sx, sy = self._intrinsics(depth_img.shape[:2])
        bbox = detection.bbox
        mask = obb_mask(
            depth_img.shape[:2],
            (bbox.center.position.x * sx, bbox.center.position.y * sy),
            (bbox.size.x * sx, bbox.size.y * sy),
            bbox.center.theta,
            self.obb_shrink,
        )
        if self.pixel_stride > 1:
            stride_mask = np.zeros_like(mask)
            stride_mask[:: self.pixel_stride, :: self.pixel_stride] = True
            mask &= stride_mask
        v, u = np.nonzero(mask)
        z = depth_img[v, u].astype(np.float64)
        if self.depth_scale > 0.0:
            z *= self.depth_scale
        elif np.issubdtype(depth_img.dtype, np.integer):
            z /= 1000.0
        valid = np.isfinite(z) & (z >= self.min_depth) & (z <= self.max_depth)
        u, v, z = u[valid], v[valid], z[valid]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.column_stack((x, y, z))

    def _depth_plane(self, detection, depth_img, transform):
        """Return (centroid, horizontal normal) in target_frame from stereo depth."""
        camera_points = self._camera_points(detection, depth_img)
        if len(camera_points) < self.min_points:
            return None
        fitted = fit_vertical_plane(
            transform_points(camera_points, transform),
            self.plane_distance_threshold,
            self.max_plane_tilt,
            self.ransac_iterations,
            self.rng,
        )
        if fitted is None:
            return None
        centroid, normal, rmse, inlier_ratio = fitted
        if inlier_ratio < self.min_inlier_ratio or rmse > self.max_plane_rmse:
            self.get_logger().debug(
                f"Rejected plane: inlier_ratio={inlier_ratio:.2f}, "
                f"rmse={rmse:.3f} m"
            )
            return None
        return centroid, normal

    def _pnp_plane(self, detection, transform):
        """Return (centroid, horizontal normal) in target_frame from planar PnP."""
        object_points = WINDOW_WHITE_PLANE_DICT.get(detection.class_name)
        if object_points is None:
            object_points = next(iter(WINDOW_WHITE_PLANE_DICT.values()))
        # Object points are TL, TR, BR, BL with y down, matching order_corners.
        object_points = np.asarray(object_points, dtype=np.float64)

        info = self.camera_info
        if self.use_rectified_intrinsics:
            camera_matrix = np.array(info.p, dtype=np.float64).reshape(3, 4)[:, :3]
            dist_coeffs = np.zeros(5)
        else:
            camera_matrix = np.array(info.k, dtype=np.float64).reshape(3, 3)
            dist_coeffs = np.array(info.d, dtype=np.float64)

        bbox = detection.bbox
        image_points = order_corners(
            obb_corners(
                (bbox.center.position.x, bbox.center.position.y),
                (bbox.size.x, bbox.size.y),
                bbox.center.theta,
            )
        ).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not ok:
            return None

        rotation, _ = cv2.Rodrigues(rvec)
        center = rotation @ object_points.mean(axis=0) + tvec.ravel()
        # Panel normal is the object z axis; rotate it only (no translation).
        ends = transform_points(np.vstack((center, center + rotation[:, 2])), transform)
        normal_xy = (ends[1] - ends[0])[:2]
        normal_xy_norm = np.linalg.norm(normal_xy)
        if normal_xy_norm < 1e-6:
            return None
        return ends[0], normal_xy / normal_xy_norm

    def _estimate_pose(self, detection, header, depth_img=None) -> PoseStamped | None:
        try:
            camera_frame = header.frame_id or self.camera_info.header.frame_id
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                camera_frame,
                Time.from_msg(header.stamp),
                timeout=Duration(seconds=0.1),
            )
            camera_origin = transform_points(np.zeros((1, 3)), transform)[0]

            if depth_img is None:
                fitted = self._pnp_plane(detection, transform)
            else:
                fitted = self._depth_plane(detection, depth_img, transform)
            if fitted is None:
                return None
            centroid, normal = fitted

            # Orient the normal into the panel (pointing away from the camera).
            inward = normal
            if inward @ (centroid[:2] - camera_origin[:2]) < 0.0:
                inward = -inward
            yaw = math.atan2(inward[1], inward[0]) + math.pi # panel faces away from the camera

            pose = PoseStamped()
            pose.header.stamp = header.stamp
            pose.header.frame_id = self.target_frame
            pose.pose.position.x = float(centroid[0])
            pose.pose.position.y = float(centroid[1])
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
            return pose
        except tf2_ros.TransformException as error:
            self.get_logger().warn(f"Dock panel transform unavailable: {error}")
        except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
            self.get_logger().warn(f"Dock panel fit failed: {error}")
        return None


def main(args=None):
    rclpy.init(args=args)
    node = DockPlanarPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
