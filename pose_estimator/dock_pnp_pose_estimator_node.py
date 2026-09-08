"""Estimate the dock pose from its six window detections.

Physical layout (all dimensions are centre-to-centre unless stated otherwise):

* Each window is 280 mm wide (X) by 360 mm high (Y).
* Adjacent windows in the same row are separated by 2000 mm in X.
* Each top window is 440 mm left and 240 mm above its paired bottom window.
* The PnP object-frame origin is the centroid of all six window centres.
* PnP object-frame X points right, Y points down, and Z points into the dock.
* Published dock-pose X points into the dock, Y points left, and Z points up.

The six window centres, expressed in metres about the dock-frame origin, are:

* Top row:    (-2.22, -0.12), (-0.22, -0.12), (1.78, -0.12)
* Bottom row: (-1.78,  0.12), ( 0.22,  0.12), (2.22,  0.12)
"""

import cv2
import numpy as np
import rclpy
import tf2_geometry_msgs
import tf2_ros
from geometry_msgs.msg import PoseStamped
from rclpy.publisher import Publisher
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from std_msgs.msg import Header
from tf_transformations import quaternion_multiply
from yolo_msgs.msg import DetectionArray

from pose_estimator.utils.detections import (
    filter_detections_by_num_points,
    get_detection_centroid,
    get_detection_obb,
    get_top_k_detections_per_class,
    match_polygon_points_sequence,
)
from pose_estimator.utils.pose_estimator import get_object_pose, refine_object_pose
from pose_estimator.utils.pose_estimator_node import (
    PoseEstimatorPosePubNode,
    get_translation_quaternion,
)
from pose_estimator.utils.ros_messages import get_pose_stamped


WINDOW_COUNT = 6
WINDOW_WIDTH_M = 0.280
WINDOW_HEIGHT_M = 0.360
DEFAULT_TARGET_FRAME_ID = "asv5/odom"

# LiDAR publishes a dock pose in an FLU-compatible frame: X into the dock,
# Y left, Z up. This quaternion maps that dock-pose frame into PnP axes
# (X image-right, Y image-down, Z into the dock). Right-multiply it after the
# camera pose transform to keep PnP and LiDAR pose conventions identical.
DOCK_POSE_TO_PNP_QUATERNION = (0.5, -0.5, 0.5, 0.5)

# Ordered top-left to top-right, followed by bottom-left to bottom-right.
DOCK_WINDOW_CENTERS = np.array(
    [
        [-2.22, -0.12],
        [-0.22, -0.12],
        [1.78, -0.12],
        [-1.78, 0.12],
        [0.22, 0.12],
        [2.22, 0.12],
    ],
    dtype=np.float32,
)


def make_window_object_polygons(
    centers: np.ndarray = DOCK_WINDOW_CENTERS,
) -> np.ndarray:
    """Return one four-corner planar polygon for each window centre."""
    half_width = WINDOW_WIDTH_M / 2.0
    half_height = WINDOW_HEIGHT_M / 2.0
    corner_offsets = np.array(
        [
            [-half_width, -half_height],
            [-half_width, half_height],
            [half_width, half_height],
            [half_width, -half_height],
        ],
        dtype=np.float32,
    )
    return centers[:, np.newaxis, :] + corner_offsets


DOCK_WINDOW_OBJECT_POLYGONS = make_window_object_polygons()


def get_ordered_detection_indices(centroids: np.ndarray) -> np.ndarray:
    """Order six image centroids by top row then bottom row, each left-to-right."""
    centroids = np.asarray(centroids, dtype=np.float32)
    if centroids.shape != (WINDOW_COUNT, 2):
        raise ValueError(
            f"Expected {WINDOW_COUNT} detection centroids, got shape {centroids.shape}"
        )
    if not np.isfinite(centroids).all():
        raise ValueError("Detection centroids must contain only finite values")

    row_order = np.argsort(centroids[:, 1], kind="stable")
    top_indices = row_order[:3]
    bottom_indices = row_order[3:]
    top_indices = top_indices[np.argsort(centroids[top_indices, 0], kind="stable")]
    bottom_indices = bottom_indices[
        np.argsort(centroids[bottom_indices, 0], kind="stable")
    ]
    return np.concatenate([top_indices, bottom_indices])


def get_dock_correspondences(
    msg: DetectionArray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build planar object/image point correspondences from six window detections."""
    filtered_detections = filter_detections_by_num_points(msg, 3)
    detections = get_top_k_detections_per_class(
        filtered_detections, {"dock_window": WINDOW_COUNT}
    )["dock_window"]

    if len(detections) != WINDOW_COUNT:
        raise ValueError(
            f"Insufficient dock windows. Received: {len(detections)}, "
            f"require: {WINDOW_COUNT}."
        )

    centroids = np.array(
        [get_detection_centroid(detection) for detection in detections],
        dtype=np.float32,
    )
    order = get_ordered_detection_indices(centroids)
    detected_polygons = [
        get_detection_obb(detections[index])[1] for index in order
    ]
    object_points, image_points = match_polygon_points_sequence(
        DOCK_WINDOW_OBJECT_POLYGONS, detected_polygons
    )
    object_points = np.column_stack(
        [object_points, np.zeros(len(object_points), dtype=np.float32)]
    )
    return object_points, image_points


def estimate_dock_pose(camera, object_points, image_points):
    """Estimate and refine a positive-depth planar dock pose."""
    rvec, tvec, inliers = get_object_pose(
        camera,
        object_points,
        image_points,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if inliers is None or len(inliers) < 4:
        raise ValueError("No valid PnP inliers found")
    if tvec is None or float(np.asarray(tvec).reshape(-1)[2]) <= 0.0:
        raise ValueError("Estimated dock pose is behind the camera")

    rvec, tvec = refine_object_pose(
        camera, object_points, image_points, rvec, tvec
    )
    if float(np.asarray(tvec).reshape(-1)[2]) <= 0.0:
        raise ValueError("Refined dock pose is behind the camera")
    return rvec, tvec, inliers


def estimate_planar_dock_poses(camera, object_points, image_points):
    """Return up to two positive-depth IPPE solutions for the planar dock."""
    _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        object_points,
        image_points,
        camera.camera_matrix(),
        camera.dist_coeffs(),
        flags=cv2.SOLVEPNP_IPPE,
    )

    poses = []
    for rvec, tvec in zip(rvecs, tvecs):
        if float(np.asarray(tvec).reshape(-1)[2]) <= 0.0:
            continue
        rvec, tvec = refine_object_pose(
            camera, object_points, image_points, rvec, tvec
        )
        if float(np.asarray(tvec).reshape(-1)[2]) > 0.0:
            poses.append((rvec, tvec))

    if not poses:
        raise ValueError("No positive-depth planar PnP solution found")
    return poses[:2]


class DockPnpPoseEstimator(PoseEstimatorPosePubNode):
    def __init__(
        self,
        node_name: str = "dock_pnp_pose_estimator_node",
        default_pose_topic: str = "pnp/pose",
    ):
        super().__init__(node_name)

        self.target_frame_id = (
            self.declare_parameter("target_frame_id", DEFAULT_TARGET_FRAME_ID)
            .get_parameter_value()
            .string_value
        )
        detections_topic = (
            self.declare_parameter("input_detections_topic", "yolo/detections")
            .get_parameter_value()
            .string_value
        )
        pose_topic = (
            self.declare_parameter("pose_topic", default_pose_topic)
            .get_parameter_value()
            .string_value
        )

        self._dock_pose_publisher = self.create_publisher(
            PoseStamped, pose_topic, qos_profile_sensor_data
        )
        self.detections_sub = self.create_subscription(
            DetectionArray,
            detections_topic,
            self.detections_callback,
            qos_profile_sensor_data,
        )
        self.tf_buffer = tf2_ros.Buffer(node=self)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    @property
    def pose_publishers(self) -> dict[str, Publisher]:
        return {"dock": self._dock_pose_publisher}

    def publish_data(
        self,
        tvec: np.ndarray,
        rvec: np.ndarray,
        object_points: np.ndarray,
        header: Header,
        object_frame_id: str,
    ) -> None:
        """Transform the camera-frame pose into the configured odom frame."""
        del object_points, object_frame_id

        self._publish_pose(tvec, rvec, header, self._dock_pose_publisher)

    def _transform_pose(
        self,
        tvec: np.ndarray,
        rvec: np.ndarray,
        header: Header,
    ) -> PoseStamped:
        translation, quaternion = get_translation_quaternion(tvec, rvec)
        camera_pose = get_pose_stamped(header, translation, quaternion)
        transform = self.tf_buffer.lookup_transform(
            self.target_frame_id,
            camera_pose.header.frame_id,
            Time(),
        )
        odom_pose = tf2_geometry_msgs.do_transform_pose_stamped(
            camera_pose,
            transform,
        )
        orientation = odom_pose.pose.orientation
        corrected_orientation = quaternion_multiply(
            (
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ),
            DOCK_POSE_TO_PNP_QUATERNION,
        )
        (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ) = corrected_orientation

        # do_transform_pose_stamped copies the transform header; retain the
        # originating detection time while publishing in the target frame.
        odom_pose.header.stamp = camera_pose.header.stamp
        odom_pose.header.frame_id = self.target_frame_id
        return odom_pose

    def _publish_pose(
        self,
        tvec: np.ndarray,
        rvec: np.ndarray,
        header: Header,
        publisher: Publisher,
    ) -> bool:
        try:
            pose = self._transform_pose(tvec, rvec, header)
        except (np.linalg.LinAlgError, tf2_ros.TransformException) as e:
            self.get_logger().warn(
                f"Failed to transform dock pose to {self.target_frame_id}: {e}"
            )
            return False
        except Exception as e:
            self.get_logger().warn(f"Dock pose conversion failed: {e}")
            return False

        publisher.publish(pose)
        return True

    def detections_callback(self, msg: DetectionArray):
        try:
            object_points, image_points = get_dock_correspondences(msg)
            rvec, tvec, _ = estimate_dock_pose(
                self.camera, object_points, image_points
            )
        except Exception as e:
            self.get_logger().warn(f"Dock pose estimation failed: {e}")
            return

        self.publish_data(
            tvec, rvec, object_points, msg.header, msg.header.frame_id
        )


def main(args=None):
    rclpy.init(args=args)
    node = DockPnpPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
