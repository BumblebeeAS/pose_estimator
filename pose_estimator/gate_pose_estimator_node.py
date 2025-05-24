from operator import attrgetter
from typing import Dict, List

import cv2
import numpy as np
import rclpy
import shapely
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import (
    Point,
    PoseStamped,
    PoseWithCovarianceStamped,
    Quaternion,
    TransformStamped,
    Vector3,
)
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import CameraInfo
from transforms3d.quaternions import mat2quat
from yolo_msgs.msg import Detection, DetectionArray

from image_matching.simple_pose_estimator_node import (
    estimate_covariance,
    get_object_pose,
)
from pose_estimator.PinholeCamera import PinholeCamera


def polygon_to_obb(points: np.ndarray) -> np.ndarray:
    """Compute the Oriented Bounding Box (OBB) of a polygon in strict canonical form.

    Args:
        points (np.ndarray): N x 2 array of points representing the polygon.

    Returns:
        np.ndarray: Oriented bounding box represented as a 4 x 2 array of points.
        The points are ordered from top-left to top-right in an anti-clockwise
        manner. Note the vertical direction is different since the y-coordinate
        increases downwards in the image coordinate system.
    """
    polygon = shapely.Polygon(points)
    min_area_rect = polygon.minimum_rotated_rectangle.normalize()
    coords_array = np.array(min_area_rect.exterior.coords)
    return coords_array


object_points_mm_dict = {
    "gate_sides_left": [
        (0, 152.40),
        (0, 152.40 + 1219.20),
        (76.20, 152.40 + 1219.20),
        (76.20, 152.40),
    ],
    "gate_sides_right": [
        (3048.00 - 76.20, 152.40),
        (3048.00 - 76.20, 152.40 + 1219.20),
        (3048.00, 152.40 + 1219.20),
        (3048.00, 152.4),
    ],
    "gate_center": [
        ((3048.00 - 50.80) / 2, 0),
        ((3048.00 - 50.80) / 2, 609.60),
        ((3048.00 + 50.80) / 2, 609.60),
        ((3048.00 + 50.80) / 2, 0),
    ],
}
object_points_dict = {}
for key, object_points_mm in object_points_mm_dict.items():
    object_points_2d = np.array(object_points_mm, dtype=np.float32) / 1000.0
    object_points = np.hstack(
        [object_points_2d, np.zeros((object_points_2d.shape[0], 1))]
    )
    object_points_dict[key] = np.array(object_points, dtype=np.float32)


# def get_object_pose(
#     camera: PinholeCamera,
#     object_points: np.ndarray,
#     image_points: np.ndarray,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Get the object pose from the camera and point correspondences.

#     Args:
#         camera (PinholeCamera):
#         object_points (np.ndarray): N x 3
#         image_points (np.ndarray): N x 2
#         max_reprojection_error (float): Maximum reprojection error for RANSAC.

#     Returns:
#         tuple[np.ndarray, np.ndarray]: (R, t)

#     Raises:
#         ValueError: If the number of object points is less than 4.
#         Exception: If cv2.solvePnPRefineLM fails.
#     """
#     # TODO: Account for equidistant distortion
#     # TODO: For the planar case, init cv2.solvePnPRefineLM directly with homography
#     if len(object_points) < 4:
#         raise ValueError(
#             f"At least 4 points needed to estimate pose, only {len(object_points)} given"
#         )

#     # TODO: Split into planar and non-planar cases.
#     # Case 1: Object points are non-planar.
#     # Use only the inliers from RANSAC as homography estimation does not apply.
#     # Case 2: Use all points for the planar case.
#     # Homography estimation filters well. RANSAC filtering is too strict, resulting
#     # in too few point correspondences and a noisy pose estimate.
#     rvec, tvec = cv2.solvePnPRefineVVS(
#         object_points,
#         image_points,
#         camera.camera_matrix(),
#         camera.dist_coeffs(),
#         rvec,
#         tvec,
#         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 1000, 1e-6),
#     )

#     return rvec, tvec


def get_best_detections_per_class(
    detection_array_msg: DetectionArray, classes: List[str]
) -> Dict[str, Detection]:
    """Returns the best detection (by score) per class.

    Note: If there are no detections for a class,
    the returned dictionary would not contain an entry for that class.

    Args:
        msg (DetectionArray): Array of Detections
        classes (List[str]): List of classes to get best detections for.

    Returns:
        Dict[str, Detection]: Dictionary of detections with class names as key.
    """

    best_detections = {}
    best_scores = {}
    classes = set(classes)
    for detection in detection_array_msg.detections:
        class_name = detection.class_name
        score = detection.score
        if class_name in classes and score > best_scores.get(class_name, -1):
            best_scores[class_name] = score
            best_detections[class_name] = detection
    return best_detections


class GatePoseEstimator(Node):
    def __init__(self):
        super().__init__("gate_detection_node")
        self.bridge = CvBridge()

        self.declare_parameter("camera_frame_id", "auv4/front_cam_optical")
        self.declare_parameter("object_frame_id", "wesley_please_come_lab")
        self.declare_parameter("camera_info_topic", "/auv4/front_cam/color/camera_info")
        self.declare_parameter(
            "detections_topic", "/auv4/front_cam/color/image/yolo/detections"
        )

        self.camera_frame_id = (
            self.get_parameter("camera_frame_id").get_parameter_value().string_value
        )
        self.object_frame_id = (
            self.get_parameter("object_frame_id").get_parameter_value().string_value
        )
        camera_info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        detections_topic = (
            self.get_parameter("detections_topic").get_parameter_value().string_value
        )

        valid, front_camera_info = wait_for_message(CameraInfo, self, camera_info_topic)
        if not valid:
            raise ValueError("Failed to get camera info")
        else:
            self.camera = PinholeCamera.from_camera_info(
                front_camera_info, rectified=False
            )

        self.detections_sub = self.create_subscription(
            DetectionArray, detections_topic, self.detections_callback, 1
        )
        self.pose_publisher = self.create_publisher(PoseStamped, "/auv4/gate/pose", 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()

    def detections_callback(self, msg: DetectionArray):
        required_objects = 2
        image_points, object_points = [], []

        best_detections = get_best_detections_per_class(
            msg, ["gate_sides_left", "gate_sides_right", "gate_center"]
        )
        num_detected_objects = len(best_detections.keys())
        if num_detected_objects < required_objects:
            self.get_logger().warn(
                f"Insufficient detected objects. Received: {num_detected_objects}, require: {required_objects}."
            )
            return

        for class_name, detection in best_detections.items():
            mask = detection.mask
            curr_object_points = object_points_dict[class_name]
            curr_image_points = [attrgetter("x", "y")(point) for point in mask.data]

            self.get_logger().info(f"{object_points}, {curr_object_points}")
            object_points.extend(curr_object_points)
            image_points.extend(curr_image_points)

            # object_points = np.concatenate([object_points, curr_object_points])
            # image_points = np.concatenate([image_points, curr_image_points])

        try:
            rvec, tvec = get_object_pose(self.camera, object_points, image_points)
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.squeeze()
        except Exception as e:
            self.get_logger().warn(f"Pose estimation failed: {e}")
            return

        try:
            covariance = estimate_covariance(object_points, rvec, tvec, self.camera)
        except np.linalg.LinAlgError as e:
            self.get_logger().warn(
                f"Covariance estimation failed, inversion for FIM matrix failed: {e}"
            )
            return

        # self.get_logger().info(
        #     f"Pose estimation std dev: {np.sqrt(covariance.diagonal())}"
        # )

        try:
            qx, qy, qz, qw = mat2quat(R)
        except np.linalg.LinAlgError as e:
            self.get_logger().warn(f"Error in mat2quat, failed to convert R: {e}")
            return

        pose = PoseWithCovarianceStamped()
        pose.header = msg.header
        pose.header.frame_id = self.camera_frame_id
        pose.pose.pose.position = Point(x=t[0], y=t[1], z=t[2])
        pose.pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        pose.pose.covariance = covariance.flatten().tolist()
        self.pose_publisher.publish(pose)

        transform_stamped = TransformStamped()
        transform_stamped.header = msg.header
        transform_stamped.child_frame_id = "wesley_please_come_lab"
        transform_stamped.transform.translation = Vector3(x=t[0], y=t[1], z=t[2])
        transform_stamped.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30), node=self)
        self.br = tf2_ros.StaticTransformBroadcaster(self)


def main(args=None):
    rclpy.init(args=args)
    node = GatePoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
