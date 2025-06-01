import cv2
import numpy as np
import rclpy
import tf2_ros
import tf_transformations
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import CameraInfo
from transforms3d.quaternions import mat2quat
from yolo_msgs.msg import DetectionArray

from pose_estimator.utils.detections import (
    OBJECT_POINTS_DICT,
    filter_detections_by_num_points,
    get_best_detections_per_class,
    get_detection_obb,
    match_polygon_points_sequence,
)
from pose_estimator.utils.PinholeCamera import PinholeCamera
from pose_estimator.utils.pose_estimator import estimate_covariance, get_object_pose
from pose_estimator.utils.ros_messages import (
    get_pose_with_covariance_stamped,
    get_transform_stamped,
)


class GatePoseEstimator(Node):
    def __init__(self):
        super().__init__("gate_pose_estimator_node")
        self.bridge = CvBridge()

        self.declare_parameter("object_frame_id", "gate")
        self.declare_parameter("camera_info_topic", "camera_info")
        self.declare_parameter("detections_topic", "yolo/detections")

        self.object_frame_id = (
            self.get_parameter("object_frame_id").get_parameter_value().string_value
        )

        camera_info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        valid, camera_info = wait_for_message(CameraInfo, self, camera_info_topic)
        if not valid:
            raise ValueError("Failed to get camera info")
        else:
            camera_info: CameraInfo
            self.camera = PinholeCamera.from_camera_info(camera_info, rectified=False)
            self.camera_frame_id = camera_info.header.frame_id

        detections_topic = (
            self.get_parameter("detections_topic").get_parameter_value().string_value
        )
        self.detections_sub = self.create_subscription(
            DetectionArray, detections_topic, self.detections_callback, 1
        )
        self.pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, "/auv4/gate/pose", 10
        )
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

    def detections_callback(self, msg: DetectionArray):
        # We require at least 3 points for polygon creation
        filtered_detections = filter_detections_by_num_points(msg, 3)

        # Only include relevant classes
        relevant_classes = ["gate_sides_left", "gate_sides_right", "gate_center"]
        required_objects = 2

        best_detections = get_best_detections_per_class(
            filtered_detections, relevant_classes
        )
        num_detected_objects = len(best_detections.keys())
        if num_detected_objects < required_objects:
            self.get_logger().warn(
                f"""Insufficient detected objects.
                Received: {num_detected_objects}, require: {required_objects}."""
            )
            return

        # Match image points and object points
        detections = list(best_detections.values())
        polygon_objects = [
            OBJECT_POINTS_DICT[detection.class_name] for detection in detections
        ]
        detected_polygons = list(map(get_detection_obb, detections))
        object_points, image_points = match_polygon_points_sequence(
            polygon_objects, detected_polygons
        )
        object_points = np.hstack(
            [object_points, np.zeros((object_points.shape[0], 1))]
        )

        assert object_points.shape[0] == image_points.shape[0], (
            "Number of object points and image points must match"
        )

        try:
            rvec, tvec, inliers = get_object_pose(
                self.camera, object_points, image_points, max_reprojection_error=100
            )

            if inliers is None:
                raise ValueError("No inliers found during pose estimation.")

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
            q = mat2quat(R)
        except np.linalg.LinAlgError as e:
            self.get_logger().warn(f"Error in mat2quat, failed to convert R: {e}")
            return

        # Apply a 180-degree rotation around the x-axis
        # TODO: Find out why this is needed
        q_rot_x_180 = tf_transformations.quaternion_from_euler(np.pi, 0, 0)
        q_rotated = tf_transformations.quaternion_multiply(q_rot_x_180, q)

        pose = get_pose_with_covariance_stamped(
            msg.header, t, q_rotated, covariance.flatten().tolist()
        )
        self.pose_publisher.publish(pose)

        transform_stamped = get_transform_stamped(
            msg.header, self.object_frame_id, t, q_rotated
        )
        self.tf_broadcaster.sendTransform(transform_stamped)


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
