from operator import attrgetter

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import (
    Point,
    PoseWithCovarianceStamped,
    Quaternion,
    TransformStamped,
    Vector3,
)
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import CameraInfo
from transforms3d.quaternions import mat2quat
from yolo_msgs.msg import DetectionArray

from pose_estimator.utils.detections import (
    OBJECT_POINTS_DICT,
    get_best_detections_per_class,
    polygon_to_obb,
)
from pose_estimator.utils.PinholeCamera import PinholeCamera
from pose_estimator.utils.pose_estimator import estimate_covariance, get_object_pose


class GatePoseEstimator(Node):
    def __init__(self):
        super().__init__("gate_pose_estimator_node")
        self.bridge = CvBridge()

        self.declare_parameter("camera_frame_id", "auv4/front_cam_optical")
        self.declare_parameter("object_frame_id", "wesley_please_come_lab")
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
            curr_object_points = OBJECT_POINTS_DICT[class_name]
            object_points.extend(curr_object_points)

            mask = detection.mask
            mask_points = [attrgetter("x", "y")(point) for point in mask.data]
            curr_image_points = polygon_to_obb(mask_points)
            image_points.extend(curr_image_points)

        object_points = np.array(object_points)
        image_points = np.array(image_points)

        assert (
            object_points.shape[0] == image_points.shape[0]
        ), "Number of object points and image points must match"

        try:
            rvec, tvec = get_object_pose(
                self.camera, object_points, image_points, max_reprojection_error=100
            )
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
