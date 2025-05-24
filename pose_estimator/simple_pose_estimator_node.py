import logging

import cv2
import numpy as np
import rclpy
import tf2_ros
from bb_perception_msgs.msg import PointCorrespondencesStamped
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
from utils.ros_np_multiarray import to_numpy_f64

from pose_estimator.utils.PinholeCamera import PinholeCamera


class SimplePoseEstimator(Node):

    def __init__(self):
        super().__init__("pose_estimator")

        self.declare_parameter("camera_info_topic", "camera_info")
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

        self.br = tf2_ros.TransformBroadcaster(self)

        self.point_subscriber = self.create_subscription(
            PointCorrespondencesStamped,
            "image_matching/point_correspondences",
            self.point_correspondences_callback,
            1,
        )

        self.pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, "image_matching/pose", 1
        )

    def point_correspondences_callback(self, msg: PointCorrespondencesStamped):
        # TODO: Filter using clustering or Kalman
        object_points = to_numpy_f64(msg.object_points)
        image_points = to_numpy_f64(msg.image_points)

        if object_points.shape[0] < 4 or image_points.shape[0] < 4:
            self.get_logger().warn(
                f"""Not enough point correspondences:
                {object_points.shape[0]} object points and {image_points.shape[0]} image points"""
            )
            return

        object_points, image_points = filter_by_homography(object_points, image_points)

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
        transform_stamped.child_frame_id = msg.object_frame_id
        transform_stamped.transform.translation = Vector3(x=t[0], y=t[1], z=t[2])
        transform_stamped.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        self.br.sendTransform(transform_stamped)


def main(args=None):
    logging.basicConfig(level=logging.INFO)
    rclpy.init(args=args)
    node = SimplePoseEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
