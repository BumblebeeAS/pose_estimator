import logging

import cv2
import numpy as np
import rclpy
import tf2_ros
import tf_transformations
from bb_perception_msgs.msg import PointCorrespondencesStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import CameraInfo
from transforms3d.quaternions import mat2quat
from utils.ros_np_multiarray import to_numpy_f64

from pose_estimator.utils.PinholeCamera import PinholeCamera
from pose_estimator.utils.pose_estimator import (
    estimate_covariance,
    filter_by_homography,
    get_object_pose,
    refine_object_pose,
)
from pose_estimator.utils.ros_messages import (
    get_pose_with_covariance_stamped,
    get_transform_stamped,
)


class PointsPoseEstimator(Node):

    def __init__(self):
        super().__init__("points_pose_estimator")

        camera_info_topic = (
            self.declare_parameter("camera_info_topic", "camera_info")
            .get_parameter_value()
            .string_value
        )
        input_points_topic = (
            self.declare_parameter("input_points_topic", "point_correspondences")
            .get_parameter_value()
            .string_value
        )
        output_pose_topic = (
            self.declare_parameter("output_pose_topic", "pose")
            .get_parameter_value()
            .string_value
        )

        valid, camera_info = wait_for_message(CameraInfo, self, camera_info_topic)
        if not valid:
            raise ValueError("Failed to get camera info")
        else:
            camera_info: CameraInfo
            self.camera = PinholeCamera.from_camera_info(camera_info, rectified=False)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.point_subscriber = self.create_subscription(
            PointCorrespondencesStamped,
            input_points_topic,
            self.point_correspondences_callback,
            1,
        )

        self.pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, output_pose_topic, 1
        )

    def point_correspondences_callback(self, msg: PointCorrespondencesStamped):
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
            # This step gives a rough estimate for pose refinement and
            # allows for quick termination if no inliers are found. This is useful
            # when there are few point correspondences and homography estimation
            # cannot determine if the points are inliers or not.

            rvec, tvec, inliers = get_object_pose(
                self.camera, object_points, image_points
            )

            if inliers is None:
                raise ValueError("No inliers found during pose estimation.")

            # We use don't filter by RANSAC inliers before refinement because RANSAC filtering
            # is too strict, resulting in a noisy pose estimate. Filtering is already done
            # by the homography.

            rvec, tvec = refine_object_pose(
                self.camera, object_points, image_points, rvec, tvec
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
            msg.header, msg.object_frame_id, t, q_rotated
        )
        self.tf_broadcaster.sendTransform(transform_stamped)


def main(args=None):
    logging.basicConfig(level=logging.INFO)
    rclpy.init(args=args)
    node = PointsPoseEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
