# Pose estimator node base class and implementations for different publishing formats.
# It is recommended to publish a pose message instead of a transform when the pose is needed in real-time.
# This is because normal ROS publishers perform better than transform broadcasters / lookups for high-frequency data.

from abc import ABC, abstractmethod

import cv2
import numpy as np
import tf2_ros
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from transforms3d.quaternions import mat2quat

from pose_estimator.utils.PinholeCamera import PinholeCamera
from pose_estimator.utils.pose_estimator import estimate_covariance
from pose_estimator.utils.ros_messages import (
    get_pose_stamped,
    get_pose_with_covariance_stamped,
    get_transform_stamped,
)


def get_translation_quaternion(tvec: np.ndarray, rvec: np.ndarray):
    """Convert translation vector and rotation vector to translation and quaternion.

    Args:
        tvec: Translation vector (3x1 NumPy array).
        rvec: Rotation vector (3x1 NumPy array).

    Raises:
        np.linalg.LinAlgError: If mat2quat fails.
        Exception: If Rodrigues conversion fails.

    Returns:
        t: Translation vector (3-element list).
        q: Quaternion (4-element list in ROS format [x, y, z, w]).
    """
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.squeeze()
    q = mat2quat(R)  # [w, x, y, z]
    q = [q[1], q[2], q[3], q[0]]  # ROS uses [x, y, z, w]
    return t, q


class PoseEstimatorNode(Node, ABC):
    """Base pose estimator node that provides functions to fetch camera info and
    get translation and quaternions from translation and rotation vectors."""

    def __init__(self, name):
        super().__init__(name)

        # Parameters
        camera_info_topic = (
            self.declare_parameter("camera_info_topic", "camera_info")
            .get_parameter_value()
            .string_value
        )
        is_image_rectified = (
            self.declare_parameter("is_image_rectified", False)
            .get_parameter_value()
            .bool_value
        )

        # Get camera info
        valid, camera_info = wait_for_message(CameraInfo, self, camera_info_topic)
        if not valid:
            raise ValueError("Failed to get camera info")
        else:
            camera_info: CameraInfo
            self.camera = PinholeCamera.from_camera_info(
                camera_info, rectified=is_image_rectified
            )
        self.camera: PinholeCamera

    @abstractmethod
    def publish_data(
        self,
        tvec: np.ndarray,
        rvec: np.ndarray,
        object_points: np.ndarray,
        header: Header,
        object_frame_id: str,
    ) -> None:
        """Publish pose data.

        Args:
            tvec: Translation vector (3x1 NumPy array).
            rvec: Rotation vector (3x1 NumPy array).
            object_points: 3D object points used for pose estimation (Nx3 NumPy array).
            header: Header for the published messages.
            object_frame_id: Frame ID for the object.
        """
        pass


class PoseEstimatorTransformPubNode(PoseEstimatorNode):
    """Publishes only the transform."""

    def __init__(self, name):
        super().__init__(name)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

    def publish_data(
        self,
        tvec: np.ndarray,
        rvec: np.ndarray,
        object_points: np.ndarray,
        header: Header,
        object_frame_id: str,
    ) -> None:
        try:
            t, q = get_translation_quaternion(tvec, rvec)
        except np.linalg.LinAlgError as e:
            self.get_logger().warn(f"Error in mat2quat, failed to convert R: {e}")
            return
        except Exception as e:
            # TODO: Not sure if an exception can occur here.
            self.get_logger().warn(f"Rodrigues conversion failed: {e}")
            return

        transform_stamped = get_transform_stamped(header, object_frame_id, t, q)
        self.tf_broadcaster.sendTransform(transform_stamped)


class PoseEstimatorPosePubNode(PoseEstimatorNode):
    """Publishes only PoseStamped (no TF)."""

    def __init__(self, name):
        super().__init__(name)

        output_pose_topic = (
            self.declare_parameter("output_pose_topic", "pose_stamped")
            .get_parameter_value()
            .string_value
        )

        self.pose_publisher = self.create_publisher(
            PoseStamped, output_pose_topic, qos_profile_sensor_data
        )

    def publish_data(
        self,
        tvec: np.ndarray,
        rvec: np.ndarray,
        object_points: np.ndarray,
        header: Header,
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

        pose = get_pose_stamped(header, t, q)
        self.pose_publisher.publish(pose)


class PoseEstimatorDataPubNode(PoseEstimatorNode):
    """
    Publish the translation vector and rotation vector as a Transform and a Pose.
    """

    def __init__(self, name):
        super().__init__(name)

        output_pose_topic = (
            self.declare_parameter("output_pose_topic", "pose")
            .get_parameter_value()
            .string_value
        )

        self.pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, output_pose_topic, qos_profile_sensor_data
        )

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

    def publish_data(
        self,
        tvec: np.ndarray,
        rvec: np.ndarray,
        object_points: np.ndarray,
        header: Header,
        object_frame_id: str,
    ) -> None:
        try:
            t, q = get_translation_quaternion(tvec, rvec)
        except np.linalg.LinAlgError as e:
            self.get_logger().warn(f"Error in mat2quat, failed to convert R: {e}")
            return
        except Exception as e:
            # TODO: Not sure if an exception can occur here.
            self.get_logger().warn(f"Rodrigues conversion failed: {e}")
            return

        transform_stamped = get_transform_stamped(header, object_frame_id, t, q)
        self.tf_broadcaster.sendTransform(transform_stamped)

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

        pose = get_pose_with_covariance_stamped(
            header, t, q, covariance.flatten().tolist()
        )
        self.pose_publisher.publish(pose)
