import cv2
import numpy as np
import tf2_ros
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from transforms3d.quaternions import mat2quat

from pose_estimator.utils.PinholeCamera import PinholeCamera
from pose_estimator.utils.ros_messages import get_transform_stamped


class PoseEstimatorNode(Node):
    """Base pose estimator node that provides functions to fetch camera info and publish transforms
    from translation and rotation vectors."""

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

        # Transforms broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

    def get_translation_quaternion(self, tvec: np.ndarray, rvec: np.ndarray):
        """Convert translation vector and rotation vector to translation and quaternion.

        Args:
            tvec: Translation vector (3x1 NumPy array).
            rvec: Rotation vector (3x1 NumPy array).

        Raises:
            Exception: If Rodrigues conversion fails.
            np.linalg.LinAlgError: If mat2quat fails.

        Returns:
            t: Translation vector (3-element list).
            q: Quaternion (4-element list in ROS format [x, y, z, w]).
        """
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.squeeze()
        q = mat2quat(R)  # [w, x, y, z]
        q = [q[1], q[2], q[3], q[0]]  # ROS uses [x, y, z, w]
        return t, q

    def publish_transform(
        self,
        tvec: np.ndarray,
        rvec: np.ndarray,
        header: Header,
        object_frame_id: str,
    ):
        """
        Publish the translation vector and rotation vector as a Transform.
        """
        try:
            t, q = self.get_translation_quaternion(tvec, rvec)
        except Exception as e:
            # TODO: Not sure if an exception can occur here.
            self.get_logger().warn(f"Rodrigues conversion failed: {e}")
            return
        except np.linalg.LinAlgError as e:
            self.get_logger().warn(f"Error in mat2quat, failed to convert R: {e}")
            return

        transform_stamped = get_transform_stamped(header, object_frame_id, t, q)
        self.tf_broadcaster.sendTransform(transform_stamped)
