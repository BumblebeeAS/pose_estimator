import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Header

from pose_estimator.utils.pose_estimator import estimate_covariance
from pose_estimator.utils.pose_estimator_node import PoseEstimatorNode
from pose_estimator.utils.ros_messages import (
    get_pose_with_covariance_stamped,
    get_transform_stamped,
)


class PoseEstimatorAndPublisherNode(PoseEstimatorNode):
    """Base pose estimator node that provides functions to fetch camera info and publish transforms
    from translation and rotation vectors.

    We publish a pose in addition to a transform when the pose is needed in real-time. This is because
    normal ROS publishers perform better than transform broadcasters for high-frequency data.
    """

    def __init__(self, name):
        super().__init__(name)

        # Parameters
        output_pose_topic = (
            self.declare_parameter("output_pose_topic", "pose")
            .get_parameter_value()
            .string_value
        )

        # Pose publisher
        self.pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, output_pose_topic, qos_profile_sensor_data
        )

    def publish_data(
        self,
        tvec: np.ndarray,
        rvec: np.ndarray,
        object_points: np.ndarray,
        header: Header,
        object_frame_id: str,
    ):
        """
        Publish the translation vector and rotation vector as a Transform and a Pose.
        """
        try:
            t, q = self.get_translation_quaternion(tvec, rvec)
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
