"""Publish both IPPE pose solutions for the planar dock target."""

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import qos_profile_sensor_data
from yolo_msgs.msg import DetectionArray

from pose_estimator.dock_pnp_pose_estimator_node import (
    DockPnpPoseEstimator,
    estimate_planar_dock_poses,
    get_dock_correspondences,
)


class DockPlanarPnpPoseEstimator(DockPnpPoseEstimator):
    """Estimate a planar dock and publish its two IPPE candidates."""

    def __init__(self):
        super().__init__(
            node_name="dock_planar_pnp_pose_estimator_node",
            default_pose_topic="planar_pnp/pose_1",
        )
        pose_topic = (
            self.declare_parameter("pose_topic_2", "planar_pnp/pose_2")
            .get_parameter_value()
            .string_value
        )
        self._second_dock_pose_publisher = self.create_publisher(
            PoseStamped, pose_topic, qos_profile_sensor_data
        )

    def detections_callback(self, msg: DetectionArray):
        try:
            object_points, image_points = get_dock_correspondences(msg)
            poses = estimate_planar_dock_poses(
                self.camera, object_points, image_points
            )
        except Exception as error:
            self.get_logger().warn(f"Planar dock pose estimation failed: {error}")
            return

        publishers = (
            self._dock_pose_publisher,
            self._second_dock_pose_publisher,
        )
        for (rvec, tvec), publisher in zip(poses, publishers):
            self._publish_pose(tvec, rvec, msg.header, publisher)


def main(args=None):
    rclpy.init(args=args)
    node = DockPlanarPnpPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
