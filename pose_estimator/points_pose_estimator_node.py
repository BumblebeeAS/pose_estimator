import logging

import rclpy
from bb_perception_msgs.msg import PointCorrespondencesStamped
from rclpy.qos import qos_profile_sensor_data
from utils.ros_np_multiarray import to_numpy_f64

from pose_estimator.utils.pose_estimator import (
    filter_by_homography,
    get_object_pose,
    refine_object_pose,
)
from pose_estimator.utils.pose_estimator_node import PoseEstimatorNode


class PointsPoseEstimator(PoseEstimatorNode):

    def __init__(self):
        super().__init__("points_pose_estimator")

        input_points_topic = (
            self.declare_parameter("input_points_topic", "point_correspondences")
            .get_parameter_value()
            .string_value
        )

        self.point_subscriber = self.create_subscription(
            PointCorrespondencesStamped,
            input_points_topic,
            self.point_correspondences_callback,
            qos_profile_sensor_data,
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

        except Exception as e:
            self.get_logger().warn(f"Pose estimation failed: {e}")
            return

        self.publish_transform(tvec, rvec, msg.header, msg.object_frame_id)


def main(args=None):
    logging.basicConfig(level=logging.INFO)
    rclpy.init(args=args)
    node = PointsPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
