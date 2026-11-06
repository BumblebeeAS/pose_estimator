import logging

import rclpy
from bb_perception_msgs.msg import PointCorrespondencesStamped
from rclpy.qos import qos_profile_sensor_data

from image_processing.utils.ros_np_multiarray import to_numpy_f64
from pose_estimator.utils.pose_estimator import (
    filter_by_homography,
    get_object_pose,
    refine_object_pose,
)
from pose_estimator.utils.pose_estimator_node import PoseEstimatorSinglePosePubNode


class PointsPoseEstimator(PoseEstimatorSinglePosePubNode):

    def __init__(self):
        super().__init__("points_pose_estimator")

        input_points_topic = (
            self.declare_parameter("input_points_topic", "point_correspondences")
            .get_parameter_value()
            .string_value
        )

        # Enable for Y-up boards viewed in Y-down image coordinates.
        self.homography_flip_object_y = self.declare_parameter("homography_flip_object_y", False).value

        self.min_points = self.declare_parameter("min_points", 4).value
        if self.min_points < 4:
            raise ValueError("min_points must be at least 4")

        self.max_reprojection_error = self.declare_parameter("max_reprojection_error", 2.0).value

        self.point_subscriber = self.create_subscription(
            PointCorrespondencesStamped,
            input_points_topic,
            self.point_correspondences_callback,
            qos_profile_sensor_data,
        )

    def point_correspondences_callback(self, msg: PointCorrespondencesStamped):
        object_points = to_numpy_f64(msg.object_points)
        image_points = to_numpy_f64(msg.image_points)

        if min(len(object_points), len(image_points)) < self.min_points:
            self.get_logger().warning(
                f"Not enough point correspondences: {len(object_points)} object points "
                f"and {len(image_points)} image points; need {self.min_points}",
                throttle_duration_sec=5.0,
            )
            return

        try:
            # Match board Y-up to image Y-down for MAGSAC; keep PnP in board coordinates.
            if self.homography_flip_object_y:
                object_points = object_points.copy()
                object_points[:, 1] *= -1
            object_points, image_points = filter_by_homography(object_points, image_points)
            if self.homography_flip_object_y:
                object_points[:, 1] *= -1  # Restore board coordinates for PnP.
            if len(object_points) < self.min_points:
                return

            # This step gives a rough estimate for pose refinement and
            # allows for quick termination if no inliers are found. This is useful
            # when there are few point correspondences and homography estimation
            # cannot determine if the points are inliers or not.

            rvec, tvec, inliers = get_object_pose(
                self.camera, object_points, image_points,
                max_reprojection_error=self.max_reprojection_error
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
            self.get_logger().warning(f"Pose estimation failed: {e}")
            return

        self.publish_data(tvec, rvec, object_points, msg.header, self.object_frame_id)


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
