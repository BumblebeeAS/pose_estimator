from operator import attrgetter

import message_filters
import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from message_filters import TimeSynchronizer
from rclpy.qos import qos_profile_sensor_data
from transforms3d.euler import euler2quat, quat2euler
from yolo_msgs.msg import DetectionArray

from pose_estimator.utils.detections import (
    get_detection_obb,
    get_top_k_detections_per_class,
)
from pose_estimator.utils.pose_estimator import get_detection_centroid_position
from pose_estimator.utils.pose_estimator_node import PoseEstimatorNode
from pose_estimator.utils.ros_messages import get_transform_stamped

_FRAME_SUFFIX = "from_table"


class TrashPoseEstimatorDepthFromTable(PoseEstimatorNode):
    def __init__(self):
        super().__init__("trash_pose_estimator_depth_from_table_node")

        input_detections_topic = (
            self.declare_parameter(
                "input_detections_topic", rclpy.Parameter.Type.STRING
            )
            .get_parameter_value()
            .string_value
        )
        input_track_detections_topic = (
            self.declare_parameter(
                "input_track_detections_topic", rclpy.Parameter.Type.STRING
            )
            .get_parameter_value()
            .string_value
        )
        table_pose_topic = (
            self.declare_parameter("table_pose_topic", "table/pose")
            .get_parameter_value()
            .string_value
        )

        detections_subscription = message_filters.Subscriber(
            self,
            DetectionArray,
            input_detections_topic,
            qos_profile=qos_profile_sensor_data,
        )
        track_detections_subscription = message_filters.Subscriber(
            self,
            DetectionArray,
            input_track_detections_topic,
            qos_profile=qos_profile_sensor_data,
        )
        table_pose_subscription = message_filters.Subscriber(
            self,
            PoseWithCovarianceStamped,
            table_pose_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.time_synchronizer = TimeSynchronizer(
            [
                detections_subscription,
                track_detections_subscription,
                table_pose_subscription,
            ],
            10,
        )
        self.time_synchronizer.registerCallback(self.detections_callback)

    def detections_callback(
        self,
        detection_array_msg: DetectionArray,
        track_detection_array_msg: DetectionArray,
        table_pose_msg: PoseWithCovarianceStamped,
    ):
        object_depth = table_pose_msg.pose.pose.position.z

        best_table_detections = get_top_k_detections_per_class(
            detection_array_msg,
            {"pink_bucket": 1, "yellow_bucket": 1},
        )
        best_track_detections = get_top_k_detections_per_class(
            track_detection_array_msg,
            {"bottle": 1, "ladle": 1},
        )
        best_detections = best_table_detections | best_track_detections
        best_detections = {k: v[0] for k, v in best_detections.items() if len(v) > 0}
        header = detection_array_msg.header

        for class_name, detection in best_detections.items():
            translation = get_detection_centroid_position(
                detection, self.camera, object_depth
            )

            if class_name in ["bottle", "ladle"]:
                # For trash objects, we want the robot to align with the long edge of the object

                angle, _ = get_detection_obb(detection)
                # Rotate 90 deg more because the object frame's x-axis points along the
                # object's short edge but we want the robot to align along the long edge
                angle += 90
                angle = (angle + 180) % 360 - 180
                object_yaw = np.deg2rad(angle)

            else:
                # To be able to deposit the objects, the long edge of the grabber needs to be aligned
                # with the long edge of the bucket. This leaves a 180 deg rotation ambiguity. However,
                # since we want the thrusters to be as far from the table as possible to minimize
                # downwash and the grabber is at the front left of the robot, we align the robot's right
                # away from the table.

                table_quat = attrgetter("w", "x", "y", "z")(
                    table_pose_msg.pose.pose.orientation
                )
                table_yaw = quat2euler(table_quat, axes="sxyz")[2]

                # The table pose is defined such that the yellow bucket is on the left
                # and the pink bucket is on the right
                if class_name == "yellow_bucket":
                    object_yaw = table_yaw - np.pi / 2
                else:
                    object_yaw = table_yaw + np.pi - np.pi / 2

            # Returns quat in ROS format [x, y, z, w]
            qw, qx, qy, qz = euler2quat(0, 0, object_yaw)
            quat = [qx, qy, qz, qw]

            if class_name in ["bottle", "ladle"]:
                frame_name = f"{class_name}_0/{_FRAME_SUFFIX}"
            else:
                frame_name = f"{class_name}/{_FRAME_SUFFIX}"

            transform_stamped = get_transform_stamped(
                header, frame_name, translation, quat
            )
            self.tf_broadcaster.sendTransform(transform_stamped)


def main(args=None):
    rclpy.init(args=args)
    node = TrashPoseEstimatorDepthFromTable()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
