import message_filters
import numpy as np
import rclpy
import tf2_ros
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.wait_for_message import wait_for_message
from yolo_msgs.msg import DetectionArray

from pose_estimator.utils.detections import (
    get_detection_centroid,
    get_top_k_detections_per_class,
)
from pose_estimator.utils.pose_estimator import backproject_pixel
from pose_estimator.utils.pose_estimator_node import PoseEstimatorNode


class TrashPoseEstimatorDepthFromOdom(PoseEstimatorNode):
    def __init__(self):
        super().__init__("trash_pose_estimator_depth_from_odom_node")

        input_detections_topic = (
            self.declare_parameter(
                "input_detections_topic", rclpy.Parameter.Type.STRING
            )
            .get_parameter_value()
            .string_value
        )
        self.input_odom_topic = (
            self.declare_parameter("input_odom_topic", rclpy.Parameter.Type.STRING)
            .get_parameter_value()
            .string_value
        )
        self.odom_frame = (
            self.declare_parameter("odom_frame", rclpy.Parameter.Type.STRING)
            .get_parameter_value()
            .string_value
        )
        self.camera_frame = (
            self.declare_parameter("camera_frame", rclpy.Parameter.Type.STRING)
            .get_parameter_value()
            .string_value
        )
        self.trash_frame_clustered = (
            self.declare_parameter("trash_frame_clustered", rclpy.Parameter.Type.STRING)
            .get_parameter_value()
            .string_value
        )

        detections_subscription = message_filters.Subscriber(
            self,
            DetectionArray,
            input_detections_topic,
            qos_profile=qos_profile_sensor_data,
        )
        odom_subscription = message_filters.Subscriber(
            self, Odometry, self.input_odom_topic, qos_profile=qos_profile_sensor_data
        )
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [detections_subscription, odom_subscription], 20, slop=0.1
        )
        self.time_synchronizer.registerCallback(self.detections_callback)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.has_setup = False

    def setup(self) -> bool:
        valid, odom_msg = wait_for_message(Odometry, self, self.input_odom_topic)
        if not valid:
            raise ValueError("Failed to get camera info")
        odom_msg: Odometry

        # TODO: @advaypakhale use odom time instead of current time for transforms
        # odom_time = Time(
        #     seconds=odom_msg.header.stamp.sec, nanoseconds=odom_msg.header.stamp.nanosec
        # )

        if not self.tf_buffer.can_transform(
            self.camera_frame, self.trash_frame_clustered, Time()
        ) or not self.tf_buffer.can_transform(
            self.odom_frame, self.camera_frame, Time()
        ):
            self.get_logger().info("Waiting for transforms...")
            return False

        object_to_camera_transform = self.tf_buffer.lookup_transform(
            self.camera_frame, self.trash_frame_clustered, Time()
        )
        camera_to_robot_transform = self.tf_buffer.lookup_transform(
            self.odom_frame, self.camera_frame, Time()
        )

        object_to_camera_depth = object_to_camera_transform.transform.translation.z
        self.camera_to_robot_depth = camera_to_robot_transform.transform.translation.z
        robot_to_surface_depth = odom_msg.pose.pose.position.z
        self.object_to_surface_depth = (
            object_to_camera_depth + self.camera_to_robot_depth + robot_to_surface_depth
        )

        self.get_logger().info(
            f"{object_to_camera_depth} + {self.camera_to_robot_depth} + {robot_to_surface_depth} = {self.object_to_surface_depth}"
        )
        return True

    def detections_callback(
        self, detection_array_msg: DetectionArray, odom_msg: Odometry
    ):
        if not self.has_setup:
            self.get_logger().info("Setting up the pose estimator...")
            success = self.setup()
            if not success:
                return
            self.has_setup = True

        start_time = self.get_clock().now()

        robot_to_surface_depth = odom_msg.pose.pose.position.z
        object_to_robot_depth = self.object_to_surface_depth - robot_to_surface_depth
        object_to_camera_depth = object_to_robot_depth - self.camera_to_robot_depth
        self.get_logger().info(f"Object to camera depth: {object_to_camera_depth} m")

        best_detections = get_top_k_detections_per_class(
            detection_array_msg,
            {"bottle": 2, "ladle": 2, "pink_bucket": 1, "yellow_bucket": 1},
        )
        header = detection_array_msg.header
        counts = {"bottle": 0, "ladle": 0}

        # We are unable to estimate orientation, so we set it to identity
        rvec = np.zeros((3, 1), dtype=np.float32)

        for class_name, detections in best_detections.items():
            # Sort detections by their track ID
            detections = sorted(detections, key=lambda d: d.id)

            for detection in detections:
                detection_centroid = get_detection_centroid(detection)
                X, Y = backproject_pixel(
                    detection_centroid[0],
                    detection_centroid[1],
                    object_to_camera_depth,
                    self.camera.camera_matrix(),
                )
                tvec = np.array([X, Y, object_to_camera_depth]).reshape((3, 1))

                if class_name in counts:
                    frame_name = f"{class_name}_2_{counts[class_name]}"
                    counts[class_name] += 1
                else:
                    frame_name = class_name

                self.publish_transform(tvec, rvec, header, frame_name)

        end_time = self.get_clock().now()
        elapsed_time = (
            end_time - start_time
        ).nanoseconds / 1e6  # Convert to milliseconds
        self.get_logger().info(
            f"Processed {len(detection_array_msg.detections)} detections in {elapsed_time:.2f} ms"
        )


def main(args=None):
    rclpy.init(args=args)
    node = TrashPoseEstimatorDepthFromOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
