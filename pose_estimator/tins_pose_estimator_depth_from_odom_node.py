import message_filters
import numpy as np
import rclpy
import tf2_ros
from bb_perception_msgs.srv import TrashToggleFrame
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.wait_for_message import wait_for_message
from yolo_msgs.msg import DetectionArray

from pose_estimator.utils.detections import (
    get_detection_obb,
    get_top_k_detections_per_class,
)
from pose_estimator.utils.pose_estimator import backproject_pixel
from pose_estimator.utils.pose_estimator_node import PoseEstimatorPosePubNode


class TinsPoseEstimatorDepthFromOdom(PoseEstimatorPosePubNode):
    def __init__(self):
        super().__init__("tins_pose_estimator_depth_from_odom_node")

        input_track_detections_topic = (
            self.declare_parameter(
                "input_track_detections_topic", rclpy.Parameter.Type.STRING
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
        time_sync_slop = (
            self.declare_parameter("time_sync_slop", 0.2)
            .get_parameter_value()
            .double_value
        )

        self.camera_frame = self.camera.frame_id

        track_detections_subscription = message_filters.Subscriber(
            self,
            DetectionArray,
            input_track_detections_topic,
            qos_profile=qos_profile_sensor_data,
        )
        odom_subscription = message_filters.Subscriber(
            self, Odometry, self.input_odom_topic, qos_profile=qos_profile_sensor_data
        )
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [track_detections_subscription, odom_subscription],
            20,
            slop=time_sync_slop,
        )
        self.time_synchronizer.registerCallback(self.detections_callback)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.toggle_frame_service = self.create_service(
            TrashToggleFrame,
            "toggle_frame_clustered",
            self.toggle_frame_callback,
        )
        self.is_enabled = False

        self._red_tins_publisher = self.create_publisher(
            PoseStamped, "red_tin_0/from_odom/pose", qos_profile_sensor_data
        )
        self._green_tins_publisher = self.create_publisher(
            PoseStamped, "green_tin_0/from_odom/pose", qos_profile_sensor_data
        )
        self._blue_tins_publisher = self.create_publisher(
            PoseStamped, "blue_tin_0/from_odom/pose", qos_profile_sensor_data
        )

    @property
    def pose_publishers(self):
        return {
            "red_tin_0": self._red_tins_publisher,
            "green_tin_0": self._green_tins_publisher,
            "blue_tin_0": self._blue_tins_publisher,
        }

    def toggle_frame_callback(
        self, request: TrashToggleFrame.Request, response: TrashToggleFrame.Response
    ):
        if not request.enable:
            self.is_enabled = False
            response.success = True
            return response

        self.trash_frame_clustered = request.trash_frame_clustered

        is_setup_success = self.setup()

        if not is_setup_success:
            message = "Failed to setup tins pose estimator from odom."
            self.get_logger().error(message)
            response.success = False
            response.message = message
            return response

        self.is_enabled = True
        response.success = True
        return response

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
            # self.get_logger().info("Waiting for transforms...")
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
        self,
        track_detection_array_msg: DetectionArray,
        odom_msg: Odometry,
    ):
        if not self.is_enabled:
            return

        start_time = self.get_clock().now()

        robot_to_surface_depth = odom_msg.pose.pose.position.z
        object_to_robot_depth = self.object_to_surface_depth - robot_to_surface_depth
        object_to_camera_depth = object_to_robot_depth - self.camera_to_robot_depth
        self.get_logger().info(f"Object to camera depth: {object_to_camera_depth} m")

        best_track_detections = get_top_k_detections_per_class(
            track_detection_array_msg,
            {"red_tin": 1, "green_tin": 1, "blue_tin": 1},
        )
        best_detections = {
            k: v[0] for k, v in best_track_detections.items() if len(v) > 0
        }
        header = track_detection_array_msg.header

        # We are unable to estimate orientation, so we set it to identity
        rvec = np.zeros((3, 1), dtype=np.float32)

        for class_name, detection in best_detections.items():
            _, obb = get_detection_obb(detection)
            centroid = np.mean(obb, axis=0)
            X, Y = backproject_pixel(
                centroid[0], centroid[1], object_to_camera_depth, self.camera
            )
            translation = np.array([X, Y, object_to_camera_depth])
            tvec = translation.reshape((3, 1))

            frame_name = f"{class_name}_0"

            self.publish_data(tvec, rvec, obb, header, frame_name)

        end_time = self.get_clock().now()
        elapsed_time = (end_time - start_time).nanoseconds / 1e6
        # self.get_logger().info(
        #     f"Processed {len(detection_array_msg.detections)} detections in {elapsed_time:.2f} ms"
        # )


def main(args=None):
    rclpy.init(args=args)
    node = TinsPoseEstimatorDepthFromOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
