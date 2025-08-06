from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import CameraInfo
from yolo_msgs.msg import Detection, DetectionArray

from pose_estimator.utils.detections import (
    get_detection_centroid,
    get_top_k_detections_per_class,
)
from pose_estimator.utils.PinholeCamera import PinholeCamera


@dataclass
class TrackObject:
    track_id: int = -1
    other_id: int = -1
    past_centroids: np.ndarray = np.zeros((1, 2))
    last_seen_time: float = -float("inf")


class PairedObjectsTrackerNode(Node):
    def __init__(self):
        super().__init__("paired_objects_tracker_node")

        # Parameters
        track_object_names = (
            self.declare_parameter(
                "track_object_names", rclpy.Parameter.Type.STRING_ARRAY
            )
            .get_parameter_value()
            .string_array_value
        )
        camera_info_topic = (
            self.declare_parameter("camera_info_topic", "camera_info")
            .get_parameter_value()
            .string_value
        )
        input_detections_topic = (
            self.declare_parameter(
                "input_detections_topic", rclpy.Parameter.Type.STRING
            )
            .get_parameter_value()
            .string_value
        )
        output_detections_topic = (
            self.declare_parameter(
                "output_detections_topic", rclpy.Parameter.Type.STRING
            )
            .get_parameter_value()
            .string_value
        )
        self.num_last_centroids = (
            self.declare_parameter("num_last_centroids", 100)
            .get_parameter_value()
            .integer_value
        )
        init_centroid = (
            self.declare_parameter("init_centroid", [0.0, 0.0])
            .get_parameter_value()
            .double_array_value
        )
        self.keep_track_alive_time = (
            self.declare_parameter("keep_track_alive_time", 5.0)
            .get_parameter_value()
            .double_value
        )

        # Using an initial centroid allows us to prioritize tracking objects
        # at a particular location in the image when there are no detections yet.
        self.track_dict: dict[str, TrackObject] = {
            track_object_name: TrackObject(np.array([init_centroid]))
            for track_object_name in track_object_names
        }

        # Get camera info
        valid, camera_info = wait_for_message(CameraInfo, self, camera_info_topic)
        if not valid:
            raise ValueError("Failed to get camera info")
        else:
            camera_info: CameraInfo
            self.camera = PinholeCamera.from_camera_info(camera_info, rectified=False)
        self.camera: PinholeCamera

        # Subscriptions and publishers
        self.input_detections_subscription = self.create_subscription(
            DetectionArray,
            input_detections_topic,
            self.detections_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self.output_detections_publisher = self.create_publisher(
            DetectionArray, output_detections_topic, qos_profile=qos_profile_sensor_data
        )

    def get_track_detections(
        self,
        best_detections: dict[str, list[Detection]],
    ) -> dict[str, Detection]:
        """Given not more than two objects per class, select the detection to track.

        We keep track of two track ids in each frame. The primary track id identifies the target
        the robot homes towards. The "other" track id exploits the fact that there are only two
        detections to track the primary target. In the current frame, if there is a detection
        with the primary track id from the previous frame, select that detection. Otherwise, if
        one of the detections has the "other" track id from the previous frame, skip the target
        for that frame if the last seen time is recent or select the other detection if the last
        seen time is too old. Finally, if both track ids are not present in the current frame,
        select the detection with the closest centroid to the target in the previous frame. Note
        that pixel distances are used.

        The idea of the "other" track id is that even if we lose track of the primary target, we
        are able to know that it is not the other object if the "other" track id is still valid.
        This allows us to re-track the primary target in subsequent frames if it reappears.

        Args:
            best_detections (dict[str, list[Detection]]): Dictionary of best detections per class.

        Returns:
            dict[str, Detection]: Dictionary of tracked detections.
        """
        track_detections = {}
        curr_time = self.get_clock().now().nanoseconds * 1e-9

        for class_name, detections in best_detections.items():
            if len(detections) == 0:
                continue

            if class_name not in self.track_dict:
                continue

            detections_by_id = {d.id: d for d in detections}
            curr_id = self.track_dict[class_name].track_id
            other_id = self.track_dict[class_name].other_id
            past_centroids = self.track_dict[class_name].past_centroids
            last_seen_time = self.track_dict[class_name].last_seen_time

            self.get_logger().info("Processing class: " + class_name)
            self.get_logger().info(f"Current track id: {curr_id}, Other id: {other_id}")
            self.get_logger().info(f"Last seen time: {last_seen_time}")

            if curr_id in detections_by_id:
                # Either select the detection with the previous frame's track_id
                selected_detection = detections_by_id[curr_id]
                last_seen_time = curr_time

            elif len(detections) == 2 and other_id in detections_by_id:
                # Or if there are two detections, and one has the previous frame's
                # other_id, the detection with the new id
                if detections[0].id != other_id:
                    selected_detection = detections[0]
                else:
                    selected_detection = detections[1]
                last_seen_time = curr_time

            elif len(detections) == 1 and other_id in detections_by_id:
                # Or if there is only one detection, and it has the previous frame's
                # other_id, don't update anything
                if curr_time - last_seen_time < self.keep_track_alive_time:
                    continue
                else:
                    # If the last seen time is too old, select the only available detection
                    selected_detection = detections[0]
                    last_seen_time = curr_time

            else:
                # Or the detection with the closest centroid to the tracked object
                track_centroid = past_centroids.mean(axis=0)
                selected_dist = float("inf")
                selected_detection = None
                for detection in detections:
                    detection_centroid = get_detection_centroid(detection)
                    dist = np.linalg.norm(track_centroid - detection_centroid)
                    if dist < selected_dist:
                        selected_dist = dist
                        selected_detection = detection
                last_seen_time = curr_time

            # Update the track object properties
            other_detection = list(
                filter(lambda d: d.id != selected_detection.id, detections)
            )
            if len(other_detection) > 0:
                other_id = other_detection[0].id
            # Else, leave other_id as is
            detection_centroid = get_detection_centroid(selected_detection)
            past_centroids = np.vstack((past_centroids, detection_centroid))
            past_centroids = past_centroids[-self.num_last_centroids :]
            self.track_dict[class_name] = TrackObject(
                selected_detection.id, other_id, past_centroids, last_seen_time
            )

            track_detections[class_name] = selected_detection

        return track_detections

    def detections_callback(self, detection_array_msg: DetectionArray):
        # Get the two best detections per track object class
        best_detections = get_top_k_detections_per_class(
            detection_array_msg,
            {track_object_name: 2 for track_object_name in self.track_dict.keys()},
        )

        track_detections = self.get_track_detections(best_detections)
        track_detections_array_msg = DetectionArray()
        track_detections_array_msg.header = detection_array_msg.header
        track_detections_array_msg.detections = list(track_detections.values())
        self.output_detections_publisher.publish(track_detections_array_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PairedObjectsTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
