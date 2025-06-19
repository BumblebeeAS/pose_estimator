import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from std_msgs.msg import String
from yolo_msgs.msg import DetectionArray

from pose_estimator.utils.detections import (
    get_best_detections_per_class,
    get_detection_centroid,
)


class SharkFishEstimator(Node):
    def __init__(self):
        super().__init__("shark_fish_estimator_node")
        self.bridge = CvBridge()

        self.declare_parameter("detections_topic", "yolo/detections")
        self.declare_parameter("output_string_topic", "shark_fish")

        detections_topic = (
            self.get_parameter("detections_topic").get_parameter_value().string_value
        )
        output_string_topic = (
            self.get_parameter("output_string_topic").get_parameter_value().string_value
        )
        self.detections_sub = self.create_subscription(
            DetectionArray, detections_topic, self.detections_callback, 1
        )
        self.string_publisher = self.create_publisher(String, output_string_topic, 10)

    def detections_callback(self, msg: DetectionArray):
        # Only include relevant classes
        relevant_classes = ["reef_shark", "sawfish"]
        required_objects = 2

        best_detections = get_best_detections_per_class(msg, relevant_classes)

        num_detected_objects = len(best_detections.keys())
        if num_detected_objects < required_objects:
            self.get_logger().warn(
                f"""Insufficient detected objects.
                Received: {num_detected_objects}, require: {required_objects}."""
            )
            return

        shark_centroid = get_detection_centroid(best_detections["reef_shark"])
        fish_centroid = get_detection_centroid(best_detections["sawfish"])

        shark_fish_string = (
            "shark_fish" if shark_centroid[0] < fish_centroid[0] else "fish_shark"
        )
        shark_fish_msg = String()
        shark_fish_msg.data = shark_fish_string
        self.string_publisher.publish(shark_fish_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SharkFishEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
