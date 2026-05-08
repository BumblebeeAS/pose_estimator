import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from yolo_msgs.msg import DetectionArray

from pose_estimator.utils.detections import (
    get_best_detections_per_class,
    get_detection_centroid,
)


class SosRepairEstimator(Node):
    def __init__(self):
        super().__init__("sos_repair_estimator_node")
        self.bridge = CvBridge()

        self.declare_parameter("detections_topic", "yolo/detections")
        self.declare_parameter("output_string_topic", "sos_repair")

        detections_topic = (
            self.get_parameter("detections_topic").get_parameter_value().string_value
        )
        output_string_topic = (
            self.get_parameter("output_string_topic").get_parameter_value().string_value
        )
        self.detections_sub = self.create_subscription(
            DetectionArray,
            detections_topic,
            self.detections_callback,
            qos_profile_sensor_data,
        )
        self.string_publisher = self.create_publisher(
            String, output_string_topic, qos_profile_sensor_data
        )

    def detections_callback(self, detections_msg: DetectionArray):
        # Only include relevant classes
        relevant_classes = ["sos", "repair"]
        required_objects = 2

        best_detections = get_best_detections_per_class(
            detections_msg, relevant_classes
        )

        num_detected_objects = len(best_detections.keys())
        if num_detected_objects < required_objects:
            self.get_logger().warn(
                f"""Insufficient detected objects.
                Received: {num_detected_objects}, require: {required_objects}."""
            )
            return

        sos_centroid = get_detection_centroid(best_detections["sos"])
        repair_centroid = get_detection_centroid(best_detections["repair"])

        sos_repair_string = (
            "sos_repair" if sos_centroid[0] < repair_centroid[0] else "repair_sos"
        )
        sos_repair_msg = String()
        sos_repair_msg.data = sos_repair_string
        self.string_publisher.publish(sos_repair_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SosRepairEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
