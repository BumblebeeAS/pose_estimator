import rclpy
from rclpy.node import Node
from yolo_msgs.msg import DetectionArray, Detection
import shapely

from pose_estimator.utils.detections import (
    get_IoA,
    get_detection_perimeter,
    get_top_k_detections_per_class,
    is_in_polygon,
)

class TrashObjectInGrabberNode(Node):
    def __init__(self):
        super().__init__("trash_object_in_grabber_node")

        input_detections_topic = (
            self.declare_parameter(
                "input_detections_topic", rclpy.Parameter.Type.STRING
            ).get_parameter_value()
            .string_value
        )

        output_detections_in_grabber_topic = (
            self.declare_parameter(
                "output_detections_in_grabber_topic", rclpy.Parameter.Type.STRING
            ).get_parameter_value()
            .string_value
        )

        grab_region_polygon_param = (
            self.declare_parameter(
                "grab_region_polygon", rclpy.Parameter.Type.DOUBLE_ARRAY
            ).get_parameter_value()
            .double_array_value
        )

        self.grab_region_polygon = np.array(grab_region_polygon_param).reshape(
            -1, 2
        )

        self.bottle_grab_area_threshold = (
            self.declare_parameter(
                "bottle_grab_area_threshold", rclpy.Parameter.Type.DOUBLE
            )
            .get_parameter_value()
            .double_value
        )
        self.ladle_grab_perimeter_threshold = (
            self.declare_parameter(
                "ladle_grab_perimeter_threshold", rclpy.Parameter.Type.DOUBLE
            )
            .get_parameter_value()
            .double_value
        ) 

        self.is_in_grabber_pub = self.create_publisher(
            DetectionArray, output_detections_in_grabber_topic
        )

        self.detections_sub = self.create_subscription(
            DetectionArray, input_detections_topic, self.detections_callback
        )

    def detetections_callback(self, detection_array_msg: DetectionArray):
        if not self.grab_region_polygon:
            self.get_logger().warn("No valid grab region provided!")

        best_detections = get_top_k_detections_per_class(
            detection_array_msg,
            {"bottle": 2, "ladle": 2, "pink_bucket": 1, "yellow_bucket": 1},
        )

        output_detection_array = DetectionArray()
        for class_name, detection in best_detections.items():
            if class_name in ["bottle", "ladle"] and is_in_polygon(
                detection, self.grab_region_polygon
            ):
                if class_name == "bottle":
                    if (
                        get_IoA(get_detection_polygon(detection), self.grab_region_polygon)
                        >= self.bottle_grab_area_threshold
                    ):
                        output_detection_array.detections.append(detection)
                elif class_name == "ladle":
                    if (
                        get_detection_perimeter(detection)
                        >= self.ladle_grab_perimeter_threshold
                    ):
                        output_detection_array.detections.append(detection)
        
        self.is_in_grabber_pub.publish(output_detection_array)

def main(args=None):
    rclpy.init(args=args)
    node = TrashObjectInGrabberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()