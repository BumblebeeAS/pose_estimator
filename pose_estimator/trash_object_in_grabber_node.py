import rclpy
from rclpy.node import Node
from yolo_msgs.msg import DetectionArray, Detection
import shapely
from foxglove_msgs.msgs import ImageAnnotations

from pose_estimator.utils.detections import (
    get_IoA,
    get_detection_perimeter,
    get_top_k_detections_per_class,
    is_in_polygon,
    get_detection_polygon
)

from image_processing.utils.image_annotations import get_image_annotations

class TrashObjectInGrabberNode(Node):
    def __init__(self):
        super().__init__("trash_object_in_grabber_node")

        input_detections_topic = (
            self.declare_parameter(
                "input_detections_topic", rclpy.Parameter.Type.STRING
            ).get_parameter_value()
            .string_value
        )

        bottle_in_grabber_topic = (
            self.declare_parameter(
                "bottle_in_grabber_topic", rclpy.Parameter.Type.STRING
            ).get_parameter_value()
            .string_value
        )

        ladle_in_grabber_topic = (
            self.declare_parameter(
                "ladle_in_grabber_topic", rclpy.Parameter.Type.STRING
            )
            .get_parameter_value()
            .string_value
        )

        bottle_grab_region_polygon_param = (
            self.declare_parameter(
                "bottle_grab_region_polygon", rclpy.Parameter.Type.DOUBLE_ARRAY
            ).get_parameter_value()
            .double_array_value
        )

        self.bottle_grab_region_polygon = np.array(bottle_grab_region_polygon_param).reshape(
            -1, 2
        )

        ladle_grab_region_polygon_param = (
            self.declare_parameter(
                "ladle_grab_region_polygon", rclpy.Parameter.Type.DOUBLE_ARRAY
            ).get_parameter_value()
            .double_array_value
        )

        self.ladle_grab_region_polygon = np.array(ladle_grab_region_polygon_param).reshape(
            -1, 2
        )

        self.bottle_grab_area_threshold = (
            self.declare_parameter(
                "bottle_grab_area_threshold", rclpy.Parameter.Type.DOUBLE
            )
            .get_parameter_value()
            .double_value
        )
        self.ladle_grab_area_threshold = (
            self.declare_parameter(
                "ladle_grab_area_threshold", rclpy.Parameter.Type.DOUBLE
            )
            .get_parameter_value()
            .double_value
        ) 

        self.bottle_in_grabber_pub = self.create_publisher(
            DetectionArray, bottle_in_grabber_topic
        )

        self.ladle_in_grabber_pub = self.create_publisher(
            DetectionArray, ladle_in_grabber_topic
        )

        self.detections_sub = self.create_subscription(
            DetectionArray, input_detections_topic, self.detections_callback
        )

        annotations_topic = (
            self.declare_parameter(
                "annotations_topic", rclpy.Parameter.Type.STRING
            )
            .get_parameter_value()
            .string_value
        )

        self.annotations_pub = self.create_publisher(
            ImageAnnotations, annotations_topic,
        )

    def detetections_callback(self, detection_array_msg: DetectionArray):
        best_detections = get_top_k_detections_per_class(
            detection_array_msg,
            {"bottle": 2, "ladle": 2, "pink_bucket": 1, "yellow_bucket": 1},
        )

        bottle_detection_array = DetectionArray()
        ladle_detection_array = DetectionArray()
        for class_name, detection in best_detections.items():
            if class_name == "bottle":
                if (get_IoA(get_detection_polygon(detection), self.bottle_grab_region_polygon)) >= self.bottle_grab_area_threshold:
                    bottle_detection_array.detections.append(detection)
            elif class_name == "ladle":
                if (
                    get_IoA(get_detection_polygon(detection), self.ladle_grab_region_polygon)
                    >= self.ladle_grab_area_threshold
                ):
                    ladle_detection_array.detections.append(detection)

        self.bottle_in_grabber_pub.publish(bottle_detection_array) 
        self.ladle_in_grabber_pub.publish(ladle_detection_array)
        image_annotations = get_image_annotations(
            detection_array_msg.header,
            [
                map(get_detection_polygon, best_detections.values())
            ]
        )
        self.annotations_pub.publish(image_annotations)

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