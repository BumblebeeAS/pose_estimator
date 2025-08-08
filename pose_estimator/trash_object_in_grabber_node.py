import numpy as np
import rclpy
from foxglove_msgs.msgs import ImageAnnotations
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from yolo_msgs.msg import DetectionArray

from image_processing.utils.image_annotations import get_image_annotations
from pose_estimator.utils.detections import (
    get_detection_polygon,
    get_IoA,
    get_top_k_detections_per_class,
)


class TrashObjectInGrabberNode(Node):
    def __init__(self):
        super().__init__("trash_object_in_grabber_node")

        input_detections_topic = (
            self.declare_parameter(
                "input_detections_topic", rclpy.Parameter.Type.STRING
            )
            .get_parameter_value()
            .string_value
        )
        output_annotations_topic = (
            self.declare_parameter(
                "output_annotations_topic", rclpy.Parameter.Type.STRING
            )
            .get_parameter_value()
            .string_value
        )
        output_objects_in_grabber_topic = (
            self.declare_parameter(
                "output_objects_in_grabber_topic", rclpy.Parameter.Type.STRING
            )
            .get_parameter_value()
            .string_value
        )
        bottle_roi_polygon = (
            self.declare_parameter(
                "bottle_roi_polygon", rclpy.Parameter.Type.DOUBLE_ARRAY
            )
            .get_parameter_value()
            .double_array_value
        )
        ladle_roi_polygon = (
            self.declare_parameter(
                "ladle_roi_polygon", rclpy.Parameter.Type.DOUBLE_ARRAY
            )
            .get_parameter_value()
            .double_array_value
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

        self.detections_sub = self.create_subscription(
            DetectionArray,
            input_detections_topic,
            self.detections_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self.objects_in_grabber_pub = self.create_publisher(
            DetectionArray,
            output_objects_in_grabber_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.annotations_pub = self.create_publisher(
            ImageAnnotations,
            output_annotations_topic,
            qos_profile=qos_profile_sensor_data,
        )

        self.bottle_grab_roi_polygon = np.array(bottle_roi_polygon).reshape(-1, 2)
        self.ladle_grab_roi_polygon = np.array(ladle_roi_polygon).reshape(-1, 2)

    def detections_callback(self, detection_array_msg: DetectionArray):
        best_detections = get_top_k_detections_per_class(
            detection_array_msg, {"bottle": 2, "ladle": 2}
        )

        detections_in_grabber: list[Detection] = []
        for class_name, detections in best_detections.items():
            for det in detections:
                containing_polygon = (
                    self.bottle_grab_roi_polygon
                    if class_name == "bottle"
                    else self.ladle_grab_roi_polygon
                )
                threshold = (
                    self.bottle_grab_area_threshold
                    if class_name == "bottle"
                    else self.ladle_grab_area_threshold
                )
                IoA = get_IoA(get_detection_polygon(det), containing_polygon)
                if IoA >= threshold:
                    detections_in_grabber.append(det)

        output_msg = DetectionArray()
        output_msg.header = detection_array_msg.header
        output_msg.detections = detections_in_grabber
        self.objects_in_grabber_pub.publish(output_msg)

        image_annotations = get_image_annotations(
            detection_array_msg.header,
            [
                map(get_detection_polygon, detections_in_grabber),
                [self.bottle_grab_roi_polygon],
                [self.ladle_grab_roi_polygon],
            ],
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
