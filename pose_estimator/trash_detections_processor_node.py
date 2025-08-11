from collections.abc import Sequence

import rclpy
from foxglove_msgs.msg import ImageAnnotations
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Header
from yolo_msgs.msg import Detection, DetectionArray

from image_processing.utils.image_annotations import get_image_annotations
from pose_estimator.utils.detections import (
    get_detection_convex_hull,
    get_detection_polygon,
    get_IoA,
    get_top_k_detections_per_class,
)


def get_single_detection(
    top_k_detections: dict[str, Sequence[Detection]], class_name: str
) -> Detection | None:
    """Get the single best detection for a given class from the top-k detections.
    If no detection is found for the class, returns None."""
    detections = top_k_detections.get(class_name, [])
    if not detections:
        return None
    return detections[0]


def is_detection_inside(
    contained_detection: Detection,
    containing_detection: Detection,
    IoA_threshold: float = 0.8,
):
    """Checks if the contained detection intersects with the convex hull of the
    containing detection."""
    # Use a convex hull since the trash may obscure corners of the containing polygon.
    convex_hull_containing_polygon = get_detection_convex_hull(containing_detection)
    contained_polygon = get_detection_polygon(contained_detection)
    return get_IoA(contained_polygon, convex_hull_containing_polygon) > IoA_threshold


def get_detection_array(
    header: Header, detections: Sequence[Detection]
) -> DetectionArray:
    """Convert a sequence of detections to a DetectionArray."""
    detection_array = DetectionArray()
    detection_array.header = header
    detection_array.detections = detections
    return detection_array


class TrashDetectionsProcessorNode(Node):
    """
    Takes trash yolo detections and processes them into three categories:
    - `on_table`: Detections that are on the trash table.
    - `in_bucket`: Detections that are in the trash bucket.
    - `uncollected`: Detections that are not in the bucket.
        (These may or may not be on the table. This is kept for cases where the table
        detections are unreliable.)
    """

    def __init__(self):
        super().__init__("trash_detections_processor_node")

        # Parameters
        input_detections_topic = (
            self.declare_parameter(
                "input_detections_topic", rclpy.Parameter.Type.STRING
            )
            .get_parameter_value()
            .string_value
        )
        self.IoA_threshold = (
            self.declare_parameter("IoA_threshold", 0.8)
            .get_parameter_value()
            .double_value
        )

        # Subscriptions and publishers
        self.input_detections_subscription = self.create_subscription(
            DetectionArray,
            input_detections_topic,
            self.detections_callback,
            qos_profile=qos_profile_sensor_data,
        )

        self.on_table_detections_publisher = self.create_publisher(
            DetectionArray,
            f"{input_detections_topic}/on_table",
            qos_profile=qos_profile_sensor_data,
        )
        self.in_bucket_detections_publisher = self.create_publisher(
            DetectionArray,
            f"{input_detections_topic}/in_bucket",
            qos_profile=qos_profile_sensor_data,
        )
        self.uncollected_detections_publisher = self.create_publisher(
            DetectionArray,
            f"{input_detections_topic}/uncollected",
            qos_profile=qos_profile_sensor_data,
        )

        self.image_annotations_publisher = self.create_publisher(
            ImageAnnotations,
            f"{input_detections_topic}/annotations",
            qos_profile=qos_profile_sensor_data,
        )

    def is_trash_in_location(
        self, trash_detection: Detection, location_detection: Detection
    ):
        return location_detection is not None and is_detection_inside(
            trash_detection, location_detection, self.IoA_threshold
        )

    def detections_callback(self, detection_array_msg: DetectionArray):
        best_detections = get_top_k_detections_per_class(
            detection_array_msg,
            {"table": 1, "bottle": 2, "ladle": 2, "yellow_bucket": 1, "pink_bucket": 1},
        )

        trash_detections = [
            det
            for k, detections in best_detections.items()
            if k in ["bottle", "ladle"]
            for det in detections
        ]

        yellow_bucket_detection = get_single_detection(best_detections, "yellow_bucket")
        pink_bucket_detection = get_single_detection(best_detections, "pink_bucket")
        table_detection = get_single_detection(best_detections, "table")

        in_bucket_detections = list()
        in_table_detections = list()
        uncollected_detections = list()

        for det in trash_detections:
            is_trash_in_bucket = self.is_trash_in_location(
                det, yellow_bucket_detection
            ) or self.is_trash_in_location(det, pink_bucket_detection)

            if is_trash_in_bucket:
                in_bucket_detections.append(det)

            if self.is_trash_in_location(det, table_detection):
                in_table_detections.append(det)

            if not is_trash_in_bucket:
                uncollected_detections.append(det)

        header = detection_array_msg.header
        in_bucket_detections_array = get_detection_array(header, in_bucket_detections)
        self.in_bucket_detections_publisher.publish(in_bucket_detections_array)

        in_table_detections_array = get_detection_array(header, in_table_detections)
        self.on_table_detections_publisher.publish(in_table_detections_array)

        uncollected_detections_array = get_detection_array(
            header, uncollected_detections
        )
        self.uncollected_detections_publisher.publish(uncollected_detections_array)

        image_annotations = get_image_annotations(
            detection_array_msg.header,
            [
                map(get_detection_polygon, in_table_detections),
                map(get_detection_polygon, in_bucket_detections),
                map(get_detection_polygon, uncollected_detections),
            ],
        )
        self.image_annotations_publisher.publish(image_annotations)


def main(args=None):
    rclpy.init(args=args)
    node = TrashDetectionsProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
