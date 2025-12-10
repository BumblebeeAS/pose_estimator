import cv2
import rclpy
from foxglove_msgs.msg import ImageAnnotations
from rclpy.qos import qos_profile_sensor_data
from yolo_msgs.msg import DetectionArray

from image_processing.utils.image_annotations import get_image_annotations
from pose_estimator.utils.detections import (
    get_detection_polygon,
    get_top_k_detections_per_class,
    polygon_to_obb,
)
from pose_estimator.utils.pose_estimator_node import PoseEstimatorNode


class FloorPoseEstimator(PoseEstimatorNode):
    def __init__(self):
        super().__init__("floor_pose_estimator_node")

        detections_topic = (
            self.declare_parameter(
                "input_detections_topic", rclpy.Parameter.Type.STRING
            )
            .get_parameter_value()
            .string_value
        )

        self.detections_sub = self.create_subscription(
            DetectionArray,
            detections_topic,
            self.detections_callback,
            qos_profile_sensor_data,
        )
        image_annotations_topic = "/auv4/floor/obb_annotations"
        self.anotations_pub = self.create_publisher(
            ImageAnnotations, image_annotations_topic, qos_profile_sensor_data
        )
        # self.is_enabled = False

    def detections_callback(self, msg: DetectionArray):
        # if not self.is_enabled:
        #     return

        # start_time = self.get_clock().now()

        # classes = ["red", "blue", "black"]

        # num_detections = len(msg.detections)

        # We are unable to estimate orientation, so we set it to identity

        # get top k
        # extract polygon points (get polygon from detection or smth)
        # undistort polygon points (take from backproject)
        # get obb of undistorted polygon points
        # get angle of obb
        # process angle depending on whether it's a black or a blue line

        best_detections = get_top_k_detections_per_class(
            msg,
            {"red": 1, "blue": 1, "black": 1},
        )

        obbs = []
        for class_name, detections in best_detections.items():
            if not detections:
                continue

            detection_polygon = get_detection_polygon(detections[0])
            undistorted = cv2.undistortPoints(
                detection_polygon,
                self.camera.camera_matrix(),
                self.camera.dist_coeffs(),
            )
            angle, obb = polygon_to_obb(undistorted)

            if class_name == "black":
                # Normalize from [-90, 90)
                angle = (angle + 90) % 180 - 90
            elif class_name == "blue":
                # Rotate 90 deg more because the angle is points along the
                # Normalize from [-90, 90)
                angle = (angle + 90 + 90) % 180 - 90

            self.get_logger().info(f"Detected {class_name} with angle {angle:.2f}")

            obbs.append(obb)

        image_annotations = get_image_annotations(msg.header, [obbs])
        self.anotations_pub.publish(image_annotations)
        # end_time = self.get_clock().now()
        # elapsed_time = (end_time - start_time).nanoseconds / 1e6
        # self.get_logger().info(
        #     f"Processed {len(msg.detections)} detections in {elapsed_time:.2f} ms"
        # )


def main(args=None):
    rclpy.init(args=args)
    node = FloorPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
