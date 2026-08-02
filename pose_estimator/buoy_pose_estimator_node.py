import cv2
import numpy as np
import rclpy
from bb_robotx_msgs.msg import BuoyDetection, BuoyDetectionArray
from foxglove_msgs.msg import ImageAnnotations
from geometry_msgs.msg import PoseStamped
from image_processing.utils.image_annotations import get_image_annotations
from pose_estimator.config.object_points import (
    BUOY_CENTER_OFFSET,
    BUOY_OBJECT_POINTS,
)
from pose_estimator.utils.detections import (
    filter_detections_by_num_points,
    get_detection_obb,
    match_polygon_points,
)
from pose_estimator.utils.pose_estimator import get_object_pose, refine_object_pose
from pose_estimator.utils.pose_estimator_node import PoseEstimatorPosePubNode
from pose_estimator.utils.ros_messages import get_pose_stamped
from rclpy.publisher import Publisher
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Header
from yolo_msgs.msg import DetectionArray

COLOR_TO_UINT8 = {
    "black_buoy": BuoyDetection.BLACK,
    "red_buoy": BuoyDetection.RED,
    "green_buoy": BuoyDetection.GREEN,
    "blue_buoy": BuoyDetection.BLUE,
}
CLASSES_OF_INTEREST = list(COLOR_TO_UINT8.keys())


class BuoyPoseEstimatorNode(PoseEstimatorPosePubNode):
    def __init__(self):
        super().__init__("buoy_pose_estimator_node")

        output_pose_topic = (
            self.declare_parameter("output_pose_topic", "/buoy_detections")
            .get_parameter_value()
            .string_value
        )
        debug_annotations_topic = (
            self.declare_parameter("debug_annotations_topic", "debug_annotations")
            .get_parameter_value()
            .string_value
        )

        self._pose_publishers = {}
        for class_name in CLASSES_OF_INTEREST:
            self._pose_publishers[class_name] = self.create_publisher(
                PoseStamped, f"{output_pose_topic}/{class_name}", qos_profile_sensor_data
            )

        self.pose_publisher: Publisher[BuoyDetectionArray] = self.create_publisher(
            BuoyDetectionArray, output_pose_topic, qos_profile_sensor_data
        )
        self.debug_annotations_publisher = self.create_publisher(
            ImageAnnotations, debug_annotations_topic, qos_profile_sensor_data
        )

        input_detections_topic = (
            self.declare_parameter(
                "input_detections_topic", rclpy.Parameter.Type.STRING
            )
            .get_parameter_value()
            .string_value
        )

        self.detection_subscription = self.create_subscription(
            DetectionArray,
            input_detections_topic,
            self.detection_callback,
            qos_profile_sensor_data,
        )

    @property
    def pose_publishers(self) -> dict[str, Publisher]:
        return self._pose_publishers

    def _get_detection_poses_in_camera_frame(self, detections: DetectionArray):
        detection_stamp = detections.header.stamp
        filtered_detections: DetectionArray = filter_detections_by_num_points(
            detections, 3
        )
        output = []
        detected_polygons = []
        for detection in filtered_detections.detections:
            if detection.class_name not in CLASSES_OF_INTEREST:
                continue
            angle, detected_points = get_detection_obb(detection)
            angle = (angle + 90) % 180 - 90

            object_points, image_points = match_polygon_points(
                BUOY_OBJECT_POINTS, detected_points, A_angle=angle
            )

            object_points = np.hstack(
                [object_points, np.zeros((object_points.shape[0], 1))]
            )

            assert (
                object_points.shape[0] == image_points.shape[0]
            ), "Number of object points and image points must match"

            try:
                rvec, tvec, inliers = get_object_pose(
                    self.camera, object_points, image_points
                )

                if inliers is None:
                    raise ValueError("No inliers found during pose estimation.")

                rvec, tvec = refine_object_pose(
                    self.camera, object_points, image_points, rvec, tvec
                )

                # Get center pose
                R, _ = cv2.Rodrigues(rvec)
                tvec = tvec.flatten() + R @ BUOY_CENTER_OFFSET

                output.append((detection.class_name, tvec, rvec, object_points))
                detected_polygons.append(detected_points)
            except Exception as e:
                self.get_logger().warn(f"Pose estimation failed: {e}")
                continue

        return detection_stamp, output, detected_polygons

    def detection_callback(self, msg: DetectionArray):
        detection_stamp, detection_poses, detected_polygons = (
            self._get_detection_poses_in_camera_frame(msg)
        )

        if detected_polygons:
            image_annotations = get_image_annotations(msg.header, [detected_polygons])
            self.debug_annotations_publisher.publish(image_annotations)

        if not detection_poses:
            return

        buoy_array_msg = BuoyDetectionArray()
        buoy_array_msg.header.stamp = msg.header.stamp
        buoy_array_msg.header.frame_id = msg.header.frame_id

        for class_name, tvec, rvec, object_points in detection_poses:
            pose_header = Header()
            pose_header.stamp = detection_stamp
            pose_header.frame_id = msg.header.frame_id
            pose_stamped = get_pose_stamped(
                pose_header, tvec.flatten(), [0.0, 0.0, 0.0, 1.0]
            )

            buoy_msg = BuoyDetection()
            buoy_msg.pose = pose_stamped
            buoy_msg.color = COLOR_TO_UINT8.get(class_name, BuoyDetection.BLACK)
            buoy_array_msg.detections.append(buoy_msg)

            self.publish_data(tvec, rvec, object_points, pose_header, class_name)

        self.pose_publisher.publish(buoy_array_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BuoyPoseEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
