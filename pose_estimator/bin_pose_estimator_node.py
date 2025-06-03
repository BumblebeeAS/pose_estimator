from operator import attrgetter

import numpy as np
import rclpy
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.qos import qos_profile_sensor_data
from yolo_msgs.msg import Detection, DetectionArray

from pose_estimator.utils.detections import (
    BIN_OBJECT_POINTS,
    filter_detections_by_num_points,
    get_best_detections_per_class,
    get_best_fit_ngon,
    get_detection_obb,
    match_polygon_points,
    polygon_to_obb,
)
from pose_estimator.utils.pose_estimator import get_object_pose, refine_object_pose
from pose_estimator.utils.pose_estimator_node import PoseEstimatorNode


def get_detection_polygon(detection: Detection, logger: RcutilsLogger) -> np.ndarray:
    mask = detection.mask
    mask_points = [attrgetter("x", "y")(point) for point in mask.data]

    try:
        polygon_points = get_best_fit_ngon(np.array(mask_points, dtype=np.int32))
    except ValueError as e:
        logger.warn(f"Failed to get best fit ngon: {e}, using OBB instead")
        polygon_points = polygon_to_obb(mask_points)

    return polygon_points


class BinPoseEstimator(PoseEstimatorNode):
    def __init__(self):
        super().__init__("bin_pose_estimator_node")

        self.object_frame_id = (
            self.declare_parameter("object_frame_id", "bin")
            .get_parameter_value()
            .string_value
        )
        detections_topic = (
            self.declare_parameter("input_detections_topic", "yolo/detections")
            .get_parameter_value()
            .string_value
        )

        self.detections_sub = self.create_subscription(
            DetectionArray,
            detections_topic,
            self.detections_callback,
            qos_profile_sensor_data,
        )

    def detections_callback(self, msg: DetectionArray):
        # We require at least 3 points for polygon creation
        filtered_detections = filter_detections_by_num_points(msg, 3)

        # Only include relevant classes
        relevant_classes = ["bin"]
        required_objects = 1

        best_detections = get_best_detections_per_class(
            filtered_detections, relevant_classes
        )
        num_detected_objects = len(best_detections.keys())
        if num_detected_objects < required_objects:
            self.get_logger().warn(
                f"""Insufficient detected objects.
                Received: {num_detected_objects}, require: {required_objects}."""
            )
            return

        # Match image points and object points
        angle, detected_points = get_detection_obb(best_detections["bin"])
        # TODO: Make get_best_fit_ngon faster -> currently takes a few seconds
        # detected_points = get_detection_polygon(best_detections["bin"], self.get_logger())

        # Rotate object points before matching by point distances because bin yaw can
        # be large and we assume perspective does not change imaged aspect ratio by much.
        object_points, image_points = match_polygon_points(
            BIN_OBJECT_POINTS, detected_points, A_angle=angle
        )

        object_points = np.hstack(
            [object_points, np.zeros((object_points.shape[0], 1))]
        )

        assert (
            object_points.shape[0] == image_points.shape[0]
        ), "Number of object points and image points must match"

        try:
            # We can set a low max re-projection error and refine pose even though
            # the segmentation mask is noisy because we only have 4 points

            rvec, tvec, inliers = get_object_pose(
                self.camera, object_points, image_points
            )

            if inliers is None:
                raise ValueError("No inliers found during pose estimation.")

            rvec, tvec = refine_object_pose(
                self.camera, object_points, image_points, rvec, tvec
            )

        except Exception as e:
            self.get_logger().warn(f"Pose estimation failed: {e}")
            return

        self.publish_data(tvec, rvec, object_points, msg.header, self.object_frame_id)


def main(args=None):
    rclpy.init(args=args)
    node = BinPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
