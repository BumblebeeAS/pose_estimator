import numpy as np
import rclpy
from cv_bridge import CvBridge
from yolo_msgs.msg import DetectionArray

from pose_estimator.config.object_points import SYMBOL_OBJECT_POINTS
from pose_estimator.utils.detections import (
    filter_detections_by_num_points,
    get_best_detections_per_class,
    get_detection_obb,
    match_polygon_points,
)
from pose_estimator.utils.pose_estimator import get_object_pose
from pose_estimator.utils.pose_estimator_node import PoseEstimatorNode

FRAME_NAME_REMAP = {
    "reef_shark": "trash/shark",
    "sawfish": "trash/fish",
}


class SharkFishPoseEstimator(PoseEstimatorNode):
    def __init__(self):
        super().__init__("shark_fish_pose_estimator_node")
        self.bridge = CvBridge()

        input_detections_topic = (
            self.declare_parameter("input_detections_topic", "yolo/detections")
            .get_parameter_value()
            .string_value
        )
        self.detections_sub = self.create_subscription(
            DetectionArray, input_detections_topic, self.detections_callback, 1
        )

    def detections_callback(self, detections_msg: DetectionArray):
        # We require at least 3 points for polygon creation
        filtered_detections = filter_detections_by_num_points(detections_msg, 3)

        # Only include relevant classes
        relevant_classes = ["reef_shark", "sawfish"]
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
        detections = list(best_detections.values())

        for detection in detections:
            object_polygon = SYMBOL_OBJECT_POINTS[detection.class_name]
            _, detected_polygon = get_detection_obb(detection)
            object_points, image_points = match_polygon_points(
                object_polygon, detected_polygon
            )
            object_points = np.hstack(
                [object_points, np.zeros((object_points.shape[0], 1))]
            )

            assert (
                object_points.shape[0] == image_points.shape[0]
            ), "Number of object points and image points must match"

            try:
                # We set a large max re-projection error as the segmentation masks are noisy and
                # the matched points for one detection may be at an offset from the matched points
                # for another. Setting a lower max re-projection error may result in a more confident
                # pose estimator, but it may also result in no inliers being found.

                rvec, tvec, inliers = get_object_pose(
                    self.camera, object_points, image_points, max_reprojection_error=100
                )

                if inliers is None:
                    raise ValueError("No inliers found during pose estimation.")

                # We do not refine the pose as the points are likely to have large re-projection error.

            except Exception as e:
                self.get_logger().warn(f"Pose estimation failed: {e}")
                return

            frame_name = FRAME_NAME_REMAP[detection.class_name]
            self.publish_data(
                tvec, rvec, object_points, detections_msg.header, frame_name
            )


def main(args=None):
    rclpy.init(args=args)
    node = SharkFishPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
