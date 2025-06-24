from typing import Dict, List

from cv_bridge import CvBridge
from yolo_msgs.msg import DetectionArray

from pose_estimator.utils.detections import (
    filter_detections_by_num_points,
    get_best_detections_per_class,
)
from pose_estimator.utils.pose_estimator import get_object_pose_from_detection_using_obb
from pose_estimator.utils.pose_estimator_node import PoseEstimatorNode


class OBBPoseEstimator(PoseEstimatorNode):
    """
    Estimates pose of each detection using oriented bounding boxes of their
    their individual masks. The most naive method of pose estimation, but works
    well when the following conditions are met:

    - The object is not rotated more than 45 degrees from its upright orientation.
    - The object is rectangular.
    - Perspective does not significantly affect the object's aspect ratio.

    Note that this node only estimates the pose for the BEST detection of each class.

    If the object has parts that can be segmented separately, consider matching
    object and image points for all parts to get a single accurate pose instead
    of using this node to estimate poses of each part which may be noisy.
    """

    def __init__(
        self,
        object_points: Dict[str, List[List[float]]],
        frame_name_remap: Dict[str, str],
    ):
        """Create a OBBPoseEstimator node.

        Args:
            object_points (Dict[str, List[List[float]]]): Dictionary mapping class names to their object points.
            frame_name_remap (Dict[str, str]): Dictionary mapping class names to their output frame names.

        Raises:
            AssertionError: If the keys of object_points and frame_name_remap do not match.
        """
        super().__init__("obb_pose_estimator_node")

        assert set(object_points.keys()) == set(frame_name_remap.keys()), (
            "Object points and frame name remap must have the same keys.",
        )
        self.input_detection_classes = list(object_points.keys())
        self.object_points = object_points
        self.frame_name_remap = frame_name_remap

        input_detections_topic = (
            self.declare_parameter("input_detections_topic", "yolo/detections")
            .get_parameter_value()
            .string_value
        )

        self.bridge = CvBridge()
        self.detections_sub = self.create_subscription(
            DetectionArray, input_detections_topic, self.detections_callback, 1
        )

    def detections_callback(self, detections_msg: DetectionArray):
        # We require at least 3 points for polygon creation
        filtered_detections = filter_detections_by_num_points(detections_msg, 3)

        # Only include relevant classes
        best_detections = get_best_detections_per_class(
            filtered_detections, self.input_detection_classes
        )

        # Match image points and object points
        detections = list(best_detections.values())

        for detection in detections:
            object_polygon = self.object_points[detection.class_name]

            try:
                rvec, tvec = get_object_pose_from_detection_using_obb(
                    self.camera, object_polygon, detection
                )
            except Exception as e:
                self.get_logger().warn(f"Pose estimation failed: {e}")
                return

            frame_name = self.frame_name_remap[detection.class_name]
            self.publish_transform(tvec, rvec, detections_msg.header, frame_name)
