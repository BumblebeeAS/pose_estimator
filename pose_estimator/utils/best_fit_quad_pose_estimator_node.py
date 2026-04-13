import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.qos import qos_profile_sensor_data
from yolo_msgs.msg import DetectionArray

from pose_estimator.utils.detections import (
    filter_detections_by_num_points,
    get_best_detections_per_class,
)
from pose_estimator.utils.pose_estimator import (
    get_object_pose_from_detection_using_best_fit_quad,
)
from pose_estimator.utils.pose_estimator_node import (
    PoseEstimatorPosePubNode,
    PoseEstimatorTransformPubNode,
)


class BestFitQuadPoseEstimator(PoseEstimatorPosePubNode):
    """
    Estimates pose of each detection using the best-fit quadrilateral of the boundary
    of its mask. See `get_object_pose_from_detection_using_best_fit_quad` for details.

    Note that this node only estimates the pose for the BEST detection of each class.

    If the object has parts that can be segmented separately, consider matching
    object and image points for all parts to get a single accurate pose instead
    of using this node to estimate poses of each part which may be noisy.
    """

    def __init__(
        self,
        object_points: dict[str, list[tuple[float, float]]],
        frame_name_remap: dict[str, str],
    ):
        """Create a BestFitQuadPoseEstimator node.

        Args:
            object_points (Dict[str, List[List[float]]]): Dictionary mapping class names to their object points.
            frame_name_remap (Dict[str, str]): Dictionary mapping class names to their output frame names.

        Raises:
            AssertionError: If the keys of object_points and frame_name_remap do not match.
        """
        super().__init__("best_fit_quad_pose_estimator_node")

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
            DetectionArray,
            input_detections_topic,
            self.detections_callback,
            qos_profile_sensor_data,
        )

    @property
    def pose_publishers(self):
        return {
            frame_name: self.create_publisher(
                PoseStamped, f"{frame_name}/pose", qos_profile_sensor_data
            )
            for frame_name in self.frame_name_remap.values()
        }

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
            object_polygon = np.array(object_polygon, dtype=np.float32)

            try:
                rvec, tvec = get_object_pose_from_detection_using_best_fit_quad(
                    self.camera, object_polygon, detection, self.get_logger()
                )
            except Exception as e:
                self.get_logger().warn(f"Pose estimation failed: {e}")
                return

            frame_name = self.frame_name_remap[detection.class_name]
            self.publish_data(
                tvec, rvec, object_polygon, detections_msg.header, frame_name
            )
