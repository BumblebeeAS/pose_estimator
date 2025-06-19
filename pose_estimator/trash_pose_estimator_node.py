import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from yolo_msgs.msg import DetectionArray

from pose_estimator.utils.detections import (
    get_detection_centroid,
    get_top_k_detections_per_class,
)
from pose_estimator.utils.pose_estimator_node import PoseEstimatorNode


def backproject_pixel(
    x: float, y: float, Z: float, K: np.ndarray
) -> tuple[float, float]:
    """
    Solve w * [x, y, 1]^T = K * [X, Y, Z]^T for X, Y, given Z, where
    w is the unknown scale factor.

    Args:
        x, y: Image coordinates
        Z: Known depth
        K: 3x3 camera intrinsic matrix

    Raises:
        np.linalg.LinAlgError: If inverse of K cannot be computed.

    Returns:
        X, Y: 3D coordinates in the camera frame
    """
    pixel = np.array([x, y, 1.0])
    K_inv = np.linalg.inv(K)
    direction = K_inv @ pixel
    direction *= Z / direction[2]
    X, Y = direction[0], direction[1]
    return X, Y


class TrashPoseEstimator(PoseEstimatorNode):
    def __init__(self):
        super().__init__("trash_pose_estimator_node")

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

        # Depth of object in camera frame
        self.object_depth = 0.5

    def detections_callback(self, detection_array_msg: DetectionArray):
        best_detections = get_top_k_detections_per_class(
            detection_array_msg,
            {"bottle": 2, "ladle": 2, "pink_bucket": 1, "yellow_bucket": 1},
        )
        header = detection_array_msg.header

        for class_name, detections in best_detections.items():
            for detection in detections:
                detection_centroid = get_detection_centroid(detection)
                X, Y = backproject_pixel(
                    detection_centroid[0],
                    detection_centroid[1],
                    self.object_depth,
                    self.camera.camera_matrix(),
                )
                tvec = np.array([X, Y, self.object_depth]).reshape((3, 1))
                rvec = np.zeros((3, 1), dtype=np.float32)

                self.publish_transform(tvec, rvec, header, class_name)


def main(args=None):
    rclpy.init(args=args)
    node = TrashPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
