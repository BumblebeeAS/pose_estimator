"""Locate colored buoy detections using footprint PnP."""

import cv2
import numpy as np
import rclpy
from bb_robotx_msgs.msg import BuoyDetection, BuoyDetectionArray
from geometry_msgs.msg import PoseArray, PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from yolo_msgs.msg import DetectionArray
from pose_estimator.config.robotx26_object_points import BUOY_FOOTPRINT_OBJECT_POINTS
from pose_estimator.utils.PinholeCamera import PinholeCamera
from pose_estimator.utils.pose_estimator import get_object_pose_from_detection_using_best_fit_quad


COLOR_IDS = {
    "off": BuoyDetection.BLACK, "red": BuoyDetection.RED,
    "green": BuoyDetection.GREEN, "blue": BuoyDetection.BLUE, "unknown": 255,
}
POSE_LABEL_COLORS = {
    "red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0),
    "off": (0.0, 0.0, 0.0), "unknown": (0.5, 0.5, 0.5),
}


class BuoyPoseEstimator(Node):
    def __init__(self):
        super().__init__("buoy_pose_estimator_node")
        param = lambda name, default: self.declare_parameter(name, default).value
        self.max_range = param("max_range", 100.0)
        if self.max_range <= 0:
            raise ValueError("Maximum buoy range must be positive")
        self.camera_info = None
        self.camera_sub = self.create_subscription(
            CameraInfo, param("camera_info_topic", "rect/camera_info"),
            self.camera_callback, qos_profile_sensor_data,
        )
        self.pose_pub = self.create_publisher(
            BuoyDetectionArray, param("output_detections_topic", "buoy/detections"), qos_profile_sensor_data
        )
        self.pose_array_pub = self.create_publisher(
            PoseArray, param("output_poses_topic", "buoy/poses"), qos_profile_sensor_data
        )
        self.pose_label_pub = self.create_publisher(
            MarkerArray, param("output_pose_labels_topic", "buoy/pose_labels"), qos_profile_sensor_data
        )
        self.detection_sub = self.create_subscription(
            DetectionArray, param("input_detections_topic", "buoy/classified_detections"),
            self.detections_callback, qos_profile_sensor_data,
        )
        self.get_logger().info("Buoy footprint PnP positions in the source camera frame")

    def camera_callback(self, message):
        self.camera_info = message

    def detections_callback(self, detections):
        output = BuoyDetectionArray(header=detections.header)
        poses = PoseArray(header=output.header)
        # Marker state persists in viewers: clear the previous frame even if
        # this frame has no accepted poses. Stamp the clear marker for image sync.
        labels = MarkerArray(markers=[Marker(header=output.header, action=Marker.DELETEALL)])
        camera = self.camera_info
        valid_camera = (camera is not None and camera.header.frame_id == detections.header.frame_id
                        and camera.width > 0 and camera.height > 0)
        if valid_camera:
            pinhole = PinholeCamera.from_camera_info(camera, rectified=True)

        for detection in detections.detections:
            if not detection.class_name.startswith("buoy_"):
                continue
            label = detection.class_name.removeprefix("buoy_")
            if label not in COLOR_IDS:
                continue
            if not valid_camera:
                continue
            try:
                _, tvec = get_object_pose_from_detection_using_best_fit_quad(
                    pinhole, BUOY_FOOTPRINT_OBJECT_POINTS[:, :2], detection, self.get_logger()
                )
                camera_center = tvec.ravel()
                if (not np.isfinite(camera_center).all() or camera_center[2] <= 0
                        or np.linalg.norm(camera_center) > self.max_range):
                    raise ValueError("Pose is invalid, behind the camera, or beyond maximum range")
            except (ValueError, AssertionError, np.linalg.LinAlgError, cv2.error) as error:
                self.get_logger().warning(f"Footprint PnP rejected: {error}", throttle_duration_sec=5.0)
                continue
            pose = PoseStamped(header=output.header)
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, camera_center)
            pose.pose.orientation.w = 1.0  # Position only; buoy orientation is unobserved.
            output.detections.append(BuoyDetection(pose=pose, color=COLOR_IDS[label]))
            poses.poses.append(pose.pose)
            marker = Marker(
                header=output.header, ns="buoy_labels", id=len(poses.poses) - 1,
                type=Marker.TEXT_VIEW_FACING, action=Marker.ADD,
                pose=pose.pose, text=f"buoy_{label}",
            )
            marker.scale.z = 0.15  # Text height in metres.
            marker.color.r, marker.color.g, marker.color.b = POSE_LABEL_COLORS[label]
            marker.color.a = 1.0
            marker.lifetime.sec = 1
            labels.markers.append(marker)
        self.pose_pub.publish(output)
        self.pose_array_pub.publish(poses)
        self.pose_label_pub.publish(labels)


def main(args=None):
    rclpy.init(args=args)
    node = BuoyPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
