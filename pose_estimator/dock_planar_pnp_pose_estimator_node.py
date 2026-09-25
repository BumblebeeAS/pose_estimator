"""Estimate dock approach yaw from the lidar returns on a target panel."""

import math

import cv2
import numpy as np
import rclpy
import tf2_ros
from bb_perception_msgs.msg import ClusteredClouds
from bb_perception_msgs.srv import GetDockPose
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from sensor_msgs_py.point_cloud2 import read_points
from tf_transformations import quaternion_matrix
from yolo_msgs.msg import DetectionArray

from pose_estimator.dock_pointcloud_roi import (
    crop_points_in_dock_boxes,
    points_in_oriented_boxes,
)


def transform_points(points: np.ndarray, transform) -> np.ndarray:
    """Transform XYZ points using a geometry_msgs TransformStamped."""
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    matrix = quaternion_matrix((rotation.x, rotation.y, rotation.z, rotation.w))
    matrix[:3, 3] = (translation.x, translation.y, translation.z)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (matrix @ homogeneous.T).T[:, :3]


def stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def fit_vertical_plane(
    points: np.ndarray,
    distance_threshold: float,
    max_tilt: float,
    iterations: int,
    rng: np.random.Generator,
):
    """Fit a vertical plane and return centroid, horizontal normal, and quality."""
    if len(points) < 3:
        return None

    best_inliers = None
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        if abs(normal[2]) > math.sin(max_tilt):
            continue
        offset = -normal @ sample[0]
        inliers = np.abs(points @ normal + offset) <= distance_threshold
        if best_inliers is None or np.count_nonzero(inliers) > np.count_nonzero(
            best_inliers
        ):
            best_inliers = inliers

    if best_inliers is None:
        return None

    inlier_points = points[best_inliers]
    centroid = np.mean(inlier_points, axis=0)
    covariance = np.cov((inlier_points - centroid).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    if abs(normal[2]) > math.sin(max_tilt):
        return None

    normal_xy = normal[:2]
    normal_xy_norm = np.linalg.norm(normal_xy)
    if normal_xy_norm < 1e-6:
        return None
    normal_xy /= normal_xy_norm

    residuals = (inlier_points - centroid) @ normal
    rmse = math.sqrt(float(np.mean(residuals**2)))
    inlier_ratio = len(inlier_points) / len(points)
    return centroid, normal_xy, rmse, inlier_ratio


class DockPlanarPnpPoseEstimator(Node):
    """Filter dock lidar points by YOLO OBBs or dock-relative crop boxes."""

    def __init__(self):
        super().__init__("dock_planar_pnp_pose_estimator_node")

        self.cluster_topic = self.declare_parameter(
            "cluster_topic", "/asv5/clustered_clouds"
        ).value
        self.dock_pose_service = self.declare_parameter(
            "dock_pose_service", "/asv5/dock/get_pose"
        ).value
        self.pose_service_name = self.declare_parameter(
            "pose_service_name", "/asv5/dock/lidar_panel/get_pose"
        ).value
        self.pose_topic = self.declare_parameter(
            "pose_topic", "/asv5/dock/lidar_planar/pose"
        ).value
        self.target_frame = self.declare_parameter(
            "target_frame", "asv5/odom"
        ).value
        self.base_frame = self.declare_parameter(
            "base_frame", "asv5/base_link"
        ).value
        self.dock_pose_timeout = float(
            self.declare_parameter("dock_pose_timeout", 10.0).value
        )
        self.roi_mode = self.declare_parameter("roi_mode", "dock_pose").value
        self.cropbox_x_offsets = np.asarray(
            self.declare_parameter(
                "cropbox_x_offsets", [-2.03, 0.0, 2.03]
            ).value,
            dtype=np.float64,
        )
        self.cropbox_y_offset = float(
            self.declare_parameter("cropbox_y_offset", 0.58).value
        )
        self.cropbox_z_offset = float(
            self.declare_parameter("cropbox_z_offset", 0.5).value
        )
        self.cropbox_size = float(
            self.declare_parameter("cropbox_size", 1.0).value
        )
        self.cropbox_padding = float(
            self.declare_parameter("cropbox_padding", 0.1).value
        )
        self.vision_detections_topic = self.declare_parameter(
            "vision_detections_topic",
            "/asv5/dock_white_planar_target/yolo/detections",
        ).value
        self.vision_camera_info_topic = self.declare_parameter(
            "vision_camera_info_topic", "/asv5/oakd/camera_info"
        ).value
        self.vision_min_score = float(
            self.declare_parameter("vision_min_score", 0.25).value
        )
        self.vision_padding_px = float(
            self.declare_parameter("vision_padding_px", 8.0).value
        )
        self.vision_max_age = float(
            self.declare_parameter("vision_max_age", 0.5).value
        )
        if self.roi_mode not in {"dock_pose", "vision"}:
            raise ValueError("roi_mode must be 'dock_pose' or 'vision'")
        if self.cropbox_size <= 0.0 or self.cropbox_padding < 0.0:
            raise ValueError("cropbox_size must be positive and padding non-negative")
        if self.vision_padding_px < 0.0 or self.vision_max_age < 0.0:
            raise ValueError("vision padding and max age must be non-negative")
        self.cropbox_centers = np.column_stack(
            (
                self.cropbox_x_offsets,
                np.full(len(self.cropbox_x_offsets), self.cropbox_y_offset),
                np.full(len(self.cropbox_x_offsets), self.cropbox_z_offset),
            )
        )
        self.min_points = int(self.declare_parameter("min_points", 20).value)
        self.plane_distance_threshold = float(
            self.declare_parameter("plane_distance_threshold", 0.05).value
        )
        self.max_plane_tilt = math.radians(
            float(self.declare_parameter("max_plane_tilt_deg", 10.0).value)
        )
        self.min_inlier_ratio = float(
            self.declare_parameter("min_inlier_ratio", 0.6).value
        )
        self.max_plane_rmse = float(
            self.declare_parameter("max_plane_rmse", 0.05).value
        )
        self.ransac_iterations = int(
            self.declare_parameter("ransac_iterations", 200).value
        )

        self.latest_cluster_message = None
        self.latest_detections = None
        self.latest_camera_info = None
        self.rng = np.random.default_rng()
        self.tf_buffer = tf2_ros.Buffer(node=self)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.service_callback_group = ReentrantCallbackGroup()
        self.pose_service = self.create_service(
            GetDockPose,
            self.pose_service_name,
            self._get_pose_callback,
            callback_group=self.service_callback_group,
        )
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            self.pose_topic,
            10,
        )
        if self.roi_mode == "dock_pose":
            self.dock_pose_client = self.create_client(
                GetDockPose,
                self.dock_pose_service,
                callback_group=self.service_callback_group,
            )
        self.cluster_subscription = self.create_subscription(
            ClusteredClouds,
            self.cluster_topic,
            self._cluster_callback,
            qos_profile_sensor_data,
        )
        if self.roi_mode == "vision":
            self.detections_subscription = self.create_subscription(
                DetectionArray,
                self.vision_detections_topic,
                self._detections_callback,
                qos_profile_sensor_data,
            )
            self.camera_info_subscription = self.create_subscription(
                CameraInfo,
                self.vision_camera_info_topic,
                self._camera_info_callback,
                qos_profile_sensor_data,
            )

        self.get_logger().info(
            f"Serving lidar panel pose on {self.pose_service_name} and "
            f"publishing on {self.pose_topic}; ROI mode={self.roi_mode}"
        )

    async def _get_pose_callback(self, request, response):
        if self.latest_cluster_message is None:
            response.success = False
            response.message = "no clustered lidar cloud available"
            return response

        if self.roi_mode == "vision" and (
            self.latest_detections is None or self.latest_camera_info is None
        ):
            response.success = False
            response.message = "no YOLO detections or camera info available"
            return response

        dock_pose = None
        if self.roi_mode == "dock_pose":
            if not self.dock_pose_client.service_is_ready():
                response.success = False
                response.message = (
                    f"dock pose service unavailable: {self.dock_pose_service}"
                )
                return response

            dock_request = GetDockPose.Request()
            dock_request.timeout_sec = (
                request.timeout_sec
                if request.timeout_sec > 0.0
                else self.dock_pose_timeout
            )
            try:
                dock_response = await self.dock_pose_client.call_async(dock_request)
            except Exception as error:
                response.success = False
                response.message = f"dock pose request failed: {error}"
                return response

            if dock_response is None or not dock_response.success:
                response.success = False
                response.message = (
                    dock_response.message
                    if dock_response is not None
                    else "empty dock pose response"
                )
                return response
            dock_pose = dock_response.pose

        pose = self._estimate_pose(
            self.latest_cluster_message,
            dock_pose,
        )
        if pose is None:
            response.success = False
            response.message = "no valid lidar panel pose found"
            return response

        response.success = True
        response.message = "lidar panel pose estimated"
        response.pose = pose
        self.pose_publisher.publish(pose)
        return response

    def _pose_in_target(self, pose: PoseStamped) -> tuple[np.ndarray, float]:
        orientation = pose.pose.orientation
        pose_matrix = quaternion_matrix(
            (orientation.x, orientation.y, orientation.z, orientation.w)
        )
        pose_matrix[:3, 3] = (
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        )
        if pose.header.frame_id != self.target_frame:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                pose.header.frame_id,
                Time.from_msg(pose.header.stamp),
                timeout=Duration(seconds=0.1),
            )
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            target_from_source = quaternion_matrix(
                (rotation.x, rotation.y, rotation.z, rotation.w)
            )
            target_from_source[:3, 3] = (
                translation.x,
                translation.y,
                translation.z,
            )
            pose_matrix = target_from_source @ pose_matrix
        position = pose_matrix[:3, 3]
        yaw = math.atan2(pose_matrix[1, 0], pose_matrix[0, 0])
        return position, yaw

    def _cloud_points_in_target(self, cloud) -> np.ndarray:
        points = np.asarray(
            [
                tuple(point)
                for point in read_points(
                    cloud, field_names=("x", "y", "z"), skip_nans=True
                )
            ],
            dtype=np.float64,
        )
        if len(points) == 0:
            return points.reshape(0, 3)
        if cloud.header.frame_id == self.target_frame:
            return points
        transform = self.tf_buffer.lookup_transform(
            self.target_frame,
            cloud.header.frame_id,
            Time.from_msg(cloud.header.stamp),
            timeout=Duration(seconds=0.1),
        )
        return transform_points(points, transform)

    def _asv_position(self, stamp) -> np.ndarray:
        transform = self.tf_buffer.lookup_transform(
            self.target_frame,
            self.base_frame,
            Time.from_msg(stamp),
            timeout=Duration(seconds=0.1),
        )
        translation = transform.transform.translation
        return np.array([translation.x, translation.y], dtype=np.float64)

    def _cluster_callback(self, message: ClusteredClouds) -> None:
        self.latest_cluster_message = message

    def _detections_callback(self, message: DetectionArray) -> None:
        self.latest_detections = message

    def _camera_info_callback(self, message: CameraInfo) -> None:
        self.latest_camera_info = message

    def _filter_with_vision(self, points: np.ndarray, header) -> np.ndarray:
        detections = self.latest_detections
        camera_info = self.latest_camera_info
        if abs(
            stamp_seconds(header.stamp) - stamp_seconds(detections.header.stamp)
        ) > self.vision_max_age:
            return points[:0]

        camera_frame = camera_info.header.frame_id or detections.header.frame_id
        transform = self.tf_buffer.lookup_transform(
            camera_frame,
            self.target_frame,
            Time.from_msg(header.stamp),
            timeout=Duration(seconds=0.1),
        )
        camera_points = transform_points(points, transform)
        in_front = camera_points[:, 2] > 0.0
        if not np.any(in_front):
            return points[:0]

        camera_points = camera_points[in_front]
        image_points, _ = cv2.projectPoints(
            camera_points,
            np.zeros(3),
            np.zeros(3),
            np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3),
            np.asarray(camera_info.d, dtype=np.float64),
        )
        image_points = image_points.reshape(-1, 2)
        boxes = np.asarray(
            [
                (
                    detection.bbox.center.position.x,
                    detection.bbox.center.position.y,
                    detection.bbox.size.x,
                    detection.bbox.size.y,
                    detection.bbox.center.theta,
                )
                for detection in detections.detections
                if detection.score >= self.vision_min_score
                and detection.class_name == "dock_white_planar_target"
                and detection.bbox.size.x > 0.0
                and detection.bbox.size.y > 0.0
            ],
            dtype=np.float64,
        )
        if len(boxes) == 0:
            return points[:0]
        inside = points_in_oriented_boxes(
            image_points,
            boxes.reshape(-1, 5),
            self.vision_padding_px,
        )
        return points[in_front][inside]

    def _estimate_pose(
        self,
        message: ClusteredClouds,
        dock_pose: PoseStamped | None,
    ) -> PoseStamped | None:
        try:
            dock_position = None
            dock_yaw = None
            if dock_pose is not None:
                dock_position, dock_yaw = self._pose_in_target(dock_pose)
            clouds = []
            header = None
            for cluster in message.clusters:
                points = self._cloud_points_in_target(cluster.cluster)
                if len(points) > 0:
                    clouds.append(points)
                    if header is None:
                        header = cluster.cluster.header
            if not clouds:
                return None
            points = np.concatenate(clouds)

            if self.roi_mode == "vision":
                points = self._filter_with_vision(points, header)
            else:
                if dock_position is None or dock_yaw is None:
                    return None
                points = crop_points_in_dock_boxes(
                    points,
                    dock_position,
                    dock_yaw,
                    self.cropbox_centers,
                    self.cropbox_size,
                    self.cropbox_padding,
                )
            if len(points) < self.min_points:
                return None
            fitted = fit_vertical_plane(
                points,
                self.plane_distance_threshold,
                self.max_plane_tilt,
                self.ransac_iterations,
                self.rng,
            )
            if fitted is None:
                return None
            centroid, normal, rmse, inlier_ratio = fitted
            if (
                inlier_ratio < self.min_inlier_ratio
                or rmse > self.max_plane_rmse
            ):
                return None

            asv_position = self._asv_position(header.stamp)
            outward = normal
            if outward @ (asv_position - centroid[:2]) < 0.0:
                outward = -outward
            inward = -outward
            yaw = math.atan2(inward[1], inward[0])

            pose = PoseStamped()
            pose.header.stamp = header.stamp
            pose.header.frame_id = self.target_frame
            pose.pose.position.x = float(centroid[0])
            pose.pose.position.y = float(centroid[1])
            pose.pose.position.z = float(centroid[2])
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
            return pose
        except tf2_ros.TransformException as error:
            self.get_logger().warn(f"Dock panel transform unavailable: {error}")
        except (ValueError, np.linalg.LinAlgError) as error:
            self.get_logger().warn(f"Dock panel fit failed: {error}")
        return None


def main(args=None):
    rclpy.init(args=args)
    node = DockPlanarPnpPoseEstimator()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
