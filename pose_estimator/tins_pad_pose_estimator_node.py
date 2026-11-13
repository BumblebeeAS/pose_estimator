"""One platform pose with fixed red, green, and blue circle child frames."""

from itertools import product
from operator import attrgetter

import cv2
import numpy as np
import rclpy
import tf2_ros
from bb_perception_msgs.srv import TrashToggleFrame, SetOrientationReference
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from frames.utils.sync_diagnostics import SyncDiagnostics
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Header
from tf_transformations import quaternion_inverse, quaternion_matrix, quaternion_multiply
from yolo_msgs.msg import DetectionArray
from tf2_msgs.msg import TFMessage

from pose_estimator.utils.detections import (
    filter_detections_by_num_points,
    get_detection_polygon,
    get_top_k_detections_per_class,
)
from pose_estimator.utils.pose_estimator_node import PoseEstimatorNode, get_translation_quaternion
from pose_estimator.utils.ros_messages import get_pose_stamped, get_transform_stamped
from pose_estimator.utils.tins_pad import estimate_tins_pad_pose


class TinsPadPoseEstimator(PoseEstimatorNode):
    def __init__(self):
        super().__init__("tins_pad_pose_estimator_node")
        self.object_frame_id = self.declare_parameter("object_frame_id", "tins_pad").value
        self.radius = self.declare_parameter("circle_diameter", 0.70).value / 2
        self.inlier_threshold = self.declare_parameter("contour_inlier_threshold", 8.0).value
        # Approximate centers read from the example drawing, NOT dimensioned CAD.
        self.centers = {
            label: self.declare_parameter(f"{label}_center_xy", xy).value
            for label, xy in {
                "red_circle": [-0.5, -0.35],
                "green_circle": [0.5, -0.35],
                "blue_circle": [0.0, 0.5],
            }.items()
        }
        model = np.asarray(list(self.centers.values()), dtype=float)
        if (model.shape != (3, 2) or not np.isfinite(model).all()
                or np.linalg.matrix_rank(model - model.mean(0)) < 2):
            raise ValueError("Circle centers must form a finite, non-collinear triangle")
        if self.radius <= 0 or not np.isfinite(self.radius):
            raise ValueError("Circle diameter must be positive")
        if self.inlier_threshold <= 0 or not np.isfinite(self.inlier_threshold):
            raise ValueError("Contour inlier threshold must be positive")

        output = self.declare_parameter("output_pose_topic", "tins_pad/pose").value
        self.pose_publisher = self.create_publisher(PoseStamped, output, qos_profile_sensor_data)
        # Full-pose and retained-orientation measurements remain separate.
        # The tree refreshes this reference from the latest pre-deposit cluster.
        self.orientation_frame = self.declare_parameter("orientation_frame", "odom_ned").value
        self.max_reference_age = self.declare_parameter("orientation_reference_max_age_s", 5.0).value
        if not np.isfinite(self.max_reference_age) or self.max_reference_age <= 0:
            raise ValueError("Orientation reference maximum age must be finite and positive")
        self.fixed_orientation = None
        self.position_publisher = self.create_publisher(
            PoseStamped, "tins_pad/position_only/pose", qos_profile_sensor_data)
        self.orientation_service = self.create_service(
            TrashToggleFrame, "tins_pad/set_orientation_reference", self.set_orientation_reference)
        self.explicit_orientation_service = self.create_service(
            SetOrientationReference, "tins_pad/set_orientation", self.set_saved_orientation)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, static_only=True)
        # A latest TF lookup stamps an all-static chain at time zero. Retain the
        # original cluster transform so freshness uses its observation timestamp.
        self.reference_transforms = {}
        self.reference_subscription = self.create_subscription(
            TFMessage, "/tf_static", self._on_reference_transforms,
            QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE))
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        header = Header(frame_id=self.object_frame_id, stamp=self.get_clock().now().to_msg())
        self.static_broadcaster.sendTransform([
            get_transform_stamped(
                header, f"{self.object_frame_id}/{label}", [float(xy[0]), float(xy[1]), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ) for label, xy in self.centers.items()
        ])

        detections_topic = self.declare_parameter("input_detections_topic", "rfdetr/detections").value
        odom_topic = self.declare_parameter("input_odom_topic", "/uav/odom_ned").value
        self.detections_sub = Subscriber(self, DetectionArray, detections_topic, qos_profile=qos_profile_sensor_data)
        self.odom_sub = Subscriber(self, Odometry, odom_topic, qos_profile=qos_profile_sensor_data)
        self.time_sync = ApproximateTimeSynchronizer([self.detections_sub, self.odom_sub], 100, slop=0.1)
        self.sync_diagnostics = SyncDiagnostics(self)
        self.time_sync.registerCallback(self.detections_callback)

    def publish_data(self, tvec, rvec, object_points, header, object_frame_id):
        translation, quaternion = get_translation_quaternion(tvec, rvec)
        self.pose_publisher.publish(get_pose_stamped(header, translation, quaternion))
        self.tf_broadcaster.sendTransform(
            get_transform_stamped(header, object_frame_id, translation, quaternion)
        )

    def set_saved_orientation(self, request, response):
        """Use an explicit cluster snapshot; approach duration does not expire it."""
        self.fixed_orientation = None
        if not request.enable:
            response.success = True
            return response
        orientation = request.orientation
        if orientation.header.frame_id != self.orientation_frame:
            response.success = False
            response.message = f"Expected orientation in {self.orientation_frame}"
            return response
        q = orientation.quaternion
        quaternion = np.array([q.x, q.y, q.z, q.w], dtype=float)
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(quaternion).all() or norm < 1e-12:
            response.success = False
            response.message = "Invalid saved orientation quaternion"
            return response
        # The tree obtained this value from the successful pre-approach cluster
        # action. Its source timestamp is retained for traceability, not TTL.
        self.fixed_orientation = quaternion / norm
        response.success = True
        response.message = "Retained explicit pre-approach cluster orientation"
        self.get_logger().info(response.message)
        return response

    def _on_reference_transforms(self, message):
        for transform in message.transforms:
            self.reference_transforms[transform.child_frame_id] = transform

    def set_orientation_reference(self, request, response):
        # Clear first: a failed refresh must never reuse an earlier mission's yaw.
        self.fixed_orientation = None
        if not request.enable:
            response.success = True
            return response
        try:
            reference = self.reference_transforms.get(request.trash_frame_clustered)
            if reference is None:
                raise ValueError("Clustered reference has not arrived on /tf_static")
            age = (self.get_clock().now().nanoseconds
                   - Time.from_msg(reference.header.stamp).nanoseconds) / 1e9
            # Allow collection finalization and both cameras' sequential service
            # callbacks; a two-second cutoff could expire between wide and narrow.
            if not 0.0 <= age <= self.max_reference_age:
                raise ValueError(
                    f"Orientation reference age {age:.2f} s exceeds the allowed range "
                    f"0..{self.max_reference_age:.2f} s")
            q = reference.transform.rotation
            quaternion = np.array([q.x, q.y, q.z, q.w], dtype=float)
            if reference.header.frame_id != self.orientation_frame:
                parent = self.tf_buffer.lookup_transform(
                    self.orientation_frame, reference.header.frame_id,
                    Time.from_msg(reference.header.stamp))
                rotation = parent.transform.rotation
                quaternion = quaternion_multiply(
                    [rotation.x, rotation.y, rotation.z, rotation.w], quaternion)
            norm = np.linalg.norm(quaternion)
            if not np.isfinite(quaternion).all() or norm < 1e-12:
                raise ValueError("Invalid reference orientation")
            self.fixed_orientation = quaternion / norm
        except (tf2_ros.TransformException, ValueError) as exc:
            response.success = False
            response.message = str(exc)
            self.get_logger().warning(f"Cannot retain deposit orientation: {exc}")
            return response
        response.success = True
        response.message = f"Retained orientation from {request.trash_frame_clustered} in {self.orientation_frame}"
        self.get_logger().info(response.message)
        return response

    def _fit_contours(self, candidates, visible_labels, rotation, fixed_yaw=None):
        best = None
        # Reject incompatible combinations rather than joining different pads.
        for combination in product(*(candidates[label] for label in visible_labels)):
            contours = {label: get_detection_polygon(detection)
                        for label, detection in zip(visible_labels, combination)}
            try:
                rvec, tvec, error = estimate_tins_pad_pose(
                    self.camera, contours, self.centers, self.radius, rotation,
                    inlier_threshold=self.inlier_threshold, fixed_yaw=fixed_yaw)
            except (ValueError, cv2.error, np.linalg.LinAlgError):
                continue
            if best is None or error < best[0]:
                best = error, rvec, tvec
        return best

    def detections_callback(self, message, odom):
        self.sync_diagnostics.record(message, odom, 'detections_to_odometry')
        if message.header.frame_id != self.camera.frame_id or not odom.child_frame_id:
            return
        candidates = get_top_k_detections_per_class(
            filter_detections_by_num_points(message, 5), {label: 2 for label in self.centers})
        visible_labels = [label for label in self.centers if candidates.get(label)]
        if not visible_labels:
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                odom.child_frame_id, self.camera.frame_id, Time.from_msg(message.header.stamp))
        except tf2_ros.TransformException:
            return
        quaternion = attrgetter("x", "y", "z", "w")
        camera_in_world_q = quaternion_multiply(
            quaternion(odom.pose.pose.orientation), quaternion(transform.transform.rotation))
        camera_in_world = quaternion_matrix(camera_in_world_q)[:3, :3]
        if len(visible_labels) >= 2:
            level_rotation = quaternion_matrix(quaternion_inverse(camera_in_world_q))[:3, :3]
            best = self._fit_contours(candidates, visible_labels, level_rotation)
            if best is not None:
                _, rvec, tvec = best
                self.publish_data(tvec, rvec, None, message.header, self.object_frame_id)

        fixed = self.fixed_orientation
        if fixed is None or odom.header.frame_id != self.orientation_frame:
            return
        # Express the retained odometry-frame orientation in this camera at
        # this observation. The solver varies translation only, even for 2+ circles.
        pad_in_camera = camera_in_world.T @ quaternion_matrix(fixed)[:3, :3]
        best = self._fit_contours(candidates, visible_labels, pad_in_camera, fixed_yaw=0.0)
        if best is None:
            return
        _, _, tvec = best
        body_rotation = quaternion_matrix(quaternion(odom.pose.pose.orientation))[:3, :3]
        body_position = odom.pose.pose.position
        camera_offset = transform.transform.translation
        camera_position = np.array([body_position.x, body_position.y, body_position.z]) + body_rotation @ np.array(
            [camera_offset.x, camera_offset.y, camera_offset.z])
        position = camera_position + camera_in_world @ tvec.ravel()
        # Publish directly in odometry so clustering preserves the exact retained
        # orientation instead of reintroducing camera/odometry synchronization error.
        header = Header(frame_id=self.orientation_frame, stamp=message.header.stamp)
        self.position_publisher.publish(get_pose_stamped(header, position, fixed))
        self.get_logger().info(
            f"Fixed-orientation pad position: circles={len(visible_labels)} "
            f"labels={','.join(visible_labels)} pixel_error={best[0]:.2f}",
            throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = TinsPadPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
