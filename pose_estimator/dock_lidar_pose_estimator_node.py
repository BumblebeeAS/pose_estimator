from math import atan2, cos, radians, sin
from pathlib import Path
from threading import Condition
from time import monotonic

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from bb_perception_msgs.srv import GetDockPose
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup,
)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker

import tf2_geometry_msgs
import tf2_ros


class DockPoseEstimator(Node):
    """Estimate dock pose from a mono BEV image using template matching.

    This node uses ``Node`` directly because it estimates a 2D lidar pose,
    not a camera-frame PnP pose. ``PoseEstimatorPosePubNode`` waits for
    camera info and expects ``rvec``/``tvec`` inputs, so it does not fit here.
    """

    def __init__(self):
        super().__init__("dock_lidar_pose_estimator_node")

        self.target_frame_id = self.declare_parameter(
            "target_frame_id", "asv5/odom"
        ).value
        
        self.input_topic = self.declare_parameter(
            "input_bev_topic", "/asv5/bev"
        ).value
        self.camera_info_topic = self.declare_parameter(
            "camera_info_topic", "/asv5/bev/camera_info"
        ).value
        self.debug_topic = self.declare_parameter(
            "debug_image_topic", "/asv5/bev_detections/image"
        ).value
        self.pose_topic = self.declare_parameter(
            "pose_topic", "/asv5/dock/pose"
        ).value
             

        self.service_timeout_sec = float(
            self.declare_parameter("service_timeout_sec", 5.0).value
        )
        self.pose_frame_id = self.declare_parameter(
            "pose_frame_id", "asv5/base_link"
        ).value

        self.dock_length_m = float(
            self.declare_parameter("dock_length_m", 3.048).value
        )
        self.dock_width_m = float(
            self.declare_parameter("dock_width_m", 6.604).value
        )
        self.bev_scale = float(self.declare_parameter("bev_scale", 0.025).value)
        self.match_threshold = float(
            self.declare_parameter("match_threshold", 0.50).value
        )
        self.search_downsample = float(
            self.declare_parameter("search_downsample", 0.5).value
        )
        self.min_template_scale = float(
            self.declare_parameter("min_template_scale", 0.8).value
        )
        self.max_template_scale = float(
            self.declare_parameter("max_template_scale", 2.0).value
        )
        self.template_scale_step = float(
            self.declare_parameter("template_scale_step", 0.05).value
        )
        self.coarse_angle_step_deg = float(
            self.declare_parameter("coarse_angle_step_deg", 15.0).value
        )
        self.fine_angle_step_deg = float(
            self.declare_parameter("fine_angle_step_deg", 3.0).value
        )
        self.entry_offset_dist = float(
            self.declare_parameter("entry_offset_dist", 0.0).value
        )
        self.pose_service_name = self.declare_parameter(
            "pose_service_name", "/asv5/dock/get_pose"
        ).value
        

        self._validate_parameters()
        default_template_path = self._default_template_path()
        template_path = self.declare_parameter(
            "template_path", default_template_path
        ).value
        self.template_source = self._load_template(template_path)
        self.nominal_template_width = max(
            1, int(round(self.dock_length_m / self.bev_scale))
        )
        self.nominal_template_height = max(
            1, int(round(self.dock_width_m / self.bev_scale))
        )

        self.bridge = CvBridge()
        self.image_center = None
        self.candidate_cache = {}
        self.image_callback_group = ReentrantCallbackGroup()
        self.service_callback_group = MutuallyExclusiveCallbackGroup()
        self.image_condition = Condition()
        self.latest_bev_image = None
        self.last_served_stamp = None

        self.pose_pub = self.create_publisher(
            PoseStamped, self.pose_topic, 10
        )
        self.marker_pub = self.create_publisher(
            Marker, f"{self.pose_topic}/marker", 10
        )
        self.debug_pub = (
            self.create_publisher(Image, self.debug_topic, 10)
            if self.debug_topic
            else None
        )
        self.bev_sub = self.create_subscription(
            Image,
            self.input_topic,
            self.cache_bev_image,
            qos_profile_sensor_data,
            callback_group=self.image_callback_group,
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
            callback_group=self.image_callback_group,
        )
        self.pose_service = self.create_service(
            GetDockPose,
            self.pose_service_name,
            self.get_dock_pose_callback,
            callback_group=self.service_callback_group,
        )

        self.get_logger().info(
            "Dock template search: nominal=%dx%d px, BEV scale=%.3f m/px"
            % (
                self.nominal_template_width,
                self.nominal_template_height,
                self.bev_scale,
            )
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    @staticmethod
    def _stamp_key(msg):
        return (msg.header.stamp.sec, msg.header.stamp.nanosec)

    def cache_bev_image(self, msg):
        """Cache the newest BEV image and wake a waiting pose request."""
        with self.image_condition:
            self.latest_bev_image = msg
            self.image_condition.notify_all()

    def _wait_for_fresh_image(self, timeout_sec):
        deadline = monotonic() + timeout_sec
        with self.image_condition:
            while (
                self.latest_bev_image is None
                or self._stamp_key(self.latest_bev_image)
                == self.last_served_stamp
            ):
                remaining = deadline - monotonic()
                if remaining <= 0.0:
                    return None
                self.image_condition.wait(timeout=remaining)

            msg = self.latest_bev_image
            self.last_served_stamp = self._stamp_key(msg)
            return msg

    def get_dock_pose_callback(self, request, response):
        """Process one fresh BEV frame and return its estimated dock pose."""
        timeout_sec = (
            request.timeout_sec
            if request.timeout_sec > 0.0
            else self.service_timeout_sec
        )
        msg = self._wait_for_fresh_image(timeout_sec)
        if msg is None:
            response.success = False
            response.message = "timed out waiting for a fresh BEV image"
            return response

        pose = self.template_callback(msg)
        if pose is None:
            response.success = False
            response.message = "no valid dock pose found in the BEV image"
            return response

        response.success = True
        response.message = "dock pose estimated"
        response.pose = pose
        return response

    def _validate_parameters(self):
        if self.bev_scale <= 0.0:
            raise ValueError("bev_scale must be positive")
        if not 0.0 < self.search_downsample <= 1.0:
            raise ValueError("search_downsample must be in (0, 1]")
        if self.min_template_scale <= 0.0:
            raise ValueError("min_template_scale must be positive")
        if self.max_template_scale < self.min_template_scale:
            raise ValueError("max_template_scale must be >= min_template_scale")
        if self.template_scale_step <= 0.0:
            raise ValueError("template_scale_step must be positive")
        if self.coarse_angle_step_deg <= 0.0 or self.fine_angle_step_deg <= 0.0:
            raise ValueError("angle steps must be positive")

    @staticmethod
    def _default_template_path():
        try:
            package_share = Path(get_package_share_directory("lidar_segmentation2"))
            return str(package_share / "templates" / "rx26_sim_platform_template2.png")
        except PackageNotFoundError:
            source_root = Path(__file__).resolve().parents[2]
            return str(
                source_root
                / "lidar_segmentation2"
                / "templates"
                / "rx26_sim_platform_template2.png"
            )

    @staticmethod
    def _load_template(template_path):
        template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(f"Could not load dock template: {template_path}")

        points = cv2.findNonZero(template)
        if points is None:
            raise ValueError(f"Dock template has no foreground: {template_path}")
        x, y, width, height = cv2.boundingRect(points)
        return template[y:y + height, x:x + width]

    def camera_info_callback(self, msg):
        if len(msg.k) == 9 and msg.k[2] > 0.0 and msg.k[5] > 0.0:
            self.image_center = (float(msg.k[2]), float(msg.k[5]))

    @staticmethod
    def _rotate_bound(image, angle_deg):
        height, width = image.shape[:2]
        center = ((width - 1) / 2.0, (height - 1) / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        abs_cos = abs(matrix[0, 0])
        abs_sin = abs(matrix[0, 1])
        bound_width = int(np.ceil(height * abs_sin + width * abs_cos))
        bound_height = int(np.ceil(height * abs_cos + width * abs_sin))
        matrix[0, 2] += (bound_width - 1) / 2.0 - center[0]
        matrix[1, 2] += (bound_height - 1) / 2.0 - center[1]
        return cv2.warpAffine(
            image,
            matrix,
            (bound_width, bound_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def _candidate(self, scale, angle_deg, downsample):
        key = (round(scale, 4), round(angle_deg % 360.0, 3), downsample)
        if key in self.candidate_cache:
            return self.candidate_cache[key]

        width = max(
            2,
            int(round(self.nominal_template_width * scale * downsample)),
        )
        height = max(
            2,
            int(round(self.nominal_template_height * scale * downsample)),
        )
        resized = cv2.resize(
            self.template_source,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        rotated = self._rotate_bound(resized, angle_deg)
        self.candidate_cache[key] = rotated
        return rotated

    def _search(self, image, angles, scales):
        best = None
        for scale in scales:
            for angle_deg in angles:
                candidate = self._candidate(
                    float(scale), float(angle_deg), self.search_downsample
                )
                if (
                    candidate.shape[0] > image.shape[0]
                    or candidate.shape[1] > image.shape[1]
                ):
                    continue
                result = cv2.matchTemplate(
                    image, candidate, cv2.TM_CCOEFF_NORMED
                )
                _, score, _, location = cv2.minMaxLoc(result)
                if np.isfinite(score) and (best is None or score > best["score"]):
                    best = {
                        "score": float(score),
                        "location": location,
                        "angle_deg": float(angle_deg) % 360.0,
                        "scale": float(scale),
                        "candidate": candidate,
                    }
        return best

    def find_best_match(self, image):
        search_image = cv2.resize(
            image,
            None,
            fx=self.search_downsample,
            fy=self.search_downsample,
            interpolation=cv2.INTER_NEAREST,
        )
        coarse_angles = np.arange(0.0, 360.0, self.coarse_angle_step_deg)
        coarse_scales = np.arange(
            self.min_template_scale,
            self.max_template_scale + self.template_scale_step * 0.5,
            self.template_scale_step,
        )
        best = self._search(search_image, coarse_angles, coarse_scales)
        if best is None:
            return None

        fine_angles = np.arange(
            best["angle_deg"] - self.coarse_angle_step_deg,
            best["angle_deg"] + self.coarse_angle_step_deg + 0.5,
            self.fine_angle_step_deg,
        )
        fine_scale_step = self.template_scale_step / 2.0
        fine_scales = np.arange(
            max(self.min_template_scale, best["scale"] - self.template_scale_step),
            min(self.max_template_scale, best["scale"] + self.template_scale_step)
            + fine_scale_step * 0.5,
            fine_scale_step,
        )
        refined = self._search(search_image, fine_angles, fine_scales)
        return refined if refined is not None else best

    @staticmethod
    def _bev_pixel_to_asv(center_u, center_v, image_center, metres_per_pixel):
        center_x, center_y = image_center
        # New BEV mapping: image x is FLU +x; image y points opposite FLU +y.
        x_forward = (center_u - center_x) * metres_per_pixel
        y_left = -(center_v - center_y) * metres_per_pixel
        return x_forward, y_left

    def compute_entry_point(self, x_forward, y_left, yaw, offset_dist):
        entry_x = x_forward - offset_dist * cos(yaw)
        entry_y = y_left - offset_dist * sin(yaw)
        return entry_x, entry_y

    def _publish_pose(self, msg, center_u, center_v, match):
        metres_per_pixel = self.bev_scale
        image_center = self.image_center
        if image_center is None:
            image_center = ((msg.width - 1) / 2.0, (msg.height - 1) / 2.0)

        x_forward, y_left = self._bev_pixel_to_asv(
            center_u, center_v, image_center, metres_per_pixel
        )
        
        bearing_yaw = atan2(y_left, x_forward)

        template_rotation = radians(match["angle_deg"])

        x_forward_w_offset, y_left_w_offset = self.compute_entry_point(
            x_forward, y_left, template_rotation, self.entry_offset_dist
        )
        
        

        source_frame = self.pose_frame_id
        target_frame = self.target_frame_id

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
            )

            pose = PoseStamped()
            pose.header.stamp = msg.header.stamp
            pose.header.frame_id = source_frame
            pose.pose.position.x = x_forward_w_offset
            pose.pose.position.y = y_left_w_offset
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = sin(template_rotation / 2.0)
            pose.pose.orientation.w = cos(template_rotation / 2.0)

            transformed_pose = tf2_geometry_msgs.do_transform_pose_stamped(
                pose,
                transform,
            )
            self.pose_pub.publish(transformed_pose)
            self.marker_pub.publish(self._dock_pose_marker(transformed_pose))

            
        except tf2_ros.TransformException as e:
            self.get_logger().error(f"Transform lookup failed: {e}")
            transformed_pose = None

        return (
            x_forward,
            y_left,
            bearing_yaw,
            metres_per_pixel,
            transformed_pose,
        )

    @staticmethod
    def _dock_pose_marker(pose):
        marker = Marker()
        marker.header = pose.header
        marker.ns = "dock_pose"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose = pose.pose
        marker.scale.z = 0.5
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.text = "Dock Pose"
        return marker

    def _draw_match(self, display, match, center_u, center_v):
        candidate = self._candidate(match["scale"], match["angle_deg"], 1.0)
        top_left_x = int(round(center_u - candidate.shape[1] / 2.0))
        top_left_y = int(round(center_v - candidate.shape[0] / 2.0))
        contours, _ = cv2.findContours(
            candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        translated_contours = []
        translation = np.array([[[top_left_x, top_left_y]]], dtype=np.int32)
        for contour in contours:
            translated_contours.append(contour.astype(np.int32) + translation)
        cv2.drawContours(display, translated_contours, -1, (0, 255, 0), 2)
        cv2.circle(
            display,
            (int(round(center_u)), int(round(center_v))),
            4,
            (0, 0, 255),
            -1,
        )

    def template_callback(self, msg):
        try:
            bev_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as error:
            self.get_logger().error(f"Could not convert BEV image: {error}")
            return None

        image_center = self.image_center
        if image_center is None:
            image_center = ((msg.width - 1) / 2.0, (msg.height - 1) / 2.0)

        display = None
        if self.debug_pub is not None:
            display = cv2.cvtColor(bev_image, cv2.COLOR_GRAY2BGR)
            cv2.drawMarker(
                display,
                (int(round(image_center[0])), int(round(image_center[1]))),
                (255, 255, 0),
                cv2.MARKER_CROSS,
                16,
                2,
            )

        match = self.find_best_match(bev_image)
        pose_result = None
        if match is not None:
            candidate = match["candidate"]
            center_u = (
                match["location"][0] + candidate.shape[1] / 2.0
            ) / self.search_downsample
            center_v = (
                match["location"][1] + candidate.shape[0] / 2.0
            ) / self.search_downsample
            if match["score"] >= self.match_threshold:
                pose_result = self._publish_pose(
                    msg, center_u, center_v, match=match
                )
            if display is not None:
                self._draw_match(display, match, center_u, center_v)
                label = "score=%.3f angle=%.1f scale=%.2f" % (
                    match["score"],
                    match["angle_deg"],
                    match["scale"],
                )
                if pose_result is not None:
                    x_forward, y_left, bearing_yaw, metres_per_pixel, _ = (
                        pose_result
                    )
                    label += " x=%.2f y=%.2f yaw=%.1f mpp=%.3f" % (
                        x_forward,
                        y_left,
                        np.degrees(bearing_yaw),
                        metres_per_pixel,
                    )
                else:
                    label += " below threshold"
                cv2.putText(
                    display,
                    label,
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        if display is not None:
            debug_msg = self.bridge.cv2_to_imgmsg(display, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)
        return pose_result[-1] if pose_result is not None else None


def main(args=None):
    rclpy.init(args=args)
    node = DockPoseEstimator()
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
