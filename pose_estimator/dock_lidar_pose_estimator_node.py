from itertools import product
from math import atan2, cos, radians, sin
from pathlib import Path
from time import monotonic

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker

import tf2_geometry_msgs
import tf2_ros


class DockPoseEstimator(Node):
    """Continuously estimate dock pose from a mono BEV image using template matching.

    The first frame (and any frame after the track is lost) runs a full
    search over the whole BEV. Once locked, each frame first searches a small
    window around the previous match's pixel position and hill-climbs angle
    and scale from the previous values, stopping as soon as no neighbour
    improves. If the local search falls below ``match_threshold``, the same
    frame falls back to a full search.

    This node uses ``Node`` directly because it estimates a 2D lidar pose,
    not a camera-frame PnP pose. ``PoseEstimatorPosePubNode`` waits for
    camera info and expects ``rvec``/``tvec`` inputs, so it does not fit here.
    """

    def __init__(self):
        super().__init__("dock_lidar_pose_estimator_node")

        self.target_frame_id = self.declare_parameter(
            "target_frame_id", "asv5/base_link"
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
        self.pose_frame_id = self.declare_parameter(
            "pose_frame_id", "asv5/base_link"
        ).value

        # Simulation uses the RX26 platform template and its larger dock
        # dimensions; real runs default to the measured dock_single template.
        # All three stay overridable via launch-file template_path /
        # dock_length_m / dock_width_m, which take precedence over use_sim.
        self.use_sim = bool(self.declare_parameter("use_sim", False).value)
        if self.use_sim:
            default_template_filename = "rx26_sim_platform_template.png"
            default_dock_length_m = 3.048
            default_dock_width_m = 6.604
        else:
            default_template_filename = "dock_single_real.png"
            default_dock_length_m = 2.5
            default_dock_width_m = 2.5
            
        self.default_template_filename = default_template_filename

        self.dock_length_m = float(
            self.declare_parameter("dock_length_m", default_dock_length_m).value
        )
        self.dock_width_m = float(
            self.declare_parameter("dock_width_m", default_dock_width_m).value
        )

        self.bev_scale = float(self.declare_parameter("bev_scale", 0.04).value)
        self.match_threshold = float(
            self.declare_parameter("match_threshold", 0.40).value
        )
        self.search_downsample = float(
            self.declare_parameter("search_downsample", 0.5).value
        )
        self.min_template_scale = float(
            self.declare_parameter("min_template_scale", 1.0).value
        )
        self.max_template_scale = float(
            self.declare_parameter("max_template_scale", 1.05).value
        )
        self.template_scale_step = float(
            self.declare_parameter("template_scale_step", 0.05).value
        )
        self.coarse_angle_step_deg = float(
            self.declare_parameter("coarse_angle_step_deg", 15.0).value
        )
        self.fine_angle_step_deg = float(
            self.declare_parameter("fine_angle_step_deg", 0.5).value
        )
        self.entry_offset_dist = float(
            self.declare_parameter("entry_offset_dist", 0.0).value
        )

        # Local tracking search around the previous match.
        self.track_search_radius_m = float(
            self.declare_parameter("track_search_radius_m", 5.0).value
        )
        self.track_angle_window_deg = float(
            self.declare_parameter("track_angle_window_deg", 30.0).value
        )
        self.track_scale_window = float(
            self.declare_parameter("track_scale_window", 0.1).value
        )

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
        # Last accepted match (full-resolution pixel centre, angle, scale).
        self.track = None

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
        # Depth 1 so frames arriving during a slow search are dropped rather
        # than queued; a mutually exclusive group serialises track updates.
        bev_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.bev_sub = self.create_subscription(
            Image,
            self.input_topic,
            self.template_callback,
            bev_qos,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            "Dock template search: use_sim=%s nominal=%dx%d px, BEV scale=%.3f m/px"
            % (
                self.use_sim,
                self.nominal_template_width,
                self.nominal_template_height,
                self.bev_scale,
            )
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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
        if self.track_search_radius_m <= 0.0:
            raise ValueError("track_search_radius_m must be positive")
        if self.track_angle_window_deg < 0.0 or self.track_scale_window < 0.0:
            raise ValueError("track windows must be non-negative")

    def _default_template_path(self):
        try:
            package_share = Path(get_package_share_directory("lidar_segmentation2"))
            return str(
                package_share / "templates" / self.default_template_filename
            )
        except PackageNotFoundError:
            source_root = Path(__file__).resolve().parents[2]
            return str(
                source_root
                / "lidar_segmentation2"
                / "templates"
                / self.default_template_filename
            )

    def _load_template(self, template_path):
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

    def _match_candidate(self, image, scale, angle_deg, roi_center, roi_radius):
        """Match one candidate, optionally within a window around ``roi_center``.

        ``image``, ``roi_center`` and ``roi_radius`` are in search-image
        (downsampled) pixels; the returned centre is in full-resolution pixels.
        """
        candidate = self._candidate(scale, angle_deg, self.search_downsample)
        cand_h, cand_w = candidate.shape[:2]

        x0 = y0 = 0
        region = image
        if roi_center is not None:
            # Window large enough for the candidate centre to land anywhere
            # within roi_radius of the previous centre.
            x0 = max(0, int(np.floor(roi_center[0] - roi_radius - cand_w / 2.0)))
            y0 = max(0, int(np.floor(roi_center[1] - roi_radius - cand_h / 2.0)))
            x1 = min(
                image.shape[1],
                int(np.ceil(roi_center[0] + roi_radius + cand_w / 2.0)) + 1,
            )
            y1 = min(
                image.shape[0],
                int(np.ceil(roi_center[1] + roi_radius + cand_h / 2.0)) + 1,
            )
            region = image[y0:y1, x0:x1]

        if cand_h > region.shape[0] or cand_w > region.shape[1]:
            return None
        result = cv2.matchTemplate(region, candidate, cv2.TM_SQDIFF_NORMED)
        # TM_SQDIFF_NORMED is a mismatch cost: 0.0 is a perfect match, so the
        # best location is the minimum. Report 1 - cost as a higher-is-better
        # similarity so the rest of the node's "> best score" logic is unchanged.
        min_val, _, min_loc, _ = cv2.minMaxLoc(result)
        if not np.isfinite(min_val):
            return None
        score = 1.0 - float(min_val)
        location = min_loc
        return {
            "score": score,
            "center_u": (x0 + location[0] + cand_w / 2.0) / self.search_downsample,
            "center_v": (y0 + location[1] + cand_h / 2.0) / self.search_downsample,
            "angle_deg": float(angle_deg) % 360.0,
            "scale": float(scale),
        }

    def _search(self, image, pairs):
        """Return the best whole-image match over ``(scale, angle_deg)`` pairs."""
        best = None
        for scale, angle_deg in pairs:
            match = self._match_candidate(
                image, float(scale), float(angle_deg), None, 0.0
            )
            if match is not None and (best is None or match["score"] > best["score"]):
                best = match
        return best

    def find_best_match(self, search_image):
        """Full coarse-to-fine search over the whole BEV."""
        coarse_angles = np.arange(0.0, 360.0, self.coarse_angle_step_deg)
        coarse_scales = np.arange(
            self.min_template_scale,
            self.max_template_scale + self.template_scale_step * 0.5,
            self.template_scale_step,
        )
        best = self._search(search_image, product(coarse_scales, coarse_angles))
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
        refined = self._search(search_image, product(fine_scales, fine_angles))
        return refined if refined is not None else best

    def _track_roi(self):
        """Local search window (centre, radius) in search-image pixels."""
        roi_center = (
            self.track["center_u"] * self.search_downsample,
            self.track["center_v"] * self.search_downsample,
        )
        roi_radius = (
            self.track_search_radius_m / self.bev_scale * self.search_downsample
        )
        return roi_center, roi_radius

    def track_match(self, search_image):
        """Hill-climb angle/scale from the previous match within a local window.

        Each step matches the 4 angle/scale neighbours of the current best
        inside the position ROI and moves to the best improvement; it stops
        as soon as no neighbour improves, so a static dock costs ~5 matches.
        """
        angle_step = self.fine_angle_step_deg
        scale_step = self.template_scale_step / 2.0
        angle_n = int(self.track_angle_window_deg // angle_step)
        scale_n = int(self.track_scale_window // scale_step)
        roi_center, roi_radius = self._track_roi()
        evaluated = {}

        def evaluate(scale_i, angle_i):
            # Integer offsets from the previous pose, so (0, 0) is exact.
            key = (scale_i, angle_i)
            if key not in evaluated:
                scale = self.track["scale"] + scale_i * scale_step
                if (
                    abs(scale_i) > scale_n
                    or abs(angle_i) > angle_n
                    or scale < self.min_template_scale - 1e-6
                    or scale > self.max_template_scale + 1e-6
                ):
                    evaluated[key] = None
                else:
                    evaluated[key] = self._match_candidate(
                        search_image,
                        scale,
                        self.track["angle_deg"] + angle_i * angle_step,
                        roi_center,
                        roi_radius,
                    )
            return evaluated[key]

        current = (0, 0)
        best = evaluate(*current)
        if best is None:
            return None
        while True:
            step = None
            for d_scale, d_angle in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                neighbour = (current[0] + d_scale, current[1] + d_angle)
                match = evaluate(*neighbour)
                if match is not None and match["score"] > best["score"]:
                    best, step = match, neighbour
            if step is None:
                return best
            current = step

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

    def _publish_pose(self, msg, image_center, match):
        metres_per_pixel = self.bev_scale
        x_forward, y_left = self._bev_pixel_to_asv(
            match["center_u"], match["center_v"], image_center, metres_per_pixel
        )
        bearing_yaw = atan2(y_left, x_forward)
        template_rotation = radians(match["angle_deg"])
        x_forward_w_offset, y_left_w_offset = self.compute_entry_point(
            x_forward, y_left, template_rotation, self.entry_offset_dist
        )

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame_id,
                self.pose_frame_id,
                Time(),
            )
        except tf2_ros.TransformException as e:
            self.get_logger().error(f"Transform lookup failed: {e}")
            return None

        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = self.pose_frame_id
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
        return x_forward, y_left, bearing_yaw, metres_per_pixel

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

    def _draw_match(self, display, match):
        center_u = match["center_u"]
        center_v = match["center_v"]
        candidate = self._candidate(match["scale"], match["angle_deg"], 1.0)
        top_left_x = int(round(center_u - candidate.shape[1] / 2.0))
        top_left_y = int(round(center_v - candidate.shape[0] / 2.0))
        translation = np.array([[[top_left_x, top_left_y]]], dtype=np.int32)
        # Green: matched dock template.
        contours, _ = cv2.findContours(
            candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        translated_contours = [
            contour.astype(np.int32) + translation for contour in contours
        ]
        cv2.drawContours(display, translated_contours, -1, (0, 255, 0), 2)
        cv2.circle(
            display,
            (int(round(center_u)), int(round(center_v))),
            4,
            (0, 0, 255),
            -1,
        )

    def _draw_track_roi(self, display):
        roi_center, roi_radius = self._track_roi()
        cv2.circle(
            display,
            (
                int(round(roi_center[0] / self.search_downsample)),
                int(round(roi_center[1] / self.search_downsample)),
            ),
            int(round(roi_radius / self.search_downsample)),
            (255, 0, 255),
            1,
        )

    def template_callback(self, msg):
        try:
            bev_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as error:
            self.get_logger().error(f"Could not convert BEV image: {error}")
            return

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
            if self.track is not None:
                self._draw_track_roi(display)

        start = monotonic()
        search_image = cv2.resize(
            bev_image,
            None,
            fx=self.search_downsample,
            fy=self.search_downsample,
            interpolation=cv2.INTER_NEAREST,
        )
        match = None
        mode = "global"
        if self.track is not None:
            match = self.track_match(search_image)
            mode = "track"
            if match is None or match["score"] < self.match_threshold:
                self.get_logger().info(
                    "Dock track lost (score=%s), falling back to full search"
                    % ("n/a" if match is None else "%.3f" % match["score"])
                )
                match = None
                mode = "global"
        if match is None:
            match = self.find_best_match(search_image)
        elapsed_ms = (monotonic() - start) * 1000.0

        accepted = match is not None and match["score"] >= self.match_threshold
        self.track = match if accepted else None

        pose_result = None
        if accepted:
            pose_result = self._publish_pose(msg, image_center, match)

        if display is None:
            return

        label = "%s %.0fms" % (mode, elapsed_ms)
        if match is not None:
            self._draw_match(display, match)
            label += " score=%.3f angle=%.1f scale=%.2f" % (
                match["score"],
                match["angle_deg"],
                match["scale"],
            )
            if pose_result is not None:
                x_forward, y_left, bearing_yaw, metres_per_pixel = pose_result
                label += " x=%.2f y=%.2f yaw=%.1f mpp=%.3f" % (
                    x_forward,
                    y_left,
                    np.degrees(bearing_yaw),
                    metres_per_pixel,
                )
            elif not accepted:
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
        debug_msg = self.bridge.cv2_to_imgmsg(display, encoding="bgr8")
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)


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
