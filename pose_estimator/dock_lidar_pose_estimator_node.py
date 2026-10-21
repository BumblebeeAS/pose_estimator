from itertools import product
from math import cos, radians, sin
from pathlib import Path
from time import monotonic

import cv2
import numpy as np
import rclpy
import tf2_geometry_msgs
import tf2_ros
from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from pose_estimator.dock_debug import (
    clear_template_marker,
    dock_pose_marker,
    draw_debug_image,
    template_marker,
)


class DockPoseEstimator(LifecycleNode):
    """Continuously estimate dock pose from a mono BEV image using template matching.

    Every frame runs a full coarse-to-fine search over the whole BEV and all
    360 deg. Nothing is carried over between frames, so a wrong match never
    locks the search onto a false target.

    Candidates are scored with ``TM_CCOEFF_NORMED`` (-1..1, higher is better),
    which is invariant to brightness gain/offset, so the greyscale height BEV
    and the greyscale template need not share an absolute intensity scale.

    This node subclasses ``LifecycleNode`` directly because it estimates a 2D
    lidar pose, not a camera-frame PnP pose. ``PoseEstimatorPosePubNode`` waits
    for camera info and expects ``rvec``/``tvec`` inputs, so it does not fit
    here.

    Lifecycle: configure creates the publishers, activate subscribes to the
    BEV (the search only runs while active), deactivate unsubscribes. Set
    ``activate_on_start`` to activate without a lifecycle manager.
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

        # 3D template marker. The BEV grey level is the highest z per cell
        # (bev_node: bev_z_min -> 1, bev_z_max -> 255, 0 = empty), so these
        # must match bev_node's bev_z_min / bev_z_max (os0_filter min/max
        # height) to lift the template pixels to their true heights.
        self.bev_z_min = float(self.declare_parameter("bev_z_min", -0.65).value)
        self.bev_z_max = float(self.declare_parameter("bev_z_max", 0.5).value)
        # Draw every Nth template pixel as a cube; 1 draws all of them.
        self.template_marker_stride = int(
            self.declare_parameter("template_marker_stride", 2).value
        )
        self.declare_parameter("activate_on_start", False)

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

        self.pose_pub = None
        self.marker_pub = None
        self.template_marker_pub = None
        self.debug_pub = None
        self.bev_sub = None
        self.camera_info_sub = None

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

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.pose_pub = self.create_lifecycle_publisher(
            PoseStamped, self.pose_topic, 10
        )
        self.marker_pub = self.create_lifecycle_publisher(
            Marker, f"{self.pose_topic}/marker", 10
        )
        self.template_marker_pub = self.create_lifecycle_publisher(
            Marker, f"{self.pose_topic}/template_marker", 10
        )
        self.debug_pub = (
            self.create_lifecycle_publisher(Image, self.debug_topic, 10)
            if self.debug_topic
            else None
        )
        self.get_logger().info("Configured")
        return super().on_configure(state)

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        # Depth 1 so frames arriving during a slow search are dropped rather
        # than queued; a mutually exclusive group serialises searches.
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
        self.get_logger().info(f"Activated, subscribed to {self.input_topic}")
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self._destroy_subscriptions()
        self.get_logger().info("Deactivated")
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self._destroy_publishers()
        self.candidate_cache = {}
        self.get_logger().info("Cleaned up")
        return super().on_cleanup(state)

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        # Shutdown can be entered from any primary state, so tear down
        # idempotently.
        self._destroy_subscriptions()
        self._destroy_publishers()
        return TransitionCallbackReturn.SUCCESS

    def _destroy_subscriptions(self):
        for name in ("bev_sub", "camera_info_sub"):
            sub = getattr(self, name)
            if sub is not None:
                self.destroy_subscription(sub)
                setattr(self, name, None)

    def _destroy_publishers(self):
        for name in ("pose_pub", "marker_pub", "template_marker_pub", "debug_pub"):
            pub = getattr(self, name)
            if pub is not None:
                self.destroy_publisher(pub)
                setattr(self, name, None)

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
        if self.bev_z_max <= self.bev_z_min:
            raise ValueError("bev_z_max must be > bev_z_min")
        if self.template_marker_stride < 1:
            raise ValueError("template_marker_stride must be >= 1")

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

    def _match_candidate(self, image, scale, angle_deg):
        """Match one candidate over the whole search image.

        ``image`` is in search-image (downsampled) pixels; the returned centre
        is in full-resolution pixels.
        """
        candidate = self._candidate(scale, angle_deg, self.search_downsample)
        cand_h, cand_w = candidate.shape[:2]

        if cand_h > image.shape[0] or cand_w > image.shape[1]:
            return None
        result = cv2.matchTemplate(image, candidate, cv2.TM_CCOEFF_NORMED)
        # TM_CCOEFF_NORMED is a similarity in [-1, 1]: the best location is
        # the maximum.
        _, max_val, _, location = cv2.minMaxLoc(result)
        if not np.isfinite(max_val):
            return None
        return {
            "score": float(max_val),
            "center_u": (location[0] + cand_w / 2.0) / self.search_downsample,
            "center_v": (location[1] + cand_h / 2.0) / self.search_downsample,
            "angle_deg": float(angle_deg) % 360.0,
            "scale": float(scale),
        }

    def _search(self, image, pairs):
        """Return the best whole-image match over ``(scale, angle_deg)`` pairs."""
        best = None
        for scale, angle_deg in pairs:
            match = self._match_candidate(image, float(scale), float(angle_deg))
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

    @staticmethod
    def _bev_pixel_to_asv(center_u, center_v, image_center, metres_per_pixel):
        center_x, center_y = image_center
        # New BEV mapping: image x is FLU +x; image y points opposite FLU +y.
        x_forward = (center_u - center_x) * metres_per_pixel
        y_left = -(center_v - center_y) * metres_per_pixel
        return x_forward, y_left

    def _publish_pose(self, msg, x_forward, y_left, angle_deg):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame_id, self.pose_frame_id, Time()
            )
        except tf2_ros.TransformException as e:
            self.get_logger().error(f"Transform lookup failed: {e}")
            return False

        yaw = radians(angle_deg)
        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = self.pose_frame_id
        pose.pose.position.x = x_forward
        pose.pose.position.y = y_left
        pose.pose.orientation.z = sin(yaw / 2.0)
        pose.pose.orientation.w = cos(yaw / 2.0)

        transformed_pose = tf2_geometry_msgs.do_transform_pose_stamped(pose, transform)
        self.pose_pub.publish(transformed_pose)
        self.marker_pub.publish(dock_pose_marker(transformed_pose))
        return True

    def template_callback(self, msg):
        try:
            bev_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as error:
            self.get_logger().error(f"Could not convert BEV image: {error}")
            return

        image_center = self.image_center or (
            (msg.width - 1) / 2.0,
            (msg.height - 1) / 2.0,
        )

        start = monotonic()
        search_image = cv2.resize(
            bev_image,
            None,
            fx=self.search_downsample,
            fy=self.search_downsample,
            interpolation=cv2.INTER_NEAREST,
        )
        match = self.find_best_match(search_image)
        elapsed_ms = (monotonic() - start) * 1000.0

        accepted = match is not None and match["score"] >= self.match_threshold
        candidate = (
            self._candidate(match["scale"], match["angle_deg"], 1.0)
            if match is not None
            else None
        )
        marker_header = Header(stamp=msg.header.stamp, frame_id=self.pose_frame_id)

        position = None
        if accepted:
            x_forward, y_left = self._bev_pixel_to_asv(
                match["center_u"], match["center_v"], image_center, self.bev_scale
            )
            if self._publish_pose(msg, x_forward, y_left, match["angle_deg"]):
                position = (x_forward, y_left)
            self.template_marker_pub.publish(
                template_marker(
                    marker_header,
                    candidate,
                    match,
                    lambda u, v: self._bev_pixel_to_asv(
                        u, v, image_center, self.bev_scale
                    ),
                    self.bev_scale,
                    self.bev_z_min,
                    self.bev_z_max,
                    self.template_marker_stride,
                )
            )
        else:
            self.template_marker_pub.publish(clear_template_marker(marker_header))

        if self.debug_pub is None:
            return
        display = draw_debug_image(
            bev_image, image_center, elapsed_ms, match, candidate, accepted, position
        )
        debug_msg = self.bridge.cv2_to_imgmsg(display, encoding="bgr8")
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DockPoseEstimator()
    node.trigger_configure()
    if node.get_parameter("activate_on_start").get_parameter_value().bool_value:
        node.trigger_activate()
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
