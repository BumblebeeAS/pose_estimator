import copy
import math
from typing import Optional

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from yolo_msgs.msg import Detection, DetectionArray

from pose_estimator.config.robotx26_object_points import (
    UPPER_FROM_LOWER_WINDOW_DX,
    UPPER_FROM_LOWER_WINDOW_DY,
    UPPER_FROM_LOWER_WINDOW_DZ,
    WINDOW_FRAME_REMAP,
    WINDOW_OBJECT_POINTS_DICT,
)
from pose_estimator.utils.detections import (
    filter_detections_by_num_points,
    filter_detections_by_score,
    get_best_detections_per_class,
    get_detection_obb,
    get_detection_polygon,
    match_polygon_points,
)
from pose_estimator.utils.pose_estimator import (
    get_detection_centroid_position,
    get_object_pose,
    get_object_pose_from_detection_using_best_fit_quad,
    refine_object_pose,
)
from pose_estimator.utils.pose_estimator_node import (
    PoseEstimatorPosePubNode,
    get_translation_quaternion,
)


def _quaternion_matrix(quaternion: Quaternion) -> np.ndarray:
    """Return a 3x3 rotation matrix for a geometry quaternion (x, y, z, w)."""
    q = np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w], dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("window pose has an invalid orientation quaternion")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _matrix_quaternion(matrix: np.ndarray) -> Quaternion:
    """Convert a 3x3 rotation matrix to a normalized geometry quaternion."""
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    return Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))


def create_standoff_pose(
    window_pose: PoseStamped,
    standoff_distance: float,
    window_width: float,
    window_height: float,
) -> PoseStamped:
    """Compute an ASV pose centered on the window normal at a fixed standoff.

    The window PnP origin is its top-left corner; its local +Y axis points down
    the window and local +Z is the plane normal. The normal is oriented toward
    the camera, and the returned ASV pose faces back toward the window.
    """
    if not math.isfinite(standoff_distance) or standoff_distance <= 0.0:
        raise ValueError("standoff_distance must be finite and positive")
    if not math.isfinite(window_width) or window_width <= 0.0:
        raise ValueError("window_width must be finite and positive")
    if not math.isfinite(window_height) or window_height <= 0.0:
        raise ValueError("window_height must be finite and positive")

    rotation = _quaternion_matrix(window_pose.pose.orientation)
    origin = np.array(
        [
            window_pose.pose.position.x,
            window_pose.pose.position.y,
            window_pose.pose.position.z,
        ],
        dtype=float,
    )
    center = origin + rotation[:, 0] * (window_width / 2.0)
    center += rotation[:, 1] * (window_height / 2.0)
    normal = rotation[:, 2]

    # PnP may choose either normal direction; select the side containing camera.
    if float(np.dot(normal, -center)) < 0.0:
        normal = -normal

    forward = -normal
    up = -rotation[:, 1]
    up -= np.dot(up, forward) * forward
    up_norm = float(np.linalg.norm(up))
    forward_norm = float(np.linalg.norm(forward))
    if up_norm < 1e-6 or forward_norm < 1e-6:
        raise ValueError("window pose does not define a usable standoff orientation")
    forward /= forward_norm
    up /= up_norm
    left = np.cross(up, forward)
    left_norm = float(np.linalg.norm(left))
    if left_norm < 1e-6:
        raise ValueError("window up and normal axes are degenerate")
    left /= left_norm
    up = np.cross(forward, left)

    target = copy.deepcopy(window_pose)
    position = center + normal * standoff_distance
    target.pose.position = Point(
        x=float(position[0]), y=float(position[1]), z=float(position[2])
    )
    target.pose.orientation = _matrix_quaternion(
        np.column_stack((forward, left, up))
    )
    return target


def window_centre_pose(
    window_pose: PoseStamped,
    window_width: float,
    window_height: float,
) -> Optional[PoseStamped]:
    """The window's centre with the PnP window axes, or None if it faces away.

    The PnP origin is the top-left corner (+x right, +y down the window, +z the
    plane normal). The centre keeps those axes, so +z points into the window,
    away from the camera; a pose whose +z points back at the camera (a
    degenerate PnP solution) returns None rather than a flipped frame.
    """
    if not (math.isfinite(window_width) and window_width > 0.0):
        raise ValueError("window_width must be finite and positive")
    if not (math.isfinite(window_height) and window_height > 0.0):
        raise ValueError("window_height must be finite and positive")
    rotation = _quaternion_matrix(window_pose.pose.orientation)
    origin = np.array(
        [
            window_pose.pose.position.x,
            window_pose.pose.position.y,
            window_pose.pose.position.z,
        ],
        dtype=float,
    )
    centre = origin + rotation[:, 0] * (window_width / 2.0)
    centre += rotation[:, 1] * (window_height / 2.0)
    # The camera sits at the optical frame origin: +z must point away from it.
    if float(np.dot(rotation[:, 2], centre)) <= 0.0:
        return None
    target = copy.deepcopy(window_pose)
    target.pose.position = Point(x=float(centre[0]), y=float(centre[1]), z=float(centre[2]))
    return target


def compute_upper_window_pose(
    lower_window_pose: PoseStamped,
    dx: float = UPPER_FROM_LOWER_WINDOW_DX,
    dy: float = UPPER_FROM_LOWER_WINDOW_DY,
    dz: float = UPPER_FROM_LOWER_WINDOW_DZ,
) -> PoseStamped:
    """Compute the upper window pose from the lower window pose using the build spec transform.

    In the window frame (+x rightwards along the board, +y downwards along the board, +z into the board):
    The upper window's top-left corner is offset from the lower window's top-left corner by:
    - dx = -0.440 m (440 mm to the left)
    - dy = -0.250 m (250 mm higher up, negative in +y down convention)
    - dz = 0.0 m (coplanar on the board)
    """
    rotation = _quaternion_matrix(lower_window_pose.pose.orientation)
    origin = np.array(
        [
            lower_window_pose.pose.position.x,
            lower_window_pose.pose.position.y,
            lower_window_pose.pose.position.z,
        ],
        dtype=float,
    )
    upper_pos = origin + rotation[:, 0] * dx + rotation[:, 1] * dy + rotation[:, 2] * dz
    target = copy.deepcopy(lower_window_pose)
    target.pose.position = Point(
        x=float(upper_pos[0]), y=float(upper_pos[1]), z=float(upper_pos[2])
    )
    return target


def is_lower_window_pose(
    window_pose: PoseStamped,
    y_threshold: float = 0.0,
    single_visible_window: bool = False,
) -> bool:
    """Classify a window as lower in the camera optical frame (+Y points down).

    This matches the firefighting report's existing window-ID rule: Y below the
    threshold is upper, and Y greater than or equal to it is lower.
    """
    y = float(window_pose.pose.position.y)
    if not math.isfinite(y):
        return False
    if single_visible_window:
        return True
    return math.isfinite(y_threshold) and y >= y_threshold


def select_lower_window_detection(
    detections: list[Detection],
    image_center_x: Optional[float] = None,
    row_tolerance: float = 0.5,
    max_column_distance: Optional[float] = None,
) -> tuple[Optional[Detection], bool]:
    """Pick the lower window to hold at from all visible window detections.

    Windows are grouped into image rows by bbox centre v (+v points down). When
    more than one row is visible, the bottom row is the lower row regardless of
    camera height, range or pitch. Within the bottom row, the window
    horizontally closest to ``image_center_x`` (the slip being faced) wins; with
    no usable centre, the highest score wins.

    When ``max_column_distance`` is set, bottom-row windows whose bbox centre is
    farther than this many pixels from ``image_center_x`` are excluded, so that
    a lower window from an adjacent dock bay is not selected.

    Returns ``(detection, rows_distinguished)``. ``rows_distinguished`` is False
    when every visible window is in one row, so upper vs lower cannot be told
    from the image and the caller should fall back to ``is_lower_window_pose``.
    """
    if not detections:
        return None, False
    lowest = max(detections, key=lambda d: d.bbox.center.position.y)
    lowest_v = float(lowest.bbox.center.position.y)
    tolerance = row_tolerance * max(float(lowest.bbox.size.y), 1.0)
    bottom_row = [
        d for d in detections if d.bbox.center.position.y >= lowest_v - tolerance
    ]
    rows_distinguished = len(bottom_row) < len(detections)
    # Hard horizontal filter: exclude windows from adjacent dock bays.
    if (
        max_column_distance is not None
        and max_column_distance > 0.0
        and image_center_x is not None
        and math.isfinite(image_center_x)
    ):
        column_filtered = [
            d for d in bottom_row
            if abs(d.bbox.center.position.x - image_center_x) <= max_column_distance
        ]
        if column_filtered:
            bottom_row = column_filtered
        else:
            return None, rows_distinguished
    if image_center_x is not None and math.isfinite(image_center_x):
        def key(d):
            return (abs(d.bbox.center.position.x - image_center_x), -d.score)
    else:
        def key(d):
            return -d.score
    return min(bottom_row, key=key), rows_distinguished


def _polygon_to_binary_mask(
    polygon: np.ndarray, resolution_wh: tuple[int, int]
) -> np.ndarray:
    """Create a binary uint8 mask (0 or 255) from polygon coordinate points."""
    w, h = int(resolution_wh[0]), int(resolution_wh[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.asarray(polygon, dtype=np.float32)
    if len(poly) >= 3:
        pts = np.round(poly).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], color=255)
    return mask


class DockwinPoseEstimator(PoseEstimatorPosePubNode):
    """Estimate dock-window poses using OBB-assisted planar PnP with optional stereo depth.

    ``best_fit_quad`` keeps the previous generic estimator available for
    comparison and fallback.  The default ``improved`` mode follows the bin
    estimator's angle-assisted point matching, which is less sensitive to the
    cyclic ordering of noisy segmentation corners.

    When ``use_depth_image`` is enabled and ``input_depth_image_topic`` is provided,
    the node extracts the median stereo depth within the eroded segmentation mask
    to anchor the 3D translation distance while retaining PnP orientation.
    """

    def __init__(self):
        super().__init__("dockwin_pose_estimator_node")

        self.object_points = {
            class_name: np.asarray(points, dtype=np.float32)
            for class_name, points in WINDOW_OBJECT_POINTS_DICT.items()
        }
        self.frame_name_remap = WINDOW_FRAME_REMAP.copy()

        if set(self.object_points) != set(self.frame_name_remap):
            raise ValueError(
                "Window object-point and output-frame mappings must have the same keys"
            )
        if len(set(self.frame_name_remap.values())) != len(self.frame_name_remap):
            raise ValueError("Window output-frame names must be unique")

        self.input_detection_classes = list(self.object_points)
        self.estimation_method = (
            self.declare_parameter("estimation_method", "improved")
            .get_parameter_value()
            .string_value
        )
        if self.estimation_method not in {"improved", "best_fit_quad"}:
            raise ValueError(
                "estimation_method must be 'improved' or 'best_fit_quad'"
            )

        self.max_reprojection_error = (
            self.declare_parameter("max_reprojection_error", 2.0)
            .get_parameter_value()
            .double_value
        )
        self.min_inliers = (
            self.declare_parameter("min_inliers", 4)
            .get_parameter_value()
            .integer_value
        )
        if self.max_reprojection_error <= 0.0:
            raise ValueError("max_reprojection_error must be positive")
        if self.min_inliers < 4:
            raise ValueError("min_inliers must be at least 4 for a quadrilateral")
        self.depth_filter_alpha = (
            self.declare_parameter("depth_filter_alpha", 0.2)
            .get_parameter_value()
            .double_value
        )
        if not 0.0 <= self.depth_filter_alpha <= 1.0:
            raise ValueError("depth_filter_alpha must be in the range [0, 1]")
        self.standoff_distance = (
            self.declare_parameter("standoff_distance", 2.5)
            .get_parameter_value()
            .double_value
        )
        if not math.isfinite(self.standoff_distance) or self.standoff_distance <= 0.0:
            raise ValueError("standoff_distance must be finite and positive")
        self.lower_window_y_threshold = (
            self.declare_parameter("lower_window_y_threshold", 0.0)
            .get_parameter_value()
            .double_value
        )
        if not math.isfinite(self.lower_window_y_threshold):
            raise ValueError("lower_window_y_threshold must be finite")
        self.assume_single_visible_window_is_lower = (
            self.declare_parameter("assume_single_visible_window_is_lower", True)
            .get_parameter_value()
            .bool_value
        )
        self.standoff_pose_topic = (
            self.declare_parameter("standoff_pose_topic", "dockwin/standoff_pose")
            .get_parameter_value()
            .string_value
        )
        if not self.standoff_pose_topic.strip():
            raise ValueError("standoff_pose_topic must not be empty")
        # Lower-window pose (top-left origin) for firefighting standoff and targeting.
        self.lower_window_pose_topic = (
            self.declare_parameter("lower_window_pose_topic", "dockwin/lower_window/pose")
            .get_parameter_value()
            .string_value
        )
        if not self.lower_window_pose_topic.strip():
            raise ValueError("lower_window_pose_topic must not be empty")

        # Upper-window pose computed from lower-window via known build spec transform.
        self.upper_window_pose_topic = (
            self.declare_parameter("upper_window_pose_topic", "dockwin/upper_window/pose")
            .get_parameter_value()
            .string_value
        )
        if not self.upper_window_pose_topic.strip():
            raise ValueError("upper_window_pose_topic must not be empty")

        detections_topic = (
            self.declare_parameter("input_detections_topic", "yolo/detections")
            .get_parameter_value()
            .string_value
        )
        self.input_depth_image_topic = (
            self.declare_parameter("input_depth_image_topic", "")
            .get_parameter_value()
            .string_value
        )
        self.use_depth_image = (
            self.declare_parameter("use_depth_image", True)
            .get_parameter_value()
            .bool_value
        )
        self.depth_sync_slop = (
            self.declare_parameter("depth_sync_slop", 0.1)
            .get_parameter_value()
            .double_value
        )
        self.mask_erosion_iters = (
            self.declare_parameter("mask_erosion_iters", 2)
            .get_parameter_value()
            .integer_value
        )
        self.min_depth = (
            self.declare_parameter("min_depth", 0.2)
            .get_parameter_value()
            .double_value
        )
        self.max_depth = (
            self.declare_parameter("max_depth", 15.0)
            .get_parameter_value()
            .double_value
        )
        self.min_confidence = (
            self.declare_parameter("min_confidence", 0.3)
            .get_parameter_value()
            .double_value
        )
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in the range [0, 1]")

        self._pose_publishers = {
            frame_name: self.create_publisher(
                PoseStamped, f"{frame_name}/pose", qos_profile_sensor_data
            )
            for frame_name in self.frame_name_remap.values()
        }
        self._standoff_pose_publisher = self.create_publisher(
            PoseStamped, self.standoff_pose_topic, qos_profile_sensor_data
        )
        self._lower_window_pose_publisher = self.create_publisher(
            PoseStamped, self.lower_window_pose_topic, qos_profile_sensor_data
        )
        self._upper_window_pose_publisher = self.create_publisher(
            PoseStamped, self.upper_window_pose_topic, qos_profile_sensor_data
        )
        self._filtered_tvecs = {}
        self._target_bay_center_x: Optional[float] = None
        self._detected_target_window_id: Optional[int] = None

        self.target_window_id_service_topic = (
            self.declare_parameter("target_window_id_service_topic", "~/get_target_window_id")
            .get_parameter_value()
            .string_value
        )
        self.default_target_window_id = (
            self.declare_parameter("default_target_window_id", 1)
            .get_parameter_value()
            .integer_value
        )
        self._target_window_id_service = self.create_service(
            Trigger,
            self.target_window_id_service_topic,
            self._get_target_window_id_callback,
        )

        self.target_window_classes = list(
            self.declare_parameter(
                "target_window_classes", ["red_window", "green_window", "blue_window"]
            )
            .get_parameter_value()
            .string_array_value
        )
        # Backward compatibility if target_window_class string parameter is provided
        self.declare_parameter("target_window_class", "")
        single_target = (
            self.get_parameter("target_window_class")
            .get_parameter_value()
            .string_value
        )
        if single_target and single_target not in self.target_window_classes:
            self.target_window_classes.insert(0, single_target)

        self.max_column_distance = (
            self.declare_parameter("max_column_distance", 0.0)
            .get_parameter_value()
            .double_value
        )
        if self.max_column_distance < 0.0:
            raise ValueError("max_column_distance must be non-negative")

        if self.use_depth_image and self.input_depth_image_topic:
            self.cv_bridge = CvBridge()
            self._detections_sub = message_filters.Subscriber(
                self,
                DetectionArray,
                detections_topic,
                qos_profile=qos_profile_sensor_data,
            )
            self._depth_sub = message_filters.Subscriber(
                self,
                Image,
                self.input_depth_image_topic,
                qos_profile=qos_profile_sensor_data,
            )
            self._time_sync = message_filters.ApproximateTimeSynchronizer(
                [self._detections_sub, self._depth_sub],
                queue_size=10,
                slop=self.depth_sync_slop,
            )
            self._time_sync.registerCallback(self.synced_detections_callback)
        else:
            self.detections_sub = self.create_subscription(
                DetectionArray,
                detections_topic,
                self.detections_callback,
                qos_profile_sensor_data,
            )

    @property
    def pose_publishers(self):
        return self._pose_publishers

    def synced_detections_callback(
        self, detections_msg: DetectionArray, depth_msg: Image
    ):
        try:
            depth_img = self.cv_bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
        except Exception as exc:
            self.get_logger().warn(
                f"Failed to convert depth image from {self.input_depth_image_topic}: {exc}"
            )
            depth_img = None

        self._process_detections(detections_msg, depth_img)

    def detections_callback(self, msg: DetectionArray):
        self._process_detections(msg, depth_img=None)

    def _process_detections(
        self, msg: DetectionArray, depth_img: Optional[np.ndarray] = None
    ):
        filtered_detections = filter_detections_by_num_points(msg, 3)
        if self.min_confidence > 0.0:
            filtered_detections = filter_detections_by_score(
                filtered_detections, self.min_confidence
            )
        best_detections = get_best_detections_per_class(
            filtered_detections, self.input_detection_classes
        )
        window_detections = [
            detection
            for detection in filtered_detections.detections
            if detection.class_name in self.input_detection_classes
        ]
        # Latch or update the horizontal pixel position of the targeted dock bay
        # so select_lower_window_detection can exclude adjacent-bay windows.
        # target_window_classes (e.g. ["red_window", "green_window"]) are checked
        # in priority order. When red turns off and green activates post-extinguish,
        # green continues tracking the bay column rather than relying on a stale x.
        target_det = None
        for cls in self.target_window_classes:
            matching = [
                d for d in filtered_detections.detections if d.class_name == cls
            ]
            if not matching:
                continue
            if self._target_bay_center_x is not None:
                if self.max_column_distance > 0.0:
                    in_column = [
                        d for d in matching
                        if abs(d.bbox.center.position.x - self._target_bay_center_x) <= self.max_column_distance
                    ]
                    if in_column:
                        matching = in_column
                target_det = min(
                    matching,
                    key=lambda d: abs(d.bbox.center.position.x - self._target_bay_center_x)
                )
            else:
                target_det = max(matching, key=lambda d: d.score)
            break

        if target_det is not None:
            self._target_bay_center_x = float(target_det.bbox.center.position.x)
        single_visible_window = (
            self.assume_single_visible_window_is_lower
            and len(window_detections) == 1
        )

        # Per-class window poses (best detection of each class), as before.
        # Keyed by id(detection) so the standoff can reuse a pose already
        # computed for the same detection.
        published_poses: dict[int, PoseStamped] = {}
        for class_name, detection in best_detections.items():
            estimate = self._estimate_detection_pose(class_name, detection, depth_img)
            if estimate is None:
                continue
            rvec, tvec = estimate
            tvec = self._filter_translation(class_name, tvec)
            frame_name = self.frame_name_remap[class_name]
            object_points = np.hstack(
                [
                    self.object_points[class_name],
                    np.zeros(
                        (len(self.object_points[class_name]), 1), dtype=np.float32
                    ),
                ]
            )
            pose = self.publish_data(
                tvec, rvec, object_points, msg.header, frame_name
            )
            if pose is not None:
                published_poses[id(detection)] = pose

        self._publish_lower_window_standoff(
            window_detections,
            published_poses,
            msg.header,
            depth_img,
            single_visible_window,
        )

    def _estimate_detection_pose(
        self,
        class_name: str,
        detection: Detection,
        depth_img: Optional[np.ndarray] = None,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """PnP (+ optional depth) pose for one detection, or None if unusable."""
        try:
            if self.estimation_method == "best_fit_quad":
                rvec, tvec = self._estimate_with_best_fit_quad(class_name, detection)
            else:
                rvec, tvec = self._estimate_with_oriented_quad(class_name, detection)
        except Exception as exc:
            self.get_logger().warn(f"Pose estimation failed for {class_name}: {exc}")
            return None

        if depth_img is not None and self.use_depth_image:
            measured_depth = self._extract_mask_depth(depth_img, detection)
            if np.isfinite(measured_depth) and measured_depth > 0.0:
                tvec = self._refine_translation_with_depth(
                    detection, tvec, measured_depth
                )

        depth = float(tvec[2, 0])
        if not np.isfinite(depth) or depth <= 0.0:
            self.get_logger().warn(
                f"Rejecting {class_name} pose with invalid depth: {depth:.3f} m"
            )
            return None
        return rvec, tvec

    def _make_pose(self, tvec: np.ndarray, rvec: np.ndarray, header) -> Optional[PoseStamped]:
        try:
            t, q = get_translation_quaternion(tvec, rvec)
        except np.linalg.LinAlgError as e:
            self.get_logger().warn(f"Error in mat2quat, failed to convert R: {e}")
            return None
        except Exception as e:
            self.get_logger().warn(f"Rodrigues conversion failed: {e}")
            return None

        pose = PoseStamped()
        pose.header = header
        pose.pose.position = Point(
            x=float(t[0]), y=float(t[1]), z=float(t[2])
        )
        pose.pose.orientation = Quaternion(
            x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])
        )
        return pose

    def publish_data(
        self,
        tvec: np.ndarray,
        rvec: np.ndarray,
        object_points: np.ndarray,
        header,
        object_frame_id: str,
    ) -> Optional[PoseStamped]:
        pose = self._make_pose(tvec, rvec, header)
        if pose is not None:
            self.pose_publishers[object_frame_id].publish(pose)
        return pose

    def _publish_lower_window_standoff(
        self,
        window_detections: list[Detection],
        published_poses: dict[int, PoseStamped],
        header,
        depth_img: Optional[np.ndarray],
        single_visible_window: bool,
    ) -> None:
        """Publish the lower window the firefighting tree holds at: its top-left
        pose on ``lower_window_pose_topic``, the derived upper-window pose on
        ``upper_window_pose_topic``, and the derived standoff pose.

        The lower window is chosen from all visible windows (not just the best
        per class) by image row; see ``select_lower_window_detection``. When
        only one row is visible, fall back to the optical-frame Y rule.

        When ``_target_bay_center_x`` has been latched (from the target window
        class), it replaces the fixed camera ``cx`` as the horizontal anchor and
        ``max_column_distance`` applies as a hard cutoff to reject windows from
        adjacent dock bays.
        """
        # Use the latched target bay x when available; fall back to camera cx.
        ref_x = (
            self._target_bay_center_x
            if self._target_bay_center_x is not None
            else float(self.camera.cx)
        )
        col_dist = self.max_column_distance if self.max_column_distance > 0.0 else None
        detection, rows_distinguished = select_lower_window_detection(
            window_detections,
            image_center_x=ref_x,
            max_column_distance=col_dist,
        )
        if detection is None:
            return
        class_name = detection.class_name
        pose = published_poses.get(id(detection))
        if pose is None:
            estimate = self._estimate_detection_pose(class_name, detection, depth_img)
            if estimate is None:
                return
            pose = self._make_pose(estimate[1], estimate[0], header)
            if pose is None:
                return
        # Single row visible: can't tell upper from lower in the image, so use
        # the camera optical frame rule (+Y points down), matching the tree's
        # existing upper/lower report selection.
        if not rows_distinguished and not is_lower_window_pose(
            pose,
            self.lower_window_y_threshold,
            single_visible_window=single_visible_window,
        ):
            return
        object_frame_id = self.frame_name_remap[class_name]
        points = self.object_points[class_name]
        window_width = float(np.max(points[:, 0]) - np.min(points[:, 0]))
        window_height = float(np.max(points[:, 1]) - np.min(points[:, 1]))
        centre_pose = window_centre_pose(pose, window_width, window_height)
        if centre_pose is None:
            self.get_logger().warning(
                f"Lower window {object_frame_id} pose faces the camera; not publishing it",
                throttle_duration_sec=5.0,
            )
            return

        # 1. Publish lower window with top-left origin convention
        self._lower_window_pose_publisher.publish(pose)

        # 2. Compute and publish upper window pose via known build spec transform
        upper_pose = compute_upper_window_pose(pose)
        self._upper_window_pose_publisher.publish(upper_pose)
        try:
            standoff_pose = create_standoff_pose(
                pose, self.standoff_distance, window_width, window_height
            )
        except ValueError as exc:
            self.get_logger().warning(
                f"Could not compute ASV standoff pose for {object_frame_id}: {exc}"
            )
            return
        self._standoff_pose_publisher.publish(standoff_pose)

        # 3. Classify target red window ID (1 = upper, 2 = lower)
        red_detections = [d for d in window_detections if d.class_name == "red_window"]
        if red_detections:
            red_det = red_detections[0]
            if lower_det is not None and id(lower_det) == id(red_det):
                self._detected_target_window_id = 2
            elif rows_distinguished and lower_det is not None:
                self._detected_target_window_id = 1
            else:
                red_pose = published_poses.get(id(red_det))
                if red_pose is not None:
                    p_red = np.array(
                        [red_pose.pose.position.x, red_pose.pose.position.y, red_pose.pose.position.z]
                    )
                    p_lower = np.array(
                        [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
                    )
                    p_upper = np.array(
                        [upper_pose.pose.position.x, upper_pose.pose.position.y, upper_pose.pose.position.z]
                    )
                    d_lower = float(np.linalg.norm(p_red - p_lower))
                    d_upper = float(np.linalg.norm(p_red - p_upper))
                    self._detected_target_window_id = 1 if d_upper < d_lower else 2

    def _get_target_window_id_callback(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Return the calculated target window ID (1 = upper, 2 = lower) for RoboCommand reporting."""
        if self._detected_target_window_id is not None:
            response.success = True
            response.message = str(self._detected_target_window_id)
        else:
            response.success = False
            response.message = str(self.default_target_window_id)
        return response

    def _extract_mask_depth(
        self, depth_img: np.ndarray, detection: Detection
    ) -> float:
        """Extract median depth in meters from the YOLO detection mask."""
        if depth_img is None or detection is None:
            return float("nan")

        mask_polygon = get_detection_polygon(detection)
        if len(mask_polygon) < 3:
            return float("nan")

        h, w = depth_img.shape[:2]
        det_w = getattr(detection.mask, "width", w) or w
        det_h = getattr(detection.mask, "height", h) or h

        if (det_w, det_h) != (w, h):
            scale_x = float(w) / float(det_w)
            scale_y = float(h) / float(det_h)
            scaled_polygon = mask_polygon * np.array([scale_x, scale_y], dtype=np.float32)
        else:
            scaled_polygon = mask_polygon

        binary_mask = _polygon_to_binary_mask(scaled_polygon, (w, h))

        if self.mask_erosion_iters > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eroded = cv2.erode(
                binary_mask, kernel, iterations=self.mask_erosion_iters
            )
            if np.any(eroded > 0):
                binary_mask = eroded

        masked_depth = depth_img[binary_mask > 0].astype(np.float32)
        if masked_depth.size == 0:
            return float("nan")

        # Auto-detect millimeter encoding (e.g. 16UC1 from OAK-D depth)
        if np.issubdtype(depth_img.dtype, np.integer) or (
            np.nanmedian(masked_depth) > 50.0
        ):
            masked_depth = masked_depth / 1000.0

        valid_mask = (
            np.isfinite(masked_depth)
            & (masked_depth >= self.min_depth)
            & (masked_depth <= self.max_depth)
        )
        valid_depths = masked_depth[valid_mask]

        if valid_depths.size == 0:
            return float("nan")

        return float(np.median(valid_depths))

    def _refine_translation_with_depth(
        self, detection: Detection, tvec: np.ndarray, measured_depth: float
    ) -> np.ndarray:
        """Refine translation vector using measured stereo depth."""
        pnp_depth = float(tvec[2, 0])
        if pnp_depth > 1e-4:
            scale = float(measured_depth) / pnp_depth
            refined_tvec = (tvec * scale).astype(np.float64)
            refined_tvec[2, 0] = float(measured_depth)
            return refined_tvec
        else:
            centroid_pos = get_detection_centroid_position(
                detection, self.camera, float(measured_depth)
            )
            return centroid_pos.reshape((3, 1)).astype(np.float64)

    def _estimate_with_oriented_quad(self, class_name, detection):
        object_points_2d = self.object_points[class_name]
        angle, image_points = get_detection_obb(detection)
        angle = (angle + 90.0) % 180.0 - 90.0

        # Rotate the known rectangle during matching, as in the bin estimator.
        # This uses the known 0.21 m x 0.29 m aspect ratio from the configuration.
        matched_object_points, image_points = self._match_points(
            object_points_2d, image_points, angle
        )
        object_points = np.hstack(
            [
                matched_object_points,
                np.zeros((len(matched_object_points), 1), dtype=np.float32),
            ]
        )

        rvec, tvec, inliers = get_object_pose(
            self.camera,
            object_points,
            image_points,
            max_reprojection_error=self.max_reprojection_error,
        )
        if inliers is None or len(inliers) < self.min_inliers:
            count = 0 if inliers is None else len(inliers)
            raise ValueError(f"Only {count} PnP inliers found")

        rvec, tvec = refine_object_pose(
            self.camera, object_points, image_points, rvec, tvec
        )
        mean_error, max_error = self._reprojection_errors(
            object_points, image_points, rvec, tvec
        )
        if max_error > self.max_reprojection_error:
            raise ValueError(
                f"Reprojection error too large: mean={mean_error:.2f}, "
                f"max={max_error:.2f} pixels"
            )
        return rvec, tvec

    def _estimate_with_best_fit_quad(self, class_name, detection):
        # Preserve the original generic implementation as an explicit fallback.
        return get_object_pose_from_detection_using_best_fit_quad(
            self.camera, self.object_points[class_name], detection, self.get_logger()
        )

    @staticmethod
    def _match_points(object_points, image_points, angle):
        return match_polygon_points(object_points, image_points, A_angle=angle)

    def _reprojection_errors(self, object_points, image_points, rvec, tvec):
        projected_points, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.camera.camera_matrix(),
            self.camera.dist_coeffs(),
        )
        errors = np.linalg.norm(
            projected_points.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1
        )
        return float(np.mean(errors)), float(np.max(errors))

    def _filter_translation(self, class_name, tvec):
        previous = self._filtered_tvecs.get(class_name)
        if (
            previous is None
            or self.depth_filter_alpha in (0.0, 1.0)
        ):
            filtered = tvec.copy()
        else:
            alpha = self.depth_filter_alpha
            filtered = alpha * tvec + (1.0 - alpha) * previous
        self._filtered_tvecs[class_name] = filtered
        return filtered


def main(args=None):
    rclpy.init(args=args)
    node = DockwinPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
