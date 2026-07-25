import time
from collections import deque
from enum import Enum

import cv2
import numpy as np
import rclpy
import tf_transformations
from geometry_msgs.msg import PoseStamped
from pose_estimator.config.object_points import (
    BUOY_CENTER_OFFSET,
    BUOY_OBJECT_POINTS,
)
from pose_estimator.utils.detections import (
    filter_detections_by_num_points,
    get_detection_obb,
    match_polygon_points,
)
from pose_estimator.utils.pose_estimator import get_object_pose, refine_object_pose
from pose_estimator.utils.pose_estimator_node import PoseEstimatorNode
from rclpy.publisher import Publisher
from rclpy.qos import qos_profile_sensor_data
from scipy import signal
from sklearn.cluster import DBSCAN
from tf2_ros import Buffer, TransformListener
from yolo_msgs.msg import DetectionArray


class Color(Enum):
    BLACK = "black"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class DetectedBuoyColor(Enum):
    BLACK = 0
    FLASHING_RED = 1
    FLASHING_GREEN = 2
    FLASHING_BLUE = 3
    SOLID_BLUE = 4
    UNKNOWN = 5


CLASSES_OF_INTEREST = [
    Color.RED.value,
    Color.GREEN.value,
    Color.BLUE.value,
    Color.BLACK.value,
]


class BuoyTrack:
    _id_counter = 0

    def __init__(self):
        self.track_id: int = BuoyTrack._id_counter
        BuoyTrack._id_counter += 1
        self.pose_history = deque()
        self.color_timeline = {
            color.value: deque() for color in Color if color != Color.BLACK
        }
        self.current_centroid: tuple[float, float] | None = None
        self.published_state: DetectedBuoyColor = DetectedBuoyColor.UNKNOWN


class TrackManager:
    def __init__(
        self,
        max_association_radius: float = 0.5,
        temporal_window: float = 5.0,
        required_density: float = 0.8,
        virtual_sampling_rate: float = 30.0,
        flashing_threshold: float = 1000.0,
        consistent_threshold: float = 0.35,
        cutoff_frequency: float = 3.0,
        fft_interval: float = 1.0,
        last_fft_time: float = 0.0,
        logger=None,
    ):
        self.max_association_radius = max_association_radius
        self.temporal_window = temporal_window
        self.required_density = required_density
        self.virtual_sampling_rate = virtual_sampling_rate
        self.flashing_threshold = flashing_threshold
        self.consistent_threshold = consistent_threshold
        self.cutoff_frequency = cutoff_frequency
        self.fft_interval = fft_interval
        self.last_fft_time = last_fft_time
        self.required_detections = int(
            temporal_window * virtual_sampling_rate * required_density
        )
        self.nyquist_frequency = virtual_sampling_rate / 2.0
        self.b, self.a = signal.butter(
            N=2, Wn=cutoff_frequency / self.nyquist_frequency, btype="low"
        )
        self.logger = logger

        self.pose_window = deque()
        self.pose_history = deque()
        self.tracks: dict[int, BuoyTrack] = {}

    def _add_to_track(self, track_id, color, world_position, timestamp):
        track = self.tracks[track_id]
        if color == Color.RED.value:
            track.color_timeline[Color.RED.value].append(1.0)
            track.color_timeline[Color.GREEN.value].append(0.0)
            track.color_timeline[Color.BLUE.value].append(0.0)
        elif color == Color.GREEN.value:
            track.color_timeline[Color.RED.value].append(0.0)
            track.color_timeline[Color.GREEN.value].append(1.0)
            track.color_timeline[Color.BLUE.value].append(0.0)
        elif color == Color.BLUE.value:
            track.color_timeline[Color.RED.value].append(0.0)
            track.color_timeline[Color.GREEN.value].append(0.0)
            track.color_timeline[Color.BLUE.value].append(1.0)
        elif color == Color.BLACK.value:
            track.color_timeline[Color.RED.value].append(0.0)
            track.color_timeline[Color.GREEN.value].append(0.0)
            track.color_timeline[Color.BLUE.value].append(0.0)
        else:
            return
        track.pose_history.append((timestamp, world_position))

    def append_detections(self, detection_stamp, detection_poses):
        # NOTE: pose_window and pose_history are of same length
        self.pose_window.extend(detection_poses)
        self.pose_history.extend([detection_stamp] * len(detection_poses))

    def prune_detections(self, current_time):
        while (
            self.pose_window
            and current_time - self.pose_history[0] > self.temporal_window
        ):
            self.pose_window.popleft()
            self.pose_history.popleft()

    def recompute_tracks(self):
        # Regnerate tracks given pose window using DBSCAN clustering
        self.tracks.clear()
        BuoyTrack._id_counter = 0
        if (
            not self.pose_window
            or len(set(self.pose_history)) < self.required_detections
        ):
            self.logger.debug(
                f"Warming up: {len(set(self.pose_history))} unique frames gathered, "
                f"waiting for {self.required_detections} to build tracks."
            )
            return
        colors = np.array([color for color, _ in self.pose_window])
        positions = np.array([pos for _, pos in self.pose_window])
        timestamps = np.array(self.pose_history)
        dbscan = DBSCAN(eps=self.max_association_radius, min_samples=3)
        labels = dbscan.fit_predict(positions)
        for label in set(labels):
            if label == -1:
                continue  # Ignore noise
            # Use label mask to get poses belonging to this track
            # NOTE: pose_window and pose_history are of same length, so we can use same labels to index into both
            mask = labels == label
            track_poses = positions[mask]
            track_timestamps = timestamps[mask]
            track_colors = colors[mask]
            track = BuoyTrack()
            track_id = track.track_id
            self.tracks[track_id] = track

            last_ts = None
            for color, world_position, timestamp in zip(
                track_colors, track_poses, track_timestamps
            ):
                # NOTE: Ideally, YOLO should not be producing detections that are this close which result in them getting associated with the same timestamp
                if last_ts == timestamp:
                    continue  # Skip duplicate timestamps within the same track
                self._add_to_track(track_id, color, world_position, timestamp)
                last_ts = timestamp

    def recompute_centroids(self, current_time):
        for track in self.tracks.values():
            if not track.pose_history:
                continue
            dt = np.array([current_time - t for t, _ in track.pose_history])
            weights = np.exp(-dt)  # Exponential decay based on age of pose
            positions = np.array([pos for _, pos in track.pose_history])
            track.current_centroid = np.average(positions, axis=0, weights=weights)

    def perform_fft(self, current_time):
        if current_time - self.last_fft_time < self.fft_interval:
            self.logger.warn(
                f"Last FFT was {(current_time - self.last_fft_time):.2f} seconds ago",
                throttle_duration_sec=1.0,
            )
            return  # Only perform FFT at specified intervals

        self.last_fft_time = current_time

        tracks_start_time = time.perf_counter()
        self.recompute_tracks()  # Ensure tracks are up to date before FFT
        tracks_end_time = time.perf_counter()
        self.logger.info(
            f"Recomputed tracks in {(tracks_end_time - tracks_start_time)*1000:.2f} ms, "
            f"{len(self.tracks)} active tracks",
            throttle_duration_sec=1.0,
        )

        if not self.tracks:
            self.logger.warn("No tracks available for FFT analysis")
            return

        centroids_start_time = time.perf_counter()
        self.recompute_centroids(
            current_time
        )  # Ensure centroids are up to date before FFT
        centroids_end_time = time.perf_counter()
        self.logger.info(
            f"Recomputed centroids in {(centroids_end_time - centroids_start_time)*1000:.2f} ms",
            throttle_duration_sec=1.0,
        )

        for track in self.tracks.values():
            if len(track.pose_history) < self.required_detections:
                self.logger.warn(
                    f"Track {track.track_id} has only {len(track.pose_history)} detections, "
                    f"which is below the required {self.required_detections} for FFT analysis",
                    throttle_duration_sec=1.0,
                )
                continue

            raw_timestamps = np.array([t for t, _ in track.pose_history])
            uniform_timestamps = np.arange(
                raw_timestamps[0],
                raw_timestamps[-1],
                1.0 / self.virtual_sampling_rate,
            )
            interpolated_red = np.interp(
                uniform_timestamps,
                raw_timestamps,
                track.color_timeline[Color.RED.value],
            )
            interpolated_green = np.interp(
                uniform_timestamps,
                raw_timestamps,
                track.color_timeline[Color.GREEN.value],
            )
            interpolated_blue = np.interp(
                uniform_timestamps,
                raw_timestamps,
                track.color_timeline[Color.BLUE.value],
            )

            # Apply low-pass Butterworth filter
            filtered_red = signal.filtfilt(self.b, self.a, interpolated_red)
            filtered_green = signal.filtfilt(self.b, self.a, interpolated_green)
            filtered_blue = signal.filtfilt(self.b, self.a, interpolated_blue)

            num_samples = len(uniform_timestamps)
            # Compute FFT and power spectral density
            fft_red_magnitude = np.abs(np.fft.rfft(filtered_red)) / num_samples
            fft_green_magnitude = np.abs(np.fft.rfft(filtered_green)) / num_samples
            fft_blue_magnitude = np.abs(np.fft.rfft(filtered_blue)) / num_samples

            frequency_bins = np.fft.rfftfreq(
                len(uniform_timestamps), d=1.0 / self.virtual_sampling_rate
            )

            red_max_idx = np.argmax(fft_red_magnitude[1:]) + 1
            red_peak_freq = frequency_bins[red_max_idx]
            red_peak_power = (
                fft_red_magnitude[red_max_idx] if 0.8 <= red_peak_freq <= 1.2 else -1.0
            )

            green_max_idx = np.argmax(fft_green_magnitude[1:]) + 1
            green_peak_freq = frequency_bins[green_max_idx]
            green_peak_power = (
                fft_green_magnitude[green_max_idx]
                if 0.8 <= green_peak_freq <= 1.2
                else -1.0
            )

            blue_max_idx = np.argmax(fft_blue_magnitude[1:]) + 1
            blue_peak_freq = frequency_bins[blue_max_idx]
            blue_peak_power = (
                fft_blue_magnitude[blue_max_idx]
                if 0.8 <= blue_peak_freq <= 1.2
                else -1.0
            )
            self.logger.info(
                f"Track {track.track_id}: Red Peak={red_peak_power:.3f} at {red_peak_freq:.2f}Hz, "
                f"Green Peak={green_peak_power:.3f} at {green_peak_freq:.2f}Hz, "
                f"Blue Peak={blue_peak_power:.3f} at {blue_peak_freq:.2f}Hz"
            )

            # NOTE: KIV target band approach
            # target_band = (frequency_bins >= 0.8) & (frequency_bins <= 1.2)

            # # Fallback if target band is empty (due to low sampling rate/window size)
            # if not np.any(target_band):
            #     track.published_state = DetectedBuoyColor.UNKNOWN
            #     continue

            # Normalized DC component represents the percentage of time that color was present.
            blue_presence = fft_blue_magnitude[0]
            red_presence = fft_red_magnitude[0]
            green_presence = fft_green_magnitude[0]

            dominant_flashing_color = max(
                [
                    (Color.RED, red_peak_power, red_presence),
                    (Color.GREEN, green_peak_power, green_presence),
                    (Color.BLUE, blue_peak_power, blue_presence),
                ],
                key=lambda x: x[1],
            )

            best_color, peak_power, dc_presence = dominant_flashing_color
            self.logger.info(
                f"Track {track.track_id}: Best Color={best_color}, Peak Power={peak_power:.3f}, DC Presence={dc_presence:.3f}"
            )

            # Claim that the light cannot be on more than 85% of the time or less than 10% of the time for it to be flashing
            is_valid_duty_cycle = 0.1 < dc_presence < 0.85

            if peak_power > self.flashing_threshold and is_valid_duty_cycle:
                track.published_state = (
                    DetectedBuoyColor.FLASHING_RED
                    if best_color == Color.RED
                    else (
                        DetectedBuoyColor.FLASHING_GREEN
                        if best_color == Color.GREEN
                        else DetectedBuoyColor.FLASHING_BLUE
                    )
                )
                continue

            # No flashing, figure out if solid blue or black.
            # Percentage of time buoy is black
            black_presence = max(
                0.0, 1.0 - (red_presence + green_presence + blue_presence)
            )
            if blue_presence > black_presence:
                if (
                    blue_presence > self.consistent_threshold
                ):  # TODO: figure out good threshold
                    track.published_state = DetectedBuoyColor.SOLID_BLUE
                else:
                    track.published_state = DetectedBuoyColor.BLACK
            else:
                track.published_state = DetectedBuoyColor.BLACK

    def log_tracks(self):
        for track_id, track in self.tracks.items():
            self.logger.info(
                f"Track {track_id}: Centroid={track.current_centroid}, State={track.published_state}"
            )


class BuoyPoseEstimatorNode(PoseEstimatorNode):
    def __init__(self):
        super().__init__("buoy_pose_estimator_node")

        max_association_radius = (
            self.declare_parameter("max_association_radius", 0.5)
            .get_parameter_value()
            .double_value
        )
        temporal_window = (
            self.declare_parameter("temporal_window", 5.0)
            .get_parameter_value()
            .double_value
        )
        required_density = (
            self.declare_parameter("required_density", 0.8)
            .get_parameter_value()
            .double_value
        )
        virtual_sampling_rate = (
            self.declare_parameter(
                "virtual_sampling_rate", 10.0
            )  # TODO: check detection rate
            .get_parameter_value()
            .double_value
        )
        fft_interval = (
            self.declare_parameter("fft_interval", 1.0)
            .get_parameter_value()
            .double_value
        )
        cutoff_frequency = (
            self.declare_parameter("cutoff_frequency", 3.0)
            .get_parameter_value()
            .double_value
        )
        flashing_threshold = (
            self.declare_parameter(
                "flashing_threshold", 0.20
            )  # TODO: figure out good threshold
            .get_parameter_value()
            .double_value
        )
        consistent_threshold = (
            self.declare_parameter(
                "consistent_threshold", 0.35
            )  # TODO: figure out good threshold
            .get_parameter_value()
            .double_value
        )
        self.world_frame_id = (
            self.declare_parameter("world_frame_id", "map_ned")
            .get_parameter_value()
            .string_value
        )

        self.track_manager = TrackManager(
            max_association_radius,
            temporal_window,
            required_density,
            virtual_sampling_rate,
            flashing_threshold,
            consistent_threshold,
            cutoff_frequency,
            fft_interval,
            logger=self.get_logger(),
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pose_publisher: Publisher[PoseStamped] = self.create_publisher(
            PoseStamped, "/buoy_poses", qos_profile_sensor_data
        )

        self.detection_subscription = self.create_subscription(
            DetectionArray,
            "/yolo/detections",
            self.detection_callback,
            qos_profile_sensor_data,
        )

    def _get_detection_poses_in_camera_frame(self, detections: DetectionArray):
        detection_stamp = detections.header.stamp
        filtered_detections: DetectionArray = filter_detections_by_num_points(
            detections, 3
        )
        output = []
        for detection in filtered_detections.detections:
            if detection.class_name not in CLASSES_OF_INTEREST:
                continue
            angle, detected_points = get_detection_obb(detection)
            angle = (angle + 90) % 180 - 90

            object_points, image_points = match_polygon_points(
                BUOY_OBJECT_POINTS, detected_points, A_angle=angle
            )

            object_points = np.hstack(
                [object_points, np.zeros((object_points.shape[0], 1))]
            )

            assert (
                object_points.shape[0] == image_points.shape[0]
            ), "Number of object points and image points must match"

            try:
                rvec, tvec, inliers = get_object_pose(
                    self.camera, object_points, image_points
                )

                if inliers is None:
                    raise ValueError("No inliers found during pose estimation.")

                rvec, tvec = refine_object_pose(
                    self.camera, object_points, image_points, rvec, tvec
                )

                # Get center pose
                R, _ = cv2.Rodrigues(rvec)
                tvec = tvec.flatten() + R @ BUOY_CENTER_OFFSET

                output.append((detection.class_name, tvec))
            except Exception as e:
                self.get_logger().warn(f"Pose estimation failed: {e}")
                continue

        return detection_stamp, output

    def _get_poses_in_world_frame(self, detection_stamp, detection_poses):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame_id,
                self.camera.frame_id,
                rclpy.time.Time.from_msg(detection_stamp),
                rclpy.duration.Duration(seconds=1),
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}, falling back to latest")
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.world_frame_id,
                    self.camera.frame_id,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=1),
                )
            except Exception as e:
                self.get_logger().error(f"Latest TF lookup also failed: {e}")
                return []

        t = transform.transform.translation
        q = transform.transform.rotation
        R = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        T = tf_transformations.translation_matrix([t.x, t.y, t.z])
        T_world_camera = T @ R

        output = []
        for class_name, tvec in detection_poses:
            R_camera_object, _ = cv2.Rodrigues(np.zeros(3))

            T_camera_object = np.eye(4)
            T_camera_object[:3, :3] = R_camera_object
            T_camera_object[:3, 3] = tvec.flatten()

            T_world_object = T_world_camera @ T_camera_object

            world_translation = T_world_object[:3, 3]

            # Only care about x and y for buoy localization
            world_translation = world_translation[:2]
            output.append((class_name, world_translation))
        return output

    def detection_callback(self, msg: DetectionArray):
        current_time = msg.header.stamp
        current_time_sec = current_time.sec + current_time.nanosec * 1e-9

        # Always prune
        prune_start_time = time.perf_counter()
        self.track_manager.prune_detections(current_time_sec)
        prune_end_time = time.perf_counter()
        self.get_logger().info(
            f"Pruned detections in {(prune_end_time - prune_start_time)*1000:.2f} ms, "
            f"{len(self.track_manager.pose_window)} active detections remain",
            throttle_duration_sec=1.0,
        )

        detection_stamp, detection_poses = self._get_detection_poses_in_camera_frame(
            msg
        )

        if not detection_poses:
            self.get_logger().warn("No valid detections for pose estimation")
        else:
            detection_poses_world = self._get_poses_in_world_frame(
                detection_stamp, detection_poses
            )

            append_start_time = time.perf_counter()
            self.track_manager.append_detections(
                current_time_sec, detection_poses_world
            )
            append_end_time = time.perf_counter()
            self.get_logger().info(
                f"Appended detections to tracks in {(append_end_time - append_start_time)*1000:.2f} ms "
                f"{len(detection_poses_world)} detections appended",
                throttle_duration_sec=1.0,
            )

        fft_start_time = time.perf_counter()
        self.track_manager.perform_fft(current_time_sec)
        fft_end_time = time.perf_counter()
        self.get_logger().info(
            f"Performed FFT analysis in {(fft_end_time - fft_start_time)*1000:.2f} ms",
            throttle_duration_sec=1.0,
        )
        self.track_manager.log_tracks()


def main(args=None):
    rclpy.init(args=args)
    node = BuoyPoseEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
