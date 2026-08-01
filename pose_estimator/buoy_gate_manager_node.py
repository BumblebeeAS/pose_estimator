#!/usr/bin/env python3
import math
import time
from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import rclpy
import tf2_ros
import tf_transformations
from bb_msgs.msg import BuoyDetectionArray
from bb_robotx_msgs.srv import (
    ExtractLocalGoal,
    GetEntry,
    GetExit,
    SetGateManagerState,
)
from geometry_msgs.msg import Point, PoseStamped, Quaternion, Vector3
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from robot_localization.srv import FromLL, ToLL
from scipy import signal
from scipy.fft import rfft, rfftfreq
from sensor_msgs.msg import NavSatFix
from sklearn.cluster import DBSCAN
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray


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


def _get_yaw_from_quaternion(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class BuoyTrack:
    _id_counter = 0

    def __init__(self):
        self.track_id: int = BuoyTrack._id_counter
        BuoyTrack._id_counter += 1
        self.pose_history = deque()  # stores (timestamp, np.array([x, y, z]))
        self.color_timeline = {
            color.value: deque() for color in Color if color != Color.BLACK
        }  # stores intensity float values per frame
        self.current_centroid: Optional[np.ndarray] = None
        self.published_state: DetectedBuoyColor = DetectedBuoyColor.UNKNOWN
        self.last_observation_time: float = 0.0

    @property
    def is_immortal(self) -> bool:
        return self.published_state in (
            DetectedBuoyColor.FLASHING_BLUE,
            DetectedBuoyColor.SOLID_BLUE,
        )


class TrackManager:
    def __init__(
        self,
        max_association_radius: float = 2.0,
        dbscan_min_samples: int = 3,
        track_timeout: float = 60.0,
        required_density: float = 0.8,
        butterworth_cutoff_freq: float = 2.5,
        butterworth_order: int = 4,
        target_freq_min: float = 0.8,
        target_freq_max: float = 1.2,
        duty_cycle_min: float = 0.1,
        duty_cycle_max: float = 0.85,
        consistent_threshold: float = 0.6,
        fft_min_magnitude: float = 0.2,
        virtual_sampling_rate: float = 10.0,
        logger=None,
    ):
        self.max_association_radius = max_association_radius
        self.dbscan_min_samples = dbscan_min_samples
        self.track_timeout = track_timeout
        self.temporal_window = 30.0
        self.required_density = required_density
        self.butterworth_cutoff_freq = butterworth_cutoff_freq
        self.butterworth_order = butterworth_order
        self.target_freq_min = target_freq_min
        self.target_freq_max = target_freq_max
        self.duty_cycle_min = duty_cycle_min
        self.duty_cycle_max = duty_cycle_max
        self.consistent_threshold = consistent_threshold
        self.fft_min_magnitude = fft_min_magnitude
        self.virtual_sampling_rate = virtual_sampling_rate
        self.logger = logger

        self.nyquist_frequency = virtual_sampling_rate / 2.0
        wn = min(0.99, max(0.01, butterworth_cutoff_freq / self.nyquist_frequency))
        self.b, self.a = signal.butter(N=butterworth_order, Wn=wn, btype="low")
        self.min_filter_samples = 3 * max(len(self.a), len(self.b)) + 1

        self.tracks: Dict[int, BuoyTrack] = {}
        self.detection_buffer: deque = deque()

    @property
    def required_detections(self) -> int:
        return int(
            self.temporal_window * self.virtual_sampling_rate * self.required_density
        )

    def clear_all_tracks(self):
        self.tracks.clear()
        self.detection_buffer.clear()
        BuoyTrack._id_counter = 0

    def associate_and_update(self, current_time: float):
        """
        Layer 2 Buffered Association: 1-shot DBSCAN spatial clustering across existing track centroids
        and accumulated incoming detections in self.detection_buffer.
        No Euclidean distance matching loop is used. Shared clusters associate buffered observations to tracks,
        new clusters instantiate tracks, and noise (label == -1) is discarded. Once absorbed, detection_buffer is cleared.
        """
        if not self.detection_buffer and not self.tracks:
            return

        point_list = []
        point_sources = []  # ('track', tid) or ('det', idx)

        # Include ONLY the current smoothed centroid of existing tracks
        for tid, track in self.tracks.items():
            if track.current_centroid is not None:
                point_list.append(track.current_centroid)
                point_sources.append(("track", tid))

        # Include all buffered raw detections
        for idx, (ts, _, pos) in enumerate(self.detection_buffer):
            point_list.append(pos)
            point_sources.append(("det", idx))

        if not point_list:
            return

        all_positions = np.array(point_list)
        dbscan = DBSCAN(eps=self.max_association_radius, min_samples=self.dbscan_min_samples)
        labels = dbscan.fit_predict(all_positions)

        for label in set(labels):
            if label == -1:
                continue  # Discard noise/outliers that fail density clustering

            cluster_mask = np.where(labels == label)[0]
            cluster_tids = set()
            cluster_det_indices = []

            for idx in cluster_mask:
                src_type, val = point_sources[idx]
                if src_type == "track":
                    cluster_tids.add(val)
                elif src_type == "det":
                    cluster_det_indices.append(val)

            if not cluster_det_indices:
                continue  # Cluster contains only existing track points, no new detections to add

            if cluster_tids:
                # Associate incoming detections to the primary existing track in this cluster
                primary_tid = min(cluster_tids)
                for idx in cluster_det_indices:
                    ts, col, p = self.detection_buffer[idx]
                    self._add_observation_to_track(primary_tid, col, p, ts)
            else:
                # Zero existing tracks in this cluster -> instantiate a brand new BuoyTrack
                new_track = BuoyTrack()
                self.tracks[new_track.track_id] = new_track
                for idx in cluster_det_indices:
                    ts, col, p = self.detection_buffer[idx]
                    self._add_observation_to_track(new_track.track_id, col, p, ts)

        # Once incoming detections are absorbed into individual buffers, clear the global buffer (Doc Section 2 Line 180)
        self.detection_buffer.clear()

        if self.logger:
            self.logger.info(
                f"[associate_and_update] DBSCAN clustered {len(point_list)} points into {len(set(labels)) - (1 if -1 in labels else 0)} clusters, "
                f"resulting in {len(self.tracks)} active tracks.",
                throttle_duration_sec=2.0,
            )

    def _add_observation_to_track(
        self, track_id: int, color: str, position: np.ndarray, timestamp: float
    ):
        track = self.tracks.get(track_id)
        if not track:
            return

        if track.pose_history and track.pose_history[-1][0] == timestamp:
            return

        track.pose_history.append((timestamp, position))
        track.last_observation_time = timestamp

        for c in track.color_timeline:
            val = 1.0 if c == color else 0.0
            track.color_timeline[c].append(val)

    def prune_temporal_window(self, current_time: float):
        if self.temporal_window <= 0.0:
            return

        cutoff = current_time - self.temporal_window
        for track in self.tracks.values():
            while track.pose_history and track.pose_history[0][0] < cutoff:
                track.pose_history.popleft()
                for c in track.color_timeline:
                    if track.color_timeline[c]:
                        track.color_timeline[c].popleft()

    def _recompute_centroids(self):
        """Compute smoothed centroid for all active tracks from their temporal pose history."""
        for track in self.tracks.values():
            if track.pose_history:
                coords = np.array([p for _, p in track.pose_history])
                track.current_centroid = np.mean(coords, axis=0)

    def prune_timeouts(self, current_time: float):
        """
        Layer 2 Timeout Pruning: Delete non-immortal tracks that haven't been observed within track_timeout.
        Immortal singletons (FLASHING_BLUE / SOLID_BLUE) are immune to timeout pruning.
        """
        to_delete = []
        for tid, track in self.tracks.items():
            if not track.is_immortal:
                if (
                    current_time - track.last_observation_time > self.track_timeout
                    or not track.pose_history
                ):
                    to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]
            if self.logger:
                self.logger.debug(f"Pruned timed-out track ID {tid}.")

        if to_delete and self.logger:
            self.logger.info(
                f"[prune_timeouts] Pruned {len(to_delete)} timed-out track(s): {to_delete}. Active tracks remaining: {len(self.tracks)}.",
                throttle_duration_sec=2.0,
            )

    def classify_tracks(self):
        """
        Layer 2 Background Classification: Rigorous time-resampling (np.interp), Butterworth low-pass filter,
        and rFFT signal analysis to verify target flashing frequency bands and duty cycles (from commit 4974cb3).
        """
        for track in self.tracks.values():
            if len(track.pose_history) < max(
                self.min_filter_samples, self.required_detections
            ):
                continue

            raw_timestamps = np.array([t for t, _ in track.pose_history])
            if raw_timestamps[-1] - raw_timestamps[0] <= 0.0:
                continue

            uniform_timestamps = np.arange(
                raw_timestamps[0],
                raw_timestamps[-1],
                1.0 / self.virtual_sampling_rate,
            )
            if len(uniform_timestamps) < self.min_filter_samples:
                continue

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
            try:
                filtered_red = signal.filtfilt(self.b, self.a, interpolated_red)
                filtered_green = signal.filtfilt(self.b, self.a, interpolated_green)
                filtered_blue = signal.filtfilt(self.b, self.a, interpolated_blue)
            except Exception:
                filtered_red = interpolated_red
                filtered_green = interpolated_green
                filtered_blue = interpolated_blue

            num_samples = len(uniform_timestamps)
            # Compute FFT and power spectral density
            fft_red_magnitude = np.abs(rfft(filtered_red)) / num_samples
            fft_green_magnitude = np.abs(rfft(filtered_green)) / num_samples
            fft_blue_magnitude = np.abs(rfft(filtered_blue)) / num_samples

            frequency_bins = rfftfreq(num_samples, d=1.0 / self.virtual_sampling_rate)

            if len(frequency_bins) <= 1:
                continue

            red_max_idx = np.argmax(fft_red_magnitude[1:]) + 1
            red_peak_freq = frequency_bins[red_max_idx]
            red_peak_power = (
                fft_red_magnitude[red_max_idx]
                if self.target_freq_min <= red_peak_freq <= self.target_freq_max
                else -1.0
            )

            green_max_idx = np.argmax(fft_green_magnitude[1:]) + 1
            green_peak_freq = frequency_bins[green_max_idx]
            green_peak_power = (
                fft_green_magnitude[green_max_idx]
                if self.target_freq_min <= green_peak_freq <= self.target_freq_max
                else -1.0
            )

            blue_max_idx = np.argmax(fft_blue_magnitude[1:]) + 1
            blue_peak_freq = frequency_bins[blue_max_idx]
            blue_peak_power = (
                fft_blue_magnitude[blue_max_idx]
                if self.target_freq_min <= blue_peak_freq <= self.target_freq_max
                else -1.0
            )

            blue_presence = fft_blue_magnitude[0]
            red_presence = fft_red_magnitude[0]
            green_presence = fft_green_magnitude[0]

            dominant_flashing_color = max(
                [
                    (Color.RED.value, red_peak_power, red_presence),
                    (Color.GREEN.value, green_peak_power, green_presence),
                    (Color.BLUE.value, blue_peak_power, blue_presence),
                ],
                key=lambda x: x[1],
            )

            best_color, peak_power, dc_presence = dominant_flashing_color

            # Verify that the light is not on more than duty_cycle_max or less than duty_cycle_min for it to be flashing
            is_valid_duty_cycle = (
                self.duty_cycle_min < dc_presence < self.duty_cycle_max
            )

            if peak_power > self.fft_min_magnitude and is_valid_duty_cycle:
                track.published_state = (
                    DetectedBuoyColor.FLASHING_RED
                    if best_color == Color.RED.value
                    else (
                        DetectedBuoyColor.FLASHING_GREEN
                        if best_color == Color.GREEN.value
                        else DetectedBuoyColor.FLASHING_BLUE
                    )
                )
                continue

            # No flashing, figure out if solid blue or black
            black_presence = max(
                0.0, 1.0 - (red_presence + green_presence + blue_presence)
            )
            if blue_presence > black_presence:
                if blue_presence > self.consistent_threshold:
                    track.published_state = DetectedBuoyColor.SOLID_BLUE
                else:
                    track.published_state = DetectedBuoyColor.BLACK
            else:
                track.published_state = DetectedBuoyColor.BLACK

        if self.logger and self.tracks:
            state_counts = {}
            for track in self.tracks.values():
                s = track.published_state.name if track.published_state else "UNCLASSIFIED"
                state_counts[s] = state_counts.get(s, 0) + 1
            self.logger.info(
                f"[classify_tracks] Active track classifications: {state_counts}",
                throttle_duration_sec=2.0,
            )

    def deduplicate_singletons(self):
        """
        Layer 2 Deduplication: Entry (FLASHING_BLUE) and Exit (SOLID_BLUE) buoys are immortal singletons.
        Merge duplicate tracks sharing milestone color classification by performing a synchronized chronological merge.
        """
        for singleton_type in (
            DetectedBuoyColor.FLASHING_BLUE,
            DetectedBuoyColor.SOLID_BLUE,
        ):
            matching_tids = [
                tid
                for tid, track in self.tracks.items()
                if track.published_state == singleton_type
            ]
            if len(matching_tids) > 1:
                matching_tids.sort()
                primary_tid = matching_tids[0]
                primary_track = self.tracks[primary_tid]
                for dup_tid in matching_tids[1:]:
                    dup_track = self.tracks[dup_tid]

                    # 1. Gather synchronized observation tuples from both tracks
                    combined = []
                    for t_track in (primary_track, dup_track):
                        for i, (t, pos) in enumerate(t_track.pose_history):
                            colors = {
                                c: t_track.color_timeline[c][i]
                                for c in t_track.color_timeline
                            }
                            combined.append((t, pos, colors))

                    # 2. Sort chronologically ascending by timestamp
                    combined.sort(key=lambda x: x[0])

                    # 3. Rebuild primary track arrays in strict chronological order, deduplicating identical timestamps
                    primary_track.pose_history.clear()
                    for c in primary_track.color_timeline:
                        primary_track.color_timeline[c].clear()

                    seen_timestamps = set()
                    for t, pos, colors in combined:
                        if t not in seen_timestamps:
                            seen_timestamps.add(t)
                            primary_track.pose_history.append((t, pos))
                            for c, val in colors.items():
                                primary_track.color_timeline[c].append(val)

                    primary_track.last_observation_time = max(
                        primary_track.last_observation_time,
                        dup_track.last_observation_time,
                    )
                    del self.tracks[dup_tid]
                    if self.logger:
                        self.logger.info(
                            f"[deduplicate_singletons] Merged duplicate singleton {singleton_type.name} track ID {dup_tid} into primary ID {primary_tid}.",
                            throttle_duration_sec=2.0,
                        )


class BuoyGateManagerNode(Node):
    def __init__(self):
        super().__init__("buoy_gate_manager_node")

        self.main_cb_group = MutuallyExclusiveCallbackGroup()
        self.client_cb_group = MutuallyExclusiveCallbackGroup()

        # Parameter declarations from Section 7 Table
        self.declare_parameter("buoy_detections_topic", "/buoy_detections")
        self.declare_parameter("marker_topic", "gate_manager_markers")
        self.declare_parameter("from_ll_service", "/fromLL")
        self.declare_parameter("to_ll_service", "/toLL")
        self.declare_parameter("set_state_service", "set_gate_manager_state")
        self.declare_parameter("reset_service", "reset_tracks")
        self.declare_parameter("extract_goal_service", "extract_local_goal")
        self.declare_parameter("get_entry_service", "get_entry")
        self.declare_parameter("get_exit_service", "get_exit")

        self.declare_parameter("world_frame_id", "map")
        self.declare_parameter("base_link_frame_id", "base_link")
        self.declare_parameter("fft_timer_period", 0.5)
        self.declare_parameter("required_density", 0.8)
        self.declare_parameter("virtual_sampling_rate", 10.0)

        self.declare_parameter("max_association_radius", 0.5)
        self.declare_parameter("dbscan_min_samples", 5)
        self.declare_parameter("track_timeout", 60.0)

        self.declare_parameter("butterworth_cutoff_freq", 3.0)
        self.declare_parameter("butterworth_order", 4)
        self.declare_parameter("target_freq_min", 0.8)
        self.declare_parameter("target_freq_max", 1.2)
        self.declare_parameter("duty_cycle_min", 0.1)
        self.declare_parameter("duty_cycle_max", 0.85)
        self.declare_parameter("consistent_threshold", 0.6)
        self.declare_parameter("fft_min_magnitude", 0.2)

        self.declare_parameter("max_gate_width", 1.0)
        self.declare_parameter("vehicle_encasing_radius", 0.8)
        self.declare_parameter("waypoint_pass_tol", 1.0)

        # Extract values
        self.buoy_detections_topic = self.get_parameter("buoy_detections_topic").value
        self.marker_topic = self.get_parameter("marker_topic").value
        self.from_ll_service_name = self.get_parameter("from_ll_service").value
        self.to_ll_service_name = self.get_parameter("to_ll_service").value
        self.set_state_service_name = self.get_parameter("set_state_service").value
        self.reset_service_name = self.get_parameter("reset_service").value
        self.extract_goal_service_name = self.get_parameter(
            "extract_goal_service"
        ).value
        self.get_entry_service_name = self.get_parameter("get_entry_service").value
        self.get_exit_service_name = self.get_parameter("get_exit_service").value

        self.world_frame_id = self.get_parameter("world_frame_id").value
        self.base_link_frame_id = self.get_parameter("base_link_frame_id").value
        self.fft_timer_period = self.get_parameter("fft_timer_period").value
        self.max_gate_width = self.get_parameter("max_gate_width").value
        self.vehicle_encasing_radius = self.get_parameter(
            "vehicle_encasing_radius"
        ).value
        self.waypoint_pass_tol = self.get_parameter("waypoint_pass_tol").value

        # Initialize TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Lifecycle and tracking state
        self.enable: bool = False
        self.last_query_pos: Optional[np.ndarray] = None

        self.track_manager = TrackManager(
            max_association_radius=self.get_parameter("max_association_radius").value,
            dbscan_min_samples=self.get_parameter("dbscan_min_samples").value,
            track_timeout=self.get_parameter("track_timeout").value,
            required_density=self.get_parameter("required_density").value,
            butterworth_cutoff_freq=self.get_parameter("butterworth_cutoff_freq").value,
            butterworth_order=self.get_parameter("butterworth_order").value,
            target_freq_min=self.get_parameter("target_freq_min").value,
            target_freq_max=self.get_parameter("target_freq_max").value,
            duty_cycle_min=self.get_parameter("duty_cycle_min").value,
            duty_cycle_max=self.get_parameter("duty_cycle_max").value,
            consistent_threshold=self.get_parameter("consistent_threshold").value,
            fft_min_magnitude=self.get_parameter("fft_min_magnitude").value,
            virtual_sampling_rate=self.get_parameter("virtual_sampling_rate").value,
            logger=self.get_logger(),
        )

        # Topic subscriptions & publishers
        self.detections_sub = self.create_subscription(
            BuoyDetectionArray,
            self.buoy_detections_topic,
            self._on_buoy_detections,
            10,
            callback_group=self.main_cb_group,
        )
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)

        # Service servers
        self.srv_set_state = self.create_service(
            SetGateManagerState,
            self.set_state_service_name,
            self._cb_set_state,
            callback_group=self.main_cb_group,
        )
        self.srv_reset = self.create_service(
            Trigger,
            self.reset_service_name,
            self._cb_reset,
            callback_group=self.main_cb_group,
        )
        self.srv_extract_goal = self.create_service(
            ExtractLocalGoal,
            self.extract_goal_service_name,
            self._cb_extract_goal,
            callback_group=self.main_cb_group,
        )
        self.srv_get_entry = self.create_service(
            GetEntry,
            self.get_entry_service_name,
            self._cb_get_entry,
            callback_group=self.main_cb_group,
        )
        self.srv_get_exit = self.create_service(
            GetExit,
            self.get_exit_service_name,
            self._cb_get_exit,
            callback_group=self.main_cb_group,
        )

        # Geodetic bridge clients
        self.from_ll_client = self.create_client(
            FromLL, self.from_ll_service_name, callback_group=self.client_cb_group
        )
        self.to_ll_client = self.create_client(
            ToLL, self.to_ll_service_name, callback_group=self.client_cb_group
        )

        # Background FFT/Pruning timer (Layer 2)
        self.fft_timer = self.create_timer(
            self.fft_timer_period, self._on_fft_timer, callback_group=self.main_cb_group
        )

        self.get_logger().info(
            f"BuoyGateManagerNode operational. Listening on '{self.buoy_detections_topic}', "
            f"world frame '{self.world_frame_id}', enable={self.enable}."
        )

    def _on_buoy_detections(self, msg: BuoyDetectionArray):
        """Layer 1: Event-driven ingestion into detection buffer with TF2 world transformation."""
        if not self.enable:
            return  # Frozen state: ignore incoming frames

        source_frame = msg.header.frame_id
        if (
            not source_frame
            and msg.detections
            and msg.detections[0].pose.header.frame_id
        ):
            source_frame = msg.detections[0].pose.header.frame_id
        if not source_frame:
            self.get_logger().warn(
                "Received BuoyDetectionArray with empty frame_id.",
                throttle_duration_sec=2.0,
            )
            return

        rot_matrix = np.eye(3)
        trans_vec = np.zeros(3)

        if source_frame != self.world_frame_id:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.world_frame_id,
                    source_frame,
                    rclpy.time.Time.from_msg(msg.header.stamp),
                    rclpy.duration.Duration(seconds=0.05),
                )
            except Exception as e:
                self.get_logger().warn(
                    f"TF lookup failed from '{source_frame}' to '{self.world_frame_id}' at timestamp, falling back to latest: {e}",
                    throttle_duration_sec=2.0,
                )
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.world_frame_id,
                        source_frame,
                        rclpy.time.Time(),
                        rclpy.duration.Duration(seconds=0.05),
                    )
                except Exception as e2:
                    self.get_logger().warn(
                        f"Latest TF lookup also failed from '{source_frame}' to '{self.world_frame_id}': {e2}",
                        throttle_duration_sec=2.0,
                    )
                    return

            t = transform.transform.translation
            q = transform.transform.rotation
            rot_matrix = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[
                :3, :3
            ]
            trans_vec = np.array([t.x, t.y, t.z])

        now_sec = self.get_clock().now().nanoseconds / 1e9
        for d in msg.detections:
            color_str = d.color.lower() if d.color else Color.BLACK.value
            if color_str in CLASSES_OF_INTEREST and color_str != Color.BLACK.value:
                pos = np.array(
                    [d.pose.position.x, d.pose.position.y, d.pose.position.z]
                )
                pos_world = rot_matrix @ pos + trans_vec
                pos_world[2] = 0.0  # Zero out Z in world frame (water surface)
                self.track_manager.detection_buffer.append(
                    (now_sec, color_str, pos_world)
                )

        # Prune raw detection buffer older than temporal window
        cutoff = now_sec - self.track_manager.temporal_window
        while (
            self.track_manager.detection_buffer
            and self.track_manager.detection_buffer[0][0] < cutoff
        ):
            self.track_manager.detection_buffer.popleft()

        self.get_logger().info(
            f"[_on_buoy_detections] Buffered {len(msg.detections)} detections, buffer size now {len(self.track_manager.detection_buffer)}.",
            throttle_duration_sec=2.0,
        )

    def _on_fft_timer(self):
        """Layer 2: Background association, FFT classification, pruning, deduplication, and visualization."""
        if not self.enable:
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        self.track_manager.associate_and_update(now_sec)
        self.track_manager.prune_temporal_window(now_sec)
        self.track_manager._recompute_centroids()
        self.track_manager.prune_timeouts(now_sec)
        self.track_manager.classify_tracks()
        self.track_manager.deduplicate_singletons()
        self.track_manager._recompute_centroids()
        self._publish_markers()

    def _cb_set_state(
        self, req: SetGateManagerState.Request, res: SetGateManagerState.Response
    ) -> SetGateManagerState.Response:
        """Layer 3: Enable/Disable lifecycle and temporal window adjustment."""
        self.enable = req.enable
        self.track_manager.temporal_window = req.temporal_window
        res.success = True
        state_str = "ENABLED" if self.enable else "FROZEN (DISABLED)"
        res.message = f"Gate manager state set to {state_str}, temporal_window={req.temporal_window}s."
        self.get_logger().info(res.message)
        return res

    def _cb_reset(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        """Layer 3: Trigger service to clear all tracks."""
        self.track_manager.clear_all_tracks()
        self.last_query_pos = None
        res.success = True
        res.message = "All tracks reset successfully."
        self.get_logger().info(res.message)
        return res

    def _convert_from_ll(self, latlon_msg: NavSatFix) -> Optional[np.ndarray]:
        if not self.from_ll_client.service_is_ready():
            if not self.from_ll_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().error(
                    f"Service {self.from_ll_service_name} unavailable."
                )
                return None
        req = FromLL.Request()
        req.ll_point = latlon_msg
        future = self.from_ll_client.call_async(req)
        while not future.done() and rclpy.ok():
            time.sleep(0.005)
        if future.result() is not None:
            pt = future.result().map_point
            return np.array([pt.x, pt.y, pt.z])
        return None

    def _convert_to_ll(self, position: np.ndarray) -> Optional[NavSatFix]:
        if not self.to_ll_client.service_is_ready():
            if not self.to_ll_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().error(
                    f"Service {self.to_ll_service_name} unavailable."
                )
                return None
        req = ToLL.Request()
        req.map_point = Point(
            x=float(position[0]), y=float(position[1]), z=float(position[2])
        )
        future = self.to_ll_client.call_async(req)
        while not future.done() and rclpy.ok():
            time.sleep(0.005)
        if future.result() is not None:
            return future.result().ll_point
        return None

    def _cb_get_entry(
        self, req: GetEntry.Request, res: GetEntry.Response
    ) -> GetEntry.Response:
        """Layer 3: Milestone query for immortal Entry buoy (FLASHING_BLUE)."""
        for track in self.track_manager.tracks.values():
            if (
                track.published_state == DetectedBuoyColor.FLASHING_BLUE
                and track.current_centroid is not None
            ):
                res.found = True
                res.message = f"Entry buoy found at track ID {track.track_id}."
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = self.world_frame_id
                pose_msg.pose.position.x = float(track.current_centroid[0])
                pose_msg.pose.position.y = float(track.current_centroid[1])
                pose_msg.pose.position.z = float(track.current_centroid[2])
                pose_msg.pose.orientation.w = 1.0
                res.buoy_pose = pose_msg
                if req.return_latlon:
                    ll = self._convert_to_ll(track.current_centroid)
                    if ll:
                        res.buoy_latlon = ll
                self.get_logger().info(f"[_cb_get_entry] {res.message}", throttle_duration_sec=2.0)
                return res
        res.found = False
        res.message = "Entry buoy (FLASHING_BLUE) not found."
        self.get_logger().info(f"[_cb_get_entry] {res.message}", throttle_duration_sec=2.0)
        return res

    def _cb_get_exit(
        self, req: GetExit.Request, res: GetExit.Response
    ) -> GetExit.Response:
        """Layer 3: Milestone query for immortal Exit buoy (SOLID_BLUE)."""
        for track in self.track_manager.tracks.values():
            if (
                track.published_state == DetectedBuoyColor.SOLID_BLUE
                and track.current_centroid is not None
            ):
                res.found = True
                res.message = f"Exit buoy found at track ID {track.track_id}."
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = self.world_frame_id
                pose_msg.pose.position.x = float(track.current_centroid[0])
                pose_msg.pose.position.y = float(track.current_centroid[1])
                pose_msg.pose.position.z = float(track.current_centroid[2])
                pose_msg.pose.orientation.w = 1.0
                res.buoy_pose = pose_msg
                if req.return_latlon:
                    ll = self._convert_to_ll(track.current_centroid)
                    if ll:
                        res.buoy_latlon = ll
                self.get_logger().info(f"[_cb_get_exit] {res.message}", throttle_duration_sec=2.0)
                return res
        res.found = False
        res.message = "Exit buoy (SOLID_BLUE) not found."
        self.get_logger().info(f"[_cb_get_exit] {res.message}", throttle_duration_sec=2.0)
        return res

    def _cb_extract_goal(
        self, req: ExtractLocalGoal.Request, res: ExtractLocalGoal.Response
    ) -> ExtractLocalGoal.Response:
        """Layer 3: Instantaneous local goal extraction with TF2 frame normalization and geodetic bridge."""
        vehicle_pos = None

        if req.input_is_latlon:
            vehicle_pos = self._convert_from_ll(req.current_latlon)
            if vehicle_pos is None:
                res.success = False
                res.goal_type = "INVALID"
                res.message = "Geodetic /fromLL conversion failed."
                return res
        else:
            in_pose = req.current_pose
            if (
                in_pose.header.frame_id
                and in_pose.header.frame_id != self.world_frame_id
            ):
                try:
                    in_pose = self.tf_buffer.transform(
                        in_pose, self.world_frame_id, timeout=Duration(seconds=1.0)
                    )
                except Exception as e:
                    res.success = False
                    res.goal_type = "INVALID"
                    res.message = f"TF transform failed from '{in_pose.header.frame_id}' to '{self.world_frame_id}': {e}"
                    return res
            vehicle_pos = np.array(
                [
                    in_pose.pose.position.x,
                    in_pose.pose.position.y,
                    in_pose.pose.position.z,
                ]
            )

        # Determine forward unit vector in world_frame_id (3-tier frame-intrinsic hierarchy)
        v_forward = None

        # Priority 1: Course axis vector from Entry to Exit buoys
        entry_pos, exit_pos = None, None
        for track in self.track_manager.tracks.values():
            if (
                track.published_state == DetectedBuoyColor.FLASHING_BLUE
                and track.current_centroid is not None
            ):
                entry_pos = track.current_centroid[:2]
            elif (
                track.published_state == DetectedBuoyColor.SOLID_BLUE
                and track.current_centroid is not None
            ):
                exit_pos = track.current_centroid[:2]
        if (
            entry_pos is not None
            and exit_pos is not None
            and np.linalg.norm(exit_pos - entry_pos) > 0.1
        ):
            diff = exit_pos - entry_pos
            v_forward = diff / np.linalg.norm(diff)

        # Priority 2: Vehicle (boat) trajectory motion vector
        if v_forward is None and self.last_query_pos is not None:
            if np.linalg.norm(vehicle_pos[:2] - self.last_query_pos[:2]) > 0.1:
                diff = vehicle_pos[:2] - self.last_query_pos[:2]
                v_forward = diff / np.linalg.norm(diff)

        # Priority 3: Fall back to drone base_link heading in world_frame_id
        if v_forward is None:
            drone_yaw = 0.0
            try:
                base_transform = self.tf_buffer.lookup_transform(
                    self.world_frame_id,
                    self.base_link_frame_id,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.1),
                )
                drone_yaw = _get_yaw_from_quaternion(base_transform.transform.rotation)
            except Exception as e:
                self.get_logger().warn(
                    f"Failed to lookup drone heading from '{self.base_link_frame_id}': {e}",
                    throttle_duration_sec=2.0,
                )
            v_forward = np.array([math.cos(drone_yaw), math.sin(drone_yaw)])
            if np.linalg.norm(v_forward) == 0.0:
                v_forward = np.array([1.0, 0.0])
            else:
                v_forward = v_forward / np.linalg.norm(v_forward)

        self.last_query_pos = vehicle_pos.copy()

        # Stateless candidate generation
        red_tracks = [
            t
            for t in self.track_manager.tracks.values()
            if t.published_state == DetectedBuoyColor.FLASHING_RED
            and t.current_centroid is not None
        ]
        green_tracks = [
            t
            for t in self.track_manager.tracks.values()
            if t.published_state == DetectedBuoyColor.FLASHING_GREEN
            and t.current_centroid is not None
        ]

        candidates: List[Tuple[str, np.ndarray]] = []
        paired_red: Set[int] = set()
        paired_green: Set[int] = set()

        # Greedy pairing within max_gate_width
        pairs = []
        for rt in red_tracks:
            for gt in green_tracks:
                dist = np.linalg.norm(rt.current_centroid[:2] - gt.current_centroid[:2])
                if dist <= self.max_gate_width:
                    pairs.append((dist, rt.track_id, gt.track_id))
        pairs.sort(key=lambda x: x[0])
        for dist, r_id, g_id in pairs:
            if r_id not in paired_red and g_id not in paired_green:
                paired_red.add(r_id)
                paired_green.add(g_id)
                midpoint = 0.5 * (
                    self.track_manager.tracks[r_id].current_centroid
                    + self.track_manager.tracks[g_id].current_centroid
                )
                candidates.append(("GATE_PAIR", midpoint))

        # Singular standoffs
        for rt in red_tracks:
            if rt.track_id not in paired_red:
                alpha = math.atan2(
                    rt.current_centroid[1] - vehicle_pos[1],
                    rt.current_centroid[0] - vehicle_pos[0],
                )
                # +90 deg CCW for RED on starboard
                standoff = rt.current_centroid.copy()
                standoff[0] += self.vehicle_encasing_radius * math.cos(
                    alpha + math.pi / 2.0
                )
                standoff[1] += self.vehicle_encasing_radius * math.sin(
                    alpha + math.pi / 2.0
                )
                candidates.append(("SINGULAR_BUOY", standoff))

        for gt in green_tracks:
            if gt.track_id not in paired_green:
                alpha = math.atan2(
                    gt.current_centroid[1] - vehicle_pos[1],
                    gt.current_centroid[0] - vehicle_pos[0],
                )
                # -90 deg CW for GREEN on port
                standoff = gt.current_centroid.copy()
                standoff[0] += self.vehicle_encasing_radius * math.cos(
                    alpha - math.pi / 2.0
                )
                standoff[1] += self.vehicle_encasing_radius * math.sin(
                    alpha - math.pi / 2.0
                )
                candidates.append(("SINGULAR_BUOY", standoff))

        # Include Exit buoy if available (Entry is NEVER included)
        for t in self.track_manager.tracks.values():
            if (
                t.published_state == DetectedBuoyColor.SOLID_BLUE
                and t.current_centroid is not None
            ):
                candidates.append(("EXIT_BUOY", t.current_centroid))

        # Stateless In-Front filtering & traversal check
        valid_candidates: List[Tuple[str, np.ndarray, float]] = []
        for c_type, c_pos in candidates:
            dist_to_cand = np.linalg.norm(c_pos[:2] - vehicle_pos[:2])
            d_vec = c_pos[:2] - vehicle_pos[:2]
            dot_val = np.dot(d_vec, v_forward)

            # Ignore waypoints that are behind the vehicle or within reaching tolerance
            if dist_to_cand < self.waypoint_pass_tol or dot_val <= 0:
                continue

            valid_candidates.append((c_type, c_pos, dist_to_cand))

        if not valid_candidates:
            res.success = False
            res.goal_type = "INVALID"
            res.message = "No valid forward waypoints found."
            self.get_logger().info(f"[_cb_extract_goal] {res.message}", throttle_duration_sec=2.0)
            return res

        # Distance Gating: minimum distance
        valid_candidates.sort(key=lambda x: x[2])
        best_type, best_pos, best_dist = valid_candidates[0]

        res.success = True
        res.goal_type = best_type
        res.message = f"Selected {best_type} goal at distance {best_dist:.2f}m."

        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = self.world_frame_id
        goal_pose.pose.position.x = float(best_pos[0])
        goal_pose.pose.position.y = float(best_pos[1])
        goal_pose.pose.position.z = float(best_pos[2])
        goal_pose.pose.orientation.w = 1.0
        res.goal_pose = goal_pose

        if req.return_latlon:
            ll = self._convert_to_ll(best_pos)
            if ll:
                res.goal_latlon = ll

        self.get_logger().info(f"[_cb_extract_goal] {res.message}", throttle_duration_sec=2.0)
        return res

    def _publish_markers(self):
        """Publish RViz visualization markers for tracks, IDs, and waypoints."""
        marker_array = MarkerArray()
        now_msg = self.get_clock().now().to_msg()

        # Clear old markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        color_map = {
            DetectedBuoyColor.FLASHING_RED: ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8),
            DetectedBuoyColor.FLASHING_GREEN: ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8),
            DetectedBuoyColor.FLASHING_BLUE: ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.9),
            DetectedBuoyColor.SOLID_BLUE: ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.9),
            DetectedBuoyColor.UNKNOWN: ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.5),
        }

        for tid, track in self.track_manager.tracks.items():
            if track.current_centroid is None:
                continue

            # Sphere marker
            m_sphere = Marker()
            m_sphere.header.frame_id = self.world_frame_id
            m_sphere.header.stamp = now_msg
            m_sphere.ns = "buoy_tracks"
            m_sphere.id = tid * 2
            m_sphere.type = Marker.SPHERE
            m_sphere.action = Marker.ADD
            m_sphere.pose.position.x = float(track.current_centroid[0])
            m_sphere.pose.position.y = float(track.current_centroid[1])
            m_sphere.pose.position.z = float(track.current_centroid[2])
            m_sphere.pose.orientation.w = 1.0
            m_sphere.scale = Vector3(x=0.6, y=0.6, z=0.6)
            m_sphere.color = color_map.get(
                track.published_state, color_map[DetectedBuoyColor.UNKNOWN]
            )
            marker_array.markers.append(m_sphere)

            # Text label marker
            m_text = Marker()
            m_text.header.frame_id = self.world_frame_id
            m_text.header.stamp = now_msg
            m_text.ns = "buoy_labels"
            m_text.id = tid * 2 + 1
            m_text.type = Marker.TEXT_VIEW_FACING
            m_text.action = Marker.ADD
            m_text.pose.position.x = float(track.current_centroid[0])
            m_text.pose.position.y = float(track.current_centroid[1])
            m_text.pose.position.z = float(track.current_centroid[2]) + 0.8
            m_text.pose.orientation.w = 1.0
            m_text.scale.z = 0.4
            m_text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            m_text.text = f"ID: {tid}\n{track.published_state.name}"
            marker_array.markers.append(m_text)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = BuoyGateManagerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
