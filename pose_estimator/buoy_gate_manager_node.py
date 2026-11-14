#!/usr/bin/env python3
"""ROS interfaces for stamped buoy observations, tracking, and gate queries."""

from collections import Counter
from copy import deepcopy
import json
import time
from typing import Optional

import numpy as np
import rclpy
from tf2_geometry_msgs import do_transform_pose_stamped
import tf2_ros
import tf_transformations
from bb_robotx_msgs.action import SendSafePassageReport
from bb_robotx_msgs.msg import BuoyDetectionArray, LatLng, SafePassageBuoy, SafePassageReport
from bb_robotx_msgs.srv import ExtractLocalGoal, GetEntry, GetExit, SetGateManagerState
from pose_estimator.buoy.geographic import Px4GeographicReference, pose_to_fix
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.clock import ClockType
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import NavSatFix
from std_srvs.srv import Trigger
from visualization_msgs.msg import MarkerArray

from pose_estimator.buoy.tracking import TrackManager, DetectedBuoyColor
from pose_estimator.buoy.observations import ObservationBuffer
from pose_estimator.buoy.planning import GatePlanning, GOAL_TYPE_INVALID
from pose_estimator.buoy.visualization import GateVisualization
from pose_estimator.utils.odometry_transforms import add_odometry_transform

# Safe Passage report for the ASV: best effort over the lossy link, volatile so only a live report counts.
REPORT_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
)


# Track state -> SafePassageBuoy.state (BeaconState vocabulary). Unknowns are
# omitted from the ASV map and guessed only in the RoboCommand report.
REPORT_STATES = {
    DetectedBuoyColor.UNKNOWN: "unknown",
    DetectedBuoyColor.BLACK: "off",
    DetectedBuoyColor.FLASHING_RED: "flashing_red",
    DetectedBuoyColor.FLASHING_GREEN: "flashing_green",
    DetectedBuoyColor.FLASHING_BLUE: "flashing_blue",
    DetectedBuoyColor.SOLID_BLUE: "steady_blue",
}

# User-provided expected state ratio for filling unknowns in the frozen report.
REPORT_GUESS_RATIO = {
    "flashing_red": 4,
    "flashing_green": 4,
    "steady_blue": 1,
    "flashing_blue": 1,
}


class BuoyGateManagerNode(GatePlanning, GateVisualization, Px4GeographicReference, Node):
    def __init__(self):
        super().__init__("buoy_gate_manager_node")

        # Reference updates, freeze/reset and publication must not overlap.
        self.main_cb_group = self.default_callback_group

        def param(name, default):
            return self.declare_parameter(name, default).value

        self.buoy_detections_topic = param("buoy_detections_topic", "buoy_detections")
        self.buoy_detections_topics = list(dict.fromkeys(
            param("buoy_detections_topics", [self.buoy_detections_topic])))
        self.marker_topic = param("marker_topic", "gate_manager_markers")
        tf_wait_timeout = param("tf_wait_timeout", 0.25)
        max_pending = param("max_pending_observations", 1000)
        self.world_frame_id = param("world_frame_id", "map")
        self.local_axes = param("local_axes", "enu")
        if self.local_axes not in ("ned", "enu"):
            raise ValueError("local_axes must be ned or enu")
        self.reference_frame_id = param("reference_frame_id", "odom_enu")
        self.setup_geographic_reference(
            param("reference_topic", "/uav/home_lat_lon"),
            param("global_position_topic", "/fmu/out/vehicle_global_position"),
        )
        self.base_link_frame_id = param("base_link_frame_id", "base_link")
        self.fft_timer_period = param("fft_timer_period", 0.5)
        self.max_gate_width = param("max_gate_width", 1.0)
        self.vehicle_encasing_radius = param("vehicle_encasing_radius", 0.8)
        self.waypoint_pass_tol = param("waypoint_pass_tol", 1.0)
        self.visualize_candidates = param("visualize_candidates", True)

        # Initialize TF2
        self.tf_buffer = tf2_ros.Buffer()
        odom_topic = param("input_odom_topic", "")
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, static_only=bool(odom_topic))
        if odom_topic:
            self.odom_tf_sub = self.create_subscription(
                Odometry, odom_topic,
                lambda msg: add_odometry_transform(self.tf_buffer, msg),
                qos_profile_sensor_data,
            )
        self.observations = ObservationBuffer(
            self.world_frame_id, self._transform_observation,
            tf_wait_timeout, max_pending,
        )
        self._reported_drops = 0
        self.observation_timer = self.create_timer(
            0.02, self._drain_observations, callback_group=self.main_cb_group
        )

        # Retain both query positions so markers can reproduce trajectory heading.
        self.enable: bool = param("enable_on_start", False)
        self.last_query_pos: Optional[np.ndarray] = None
        self.prev_query_pos: Optional[np.ndarray] = None

        self.track_manager = TrackManager(
            max_association_radius=param("max_association_radius", 0.5),
            dbscan_min_samples=param("dbscan_min_samples", 5),
            track_timeout=param("track_timeout", 60.0),
            required_density=param("required_density", 0.8),
            butterworth_cutoff_freq=param("butterworth_cutoff_freq", 3.0),
            butterworth_order=param("butterworth_order", 4),
            target_freq=param("target_freq", 0.5),  # handbook: 1 s on, 1 s off
            target_freq_tol=param("target_freq_tol", 0.2),
            duty_cycle_min=param("duty_cycle_min", 0.1),
            duty_cycle_max=param("duty_cycle_max", 0.85),
            consistent_threshold=param("consistent_threshold", 0.6),
            fft_min_magnitude=param("fft_min_magnitude", 0.2),
            virtual_sampling_rate=param("virtual_sampling_rate", 10.0),
            max_color_gap=param("max_color_gap", 0.5),
            min_flash_cycles=param("min_flash_cycles", 2.0),
            logger=self.get_logger(),
        )

        self.detections_subs = [self.create_subscription(
            BuoyDetectionArray,
            topic,
            self._on_buoy_detections,
            qos_profile_sensor_data,
            callback_group=self.main_cb_group,
        ) for topic in self.buoy_detections_topics]
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)

        # Serve the frozen map to the ASV; optionally retry delivery to RoboCommand.
        self._report_msg: Optional[SafePassageReport] = None  # RoboCommand, with frozen guesses
        self._asv_report_msg: Optional[SafePassageReport] = None  # observed states only
        self.report_pub = self.create_publisher(
            SafePassageReport, param("report_topic", "safe_passage/report"), REPORT_QOS)
        self.report_timer = self.create_timer(
            param("report_publish_period", 1.0), self._publish_report, callback_group=self.main_cb_group
        )
        self.send_report = param("send_report_on_freeze", False)
        self.report_timeout = param("report_timeout", 30.0)  # s, covers the OCS rate budget and its PUBACK wait
        if not np.isfinite(self.report_timeout) or self.report_timeout <= 0:
            raise ValueError("report_timeout must be finite and positive")
        self.report_retry_period = param("report_retry_period", 5.0)  # s after a failed attempt
        self.report_action = param("report_action", "/uav/report/safe_passage")
        self.report_client = ActionClient(
            self, SendSafePassageReport, self.report_action, callback_group=self.main_cb_group)
        self._report_unsent = False  # the current report still has to reach RoboCommand
        self._report_in_flight = False
        self._report_attempt = 0  # answers to an older attempt are ignored
        self._report_deadline = 0.0  # monotonic s
        self._report_handle = None
        self._report_next_try = 0.0  # monotonic s

        # Service servers
        self.srv_set_state = self.create_service(
            SetGateManagerState,
            param("set_state_service", "set_gate_manager_state"),
            self._cb_set_state,
            callback_group=self.main_cb_group,
        )
        self.srv_reset = self.create_service(
            Trigger,
            param("reset_service", "reset_tracks"),
            self._cb_reset,
            callback_group=self.main_cb_group,
        )
        self.srv_extract_goal = self.create_service(
            ExtractLocalGoal,
            param("extract_goal_service", "extract_local_goal"),
            self._cb_extract_goal,
            callback_group=self.main_cb_group,
        )
        self.srv_get_entry = self.create_service(
            GetEntry,
            param("get_entry_service", "get_entry"),
            self._cb_get_entry,
            callback_group=self.main_cb_group,
        )
        self.srv_get_exit = self.create_service(
            GetExit,
            param("get_exit_service", "get_exit"),
            self._cb_get_exit,
            callback_group=self.main_cb_group,
        )

        self.fft_timer = self.create_timer(
            self.fft_timer_period, self._on_fft_timer, callback_group=self.main_cb_group
        )

        self.get_logger().info(
            f"BuoyGateManagerNode operational. Listening on {self.buoy_detections_topics}, "
            f"world frame '{self.world_frame_id}', enable={self.enable}."
        )

    def _on_buoy_detections(self, msg: BuoyDetectionArray):
        if self.enable:
            self.observations.add(msg, time.monotonic())
            self._drain_observations()

    def _transform_observation(self, observation):
        # Zero-time lookups mean "latest" in TF; the input buffer rejects them.
        transform = self.tf_buffer.lookup_transform(
            self.world_frame_id, observation.frame, Time(nanoseconds=observation.stamp_ns, clock_type=ClockType.ROS_TIME)
        ).transform
        q, t = transform.rotation, transform.translation
        rotation = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        return rotation @ observation.position + np.array([t.x, t.y, t.z])

    def _drain_observations(self):
        if not self.enable:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        window = self.track_manager.temporal_window
        ready = self.observations.drain(time.monotonic())
        buffer = self.track_manager.detection_buffer
        # A delayed TF can make older observations arrive after newer ones.
        if ready:
            ordered = sorted([*buffer, *ready], key=lambda observation: observation[0])
            buffer.clear()
            buffer.extend(ordered)
        if window > 0:
            while buffer and buffer[0][0] < now - window:
                buffer.popleft()
        if self.observations.dropped != self._reported_drops:
            self.get_logger().warning(
                f"Dropped {self.observations.dropped} invalid, expired, or overflowed buoy observations",
                throttle_duration_sec=5.0,
            )
            self._reported_drops = self.observations.dropped

    def _on_fft_timer(self):
        """Update tracks, classify colors, merge singletons, and publish markers."""
        if not self.enable:
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        self._drain_observations()
        self.track_manager.associate_and_update(now_sec)
        self.track_manager.prune_temporal_window(now_sec)
        self.track_manager.recompute_centroids()
        self.track_manager.prune_timeouts(now_sec)
        previous_states = {tid: track.published_state for tid, track in self.track_manager.tracks.items()}
        self.track_manager.classify_tracks()
        for tid, track in self.track_manager.tracks.items():
            if previous_states.get(tid) != track.published_state:
                self.get_logger().info(f"Buoy track {tid}: {track.published_state.name}")
        self.track_manager.deduplicate_singletons()
        self.track_manager.recompute_centroids()
        self._publish_markers()

    def _cb_set_state(
        self, req: SetGateManagerState.Request, res: SetGateManagerState.Response
    ) -> SetGateManagerState.Response:
        """Enable or freeze collection and set the temporal window."""
        if not np.isfinite(req.temporal_window):
            res.success = False
            res.message = "Temporal window must be finite (nonpositive means unlimited)"
            return res
        if not req.enable:
            # Finish processing queued observations before freezing the result.
            if self.enable:
                self._on_fft_timer()
                tracks = []
                for tid, track in self.track_manager.tracks.items():
                    known = [o for o in track.observations if o.color is not None]
                    gaps = np.diff([o.timestamp for o in known])
                    tracks.append({
                        "id": tid, "state": track.published_state.name,
                        "position": (track.current_centroid.tolist()
                                     if track.current_centroid is not None else None),
                        "observations": len(track.observations),
                        "colors": dict(Counter(o.color or "unobservable" for o in track.observations)),
                        "known_span_s": known[-1].timestamp - known[0].timestamp if known else 0.0,
                        "max_known_gap_s": float(max(gaps)) if len(gaps) else 0.0,
                    })
                self.get_logger().info("BUOY_SEARCH_FROZEN " + json.dumps(tracks))
            self.observations.clear()
            self.track_manager.detection_buffer.clear()
            if self.enable:
                self._set_report(self._build_report_msg())
        else:
            self._set_report(None)  # collecting again: stop serving before the tracks change
        self.enable = req.enable
        self.track_manager.temporal_window = req.temporal_window
        res.success = True
        state_str = "ENABLED" if self.enable else "FROZEN (DISABLED)"
        res.message = f"Gate manager state set to {state_str}, temporal_window={req.temporal_window}s."
        if not self.enable and self._report_msg is None:
            res.message += " No Safe Passage report available; see manager logs."
        self.get_logger().info(res.message)
        return res

    def _cb_reset(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        """Clear pending observations, tracks, and query history."""
        self.observations.clear()
        self.track_manager.clear_all_tracks()
        self._set_report(None)
        self.last_query_pos = None
        self.prev_query_pos = None
        res.success = True
        res.message = "All tracks reset successfully."
        self.get_logger().info(res.message)
        return res

    def _reference(self, message):
        # The reference publisher labels its origin odom_enu even for NED users.
        # Both local views share that origin; local_axes controls the axis swap.
        if (message.header.frame_id != self.reference_frame_id
                or not (message.header.stamp.sec or message.header.stamp.nanosec)):
            self.reference = self.converter = None
            self.get_logger().warning("Invalid PX4 reference frame or timestamp", throttle_duration_sec=5.0)
            return
        super()._reference(message)

    def _geographic_ready(self):
        if self.converter is None or self.altitude_offset is None:
            self.get_logger().warning("Waiting for PX4 geographic reference and ellipsoid altitude correction",
                                      throttle_duration_sec=5.0)
            return False
        return True

    def _convert_from_ll(self, latlon_msg: NavSatFix) -> Optional[np.ndarray]:
        if not self._geographic_ready():
            return None
        latitude, longitude, altitude = latlon_msg.latitude, latlon_msg.longitude, latlon_msg.altitude
        if (not np.isfinite([latitude, longitude, altitude]).all()
                or not -90 <= latitude <= 90 or not -180 <= longitude <= 180):
            return None
        point = self.converter.forward(latitude, longitude, altitude)
        position = ([point.y, point.x, -point.z] if self.local_axes == "ned"
                    else [point.x, point.y, point.z])
        return np.array(position)

    def _convert_to_ll(self, position: np.ndarray) -> Optional[NavSatFix]:
        if not self._geographic_ready():
            return None
        try:
            return pose_to_fix(self._world_pose(position), self.converter,
                               self.local_axes, self.world_frame_id)
        except ValueError:
            return None

    def _guess_unknown_report_states(self, report):
        unknowns = [b for b in report.buoys if b.state == "unknown"]
        if not unknowns:
            return
        states = tuple(REPORT_GUESS_RATIO)
        ratio = np.array(list(REPORT_GUESS_RATIO.values()), dtype=float)
        expected = len(report.buoys) * ratio / ratio.sum()
        confirmed = Counter(b.state for b in report.buoys if b.state != "unknown")
        initial_confirmed = dict(confirmed)
        remaining = np.maximum(0.0, expected - np.array([confirmed[s] for s in states]))
        for i, state in enumerate(states):
            if state in ('steady_blue', 'flashing_blue'):
                remaining[i] = min(remaining[i], max(0, 1 - confirmed[state]))
        rng = np.random.default_rng()
        assignments = []
        for buoy in unknowns:
            available = np.array([state not in ('steady_blue', 'flashing_blue') or confirmed[state] < 1
                                  for state in states])
            weights = remaining * available
            total = float(weights.sum())
            # Excess confirmed counts cannot create negative probabilities. If
            # no positive deficit remains, fall back to the original ratio.
            if not np.isfinite(total) or total <= 0:
                weights = ratio * available  # Red/green always remain available.
                total = float(weights.sum())
            probabilities = weights / total
            index = int(rng.choice(len(states), p=probabilities))
            buoy.state = states[index]
            confirmed[buoy.state] += 1
            remaining[index] = max(0.0, remaining[index] - 1.0)
            assignments.append({'latitude': buoy.position.latitude, 'longitude': buoy.position.longitude,
                                'state': buoy.state, 'probabilities': probabilities.tolist()})
        self.get_logger().info('SAFE_PASSAGE_REPORT_GUESSES ' + json.dumps({
            'total_detected': len(report.buoys), 'states': states,
            'expected': expected.tolist(), 'confirmed': initial_confirmed, 'assignments': assignments,
        }))

    def _build_report_msg(self) -> Optional[SafePassageReport]:
        """Freeze positions and inferred report states, including an empty report.

        Select one measured candidate per blue role without changing track states.
        Unknowns stay unresolved here so each recipient can apply its own policy.
        Missing entry/exit positions remain NaN; failed position conversions are omitted.
        """
        unknown = LatLng(latitude=float("nan"), longitude=float("nan"))
        report = SafePassageReport(entry_position=unknown, exit_position=deepcopy(unknown))
        report_tracks = []
        for track in self.track_manager.tracks.values():
            state = REPORT_STATES.get(track.published_state)
            if state is None or track.current_centroid is None:
                continue
            latlon = self._convert_to_ll(track.current_centroid)
            if latlon is None:
                self.get_logger().warning(f"Safe Passage report: track {track.track_id} left out, conversion failed")
                continue
            position = LatLng(latitude=latlon.latitude, longitude=latlon.longitude)
            report.buoys.append(SafePassageBuoy(position=position, state=state))
            report_tracks.append(track)
        for state, metric in (('steady_blue', 'blue_presence'), ('flashing_blue', 'blue_power')):
            candidates = [i for i, buoy in enumerate(report.buoys) if buoy.state == state]
            if not candidates:
                continue
            selected = max(candidates, key=lambda i: (
                getattr(report_tracks[i], metric, 0.0),
                getattr(report_tracks[i], 'blue_duration', 0.0), -report_tracks[i].track_id))
            for i in candidates:
                if i != selected:
                    report.buoys[i].state = 'unknown'
            self.get_logger().info('SAFE_PASSAGE_BLUE_SELECTION ' + json.dumps({
                'state': state, 'selected_track': report_tracks[selected].track_id,
                'candidate_tracks': [report_tracks[i].track_id for i in candidates],
            }))
        self._set_report_endpoints(report)
        self.get_logger().info("SAFE_PASSAGE_REPORT " + json.dumps(
            [[b.state, round(b.position.latitude, 7), round(b.position.longitude, 7)] for b in report.buoys]))
        return report

    @staticmethod
    def _set_report_endpoints(report):
        unknown = LatLng(latitude=float("nan"), longitude=float("nan"))
        for state, field in (('flashing_blue', 'entry_position'), ('steady_blue', 'exit_position')):
            selected = next((b for b in report.buoys if b.state == state), None)
            setattr(report, field, deepcopy(selected.position) if selected is not None else deepcopy(unknown))

    def _cancel_report_attempt(self):
        """Invalidate callbacks before requesting cancellation of an old send."""
        self._report_attempt += 1
        if self._report_handle is not None:
            self._report_handle.cancel_goal_async()
        self._report_in_flight, self._report_handle = False, None

    def _set_report(self, report: Optional[SafePassageReport]):
        self._cancel_report_attempt()
        self._asv_report_msg = deepcopy(report)
        self._report_msg = deepcopy(report)
        if report is not None:
            self._asv_report_msg.buoys = [b for b in self._asv_report_msg.buoys if b.state != 'unknown']
            self._set_report_endpoints(self._asv_report_msg)
            self._guess_unknown_report_states(self._report_msg)
            self._set_report_endpoints(self._report_msg)
        self._report_unsent = report is not None and self.send_report
        self._report_next_try = 0.0

    def _publish_report(self):
        """Publish observed states to ASV; send the frozen guessed report to RoboCommand."""
        report = self._report_msg
        if report is None or self.enable:
            return
        self.report_pub.publish(self._asv_report_msg)
        now = time.monotonic()
        if self._report_in_flight and now > self._report_deadline:
            self.get_logger().warning(f"Safe Passage report unanswered after {self.report_timeout:.0f} s; retrying")
            self._cancel_report_attempt()
            self._report_next_try = now + self.report_retry_period
        if self._report_unsent and not self._report_in_flight and now >= self._report_next_try:
            self._send_report(report)

    def _send_report(self, report: SafePassageReport):
        if not self.report_client.server_is_ready():
            self.get_logger().warning(f"{self.report_action} unavailable, report not sent yet",
                                      throttle_duration_sec=10.0)
            self._report_next_try = time.monotonic() + self.report_retry_period
            return
        self._report_in_flight = True
        self._report_attempt += 1
        attempt = self._report_attempt
        self._report_deadline = time.monotonic() + self.report_timeout

        def finished(ok: bool, why: str):
            if attempt != self._report_attempt:
                return  # timed out and superseded
            self._report_in_flight, self._report_handle = False, None
            if ok:
                self._report_unsent = False
                self.get_logger().info(f"Safe Passage report sent to RoboCommand ({why})")
            else:
                self._report_next_try = time.monotonic() + self.report_retry_period
                self.get_logger().warning(f"Safe Passage report not sent: {why}; retrying")

        def on_result(future):
            result = future.result().result
            finished(not result.error, f"seq {result.seq}" if not result.error else result.error)

        def on_goal(future):
            handle = future.result()
            if not handle.accepted:
                finished(False, "bridge rejected the goal")
                return
            if attempt != self._report_attempt:
                handle.cancel_goal_async()  # accepted after we gave up on it
                return
            self._report_handle = handle
            handle.get_result_async().add_done_callback(on_result)

        self.report_client.send_goal_async(SendSafePassageReport.Goal(report=report)).add_done_callback(on_goal)

    def _cb_get_entry(self, req: GetEntry.Request, res: GetEntry.Response) -> GetEntry.Response:
        return self._milestone_response(req, res, DetectedBuoyColor.FLASHING_BLUE, "Entry")

    def _cb_get_exit(self, req: GetExit.Request, res: GetExit.Response) -> GetExit.Response:
        return self._milestone_response(req, res, DetectedBuoyColor.SOLID_BLUE, "Exit")

    def _world_pose(self, position):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.world_frame_id
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, position)
        pose.pose.orientation.w = 1.0
        return pose

    def _milestone_response(self, req, res, state, name):
        track = next((track for track in self.track_manager.tracks.values()
                      if track.published_state == state and track.current_centroid is not None), None)
        res.found = track is not None
        if track is None:
            res.message = f"{name} buoy ({state.name}) not found."
        else:
            res.message = f"{name} buoy found at track ID {track.track_id}."
            res.buoy_pose = self._world_pose(track.current_centroid)
            if req.return_latlon:
                latlon = self._convert_to_ll(track.current_centroid)
                if latlon is not None:
                    res.buoy_latlon = latlon
                else:
                    res.found = False
                    res.message = f"{name} buoy found, but geographic conversion failed."
        self.get_logger().info(res.message, throttle_duration_sec=2.0)
        return res

    def _cb_extract_goal(
        self, req: ExtractLocalGoal.Request, res: ExtractLocalGoal.Response
    ) -> ExtractLocalGoal.Response:
        """Normalize the query pose and return the nearest valid forward goal."""
        vehicle_pos = None

        if req.input_is_latlon:
            vehicle_pos = self._convert_from_ll(req.current_latlon)
            if vehicle_pos is None:
                res.success = False
                res.goal_type = GOAL_TYPE_INVALID
                res.message = "Geographic input conversion failed."
                return res
        else:
            in_pose = req.current_pose
            if (
                in_pose.header.frame_id
                and in_pose.header.frame_id != self.world_frame_id
            ):
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.world_frame_id, in_pose.header.frame_id,
                        Time.from_msg(in_pose.header.stamp), timeout=Duration(seconds=1.0),
                    )
                    in_pose = do_transform_pose_stamped(in_pose, transform)
                except Exception as e:
                    res.success = False
                    res.goal_type = GOAL_TYPE_INVALID
                    res.message = f"TF transform failed from '{in_pose.header.frame_id}' to '{self.world_frame_id}': {e}"
                    return res
            vehicle_pos = np.array(
                [
                    in_pose.pose.position.x,
                    in_pose.pose.position.y,
                    in_pose.pose.position.z,
                ]
            )

        # Share candidate evaluation with marker rendering.
        _, forward_source, candidates = self._evaluate_candidates(
            vehicle_pos, self.last_query_pos
        )

        # Evaluate against the preceding query before advancing history.
        self.prev_query_pos = self.last_query_pos
        self.last_query_pos = vehicle_pos.copy()

        best = self._select_best_candidate(candidates)
        if best is None:
            res.success = False
            res.goal_type = GOAL_TYPE_INVALID
            res.message = (
                f"No valid forward waypoints found among {len(candidates)} candidate(s) "
                f"(heading source {forward_source})."
            )
            self.get_logger().info(
                f"[_cb_extract_goal] {res.message}", throttle_duration_sec=2.0
            )
            return res

        res.success = True
        res.goal_type = best.goal_type
        res.message = (
            f"Selected {best.goal_type} goal at distance {best.distance:.2f}m "
            f"(heading source {forward_source})."
        )

        res.goal_pose = self._world_pose(best.position)

        if req.return_latlon:
            ll = self._convert_to_ll(best.position)
            if ll is not None:
                res.goal_latlon = ll
            else:
                res.success = False
                res.goal_type = GOAL_TYPE_INVALID
                res.message = "Goal found, but geographic conversion failed."

        self.get_logger().info(
            f"[_cb_extract_goal] {res.message}", throttle_duration_sec=2.0
        )
        return res


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
