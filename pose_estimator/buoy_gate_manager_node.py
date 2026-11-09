#!/usr/bin/env python3
"""ROS interfaces for stamped buoy observations, tracking, and gate queries."""

from collections import Counter
import json
import time
from threading import Event
from typing import Optional

import numpy as np
import rclpy
from tf2_geometry_msgs import do_transform_pose_stamped
import tf2_ros
import tf_transformations
from bb_robotx_msgs.msg import BuoyDetectionArray
from bb_robotx_msgs.srv import ExtractLocalGoal, GetEntry, GetExit, SetGateManagerState
from geographic_msgs.msg import GeoPath, GeoPoseStamped
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import ClockType
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data,
)
from rclpy.time import Time
from robot_localization.srv import FromLL, ToLL
from sensor_msgs.msg import NavSatFix
from std_srvs.srv import Trigger
from visualization_msgs.msg import MarkerArray

from pose_estimator.buoy.tracking import TrackManager, DetectedBuoyColor
from pose_estimator.buoy.observations import ObservationBuffer
from pose_estimator.buoy.planning import GatePlanning, GOAL_TYPE_INVALID
from pose_estimator.buoy.visualization import GateVisualization
from pose_estimator.utils.odometry_transforms import add_odometry_transform

# Safe Passage path for the ASV: best effort over the lossy link, volatile so only a live path counts.
PATH_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
)


class BuoyGateManagerNode(GatePlanning, GateVisualization, Node):
    def __init__(self):
        super().__init__("buoy_gate_manager_node")

        self.main_cb_group = MutuallyExclusiveCallbackGroup()
        self.client_cb_group = MutuallyExclusiveCallbackGroup()

        def param(name, default):
            return self.declare_parameter(name, default).value

        self.buoy_detections_topic = param("buoy_detections_topic", "buoy_detections")
        self.marker_topic = param("marker_topic", "gate_manager_markers")
        tf_wait_timeout = param("tf_wait_timeout", 0.25)
        max_pending = param("max_pending_observations", 1000)
        self.from_ll_service_name = param("from_ll_service", "fromLL")  # relative: /uav/fromLL
        self.to_ll_service_name = param("to_ll_service", "toLL")  # relative: /uav/toLL
        self.geodetic_timeout = param("geodetic_timeout", 3.0)
        if not np.isfinite(self.geodetic_timeout) or self.geodetic_timeout <= 0:
            raise ValueError("Geodetic timeout must be finite and positive")
        self.world_frame_id = param("world_frame_id", "map")
        self.base_link_frame_id = param("base_link_frame_id", "base_link")
        self.fft_timer_period = param("fft_timer_period", 0.5)
        self.max_gate_width = param("max_gate_width", 1.0)
        # Safe Passage path standoff for unpaired buoys. Keep above the ASV's sample_radius.
        self.standoff_distance = param("standoff_distance", 1.5)
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

        self.detections_sub = self.create_subscription(
            BuoyDetectionArray,
            self.buoy_detections_topic,
            self._on_buoy_detections,
            qos_profile_sensor_data,
            callback_group=self.main_cb_group,
        )
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)

        # Safe Passage path: built when frozen, served only while frozen
        self._path_msg: Optional[GeoPath] = None
        self.path_pub = self.create_publisher(GeoPath, param("path_topic", "safe_passage/path"), PATH_QOS)
        self.path_timer = self.create_timer(
            param("path_publish_period", 1.0), self._publish_path, callback_group=MutuallyExclusiveCallbackGroup()
        )

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

        # Geodetic bridge clients
        self.from_ll_client = self.create_client(
            FromLL, self.from_ll_service_name, callback_group=self.client_cb_group
        )
        self.to_ll_client = self.create_client(
            ToLL, self.to_ll_service_name, callback_group=self.client_cb_group
        )

        self.fft_timer = self.create_timer(
            self.fft_timer_period, self._on_fft_timer, callback_group=self.main_cb_group
        )

        self.get_logger().info(
            f"BuoyGateManagerNode operational. Listening on '{self.buoy_detections_topic}', "
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
            self._path_msg = self._build_path_msg()
        else:
            self._path_msg = None  # collecting again: stop serving before the tracks change
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
        """Clear pending observations, tracks, and query history."""
        self.observations.clear()
        self.track_manager.clear_all_tracks()
        self._path_msg = None
        self.last_query_pos = None
        self.prev_query_pos = None
        res.success = True
        res.message = "All tracks reset successfully."
        self.get_logger().info(res.message)
        return res

    def _call_geodetic(self, client, request, service_name):
        """Bound service discovery and response waiting by one monotonic deadline."""
        deadline = time.monotonic() + self.geodetic_timeout
        if not client.wait_for_service(timeout_sec=self.geodetic_timeout):
            self.get_logger().error(f"Service {service_name} unavailable.")
            return None
        future = client.call_async(request)
        completed = Event()
        future.add_done_callback(lambda _: completed.set())
        if not completed.wait(max(0.0, deadline - time.monotonic())):
            future.cancel()
            self.get_logger().error(f"Service {service_name} timed out.")
            return None
        try:
            return future.result()
        except Exception as error:
            self.get_logger().error(f"Service {service_name} failed: {error}")
            return None

    def _convert_from_ll(self, latlon_msg: NavSatFix) -> Optional[np.ndarray]:
        request = FromLL.Request()
        request.ll_point.latitude = latlon_msg.latitude
        request.ll_point.longitude = latlon_msg.longitude
        request.ll_point.altitude = latlon_msg.altitude
        response = self._call_geodetic(self.from_ll_client, request, self.from_ll_service_name)
        if response is not None:
            p = response.map_point
            return np.array([p.x, p.y, p.z])
        return None

    def _convert_to_ll(self, position: np.ndarray) -> Optional[NavSatFix]:
        response = self._call_geodetic(
            self.to_ll_client, ToLL.Request(map_point=Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))),
            self.to_ll_service_name,
        )
        if response is None:
            return None
        point = response.ll_point
        result = NavSatFix(latitude=point.latitude, longitude=point.longitude, altitude=point.altitude)
        result.header.stamp = self.get_clock().now().to_msg()
        result.header.frame_id = self.world_frame_id
        return result

    def _build_path_msg(self) -> Optional[GeoPath]:
        """GeoPath of build_path() in lat/lon: poses[0] entry, poses[-1] exit, the rest pass-through points."""
        points = self.build_path()
        if points is None:
            self.get_logger().warning("No Safe Passage path: entry and exit buoys are not both classified")
            return None
        msg = GeoPath()
        msg.header.stamp = self.get_clock().now().to_msg()
        for point in points:
            latlon = self._convert_to_ll(point)
            if latlon is None:
                self.get_logger().error("No Safe Passage path: toLL conversion failed")
                return None
            pose = GeoPoseStamped()
            pose.header.stamp = msg.header.stamp
            pose.pose.position.latitude, pose.pose.position.longitude = latlon.latitude, latlon.longitude
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.get_logger().info(
            f"Safe Passage path: entry, {len(points) - 2} pass-through point(s), exit: "
            + json.dumps([[round(p.pose.position.latitude, 7), round(p.pose.position.longitude, 7)] for p in msg.poses])
        )
        return msg

    def _publish_path(self):
        msg = self._path_msg
        if msg is not None and not self.enable:
            msg.header.stamp = self.get_clock().now().to_msg()
            self.path_pub.publish(msg)

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
                res.message = f"Geodetic {self.from_ll_service_name} conversion failed."
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
            if ll:
                res.goal_latlon = ll

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
