"""Marker rendering for buoy tracks and gate candidates."""

from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from pose_estimator.utils.buoy_tracking import DetectedBuoyColor
from pose_estimator.utils.buoy_gate_planning import (
    GOAL_TYPE_GATE_PAIR, GOAL_TYPE_SINGULAR_BUOY, GOAL_TYPE_EXIT_BUOY,
)

TRACK_COLOR_MAP = {
    DetectedBuoyColor.FLASHING_RED: ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8),
    DetectedBuoyColor.FLASHING_GREEN: ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8),
    DetectedBuoyColor.FLASHING_BLUE: ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.9),
    DetectedBuoyColor.SOLID_BLUE: ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.9),
    DetectedBuoyColor.UNKNOWN: ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.5),
}


CANDIDATE_COLOR_MAP = {
    GOAL_TYPE_GATE_PAIR: ColorRGBA(r=1.0, g=0.9, b=0.1, a=0.9),
    GOAL_TYPE_SINGULAR_BUOY: ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.9),
    GOAL_TYPE_EXIT_BUOY: ColorRGBA(r=0.0, g=0.9, b=0.9, a=0.9),
}


REJECTED_CANDIDATE_COLOR = ColorRGBA(r=0.4, g=0.4, b=0.4, a=0.35)


SELECTED_GOAL_COLOR = ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.6)


class GateVisualization:
    def _new_marker(self, ns: str, marker_id: int, marker_type: int, stamp) -> Marker:
        """Build an ADD marker in world_frame_id with an identity pose."""
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def _get_visualisation_query_pos(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Reference position the candidate visualisation is generated from.

        Prefers the vehicle position of the most recent extract-goal call, together
        with the one before it, so the markers reproduce that call's reasoning exactly.
        Before any call has arrived, falls back to the drone's own position so the
        candidate set is still observable during overfly.

        Returns:
            The query position and preceding query position (None when unavailable).
        """
        if self.last_query_pos is not None:
            return self.last_query_pos, self.prev_query_pos

        try:
            base_transform = self.tf_buffer.lookup_transform(
                self.world_frame_id,
                self.base_link_frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1),
            )
        except Exception as e:
            self.get_logger().warning(
                f"No query pose for candidate visualisation, and TF lookup of "
                f"'{self.base_link_frame_id}' failed: {e}",
                throttle_duration_sec=5.0,
            )
            return None, None

        t = base_transform.transform.translation
        return np.array([t.x, t.y, t.z]), None

    def _shape(self, ns, marker_id, stamp, position, marker_type, scale, color):
        marker = self._new_marker(ns, marker_id, marker_type, stamp)
        marker.pose.position = Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))
        marker.scale = Vector3(x=scale[0], y=scale[1], z=scale[2])
        marker.color = color
        return marker

    def _text(self, ns, marker_id, stamp, position, text, height, offset, color):
        position = np.asarray(position) + [0.0, 0.0, offset]
        marker = self._shape(ns, marker_id, stamp, position, Marker.TEXT_VIEW_FACING,
                             (0.0, 0.0, height), color)
        marker.text = text
        return marker

    def _line(self, ns, marker_id, stamp, positions, width, color):
        marker = self._new_marker(ns, marker_id, Marker.LINE_LIST, stamp)
        marker.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in positions]
        marker.scale = Vector3(x=width)
        marker.color = color
        return marker

    def _append_candidate_markers(self, marker_array: MarkerArray, now_msg) -> None:
        query_pos, previous_pos = self._get_visualisation_query_pos()
        if query_pos is None:
            return
        _, _, candidates = self._evaluate_candidates(query_pos, previous_pos)
        best = self._select_best_candidate(candidates)
        markers = marker_array.markers
        for idx, cand in enumerate(candidates):
            color = CANDIDATE_COLOR_MAP[cand.goal_type] if cand.is_valid else REJECTED_CANDIDATE_COLOR
            markers.append(self._shape(
                "gate_candidates", idx, now_msg, cand.position, Marker.CYLINDER, (0.25, 0.25, 0.05), color,
            ))
            rejected = "" if cand.is_valid else f"\n{cand.reject_reason}"
            width = "" if cand.gate_width is None else f" | gate {cand.gate_width:.2f}m"
            text = (f"{cand.goal_type} {cand.distance:.1f}m{width}{rejected}"
                    f"\ntracks: {', '.join(str(t) for t in cand.track_ids)}")
            markers.append(self._text(
                "gate_candidate_labels", idx, now_msg, cand.position, text, 0.15, 0.2, color,
            ))
            if cand.goal_type == GOAL_TYPE_GATE_PAIR and len(cand.track_ids) == 2:
                endpoints = [self.track_manager.tracks[tid].current_centroid for tid in cand.track_ids
                             if tid in self.track_manager.tracks
                             and self.track_manager.tracks[tid].current_centroid is not None]
                if len(endpoints) == 2:
                    markers.append(self._line("gate_pair_links", idx, now_msg, endpoints, 0.02, color))
        if best is not None:
            markers.append(self._shape(
                "gate_selected_goal", 0, now_msg, best.position, Marker.SPHERE,
                (0.35, 0.35, 0.35), SELECTED_GOAL_COLOR,
            ))

    def _publish_markers(self):
        marker_array = MarkerArray()
        now_msg = self.get_clock().now().to_msg()
        # An empty namespace clears old labels and candidates as well as track spheres.
        delete_marker = self._new_marker("", 0, Marker.SPHERE, now_msg)
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        for tid, track in self.track_manager.tracks.items():
            if track.current_centroid is None:
                continue
            color = TRACK_COLOR_MAP.get(track.published_state, TRACK_COLOR_MAP[DetectedBuoyColor.UNKNOWN])
            marker_array.markers.append(self._shape(
                "buoy_tracks", tid, now_msg, track.current_centroid, Marker.SPHERE, (0.2, 0.2, 0.2), color,
            ))
            marker_array.markers.append(self._text(
                "buoy_labels", tid, now_msg, track.current_centroid,
                f"ID: {tid}\n{track.published_state.name}", 0.15, 0.25,
                ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0),
            ))
        if self.visualize_candidates:
            self._append_candidate_markers(marker_array, now_msg)
        self.marker_pub.publish(marker_array)
