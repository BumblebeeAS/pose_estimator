"""Gate candidate generation and selection shared by services and visualization."""

import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion
from tf2_ros import TransformException

from pose_estimator.utils.buoy_tracking import DetectedBuoyColor

GOAL_TYPE_GATE_PAIR = "GATE_PAIR"
GOAL_TYPE_SINGULAR_BUOY = "SINGULAR_BUOY"
GOAL_TYPE_EXIT_BUOY = "EXIT_BUOY"
GOAL_TYPE_INVALID = "INVALID"


def _get_yaw_from_quaternion(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class GoalCandidate:
    """A waypoint proposal shared by goal extraction and marker rendering."""

    goal_type: str
    position: np.ndarray
    track_ids: Tuple[int, ...] = ()
    distance: float = float("inf")
    is_valid: bool = False
    reject_reason: str = "UNEVALUATED"
    gate_width: Optional[float] = None


class GatePlanning:
    def _get_milestone_centroids(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return the (entry, exit) centroids, i.e. FLASHING_BLUE and SOLID_BLUE."""
        entry_pos, exit_pos = None, None
        for track in self.track_manager.tracks.values():
            if track.current_centroid is None:
                continue
            if track.published_state == DetectedBuoyColor.FLASHING_BLUE:
                entry_pos = track.current_centroid
            elif track.published_state == DetectedBuoyColor.SOLID_BLUE:
                exit_pos = track.current_centroid
        return entry_pos, exit_pos

    def _compute_forward_vector(
        self, vehicle_pos: np.ndarray, previous_pos: Optional[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], str]:
        """Return the XY unit heading and source: course axis, query motion, or drone TF."""
        entry_pos, exit_pos = self._get_milestone_centroids()
        course_axis = (exit_pos[:2] - entry_pos[:2]
                       if entry_pos is not None and exit_pos is not None else None)
        trajectory = vehicle_pos[:2] - previous_pos[:2] if previous_pos is not None else None
        for direction, source in ((course_axis, "COURSE_AXIS"), (trajectory, "VEHICLE_TRAJECTORY")):
            if direction is not None:
                length = np.linalg.norm(direction)
                if length > 0.1:
                    return direction / length, source

        try:
            base_transform = self.tf_buffer.lookup_transform(
                self.world_frame_id,
                self.base_link_frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1),
            )
            drone_yaw = _get_yaw_from_quaternion(base_transform.transform.rotation)
        except TransformException as e:
            self.get_logger().warning(
                f"Failed to lookup drone heading from '{self.base_link_frame_id}': {e}",
                throttle_duration_sec=2.0,
            )
            return None, "UNAVAILABLE"
        return np.array([math.cos(drone_yaw), math.sin(drone_yaw)]), "DRONE_HEADING"

    def _generate_candidates(self, vehicle_pos: np.ndarray) -> List[GoalCandidate]:
        """Pair nearby red/green tracks, offset unpaired buoys, and include the exit."""
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

        candidates: List[GoalCandidate] = []
        paired_red: Set[int] = set()
        paired_green: Set[int] = set()

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
                candidates.append(
                    GoalCandidate(
                        goal_type=GOAL_TYPE_GATE_PAIR,
                        position=midpoint,
                        track_ids=(r_id, g_id),
                        gate_width=float(dist),
                    )
                )

        # RED and GREEN use opposite perpendicular standoffs.
        for tracks, paired, side in ((red_tracks, paired_red, 1), (green_tracks, paired_green, -1)):
            for track in tracks:
                if track.track_id in paired:
                    continue
                delta = track.current_centroid[:2] - vehicle_pos[:2]
                angle = math.atan2(delta[1], delta[0]) + side * math.pi / 2
                position = track.current_centroid.copy()
                position[:2] += self.vehicle_encasing_radius * np.array([math.cos(angle), math.sin(angle)])
                candidates.append(GoalCandidate(
                    goal_type=GOAL_TYPE_SINGULAR_BUOY, position=position, track_ids=(track.track_id,),
                ))

        for tid, t in self.track_manager.tracks.items():
            if (
                t.published_state == DetectedBuoyColor.SOLID_BLUE
                and t.current_centroid is not None
            ):
                candidates.append(
                    GoalCandidate(
                        goal_type=GOAL_TYPE_EXIT_BUOY,
                        position=t.current_centroid,
                        track_ids=(tid,),
                    )
                )

        return candidates

    def _filter_candidates(
        self,
        candidates: List[GoalCandidate],
        vehicle_pos: np.ndarray,
        v_forward: Optional[np.ndarray],
    ) -> None:
        """Mark candidates invalid when heading is missing, reached, or behind the vehicle."""
        for cand in candidates:
            d_vec = cand.position[:2] - vehicle_pos[:2]
            cand.distance = float(np.linalg.norm(d_vec))
            if v_forward is None:
                cand.is_valid = False
                cand.reject_reason = "HEADING_UNAVAILABLE"
            elif cand.distance < self.waypoint_pass_tol:
                cand.is_valid = False
                cand.reject_reason = "WITHIN_PASS_TOL"
            elif np.dot(d_vec, v_forward) <= 0.0:
                cand.is_valid = False
                cand.reject_reason = "BEHIND_VEHICLE"
            else:
                cand.is_valid = True
                cand.reject_reason = ""

    def _evaluate_candidates(
        self, vehicle_pos: np.ndarray, previous_pos: Optional[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], str, List[GoalCandidate]]:
        """Return heading, source, and candidates sorted valid-first then nearest-first."""
        v_forward, forward_source = self._compute_forward_vector(
            vehicle_pos, previous_pos
        )
        candidates = self._generate_candidates(vehicle_pos)
        self._filter_candidates(candidates, vehicle_pos, v_forward)
        candidates.sort(key=lambda c: (not c.is_valid, c.distance))
        return v_forward, forward_source, candidates

    @staticmethod
    def _select_best_candidate(
        candidates: List[GoalCandidate],
    ) -> Optional[GoalCandidate]:
        """Return the first valid candidate from the sorted list."""
        return next((candidate for candidate in candidates if candidate.is_valid), None)
