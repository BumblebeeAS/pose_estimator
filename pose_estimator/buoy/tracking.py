"""Spatial buoy tracks and temporal beacon classification, independent of ROS."""

from bisect import bisect_left
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from sklearn.cluster import DBSCAN


class DetectedBuoyColor(Enum):
    BLACK = 0
    FLASHING_RED = 1
    FLASHING_GREEN = 2
    FLASHING_BLUE = 3
    SOLID_BLUE = 4
    UNKNOWN = 5


@dataclass
class Observation:
    timestamp: float
    position: np.ndarray
    color: Optional[str]


class BuoyTrack:
    _id_counter = 0

    def __init__(self):
        self.track_id: int = BuoyTrack._id_counter
        BuoyTrack._id_counter += 1
        self.observations = deque()
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
        target_freq: float = 1.0,
        target_freq_tol: float = 0.2,
        duty_cycle_min: float = 0.1,
        duty_cycle_max: float = 0.85,
        consistent_threshold: float = 0.6,
        fft_min_magnitude: float = 0.2,
        virtual_sampling_rate: float = 10.0,
        logger=None,
        max_color_gap: float = 0.5,
        min_flash_cycles: float = 2.0,
    ):
        self.max_association_radius = max_association_radius
        self.dbscan_min_samples = dbscan_min_samples
        self.track_timeout = track_timeout
        self.temporal_window = 30.0
        self.required_density = required_density
        self.target_freq = target_freq
        self.target_freq_tol = target_freq_tol
        self.duty_cycle_min = duty_cycle_min
        self.duty_cycle_max = duty_cycle_max
        self.consistent_threshold = consistent_threshold
        self.fft_min_magnitude = fft_min_magnitude
        self.virtual_sampling_rate = virtual_sampling_rate
        self.logger = logger
        if not all(np.isfinite(v) and v > 0 for v in
                   (target_freq, virtual_sampling_rate, max_color_gap, min_flash_cycles)):
            raise ValueError("FFT frequency, sampling rate, gap and cycle bounds must be positive")
        self.max_color_gap = max_color_gap
        self.min_flash_cycles = min_flash_cycles

        wn = min(0.99, max(0.01, 2.0 * butterworth_cutoff_freq / virtual_sampling_rate))
        self.b, self.a = signal.butter(N=butterworth_order, Wn=wn, btype="low")
        self.min_filter_samples = 3 * max(len(self.a), len(self.b)) + 1

        self.tracks: Dict[int, BuoyTrack] = {}
        self.detection_buffer: deque = deque()

    def clear_all_tracks(self):
        self.tracks.clear()
        self.detection_buffer.clear()
        BuoyTrack._id_counter = 0

    def associate_and_update(self, current_time: float):
        """Match established tracks first; accumulate unmatched points for new tracks."""
        known = [t for t in self.tracks.values() if t.current_centroid is not None]
        unmatched = []
        while self.detection_buffer:
            timestamp, color, position = self.detection_buffer.popleft()
            if self.temporal_window > 0 and timestamp < current_time - self.temporal_window:
                continue
            track = min(known, key=lambda t: np.linalg.norm(
                position[:2] - t.current_centroid[:2]), default=None)
            if track is not None and np.linalg.norm(
                    position[:2] - track.current_centroid[:2]) <= self.max_association_radius:
                self._add_observation_to_track(track.track_id, color, position, timestamp)
            else:
                unmatched.append((timestamp, color, position))
        if not unmatched:
            return
        # Match on XY while keeping the measured height in every observation.
        positions = np.array([p[:2] for _, _, p in unmatched])
        labels = DBSCAN(eps=self.max_association_radius,
                        min_samples=self.dbscan_min_samples).fit_predict(positions)
        for label in set(labels) - {-1}:
            track = BuoyTrack()
            self.tracks[track.track_id] = track
            for index in np.flatnonzero(labels == label):
                timestamp, color, position = unmatched[index]
                self._add_observation_to_track(track.track_id, color, position, timestamp)
        # Sparse batches can establish a track over several timer periods.
        self.detection_buffer.extend(o for o, label in zip(unmatched, labels) if label == -1)

    def _add_observation_to_track(
        self, track_id: int, color: Optional[str], position: np.ndarray, timestamp: float
    ):
        track = self.tracks.get(track_id)
        if not track:
            return

        index = bisect_left([o.timestamp for o in track.observations], timestamp)
        if index < len(track.observations) and track.observations[index].timestamp == timestamp:
            # A later delivery may add color evidence to an unknown observation.
            if track.observations[index].color is None and color is not None:
                track.observations[index].color = color
            return
        track.observations.insert(index, Observation(timestamp, position, color))
        track.last_observation_time = max(track.last_observation_time, timestamp)

    def prune_temporal_window(self, current_time: float):
        if self.temporal_window <= 0.0:
            return

        cutoff = current_time - self.temporal_window
        for track in self.tracks.values():
            while track.observations and track.observations[0].timestamp < cutoff:
                track.observations.popleft()

    def recompute_centroids(self):
        """Compute smoothed centroid for all active tracks from their temporal pose history."""
        for track in self.tracks.values():
            if track.observations:
                coords = np.array([o.position for o in track.observations])
                track.current_centroid = np.mean(coords, axis=0)

    def prune_timeouts(self, current_time: float):
        """Remove expired tracks, retaining the entry and exit singletons."""
        expired = [
            tid for tid, track in self.tracks.items()
            if not track.is_immortal and (
                not track.observations
                or (self.track_timeout > 0
                    and current_time - track.last_observation_time > self.track_timeout)
            )
        ]
        for tid in expired:
            del self.tracks[tid]
        if expired and self.logger:
            self.logger.info(
                f"Pruned timed-out tracks {expired}; {len(self.tracks)} remain.",
                throttle_duration_sec=2.0,
            )

    @staticmethod
    def _band_power(magnitudes: np.ndarray, band_mask: np.ndarray) -> float:
        """Combine target-band energy so flashes between FFT bins retain their strength."""
        return float(np.sqrt(np.square(magnitudes[band_mask]).sum()))

    def classify_tracks(self):
        """Count off versus color, vote for the hue, and time only blue's light state."""
        flashing_states = {
            'red': DetectedBuoyColor.FLASHING_RED,
            'green': DetectedBuoyColor.FLASHING_GREEN,
            'blue': DetectedBuoyColor.FLASHING_BLUE,
        }
        for track in self.tracks.values():
            known = [o for o in track.observations if o.color is not None]
            if not known:
                continue  # Unknown observations carry position, not color evidence.
            votes = Counter(o.color for o in known if o.color in flashing_states)
            # Off is evidence against an active light, not a competing hue.
            # Reuse the configured minimum on-fraction (10% on the UAV).
            if sum(votes.values()) / len(known) <= self.duty_cycle_min:
                track.published_state = DetectedBuoyColor.BLACK
                continue
            color = max(votes, key=votes.get)
            if sum(count == votes[color] for count in votes.values()) > 1:
                continue  # A tied vote retains the previous state.
            if color in ('red', 'green'):
                # Each red/green beacon flashes only its own color. Timing is
                # needed for blue entry/exit, not to identify this color.
                track.published_state = flashing_states[color]
                continue
            times = np.array([o.timestamp for o in known])
            # Keep the original time axis: never bridge an unseen interval or splice
            # separate sightings into a fictitious continuous flashing signal.
            segments = np.split(np.arange(len(known)),
                                np.flatnonzero(np.diff(times) > self.max_color_gap) + 1)
            minimum_duration = self.min_flash_cycles / self.target_freq
            evidence = {color: np.zeros(2) for color in flashing_states}
            total_duration = 0.0
            for indices in segments:
                segment_times = times[indices]
                duration = segment_times[-1] - segment_times[0]
                if duration < minimum_duration:
                    continue
                uniform = np.arange(segment_times[0], segment_times[-1],
                                    1.0 / self.virtual_sampling_rate)
                if len(uniform) < self.min_filter_samples:
                    continue
                frequencies = rfftfreq(len(uniform), d=1.0 / self.virtual_sampling_rate)
                bin_width = frequencies[1] - frequencies[0]
                half_width = max(self.target_freq_tol, bin_width) + 0.5 * bin_width
                band = np.abs(frequencies - self.target_freq) <= half_width
                # Average near-simultaneous camera observations before resampling.
                # This avoids treating their arrival order as on/off transitions.
                bins = np.floor((segment_times - segment_times[0])
                                * self.virtual_sampling_rate).astype(int)
                occupied, inverse = np.unique(bins, return_inverse=True)
                counts = np.bincount(inverse)
                for candidate in flashing_states:
                    values = [float(known[i].color == candidate) for i in indices]
                    means = np.bincount(inverse, weights=values) / counts
                    samples = np.interp(uniform, segment_times[0] + occupied
                                        / self.virtual_sampling_rate, means)
                    samples = signal.filtfilt(self.b, self.a, samples)
                    spectrum = np.abs(rfft(samples)) / len(uniform)
                    power = self._band_power(spectrum, band)
                    # Duration-weighted energy and presence, independent of phase
                    # between sightings. Short sightings contribute neither.
                    evidence[candidate] += duration * np.array([power ** 2, spectrum[0]])
                total_duration += duration
            if not total_duration:
                continue  # Retain the last state until a continuous sighting qualifies.
            evidence = {color: (np.sqrt(value[0] / total_duration), value[1] / total_duration)
                        for color, value in evidence.items()}
            power, presence = evidence[color]
            if power > self.fft_min_magnitude and self.duty_cycle_min < presence < self.duty_cycle_max:
                track.published_state = flashing_states[color]
            else:
                blue = evidence['blue'][1]
                if color == 'blue' and blue > self.consistent_threshold:
                    track.published_state = DetectedBuoyColor.SOLID_BLUE
                else:
                    track.published_state = DetectedBuoyColor.UNKNOWN

    def deduplicate_singletons(self):
        """Merge entry/exit duplicates into the oldest track, preserving source time."""
        for state in (DetectedBuoyColor.FLASHING_BLUE, DetectedBuoyColor.SOLID_BLUE):
            matches = sorted(
                (track for track in self.tracks.values() if track.published_state == state),
                key=lambda track: track.track_id,
            )
            if len(matches) < 2:
                continue
            primary, *duplicates = matches
            for duplicate in duplicates:
                for observation in duplicate.observations:
                    self._add_observation_to_track(
                        primary.track_id, observation.color,
                        observation.position, observation.timestamp,
                    )
                del self.tracks[duplicate.track_id]
                if self.logger:
                    self.logger.info(
                        f"Merged {state.name} track {duplicate.track_id} into {primary.track_id}.",
                        throttle_duration_sec=2.0,
                    )
