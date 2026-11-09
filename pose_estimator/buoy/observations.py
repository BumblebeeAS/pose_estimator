"""Bounded waiting for transforms at buoy observation timestamps."""

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from bb_robotx_msgs.msg import BuoyDetection
from tf2_ros import TransformException

COLOR_NAMES = {
    BuoyDetection.BLACK: 'black',
    BuoyDetection.RED: 'red',
    BuoyDetection.GREEN: 'green',
    BuoyDetection.BLUE: 'blue',
    255: None,  # Unknown position observations carry no color evidence.
}


@dataclass
class PendingObservation:
    stamp_ns: int
    frame: str
    position: np.ndarray
    color: Optional[str]
    deadline: float


class ObservationBuffer:
    """Use monotonic deadlines; never substitute arrival time or latest TF."""

    def __init__(self, world_frame, transform_position, wait_timeout=0.25, max_pending=1000):
        if not world_frame or not math.isfinite(wait_timeout) or wait_timeout <= 0 or max_pending <= 0:
            raise ValueError("Expected a world frame and positive TF wait and queue bounds")
        self.world_frame = world_frame
        self.transform_position = transform_position
        self.wait_timeout = wait_timeout
        self.max_pending = max_pending
        self.pending = deque()
        self.dropped = 0

    def clear(self):
        self.pending.clear()

    def add(self, message, received_at):
        for detection in message.detections:
            if detection.color not in COLOR_NAMES:
                self.dropped += 1
                continue
            header = detection.pose.header
            frame = header.frame_id or message.header.frame_id
            stamp = header.stamp if header.stamp.sec or header.stamp.nanosec else message.header.stamp
            stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
            p = detection.pose.pose.position
            position = np.array([p.x, p.y, p.z])
            if not frame or stamp_ns <= 0 or not np.isfinite(position).all():
                self.dropped += 1
                continue
            if len(self.pending) >= self.max_pending:
                self.pending.popleft()
                self.dropped += 1
            self.pending.append(PendingObservation(
                stamp_ns, frame, position, COLOR_NAMES[detection.color], received_at + self.wait_timeout
            ))

    def drain(self, now):
        ready = []
        for _ in range(len(self.pending)):
            observation = self.pending.popleft()
            if now >= observation.deadline:
                self.dropped += 1
                continue
            try:
                position = (observation.position.copy() if observation.frame == self.world_frame
                            else self.transform_position(observation))
            except TransformException:
                self.pending.append(observation)
                continue
            if not np.isfinite(position).all():
                self.dropped += 1
                continue
            ready.append((observation.stamp_ns / 1e9, observation.color, position))
        return sorted(ready, key=lambda item: item[0])
