"""Bounded, asynchronous delivery of immutable Safe Passage reports."""

from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
import time
from typing import Any

from action_msgs.msg import GoalStatus
from bb_robotx_msgs.action import SendSafePassageReport
from bb_robotx_msgs.msg import SafePassageReport
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import Clock, ClockType


@dataclass
class ReportAttempt:
    report: SafePassageReport
    deadline: float
    phase: str = "discovery"
    future: Any = None
    handle: Any = None


class FrozenReportSender:
    """One attempt per submission; no blocking waits or automatic retries."""

    def __init__(self, node, action_name, timeout):
        self.logger = node.get_logger()
        self.timeout = timeout
        self._lock = Lock()
        self._attempts = []
        group = MutuallyExclusiveCallbackGroup()
        self.client = ActionClient(node, SendSafePassageReport, action_name, callback_group=group)
        self.timer = node.create_timer(
            0.1, self._tick, callback_group=group,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )

    def submit(self, report):
        with self._lock:
            self._attempts.append(ReportAttempt(deepcopy(report), time.monotonic() + self.timeout))
        self.logger.info("SAFE_PASSAGE_REPORT_QUEUED")

    def _cancel_late_goal(self, future):
        """An acceptance arriving after our deadline must not leave an orphan goal."""
        try:
            handle = future.result()
            if handle.accepted:
                handle.cancel_goal_async()
        except Exception as error:
            self.logger.error(f"Safe Passage report cancellation failed: {error}")

    def _advance(self, attempt):
        try:
            if time.monotonic() >= attempt.deadline:
                self.logger.error(f"SAFE_PASSAGE_REPORT_FAILED: timeout during {attempt.phase}")
                if attempt.handle is not None:
                    attempt.handle.cancel_goal_async()
                elif attempt.future is not None:
                    attempt.future.add_done_callback(self._cancel_late_goal)
                return False
            if attempt.phase == "discovery":
                if not self.client.server_is_ready():
                    return True
                attempt.future = self.client.send_goal_async(
                    SendSafePassageReport.Goal(report=attempt.report))
                attempt.phase = "acceptance"
                return True
            if not attempt.future.done():
                return True
            if attempt.phase == "acceptance":
                attempt.handle = attempt.future.result()
                if not attempt.handle.accepted:
                    self.logger.error("SAFE_PASSAGE_REPORT_FAILED: goal rejected")
                    return False
                self.logger.info("SAFE_PASSAGE_REPORT_ACCEPTED")
                attempt.future = attempt.handle.get_result_async()
                attempt.phase = "result"
                return True
            response = attempt.future.result()
            if response.status == GoalStatus.STATUS_SUCCEEDED and not response.result.error:
                self.logger.info(f"SAFE_PASSAGE_REPORT_SENT seq={response.result.seq}")
            else:
                self.logger.error(
                    f"SAFE_PASSAGE_REPORT_FAILED: status={response.status}, error={response.result.error}")
            return False
        except Exception as error:
            self.logger.error(f"SAFE_PASSAGE_REPORT_FAILED during {attempt.phase}: {error}")
            return False

    def _tick(self):
        with self._lock:
            self._attempts = [attempt for attempt in self._attempts if self._advance(attempt)]
