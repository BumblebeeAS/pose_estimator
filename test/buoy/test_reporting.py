"""Freeze validation and one-shot action delivery, without an external broker."""

from concurrent.futures import Future
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

from action_msgs.msg import GoalStatus
from bb_robotx_msgs.action import SendSafePassageReport
from bb_robotx_msgs.msg import SafePassageReport
from bb_robotx_msgs.srv import SetGateManagerState
import numpy as np
import pytest
from std_srvs.srv import Trigger

from pose_estimator.buoy import reporting
from pose_estimator.buoy.reporting import FrozenReportSender
from pose_estimator.buoy.tracking import DetectedBuoyColor
from test_geographic import manager










def finished(value):
    future = Future()
    future.set_result(value)
    return future


@pytest.fixture
def sender(manager, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(reporting, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    sender = FrozenReportSender(manager, '/test_report', 1.)
    sender.timer.cancel()
    sender.client = Mock()
    sender.client.server_is_ready.return_value = True
    sender.logger = Mock()
    sender.test_clock = clock
    return sender


@pytest.mark.parametrize('status,error,success', [
    (GoalStatus.STATUS_SUCCEEDED, '', True),
    (GoalStatus.STATUS_SUCCEEDED, 'broker error', False),
    (GoalStatus.STATUS_ABORTED, 'rejected by OCS', False),
    (GoalStatus.STATUS_CANCELED, '', False),
])
def test_action_results_and_immutable_payload(sender, status, error, success):
    result = SimpleNamespace(status=status, result=SendSafePassageReport.Result(seq=42, error=error))
    handle = Mock(accepted=True)
    handle.get_result_async.return_value = finished(result)
    sender.client.send_goal_async.return_value = finished(handle)
    report = SafePassageReport()
    report.entry_position.latitude = 1.3
    sender.submit(report)
    report.entry_position.latitude = 9.
    for _ in range(4):
        sender._tick()
    sender.client.send_goal_async.assert_called_once()
    assert sender.client.send_goal_async.call_args.args[0].report.entry_position.latitude == 1.3
    assert not sender._attempts
    if success:
        assert any('SENT seq=42' in c.args[0] for c in sender.logger.info.call_args_list)
        sender.logger.error.assert_not_called()
    else:
        sender.logger.error.assert_called_once()


@pytest.mark.parametrize('phase', ['discovery', 'acceptance', 'result'])
def test_timeout_is_bounded_without_retry_and_cancels_late_acceptance(sender, phase):
    acceptance = Future()
    handle = Mock(accepted=True)
    handle.get_result_async.return_value = Future()
    sender.client.server_is_ready.return_value = phase != 'discovery'
    sender.client.send_goal_async.return_value = acceptance
    sender.submit(SafePassageReport())
    sender._tick()
    if phase == 'result':
        acceptance.set_result(handle)
        sender._tick()
    sender.test_clock[0] = 2.
    sender._tick()
    sender._tick()
    assert not sender._attempts
    sender.logger.error.assert_called_once()
    if phase == 'discovery':
        sender.client.send_goal_async.assert_not_called()
    else:
        sender.client.send_goal_async.assert_called_once()
        if phase == 'acceptance':
            acceptance.set_result(handle)
        handle.cancel_goal_async.assert_called_once()
