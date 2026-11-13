"""The manager's Safe Passage report, exercised without constructing the node."""
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from bb_robotx_msgs.action import SendSafePassageReport

from pose_estimator.buoy_gate_manager_node import BuoyGateManagerNode
from pose_estimator.buoy.tracking import DetectedBuoyColor


def bare(states, convert=None):
    node = BuoyGateManagerNode.__new__(BuoyGateManagerNode)
    node.track_manager = SimpleNamespace(tracks={
        i: SimpleNamespace(track_id=i, published_state=s, current_centroid=np.array(p, float))
        for i, (s, p) in enumerate(states)})
    node._convert_to_ll = convert or (lambda p: SimpleNamespace(latitude=1.0 + p[0] * 1e-5,
                                                                 longitude=103.0 + p[1] * 1e-5))
    logger = Mock()
    node.get_logger = lambda: logger
    node.send_report, node.report_retry_period, node.report_timeout = True, 0.0, 30.0
    node.report_action = "/uav/report/safe_passage"
    node._report_in_flight, node._report_attempt, node._report_handle = False, 0, None
    node._report_deadline, node._report_next_try, node.enable = 0.0, 0.0, False
    node.report_pub = Mock()
    node.report_client = Mock()
    node.report_client.server_is_ready.return_value = True
    return node


class Bridge:
    """Records goals; answer(i, ...) plays the bridge's reply to goal i."""

    def __init__(self, node):
        self.sent = []
        node.report_client.send_goal_async.side_effect = self.send

    def send(self, goal):
        future, callbacks = Mock(), []
        future.add_done_callback.side_effect = callbacks.append
        self.sent.append((goal, callbacks))
        return future

    def answer(self, index, accepted=True, error=None):
        _, callbacks = self.sent[index]
        result_future, result_callbacks = Mock(), []
        result_future.add_done_callback.side_effect = result_callbacks.append
        handle = Mock(accepted=accepted)
        handle.get_result_async.return_value = result_future
        callbacks[0](SimpleNamespace(result=lambda: handle))
        if accepted and error is not None:
            result = SendSafePassageReport.Result(seq=7, error=error)
            result_callbacks[0](SimpleNamespace(result=lambda: SimpleNamespace(result=result)))
        return handle


def test_failed_conversion_leaves_only_that_track_out():
    node = bare([(DetectedBuoyColor.FLASHING_RED, [0, 0, 0]), (DetectedBuoyColor.FLASHING_GREEN, [5, 0, 0])],
                convert=lambda p: None if p[0] > 1 else SimpleNamespace(latitude=1.0, longitude=103.0))
    assert [b.state for b in node._build_report_msg().buoys] == ["flashing_red"]


def test_unanswered_attempt_times_out_is_canceled_and_retried():
    node = bare([(DetectedBuoyColor.FLASHING_RED, [0, 0, 0])])
    bridge = Bridge(node)
    node._set_report(node._build_report_msg())
    node._publish_report()
    handle = bridge.answer(0)  # accepted, never answered
    node._report_deadline = time.monotonic() - 1.0
    node._publish_report()
    handle.cancel_goal_async.assert_called_once()
    assert len(bridge.sent) == 2  # retried on the same tick (retry period 0)
    bridge.answer(1, error="")
    assert not node._report_unsent


def test_new_report_abandons_the_old_attempt_and_collecting_stops_serving():
    node = bare([(DetectedBuoyColor.FLASHING_RED, [0, 0, 0])])
    bridge = Bridge(node)
    node._set_report(node._build_report_msg())
    node._publish_report()
    handle = bridge.answer(0)
    late_result = handle.get_result_async.return_value.add_done_callback.call_args.args[0]
    node._set_report(node._build_report_msg())  # refrozen
    handle.cancel_goal_async.assert_called_once()
    node._publish_report()
    assert len(bridge.sent) == 2
    late_result(SimpleNamespace(result=lambda: SimpleNamespace(
        result=SendSafePassageReport.Result(seq=7, error=""))))
    assert node._report_unsent and node._report_in_flight  # Old success cannot finish the new send.
    node._set_report(None)
    node.enable = True
    node._publish_report()
    assert node.report_pub.publish.call_count == 2
