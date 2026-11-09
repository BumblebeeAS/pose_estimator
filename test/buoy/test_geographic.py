"""Physical NED/ENU equivalence and a single geographic snapshot per freeze."""

from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest
import rclpy
from bb_robotx_msgs.srv import ExtractLocalGoal, GetEntry, GetExit, SetGateManagerState
from geographic_msgs.msg import GeoPointStamped
from px4_msgs.msg import VehicleGlobalPosition
from sensor_msgs.msg import NavSatFix
from std_srvs.srv import Trigger

from pose_estimator.buoy_gate_manager_node import BuoyGateManagerNode
from pose_estimator.buoy.tracking import BuoyTrack, DetectedBuoyColor
from test_manager import planner_with_tracks


@pytest.fixture
def manager():
    rclpy.init()
    node = BuoyGateManagerNode()
    try:
        yield node
    finally:
        node.destroy_node()
        rclpy.shutdown()


def reference(node, latitude=1.3, altitude=10.):
    message = GeoPointStamped()
    message.header.frame_id = 'odom_enu'
    message.header.stamp.sec = 42
    message.position.latitude = latitude
    message.position.longitude = 103.8
    message.position.altitude = altitude
    node._reference(message)
    return message


def correction(node, valid=True):
    message = VehicleGlobalPosition(timestamp=1, alt_valid=valid, alt=20., alt_ellipsoid=26.)
    node._global_position(message)


def add_track(node, state, position):
    track = BuoyTrack()
    track.published_state = state
    track.current_centroid = np.array(position, dtype=float)
    node.track_manager.tracks[track.track_id] = track


def swap_axes(position):
    return np.array([position[1], position[0], -position[2]])


@pytest.mark.parametrize('direction', [[1., 0., 0.], [0., 1., 0.], [-1., -2., 0.]])
def test_standoffs_and_paths_are_physically_equivalent(direction):
    forward = np.array(direction)
    side = np.array([-forward[1], forward[0], 0.])
    states = [
        (DetectedBuoyColor.FLASHING_BLUE, np.zeros(3)),
        (DetectedBuoyColor.SOLID_BLUE, 30 * forward),
        (DetectedBuoyColor.FLASHING_RED, 10 * forward - side),
        (DetectedBuoyColor.FLASHING_GREEN, 10 * forward + side),
        (DetectedBuoyColor.FLASHING_RED, 20 * forward),
        (DetectedBuoyColor.FLASHING_GREEN, 25 * forward + 10 * side),
    ]
    enu = planner_with_tracks(states)
    ned = planner_with_tracks([(state, swap_axes(p)) for state, p in states])
    enu.local_axes, ned.local_axes = 'enu', 'ned'
    enu.standoff_distance = ned.standoff_distance = 1.5
    np.testing.assert_allclose(ned.build_path(), [swap_axes(p) for p in enu.build_path()])
    for a, b in zip(enu._generate_candidates(np.zeros(3)), ned._generate_candidates(np.zeros(3))):
        assert a.goal_type == b.goal_type
        np.testing.assert_allclose(b.position, swap_axes(a.position), atol=1e-9)


@pytest.mark.parametrize('axes', ['ned', 'enu'])
@pytest.mark.parametrize('correction_first', [True, False])
def test_geographic_round_trip_and_axis_directions(manager, axes, correction_first):
    node = manager
    node.local_axes = axes
    if correction_first:
        correction(node)
    reference(node)
    if not correction_first:
        assert node._convert_to_ll(np.zeros(3)) is None
        correction(node)
    # 100 m north, 50 m east and 3 m up, with origin ellipsoid height 16 m.
    position = np.array([100., 50., -3.]) if axes == 'ned' else np.array([50., 100., 3.])
    fix = node._convert_to_ll(position)
    assert fix.latitude > 1.3 and fix.longitude > 103.8
    assert fix.altitude == pytest.approx(19., abs=.002)
    assert fix.status.status == fix.status.STATUS_FIX
    np.testing.assert_allclose(node._convert_from_ll(fix), position, atol=1e-6)
    assert node._convert_from_ll(NavSatFix(latitude=91., longitude=103.8)) is None
    assert node._convert_to_ll(np.array([float('nan'), 0., 0.])) is None
    correction(node, valid=False)
    assert node._convert_to_ll(position) is None


@pytest.mark.parametrize('invalid', ['frame', 'stamp', 'latitude'])
def test_invalid_reference_disables_conversion(manager, invalid):
    correction(manager)
    message = reference(manager)
    if invalid == 'frame':
        message.header.frame_id = 'home_ned'
    elif invalid == 'stamp':
        message.header.stamp.sec = 0
    else:
        message.position.latitude = float('nan')
    manager._reference(message)
    assert manager._convert_to_ll(np.zeros(3)) is None


@pytest.mark.parametrize('service,state,callback', [
    (GetEntry, DetectedBuoyColor.FLASHING_BLUE, '_cb_get_entry'),
    (GetExit, DetectedBuoyColor.SOLID_BLUE, '_cb_get_exit'),
])
def test_geographic_milestone_failure_is_explicit(manager, service, state, callback):
    add_track(manager, state, [10., 0., 0.])
    result = getattr(manager, callback)(service.Request(return_latlon=True), service.Response())
    assert not result.found and 'conversion failed' in result.message
    assert result.buoy_pose.pose.position.x == 10.
    local = getattr(manager, callback)(service.Request(return_latlon=False), service.Response())
    assert local.found


def test_geographic_goal_failures_are_explicit(manager):
    for geographic_input in (False, True):
        add_track(manager, DetectedBuoyColor.FLASHING_BLUE, [0., 0., 0.])
        add_track(manager, DetectedBuoyColor.SOLID_BLUE, [10., 0., 0.])
        request = ExtractLocalGoal.Request(input_is_latlon=geographic_input, return_latlon=True)
        result = manager._cb_extract_goal(request, ExtractLocalGoal.Response())
        assert not result.success and result.goal_type == 'INVALID'
        assert 'conversion failed' in result.message


@pytest.mark.parametrize('ready_at_freeze', [True, False])
def test_freeze_is_one_snapshot_without_retries(manager, ready_at_freeze):
    node = manager
    node.local_axes = 'ned'
    node.world_frame_id = 'odom_ned'
    node._on_fft_timer = Mock()  # Keep the deliberately seeded classifications.
    node.path_pub = Mock()
    add_track(node, DetectedBuoyColor.FLASHING_BLUE, [0., 0., -3.])
    add_track(node, DetectedBuoyColor.SOLID_BLUE, [30., 0., -3.])
    if ready_at_freeze:
        reference(node)
        correction(node)
    node.enable = True
    result = node._cb_set_state(SetGateManagerState.Request(enable=False), SetGateManagerState.Response())
    assert result.success and not node.enable
    snapshot = deepcopy(node._path_msg)
    if ready_at_freeze:
        assert len(snapshot.poses) == 2
        assert snapshot.poses[0].pose.position.altitude == pytest.approx(19.)
    else:
        assert snapshot is None and 'No geographic path' in result.message
    reference(node, latitude=1.4, altitude=100.)
    correction(node)
    node._cb_set_state(SetGateManagerState.Request(enable=False), SetGateManagerState.Response())
    node._publish_path()
    if ready_at_freeze:
        assert node._path_msg.poses == snapshot.poses
        node.path_pub.publish.assert_called_once()
    else:
        assert node._path_msg is None
        node.path_pub.publish.assert_not_called()
    node._cb_set_state(SetGateManagerState.Request(enable=True), SetGateManagerState.Response())
    node.path_pub.reset_mock()
    node._publish_path()
    assert node._path_msg is None
    node.path_pub.publish.assert_not_called()
    node._cb_set_state(SetGateManagerState.Request(enable=False), SetGateManagerState.Response())
    assert node._path_msg is not None  # A fresh collection/freeze can convert again.
    node._cb_reset(Trigger.Request(), Trigger.Response())
    node.path_pub.reset_mock()
    node._publish_path()
    assert node._path_msg is None
    node.path_pub.publish.assert_not_called()


def test_uav_parameters_load_as_ned():
    # Exercise actual ROS parameter overrides rather than assigning attributes.
    from pathlib import Path
    import yaml
    path = Path('/workspaces/isaac_ros-dev/src/vision_pipeline/config/uav/buoy.yaml')
    if not path.exists():
        path = Path(__file__).resolve().parents[3] / 'vision_pipeline/config/uav/buoy.yaml'
    parameters = yaml.safe_load(path.read_text())['/**/buoy_gate_manager_node']['ros__parameters']
    args = ['--ros-args']
    for name, value in parameters.items():
        args += ['-p', f'{name}:={yaml.safe_dump(value, default_flow_style=True).splitlines()[0]}']
    rclpy.init(args=args)
    node = None
    try:
        node = BuoyGateManagerNode()
        assert node.world_frame_id == 'odom_ned' and node.local_axes == 'ned'
        assert node.base_link_frame_id == 'uav/base_link_frd'
        assert node.path_pub.topic_name == '/uav/buoy/safe_passage/path'
        assert node.path_timer.callback_group is node.srv_reset.callback_group
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_invalid_axes_rejected():
    rclpy.init(args=['--ros-args', '-p', 'local_axes:=invalid'])
    try:
        with pytest.raises(ValueError, match='local_axes'):
            BuoyGateManagerNode()
    finally:
        rclpy.shutdown()
