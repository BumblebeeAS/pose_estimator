"""Observation-time ingestion, missing color evidence, and manager behavior."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from bb_robotx_msgs.msg import BuoyDetection, BuoyDetectionArray
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from tf2_ros import TransformException

from pose_estimator.buoy.observations import ObservationBuffer
from pose_estimator.buoy.tracking import BuoyTrack, DetectedBuoyColor, TrackManager


def message(color=BuoyDetection.RED, stamp=42, frame="camera", position=(1., 2., 3.)):
    header = Header(frame_id=frame)
    header.stamp.sec = stamp
    pose = PoseStamped(header=header)
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = position
    return BuoyDetectionArray(header=header, detections=[BuoyDetection(pose=pose, color=color)])


def test_expired_transform_is_discarded_without_latest_fallback():
    transform = Mock(side_effect=TransformException("missing"))
    queue = ObservationBuffer("map", transform, wait_timeout=.2)
    queue.add(message(), received_at=10.)
    assert queue.drain(10.1) == []
    transform.side_effect = None
    transform.return_value = np.ones(3)
    assert queue.drain(10.21) == []
    assert transform.call_count == 1 and queue.dropped == 1


def test_queue_overflow_is_bounded_and_uses_each_pose_header():
    queue = ObservationBuffer("map", Mock(), max_pending=2)
    for stamp in (3, 1, 2):
        msg = message(stamp=stamp, frame="map")
        msg.header = Header(frame_id="wrong_array_frame")
        queue.add(msg, received_at=10.)
    ready = queue.drain(10.1)
    assert [item[0] for item in ready] == [1., 2.]
    assert queue.dropped == 1
    queue.transform_position.assert_not_called()


@pytest.mark.parametrize("invalid", ["stamp", "frame", "position", "color"])
def test_invalid_observations_do_not_enter_queue(invalid):
    msg = message()
    if invalid == "stamp": msg.header.stamp.sec = 0
    if invalid == "frame": msg.header.frame_id = ""
    if invalid == "position": msg.detections[0].pose.pose.position.x = float("nan")
    if invalid == "color": msg.detections[0].color = 254
    queue = ObservationBuffer("map", Mock())
    queue.add(msg, received_at=1.)
    assert not queue.pending and queue.dropped == 1


def test_array_header_is_used_when_pose_header_is_empty():
    msg = message(frame="map")
    msg.detections[0].pose.header = Header()
    queue = ObservationBuffer("map", Mock())
    queue.add(msg, received_at=1.)
    [(stamp, _, _)] = queue.drain(1.1)
    assert stamp == 42


def new_track(**options):
    manager = TrackManager(**options)
    track = BuoyTrack()
    manager.tracks[track.track_id] = track
    return manager, track


@pytest.mark.parametrize("kind,expected", [
    ("off", DetectedBuoyColor.BLACK),
    ("blue", DetectedBuoyColor.SOLID_BLUE),
    ("flashing_red", DetectedBuoyColor.FLASHING_RED),
    ("flashing_green", DetectedBuoyColor.FLASHING_GREEN),
    ("flashing_blue", DetectedBuoyColor.FLASHING_BLUE),
    ("unknown", DetectedBuoyColor.UNKNOWN),
    ("red_unknown", DetectedBuoyColor.UNKNOWN),
    ("off_unknown", DetectedBuoyColor.UNKNOWN),
])
def test_beacon_classification_uses_only_known_color_samples(kind, expected):
    manager, track = new_track()
    for i in range(200):
        on = i % 10 < 5
        if kind == "off": color = "black"
        elif kind == "blue": color = "blue"
        elif kind == "unknown": color = None
        elif kind == "red_unknown": color = "red" if on else None
        elif kind == "off_unknown": color = "black" if i % 4 == 0 else None
        else: color = kind.removeprefix("flashing_") if on else "black"
        manager._add_observation_to_track(track.track_id, color, np.zeros(3), 100 + i / 10)
    manager.classify_tracks()
    assert track.published_state == expected


def test_singleton_tracks_survive_timeout_and_keep_aligned_history():
    manager, track = new_track()
    manager._add_observation_to_track(track.track_id, "blue", np.zeros(3), 1.)
    track.published_state = DetectedBuoyColor.SOLID_BLUE
    manager.prune_temporal_window(100.)
    manager.prune_timeouts(100.)
    assert track.track_id in manager.tracks


def planner_with_tracks(states):
    from pose_estimator.buoy.planning import GatePlanning
    planner = GatePlanning()
    planner.track_manager = TrackManager(logger=Mock())
    planner.max_gate_width = 3.
    planner.vehicle_encasing_radius = 1.
    planner.waypoint_pass_tol = .5
    for state, position in states:
        track = BuoyTrack()
        track.published_state = state
        track.current_centroid = np.array(position, dtype=float)
        planner.track_manager.tracks[track.track_id] = track
    return planner


def test_gate_selection_uses_course_axis_and_nearest_valid_pair():
    from pose_estimator.buoy.planning import GOAL_TYPE_GATE_PAIR
    planner = planner_with_tracks([
        (DetectedBuoyColor.FLASHING_BLUE, [0, 0, 0]),
        (DetectedBuoyColor.SOLID_BLUE, [20, 0, 0]),
        (DetectedBuoyColor.FLASHING_RED, [10, -1, 0]),
        (DetectedBuoyColor.FLASHING_GREEN, [10, 1, 0]),
    ])
    direction, source, candidates = planner._evaluate_candidates(np.array([2., 0., 0.]), None)
    np.testing.assert_array_equal(direction, [1., 0.])
    assert source == "COURSE_AXIS"
    best = planner._select_best_candidate(candidates)
    assert best.goal_type == GOAL_TYPE_GATE_PAIR
    np.testing.assert_array_equal(best.position, [10., 0., 0.])


def test_single_red_and_green_standoffs_use_opposite_sides():
    planner = planner_with_tracks([
        (DetectedBuoyColor.FLASHING_RED, [10, 0, 0]),
        (DetectedBuoyColor.FLASHING_GREEN, [15, 0, 0]),
    ])
    candidates = planner._generate_candidates(np.zeros(3))
    np.testing.assert_allclose(candidates[0].position, [10, 1, 0], atol=1e-9)
    np.testing.assert_allclose(candidates[1].position, [15, -1, 0], atol=1e-9)


def test_candidate_filter_rejects_reached_and_behind_goals():
    from pose_estimator.buoy.planning import GoalCandidate
    planner = planner_with_tracks([])
    candidates = [GoalCandidate("test", np.array(p)) for p in ([.1, 0, 0], [-2, 0, 0], [2, 0, 0])]
    planner._filter_candidates(candidates, np.zeros(3), np.array([1., 0.]))
    assert [c.reject_reason for c in candidates] == ["WITHIN_PASS_TOL", "BEHIND_VEHICLE", ""]
    assert [c.is_valid for c in candidates] == [False, False, True]


@pytest.fixture
def ros_manager():
    import rclpy
    from pose_estimator.buoy_gate_manager_node import BuoyGateManagerNode
    rclpy.init(args=[])
    node = BuoyGateManagerNode()
    try:
        yield node
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_manager_ingestion_keeps_source_time_and_queues_tf(ros_manager, monkeypatch):
    from geometry_msgs.msg import TransformStamped
    node = ros_manager
    node.enable = True
    monkeypatch.setattr('pose_estimator.buoy_gate_manager_node.time.monotonic', lambda: 100.)
    node.get_clock = Mock(return_value=SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=45_000_000_000)))
    node.tf_buffer = Mock()
    node.tf_buffer.lookup_transform.side_effect = TransformException("not yet")
    node._on_buoy_detections(message(color=255))
    assert len(node.observations.pending) == 1
    assert not node.track_manager.detection_buffer
    transform = TransformStamped()
    transform.transform.rotation.w = 1.
    transform.transform.translation.x = 10.
    node.tf_buffer.lookup_transform.side_effect = None
    node.tf_buffer.lookup_transform.return_value = transform
    node._drain_observations()
    [(stamp, color, position)] = node.track_manager.detection_buffer
    assert stamp == 42. and color is None
    np.testing.assert_array_equal(position, [11., 2., 3.])
    args = node.tf_buffer.lookup_transform.call_args.args
    assert args[:2] == ('map', 'camera')
    assert args[2].nanoseconds == 42_000_000_000


def test_manager_disable_and_reset_clear_pending_observations(ros_manager):
    from bb_robotx_msgs.srv import SetGateManagerState
    from std_srvs.srv import Trigger
    node = ros_manager
    node.observations.add(message(), 1.)
    node.track_manager.detection_buffer.append((42., 'red', np.zeros(3)))
    response = node._cb_set_state(SetGateManagerState.Request(enable=False, temporal_window=30.), SetGateManagerState.Response())
    assert response.success and not node.enable
    assert not node.observations.pending and not node.track_manager.detection_buffer
    node.observations.add(message(), 1.)
    track = BuoyTrack()
    node.track_manager.tracks[track.track_id] = track
    assert node._cb_reset(Trigger.Request(), Trigger.Response()).success
    assert not node.observations.pending and not node.track_manager.tracks


@pytest.mark.parametrize('entry', [True, False])
def test_milestone_services_return_world_pose(ros_manager, entry):
    from bb_robotx_msgs.srv import GetEntry, GetExit
    node = ros_manager
    track = BuoyTrack()
    track.published_state = DetectedBuoyColor.FLASHING_BLUE if entry else DetectedBuoyColor.SOLID_BLUE
    track.current_centroid = np.array([1., 2., 0.])
    node.track_manager.tracks[track.track_id] = track
    service = GetEntry if entry else GetExit
    callback = node._cb_get_entry if entry else node._cb_get_exit
    response = callback(service.Request(return_latlon=False), service.Response())
    assert response.found and response.buoy_pose.header.frame_id == 'map'
    assert response.buoy_pose.pose.position.x == 1.


def test_singleton_merge_keeps_observations_sorted_and_upgrades_unknown_color():
    manager, primary = new_track()
    duplicate = BuoyTrack()
    manager.tracks[duplicate.track_id] = duplicate
    for track in (primary, duplicate):
        track.published_state = DetectedBuoyColor.SOLID_BLUE
    for track, stamp, color in [(primary, 3., 'blue'), (primary, 2., None),
                                (duplicate, 1., 'black'), (duplicate, 2., 'blue')]:
        manager._add_observation_to_track(track.track_id, color, np.array([stamp, 0., 0.]), stamp)
    manager.deduplicate_singletons()
    assert list(manager.tracks) == [primary.track_id]
    assert [(o.timestamp, o.color) for o in primary.observations] == [
        (1., 'black'), (2., 'blue'), (3., 'blue'),
    ]
    assert primary.last_observation_time == 3.
    manager.recompute_centroids()
    np.testing.assert_array_equal(primary.current_centroid, [2., 0., 0.])
    manager.temporal_window = 2.
    manager.prune_temporal_window(4.1)
    assert [(o.timestamp, o.color) for o in primary.observations] == [(3., 'blue')]


@pytest.mark.parametrize('outcome', ['unavailable', 'timeout', 'error'])
def test_geodetic_failures_are_bounded(ros_manager, outcome):
    from concurrent.futures import Future
    node = ros_manager
    node.geodetic_timeout = .01
    future = Future()
    if outcome == 'error':
        future.set_exception(RuntimeError('conversion failed'))
    client = Mock()
    client.wait_for_service.return_value = outcome != 'unavailable'
    client.call_async.return_value = future
    assert node._call_geodetic(client, object(), '/test_conversion') is None
    if outcome == 'unavailable':
        client.call_async.assert_not_called()
    elif outcome == 'timeout':
        assert future.cancelled()


def test_candidate_markers_keep_geometry_and_labels(ros_manager):
    node = ros_manager
    node.marker_pub = Mock()
    node.last_query_pos = np.zeros(3)
    node.prev_query_pos = np.array([-1., 0., 0.])
    for state, position in [(DetectedBuoyColor.FLASHING_RED, [2., -.25, 0.]),
                            (DetectedBuoyColor.FLASHING_GREEN, [2., .25, 0.])]:
        track = BuoyTrack()
        track.published_state = state
        track.current_centroid = np.array(position)
        node.track_manager.tracks[track.track_id] = track
    node._publish_markers()
    markers = node.marker_pub.publish.call_args.args[0].markers
    assert markers[0].action == markers[0].DELETEALL and markers[0].ns == ''
    by_ns = {m.ns: m for m in markers}
    assert all(m.header.frame_id == 'map' and m.header.stamp.sec > 0 for m in markers)
    assert by_ns['gate_selected_goal'].pose.position.x == 2.
    assert by_ns['gate_pair_links'].points[0].y == -.25
    assert by_ns['gate_pair_links'].points[1].y == .25
    assert not {'gate_query', 'gate_forward', 'gate_forward_label', 'gate_candidate_rays'} & by_ns.keys()
    assert by_ns['gate_candidate_labels'].pose.position.z == .2
    assert 'GATE_PAIR' in by_ns['gate_candidate_labels'].text


def test_geodetic_bridge_round_trip_with_real_ros_services(ros_manager):
    from threading import Thread
    from rclpy.executors import MultiThreadedExecutor
    from robot_localization.srv import FromLL, ToLL
    from sensor_msgs.msg import NavSatFix
    node = ros_manager
    received = []

    def from_ll(request, response):
        received.append(request.ll_point)
        response.map_point.x = 10.
        response.map_point.y = 20.
        response.map_point.z = 30.
        return response

    def to_ll(request, response):
        received.append(request.map_point)
        response.ll_point.latitude = 1.
        response.ll_point.longitude = 2.
        response.ll_point.altitude = 3.
        return response

    node.create_service(FromLL, node.from_ll_service_name, from_ll)
    node.create_service(ToLL, node.to_ll_service_name, to_ll)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = Thread(target=executor.spin)
    thread.start()
    try:
        local = node._convert_from_ll(NavSatFix(latitude=1., longitude=2., altitude=3.))
        np.testing.assert_array_equal(local, [10., 20., 30.])
        global_position = node._convert_to_ll(local)
        assert isinstance(global_position, NavSatFix)
        assert (global_position.latitude, global_position.longitude, global_position.altitude) == (1., 2., 3.)
        assert global_position.header.stamp.sec > 0
        assert (received[0].latitude, received[0].longitude, received[0].altitude) == (1., 2., 3.)
        assert (received[1].x, received[1].y, received[1].z) == (10., 20., 30.)
    finally:
        executor.shutdown()
        thread.join(timeout=2.)


def test_unknown_positions_establish_track_and_preserve_height_in_markers(ros_manager):
    node = ros_manager
    node.marker_pub = Mock()
    node.visualize_candidates = False
    node.track_manager.dbscan_min_samples = 3
    node.track_manager.detection_buffer.extend(
        (42. + i / 10, None, np.array([1., 2., height]))
        for i, height in enumerate((3., 4., 5.))
    )
    node.track_manager.associate_and_update(43.)
    node.track_manager.recompute_centroids()
    assert len(node.track_manager.tracks) == 1
    track = next(iter(node.track_manager.tracks.values()))
    np.testing.assert_array_equal(track.current_centroid, [1., 2., 4.])
    assert all(o.color is None for o in track.observations)
    assert track.published_state == DetectedBuoyColor.UNKNOWN
    node._publish_markers()
    markers = node.marker_pub.publish.call_args.args[0].markers
    sphere = next(m for m in markers if m.ns == 'buoy_tracks' and m.action == m.ADD)
    assert sphere.pose.position.z == 4.


def test_nearby_tracks_receive_their_own_color_observations():
    manager = TrackManager(max_association_radius=2., dbscan_min_samples=5)
    tracks = []
    for x in (0., 1.5):
        track = BuoyTrack()
        track.current_centroid = np.array([x, 0., 0.])
        manager.tracks[track.track_id] = track
        tracks.append(track)
    for i, (x, color) in enumerate([(.05, 'red'), (1.45, 'green')]):
        manager.detection_buffer.append((42. + i / 10, color, np.array([x, 0., 0.])))
    manager.associate_and_update(43.)
    assert [[o.color for o in t.observations] for t in tracks] == [['red'], ['green']]
    assert not manager.detection_buffer
    assert len(manager.tracks) == 2


def test_sparse_batches_accumulate_to_establish_new_track_and_expire_outliers():
    manager = TrackManager(dbscan_min_samples=5)
    for i in range(5):
        manager.detection_buffer.append((42. + i, 'red', np.array([1., 2., 3.])))
        manager.associate_and_update(42. + i)
        if i < 4:
            assert not manager.tracks
            assert len(manager.detection_buffer) == i + 1
    track = next(iter(manager.tracks.values()))
    assert len(track.observations) == 5
    assert not manager.detection_buffer
    manager.detection_buffer.append((47., 'green', np.array([100., 100., 0.])))
    manager.associate_and_update(47.)
    assert len(manager.detection_buffer) == 1
    manager.associate_and_update(78.)
    assert not manager.detection_buffer


def test_missing_heading_rejects_goals_without_inventing_a_direction(ros_manager):
    from bb_robotx_msgs.srv import ExtractLocalGoal
    node = ros_manager
    node.tf_buffer = Mock()
    node.tf_buffer.lookup_transform.side_effect = TransformException('missing heading')
    track = BuoyTrack()
    track.published_state = DetectedBuoyColor.FLASHING_RED
    track.current_centroid = np.array([10., 0., 0.])
    node.track_manager.tracks[track.track_id] = track
    direction, source, candidates = node._evaluate_candidates(np.zeros(3), None)
    assert direction is None and source == 'UNAVAILABLE'
    assert candidates and all(c.reject_reason == 'HEADING_UNAVAILABLE' for c in candidates)
    assert node._select_best_candidate(candidates) is None
    request = ExtractLocalGoal.Request()
    request.current_pose.header.frame_id = node.world_frame_id
    response = node._cb_extract_goal(request, ExtractLocalGoal.Response())
    assert not response.success and response.goal_type == 'INVALID'
    assert 'UNAVAILABLE' in response.message
    # Actual motion remains a usable heading source even when TF is missing.
    direction, source = node._compute_forward_vector(np.array([1., 0., 0.]), np.zeros(3))
    np.testing.assert_array_equal(direction, [1., 0.])
    assert source == 'VEHICLE_TRAJECTORY'


@pytest.mark.parametrize('rate,duration,expected', [
    (100., 1., DetectedBuoyColor.UNKNOWN),  # Many samples cannot replace elapsed cycles.
    (5., 5., DetectedBuoyColor.FLASHING_RED),  # Fewer than 40 samples can be sufficient.
    (10., 5., DetectedBuoyColor.FLASHING_RED),
])
def test_fft_requires_observed_cycles_instead_of_sample_count(rate, duration, expected):
    manager, track = new_track(target_freq=.5, fft_min_magnitude=.15)
    for t in np.arange(0., duration, 1. / rate):
        color = 'red' if t % 2 < 1 else 'black'
        manager._add_observation_to_track(track.track_id, color, np.zeros(3), 100 + t)
    manager.classify_tracks()
    assert track.published_state == expected


def test_short_sightings_are_not_spliced_across_visibility_gaps():
    manager, track = new_track(target_freq=.5, fft_min_magnitude=.15)
    for start in np.arange(0., 20., 2.):
        # Alternating short sightings would look periodic if interpolated over gaps.
        for t in np.arange(0., .6, .1):
            color = 'red' if start % 4 == 0 else 'black'
            manager._add_observation_to_track(track.track_id, color, np.zeros(3), 100 + start + t)
    manager.classify_tracks()
    assert track.published_state == DetectedBuoyColor.UNKNOWN


def test_separate_flashing_segments_combine_evidence_without_phase_alignment():
    manager, track = new_track(target_freq=.5, fft_min_magnitude=.15)
    for start, phase in [(0., 0.), (10., .7)]:
        for t in np.arange(0., 5., .1):
            color = 'green' if (t + phase) % 2 < 1 else 'black'
            manager._add_observation_to_track(track.track_id, color, np.zeros(3), 100 + start + t)
    manager.classify_tracks()
    assert track.published_state == DetectedBuoyColor.FLASHING_GREEN
