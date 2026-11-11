import numpy as np
from geometry_msgs.msg import PoseStamped

from pose_estimator.dockwin_pose_estimator_node import (
    _quaternion_matrix,
    create_standoff_pose,
    is_lower_window_pose,
    select_lower_window_detection,
    window_centre_pose,
)
from yolo_msgs.msg import Detection


def test_standoff_pose_is_centered_and_faces_window():
    window = PoseStamped()
    window.header.frame_id = "asv5/oakd_link_optical"
    window.pose.position.z = 3.0
    window.pose.orientation.w = 1.0

    target = create_standoff_pose(
        window,
        standoff_distance=2.5,
        window_width=0.28,
        window_height=0.36,
    )

    assert target.header.frame_id == window.header.frame_id
    assert np.isclose(target.pose.position.x, 0.14)
    assert np.isclose(target.pose.position.y, 0.18)
    assert np.isclose(target.pose.position.z, 0.5)

    center = np.array([0.14, 0.18, 3.0])
    position = np.array(
        [target.pose.position.x, target.pose.position.y, target.pose.position.z]
    )
    assert np.isclose(np.linalg.norm(center - position), 2.5)

    body_forward = _quaternion_matrix(target.pose.orientation)[:, 0]
    toward_window = (center - position) / np.linalg.norm(center - position)
    assert np.dot(body_forward, toward_window) > 0.999


def test_standoff_pose_rejects_nonpositive_distance():
    window = PoseStamped()
    window.pose.orientation.w = 1.0

    try:
        create_standoff_pose(window, 0.0, 0.28, 0.36)
    except ValueError as exc:
        assert "standoff_distance" in str(exc)
    else:
        raise AssertionError("nonpositive standoff distance must be rejected")


def test_lower_window_uses_positive_optical_y():
    pose = PoseStamped()
    pose.pose.position.y = -0.1
    assert not is_lower_window_pose(pose, y_threshold=0.0)

    pose.pose.position.y = 0.0
    assert is_lower_window_pose(pose, y_threshold=0.0)

    pose.pose.position.y = 0.2
    assert is_lower_window_pose(pose, y_threshold=0.0)


def test_single_visible_window_can_be_assumed_lower():
    pose = PoseStamped()
    pose.pose.position.y = -0.2

    assert not is_lower_window_pose(pose, y_threshold=0.0)
    assert is_lower_window_pose(
        pose, y_threshold=0.0, single_visible_window=True
    )


def _det(u, v, score=0.9, cls="null_window", h=60.0):
    d = Detection()
    d.class_name = cls
    d.score = score
    d.bbox.center.position.x = float(u)
    d.bbox.center.position.y = float(v)
    d.bbox.size.x = 50.0
    d.bbox.size.y = h
    return d


def test_lower_window_is_bottom_row_even_when_all_above_optical_axis():
    # 2x2 grid entirely in the top of a 640x400 image (camera below the dock).
    upper_l, upper_r = _det(250, 60, 0.95), _det(390, 62, 0.95, "red_window")
    lower_l, lower_r = _det(252, 150, 0.90), _det(388, 148, 0.93)
    chosen, rows = select_lower_window_detection(
        [upper_l, upper_r, lower_l, lower_r], image_center_x=320.0
    )
    assert rows
    assert chosen in (lower_l, lower_r)


def test_lower_window_prefers_bottom_row_window_nearest_image_centre():
    left, centre, right = _det(100, 200), _det(330, 205), _det(560, 198)
    upper = _det(330, 80, 0.99)
    chosen, rows = select_lower_window_detection(
        [upper, left, centre, right], image_center_x=320.0
    )
    assert rows
    assert chosen is centre


def test_lower_window_falls_back_to_score_without_image_centre():
    a, b = _det(100, 200, 0.6), _det(500, 205, 0.8)
    chosen, rows = select_lower_window_detection([_det(300, 50), a, b])
    assert rows
    assert chosen is b


def test_single_row_is_not_distinguished():
    a, b = _det(200, 150), _det(400, 160)
    chosen, rows = select_lower_window_detection([a, b], image_center_x=320.0)
    assert not rows
    assert chosen in (a, b)

    only, rows = select_lower_window_detection([a], image_center_x=320.0)
    assert only is a and not rows


def test_no_windows_selects_nothing():
    assert select_lower_window_detection([], image_center_x=320.0) == (None, False)


def test_column_distance_excludes_adjacent_bay_window():
    """Bottom row has windows from two bays; only the one within column distance wins."""
    target_bay = _det(300, 200, 0.8, cls="red_window")    # target bay lower
    adjacent_bay = _det(550, 205, 0.95, cls="null_window")  # adjacent bay lower, higher score
    upper = _det(310, 80, 0.9)
    chosen, rows = select_lower_window_detection(
        [upper, target_bay, adjacent_bay],
        image_center_x=300.0,
        max_column_distance=100.0,
    )
    assert rows
    assert chosen is target_bay


def test_column_distance_all_excluded_returns_none():
    """When every bottom-row window is outside the column distance, return None."""
    far_left = _det(50, 200, 0.9)
    far_right = _det(600, 205, 0.95)
    upper = _det(320, 80, 0.8)
    chosen, rows = select_lower_window_detection(
        [upper, far_left, far_right],
        image_center_x=320.0,
        max_column_distance=50.0,
    )
    assert chosen is None
    assert rows  # rows were distinguished even though column filter emptied it


def test_column_distance_none_preserves_existing_behaviour():
    """With max_column_distance=None, the farthest window can still win by score."""
    near = _det(310, 200, 0.5)
    far = _det(550, 205, 0.95)
    upper = _det(320, 80, 0.8)
    # Without column filter, nearest to image_center_x wins (soft tiebreaker).
    chosen_no_filter, _ = select_lower_window_detection(
        [upper, near, far], image_center_x=320.0, max_column_distance=None,
    )
    assert chosen_no_filter is near
    # Without image_center_x, highest score wins — far has 0.95.
    chosen_by_score, _ = select_lower_window_detection(
        [upper, near, far], image_center_x=None, max_column_distance=None,
    )
    assert chosen_by_score is far


def _window(z=3.0, yaw_deg=0.0):
    """Window corner pose in the optical frame, rotated about the optical y axis."""
    window = PoseStamped()
    window.header.frame_id = "asv5/oakd_link_optical"
    window.pose.position.z = z
    half = np.radians(yaw_deg) / 2.0
    window.pose.orientation.y = float(np.sin(half))
    window.pose.orientation.w = float(np.cos(half))
    return window


def test_window_centre_pose_keeps_axes_and_moves_to_centre():
    window = _window(yaw_deg=20.0)
    centre = window_centre_pose(window, 0.28, 0.36)

    rotation = _quaternion_matrix(window.pose.orientation)
    expected = np.array([0.0, 0.0, 3.0]) + rotation[:, 0] * 0.14 + rotation[:, 1] * 0.18
    got = np.array([centre.pose.position.x, centre.pose.position.y, centre.pose.position.z])
    assert np.allclose(got, expected)
    assert centre.pose.orientation == window.pose.orientation
    assert centre.header.frame_id == window.header.frame_id


def test_window_centre_pose_rejects_window_facing_the_camera():
    # Rotated 180 deg about y: +z points back at the camera.
    assert window_centre_pose(_window(yaw_deg=180.0), 0.28, 0.36) is None


def test_standoff_is_on_the_centre_normal():
    window = _window(yaw_deg=-25.0)
    centre = window_centre_pose(window, 0.28, 0.36)
    standoff = create_standoff_pose(window, 2.5, 0.28, 0.36)

    normal = _quaternion_matrix(centre.pose.orientation)[:, 2]
    c = np.array([centre.pose.position.x, centre.pose.position.y, centre.pose.position.z])
    st = np.array([standoff.pose.position.x, standoff.pose.position.y, standoff.pose.position.z])
    assert np.allclose(st, c - 2.5 * normal)
