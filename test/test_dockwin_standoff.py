import numpy as np
from geometry_msgs.msg import PoseStamped

from pose_estimator.dockwin_pose_estimator_node import (
    _quaternion_matrix,
    create_standoff_pose,
    is_lower_window_pose,
)


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
