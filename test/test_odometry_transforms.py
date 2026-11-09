"""Vehicle motion from odometry must compose with fixed camera extrinsics."""
from math import sqrt

import pytest
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.time import Time
from tf2_ros import Buffer

from pose_estimator.utils.odometry_transforms import add_odometry_transform


def sample():
    msg = Odometry()
    msg.header.frame_id = 'odom'
    msg.header.stamp.sec = 10
    msg.child_frame_id = 'body'
    msg.pose.pose.position.x = 10.0
    msg.pose.pose.position.y = 20.0
    msg.pose.pose.orientation.z = sqrt(0.5)
    msg.pose.pose.orientation.w = sqrt(0.5)
    return msg


def test_composition_and_timestamped_interpolation_without_dynamic_tf():
    buffer = Buffer()
    camera = TransformStamped()
    camera.header.frame_id = 'body'
    camera.child_frame_id = 'camera'
    camera.transform.translation.x = 1.0
    camera.transform.rotation.w = 1.0
    buffer.set_transform_static(camera, 'test')
    msg = sample()
    assert add_odometry_transform(buffer, msg)
    msg.header.stamp.sec = 12
    msg.pose.pose.position.x = 14.0
    assert add_odometry_transform(buffer, msg)
    result = buffer.lookup_transform('odom', 'camera', Time(seconds=11))
    assert result.transform.translation.x == pytest.approx(12.0)
    assert result.transform.translation.y == pytest.approx(21.0)
    assert result.header.stamp.sec == 11


@pytest.mark.parametrize('invalid', ['parent', 'child', 'same', 'stamp', 'nan', 'quaternion'])
def test_invalid_odometry_is_rejected(invalid):
    msg = sample()
    if invalid == 'parent': msg.header.frame_id = ''
    if invalid == 'child': msg.child_frame_id = ''
    if invalid == 'same': msg.child_frame_id = msg.header.frame_id
    if invalid == 'stamp': msg.header.stamp.sec = 0
    if invalid == 'nan': msg.pose.pose.position.x = float('nan')
    if invalid == 'quaternion':
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 0.0
    assert not add_odometry_transform(Buffer(), msg)
