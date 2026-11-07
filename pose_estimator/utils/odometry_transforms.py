"""Populate a local transform buffer from stamped odometry, without /tf traffic."""

from math import isfinite

from geometry_msgs.msg import TransformStamped


def add_odometry_transform(buffer, odometry):
    """Use the message's parent, child and timestamp; reject unusable samples."""
    header = odometry.header
    child = odometry.child_frame_id
    pose = odometry.pose.pose
    rotation = pose.orientation
    values = (pose.position.x, pose.position.y, pose.position.z,
              rotation.x, rotation.y, rotation.z, rotation.w)
    if (not header.frame_id or not child or header.frame_id == child
            or (header.stamp.sec == 0 and header.stamp.nanosec == 0)
            or not all(isfinite(value) for value in values)
            or sum(value * value for value in values[3:]) < 1e-12):
        return False
    transform = TransformStamped()
    transform.header = header
    transform.child_frame_id = child
    transform.transform.translation.x = pose.position.x
    transform.transform.translation.y = pose.position.y
    transform.transform.translation.z = pose.position.z
    transform.transform.rotation = rotation
    buffer.set_transform(transform, 'odometry_message')
    return True
