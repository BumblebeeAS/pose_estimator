from typing import List

from geometry_msgs.msg import (
    Point,
    PoseWithCovarianceStamped,
    Quaternion,
    TransformStamped,
    Vector3,
)
from numpy.typing import ArrayLike
from std_msgs.msg import Header


def get_pose_with_covariance_stamped(
    header: Header, t: ArrayLike, q: ArrayLike, covariance: List[float]
):
    pose = PoseWithCovarianceStamped()
    pose.header = header
    pose.pose.pose.position = Point(x=t[0], y=t[1], z=t[2])
    pose.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
    pose.pose.covariance = covariance
    return pose


def get_transform_stamped(
    header: Header, child_frame_id: str, t: ArrayLike, q: ArrayLike
):
    """Gets transform stamped message, where q is in ROS format [x, y, z, w]."""
    transform_stamped = TransformStamped()
    transform_stamped.header = header
    transform_stamped.child_frame_id = child_frame_id
    transform_stamped.transform.translation = Vector3(x=t[0], y=t[1], z=t[2])
    transform_stamped.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
    return transform_stamped
