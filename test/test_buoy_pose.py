"""Check footprint PnP and stamped colored-detection outputs."""

import cv2
import numpy as np
import pytest

from unittest.mock import Mock
from pose_estimator.utils.PinholeCamera import PinholeCamera
from pose_estimator.utils.pose_estimator import get_object_pose_from_detection_using_best_fit_quad
from yolo_msgs.msg import Detection, Point2D

@pytest.mark.parametrize("rvec", [[0.1, -0.25, 0.2], [-0.5, 0.4, -0.6], [0.0, 0.0, 0.0]])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("shift", [0, 10, 20, 30])
def test_footprint_pnp_recovers_bottom_center_from_perspective_contour(rvec, reverse, shift):
    from pose_estimator.config.robotx26_object_points import (
        BUOY_FOOTPRINT_OBJECT_POINTS,
    )

    camera = np.array([[700., 0., 320.], [0., 710., 180.], [0., 0., 1.]])
    rvec = np.asarray(rvec, dtype=float)
    translation = np.array([0.1, -0.15, 3.0])
    vertices = BUOY_FOOTPRINT_OBJECT_POINTS
    # Sample actual perspective edges rather than replacing them with a rectangle.
    edges = np.concatenate([
        a + np.linspace(0, 1, 10, endpoint=False)[:, None] * (b - a)
        for a, b in zip(vertices, np.roll(vertices, -1, axis=0))
    ])
    pixels = cv2.projectPoints(edges, rvec, translation, camera, None)[0][:, 0]
    pixels = np.roll(pixels[::-1] if reverse else pixels, shift, axis=0)
    detection = Detection()
    detection.mask.data = [Point2D(x=float(x), y=float(y)) for x, y in pixels]
    pinhole = PinholeCamera("camera", 640, 360, 700., 710., 320., 180.)
    _, actual = get_object_pose_from_detection_using_best_fit_quad(
        pinhole, vertices[:, :2], detection, Mock()
    )
    np.testing.assert_allclose(actual.ravel(), translation, atol=0.002)


@pytest.mark.parametrize("reject_pnp", [False, True])
def test_footprint_callback_publishes_pose_only_when_pnp_succeeds(reject_pnp):
    from copy import deepcopy
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    from sensor_msgs.msg import CameraInfo
    from std_msgs.msg import Header
    from yolo_msgs.msg import Detection, DetectionArray, Point2D
    from pose_estimator.buoy_pose_estimator_node import BuoyPoseEstimator

    header = Header(frame_id="camera")
    header.stamp.sec = 42
    camera = CameraInfo(header=header, width=640, height=360)
    camera.p = [700., 0., 320., 0., 0., 710., 180., 0., 0., 0., 1., 0.]
    node = SimpleNamespace(
        max_range=100., camera_info=camera,
        pose_pub=Mock(), pose_array_pub=Mock(), pose_label_pub=Mock(),
        get_logger=Mock(return_value=Mock()),
    )
    buoy = Detection(class_name="buoy_red", class_id=2, score=0.9)
    buoy.bbox.center.position.x = 320.
    buoy.bbox.center.position.y = 180.
    buoy.bbox.size.x = buoy.bbox.size.y = 40.
    buoy.mask.data = [Point2D(x=x, y=y) for x, y in [(300., 160.), (340., 160.), (340., 200.), (300., 200.)]]
    green = deepcopy(buoy)
    green.class_name = "buoy_green"
    detections = DetectionArray(header=header, detections=[
        buoy, green, deepcopy(buoy), Detection(class_name="circle_1"), Detection(class_name="buoy"),
    ])
    with patch("pose_estimator.buoy_pose_estimator_node.get_object_pose_from_detection_using_best_fit_quad") as estimate:
        if reject_pnp:
            estimate.side_effect = ValueError("unsuitable contour")
        else:
            estimate.return_value = (np.zeros((3, 1)), np.array([[1.], [2.], [3.]]))
        BuoyPoseEstimator.detections_callback(node, detections)
        assert estimate.call_count == 3
    result = node.pose_pub.publish.call_args.args[0]
    assert result.header.frame_id == "camera"
    assert result.header.stamp == header.stamp
    poses = node.pose_array_pub.publish.call_args.args[0]
    markers = node.pose_label_pub.publish.call_args.args[0].markers
    assert poses.header == result.header
    assert poses.poses == [detection.pose.pose for detection in result.detections]
    from visualization_msgs.msg import Marker
    assert markers[0].action == Marker.DELETEALL
    assert all(marker.header == header for marker in markers)
    texts = markers[1:]
    assert len(texts) == len(poses.poses)
    if reject_pnp:
        assert not result.detections
    else:
        pose = result.detections[0].pose.pose
        np.testing.assert_allclose([pose.position.x, pose.position.y, pose.position.z], [1., 2., 3.])
        assert result.detections[0].color == 1
        text = texts[0]
        assert [text.text for text in texts] == ["buoy_red", "buoy_green", "buoy_red"]
        assert len(poses.poses) == 3
        assert [detection.color for detection in result.detections] == [1, 2, 1]
        assert text.text == "buoy_red"
        assert text.color.r == 1.0 and text.color.a == 1.0
        assert [marker.id for marker in texts] == [0, 1, 2]
        assert all(marker.type == Marker.TEXT_VIEW_FACING for marker in texts)
        assert all(marker.action == Marker.ADD for marker in texts)
        assert [marker.pose for marker in texts] == poses.poses
        assert text.scale.z > 0 and text.lifetime.sec == 1


@pytest.mark.parametrize("calibration", ["missing", "wrong_frame", "invalid_size"])
def test_invalid_calibration_publishes_empty_stamped_outputs(calibration):
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    from sensor_msgs.msg import CameraInfo
    from std_msgs.msg import Header
    from yolo_msgs.msg import Detection, DetectionArray
    from pose_estimator.buoy_pose_estimator_node import BuoyPoseEstimator

    header = Header(frame_id="camera")
    header.stamp.sec = 42
    camera = CameraInfo(header=Header(frame_id="other" if calibration == "wrong_frame" else "camera"),
                        width=0 if calibration == "invalid_size" else 640, height=360)
    node = SimpleNamespace(camera_info=None if calibration == "missing" else camera,
                           pose_pub=Mock(), pose_array_pub=Mock(), pose_label_pub=Mock())
    detections = DetectionArray(header=header, detections=[Detection(class_name="buoy_red")])
    with patch("pose_estimator.buoy_pose_estimator_node.get_object_pose_from_detection_using_best_fit_quad") as estimate:
        BuoyPoseEstimator.detections_callback(node, detections)
        estimate.assert_not_called()
    output = node.pose_pub.publish.call_args.args[0]
    poses = node.pose_array_pub.publish.call_args.args[0]
    assert output.header == poses.header == header
    assert not output.detections and not poses.poses
    from visualization_msgs.msg import Marker
    markers = node.pose_label_pub.publish.call_args.args[0].markers
    assert len(markers) == 1
    assert markers[0].action == Marker.DELETEALL
    assert markers[0].header == header


@pytest.mark.parametrize("polygon", [
    [[0, 20], [80, 20], [80, 100], [0, 100]],
    np.array([320, 180]) + 30 * np.column_stack([
        np.cos(np.linspace(0, 2 * np.pi, 40, endpoint=False)),
        np.sin(np.linspace(0, 2 * np.pi, 40, endpoint=False)),
    ]),
])
def test_contours_are_not_rejected_for_border_contact_or_quad_coverage(polygon):

    from pose_estimator.config.robotx26_object_points import BUOY_FOOTPRINT_OBJECT_POINTS

    detection = Detection()
    detection.mask.data = [Point2D(x=float(x), y=float(y)) for x, y in polygon]
    camera = PinholeCamera("camera", 640, 360, 700., 700., 320., 180.)
    _, center = get_object_pose_from_detection_using_best_fit_quad(
        camera, BUOY_FOOTPRINT_OBJECT_POINTS[:, :2], detection, Mock()
    )
    assert np.isfinite(center).all() and center.ravel()[2] > 0
