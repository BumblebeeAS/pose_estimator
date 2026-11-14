"""Calibrated mask footprints preserve all tins and reject unusable geometry."""
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.time import Time
from rclpy.serialization import serialize_message
from tf_transformations import euler_matrix
from yolo_msgs.msg import Detection, DetectionArray

from pose_estimator import tins_pose_estimator_depth_from_odom_node as module
from pose_estimator.utils.PinholeCamera import PinholeCamera


def test_metric_projection_agrees_across_distance_intrinsics_and_tilt():
    polygon = np.array([[-.05,-.05,0.], [.05,-.05,0.], [.05,.05,0.], [-.05,.05,0.]])
    pixel_areas = []
    for depth, focal, tilt in [(2.,400.,0.), (3.,600.,.3)]:
        camera = PinholeCamera('camera', 640, 480, focal, focal, 320., 240.)
        world_camera = euler_matrix(tilt, -.1, 0.)
        world_camera[2, 3] = -depth
        camera_polygon = (polygon-world_camera[:3, 3]) @ world_camera[:3, :3]
        pixels, _ = cv2.projectPoints(camera_polygon, np.zeros(3), np.zeros(3), camera.camera_matrix(), camera.dist_coeffs())
        result, area = module.project_mask_to_plane(pixels, camera, world_camera, (np.zeros(3), np.array([0.,0.,1.])))
        np.testing.assert_allclose(result, camera_polygon, atol=1e-6)
        assert area == pytest.approx(.01, rel=1e-5)
        pixel_areas.append(cv2.contourArea(pixels.astype(np.float32)))
    assert pixel_areas[0] != pytest.approx(pixel_areas[1], rel=.01)


def test_count_and_metric_batches_preserve_candidates_and_invalid_depth_is_not_empty(monkeypatch):
    def polygon(d):
        return module.get_circle_points(d.score*100, 100., 10 if d.score < .8 else 5, 36).astype(np.float32)
    monkeypatch.setattr(module, 'get_detection_obb', lambda d: (None, polygon(d)))
    monkeypatch.setattr(module, 'get_detection_polygon', polygon)
    publishers = {color: Mock() for color in ('red', 'green', 'blue')}
    metric, largest = ({color: Mock() for color in publishers} for _ in range(2))
    clock = Mock()
    clock.now.return_value = Time(seconds=1.)
    extrinsic = TransformStamped()
    extrinsic.transform.rotation.w = 1.
    buffer = Mock()
    buffer.lookup_transform.return_value = extrinsic
    node = SimpleNamespace(
        is_enabled=True, pad_plane=(np.array([0.,0.,2.]), np.array([0.,0.,1.])), plane_frame='odom_ned',
        tf_buffer=buffer, camera_frame='camera', camera=PinholeCamera('camera',640,480,400.,400.,320.,240.),
        all_tins_publishers=publishers, metric_tins_publishers=metric,
        largest_tins_publishers=largest, publish_data=Mock(),
        get_clock=lambda: clock, get_logger=lambda: Mock())
    message = DetectionArray(detections=[Detection(class_name='red_tin', score=.7),
                                          Detection(class_name='red_tin', score=.9)])
    message.header.frame_id = 'camera'
    odom = Odometry()
    odom.header.frame_id = 'odom_ned'
    odom.pose.pose.orientation.w = 1.
    module.TinsPoseEstimatorDepthFromOdom.detections_callback(node, message, odom)
    poses = publishers['red'].publish.call_args.args[0]
    objects = metric['red'].publish.call_args.args[0]
    assert len(poses.poses) == len(objects.objects) == 2
    assert largest['red'].publish.call_args.args[0].pose == poses.poses[1]
    assert node.publish_data.call_count == 1  # Legacy best-confidence output retained.
    assert serialize_message(poses) and serialize_message(objects)
    for obj, radius in zip(objects.objects, [.025, .05]):
        p = np.array([[v.x,v.y,v.z] for v in obj.hypothesis.shape.footprint.points])
        area = np.linalg.norm(np.cross(p, np.roll(p,-1,axis=0)).sum(axis=0))/2
        expected = 16*np.sin(2*np.pi/32)*radius**2
        assert area == pytest.approx(expected, rel=1e-4)
    empty = DetectionArray(header=message.header)
    module.TinsPoseEstimatorDepthFromOdom.detections_callback(node, empty, odom)
    assert all(not pub.publish.call_args.args[0].objects for pub in metric.values())
    monkeypatch.setattr(module, 'get_detection_polygon', lambda d: np.array(
        [[50.,50.],[60.,50.],[60.,60.],[50.,60.]], dtype=np.float32))
    module.TinsPoseEstimatorDepthFromOdom.detections_callback(node, message, odom)
    unsupported = metric['red'].publish.call_args.args[0].objects
    assert len(unsupported) == 2 and all(not obj.hypothesis.shape.footprint.points for obj in unsupported)
    assert largest['red'].publish.call_count == 1
    for depth in [-.01, 0., float('nan')]:
        node.pad_plane = (np.array([0.,0.,depth]), np.array([0.,0.,1.]))
        module.TinsPoseEstimatorDepthFromOdom.detections_callback(node, empty, odom)
    assert all(pub.publish.call_count == 3 for pub in metric.values())
    assert all(pub.publish.call_count == 3 for pub in publishers.values())


@pytest.mark.parametrize('tilt', [0., .3])
def test_cropped_arc_recovers_full_diameter_and_centre(tilt):
    camera = PinholeCamera('camera',640,480,400.,400.,320.,240.)
    world_camera = euler_matrix(tilt, 0., 0.)
    world_camera[2,3] = -2.
    centre = np.array([-318*2*np.cos(tilt)/400, 0., 0.])
    circle = np.column_stack((module.get_circle_points(*centre[:2],.06,100), np.zeros(100)))
    camera_points = (circle-world_camera[:3,3]) @ world_camera[:3,:3]
    pixels, _ = cv2.projectPoints(camera_points, np.zeros(3), np.zeros(3), camera.camera_matrix(), camera.dist_coeffs())
    pixels = pixels.reshape(-1,2)
    pixels += np.random.default_rng(15).normal(0.,.3,pixels.shape)
    pixels[:,0] = np.maximum(0., pixels[:,0])  # Artificial straight image-border edge.
    assert np.any(pixels[:,0] == 0.)
    np.random.seed(8)
    fitted, footprint = module.fit_tin_circle(pixels, camera, world_camera, (np.zeros(3), np.array([0.,0.,1.])))
    np.testing.assert_allclose(fitted, (centre-world_camera[:3,3]) @ world_camera[:3,:3], atol=.005)
    np.testing.assert_allclose(np.linalg.norm(footprint-fitted,axis=1),.06,atol=.005)


def test_short_arc_and_straight_boundary_cannot_supply_size():
    camera = PinholeCamera('camera',640,480,400.,400.,320.,240.)
    world_camera = np.eye(4)
    plane = (np.array([0.,0.,2.]),np.array([0.,0.,1.]))
    angles = np.linspace(0.,np.pi/3,30)
    short_arc = np.column_stack((320+15*np.cos(angles),240+15*np.sin(angles)))
    line = np.column_stack((np.linspace(300.,350.,30),np.full(30,240.)))
    for pixels in [short_arc,line]:
        with pytest.raises(ValueError):
            module.fit_tin_circle(pixels,camera,world_camera,plane)
    # The shared RANSAC helper must also terminate on a degenerate input.
    with pytest.raises(ValueError):
        module.fit_circle_RANSAC(line,100,1.)
