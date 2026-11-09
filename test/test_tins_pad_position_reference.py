"""Single-circle rechecks preserve the shared world orientation across camera motion."""
from types import SimpleNamespace, MethodType
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import tf2_ros
from bb_perception_msgs.srv import TrashToggleFrame
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.time import Time
from tf_transformations import quaternion_from_euler, quaternion_matrix

from pose_estimator import tins_pad_pose_estimator_node as module
from pose_estimator.utils.PinholeCamera import PinholeCamera
CENTERS = {'red_circle':[-.5,-.35], 'green_circle':[.5,-.35], 'blue_circle':[0.,.5]}


def reference_node(now=100.):
    return SimpleNamespace(
        reference_transforms={}, fixed_orientation=None, orientation_frame='odom_ned',
        max_reference_age=5., tf_buffer=tf2_ros.Buffer(), get_logger=lambda: Mock(),
        get_clock=lambda: SimpleNamespace(now=lambda: Time(seconds=now)))


def projected_contours(camera, rotation, translation):
    angles=np.linspace(0,2*np.pi,100,endpoint=False)
    points=np.column_stack([-.5+.35*np.cos(angles),-.35+.35*np.sin(angles),np.zeros_like(angles)])
    pixels,_=cv2.projectPoints(points,cv2.Rodrigues(rotation)[0],translation,
                              camera.camera_matrix(),camera.dist_coeffs())
    return {'red_circle':pixels[:,0]}


def test_reference_is_fresh_copied_and_cleared_on_failed_refresh():
    tf = TransformStamped()
    tf.header.stamp = Time(seconds=99.5).to_msg()
    tf.header.frame_id = "odom_ned"
    tf.child_frame_id = "pad/clustered_0"
    tf.transform.rotation.z, tf.transform.rotation.w = .6, .8
    node = reference_node()
    node.reference_transforms = {'pad/clustered_0': tf}
    request = TrashToggleFrame.Request(enable=True, trash_frame_clustered='pad/clustered_0')
    call = module.TinsPadPoseEstimator.set_orientation_reference
    assert call(node, request, TrashToggleFrame.Response()).success
    np.testing.assert_allclose(node.fixed_orientation, [0,0,.6,.8])
    tf.transform.rotation.z = 0.
    np.testing.assert_allclose(node.fixed_orientation, [0,0,.6,.8])
    node.reference_transforms.clear()
    assert not call(node, request, TrashToggleFrame.Response()).success
    assert node.fixed_orientation is None
    node.reference_transforms['pad/clustered_0'] = tf
    tf.header.stamp = Time(seconds=90).to_msg()
    assert not call(node, request, TrashToggleFrame.Response()).success
    assert node.fixed_orientation is None
    node.fixed_orientation = np.array([0,0,0,1])
    request.enable = False
    assert call(node, request, TrashToggleFrame.Response()).success
    assert node.fixed_orientation is None


@pytest.mark.parametrize('body_angles,offset,fx', [
    ([0.,0.,0.], [.175,-.04,-.11], 390.),
    ([.2,-.15,.7], [.176,.033,-.097], 595.),
])
def test_one_circle_publishes_world_position_with_exact_retained_quaternion(
        monkeypatch, body_angles, offset, fx):
    camera = PinholeCamera('camera',640,360,fx,fx,320,180)
    fixed_q = np.asarray(quaternion_from_euler(0,0,.9))
    odom = Odometry()
    odom.header.frame_id = 'odom_ned';odom.child_frame_id = 'body'
    odom.pose.pose.position.x,odom.pose.pose.position.y,odom.pose.pose.position.z = .4,-.3,.6
    body_q = quaternion_from_euler(*body_angles)
    q = odom.pose.pose.orientation
    q.x,q.y,q.z,q.w = body_q
    tf = TransformStamped();tf.transform.rotation.w = 1.
    tf.transform.translation.x,tf.transform.translation.y,tf.transform.translation.z = offset
    world_rotation = quaternion_matrix(body_q)[:3,:3]
    camera_position = np.array([.4,-.3,.6])+world_rotation@np.array(offset)
    pad_position = np.array([.3,-.2,4.])
    rotation = world_rotation.T@quaternion_matrix(fixed_q)[:3,:3]
    translation = world_rotation.T@(pad_position-camera_position)
    contours = projected_contours(camera,rotation,translation)
    candidates = {'red_circle':[contours['red_circle']]}
    monkeypatch.setattr(module,'filter_detections_by_num_points',lambda msg,n:msg)
    monkeypatch.setattr(module,'get_top_k_detections_per_class',lambda msg,counts:candidates)
    monkeypatch.setattr(module,'get_detection_polygon',lambda contour:contour)
    node = SimpleNamespace(camera=camera,centers=CENTERS,radius=.35,inlier_threshold=8.,
                           sync_diagnostics=Mock(),
                           fixed_orientation=fixed_q.copy(),orientation_frame='odom_ned', max_reference_age=5.0,
                           tf_buffer=Mock(),publish_data=Mock(),position_publisher=Mock(),
                           get_logger=lambda:Mock())
    node.tf_buffer.lookup_transform.return_value = tf
    node._fit_contours = MethodType(module.TinsPadPoseEstimator._fit_contours,node)
    message = SimpleNamespace(header=SimpleNamespace(frame_id='camera',stamp=Time(seconds=100).to_msg()))
    np.random.seed(0)
    module.TinsPadPoseEstimator.detections_callback(node,message,odom)
    node.publish_data.assert_not_called()  # One circle never pretends to measure full pose.
    node.position_publisher.publish.assert_called_once()
    pose = node.position_publisher.publish.call_args.args[0]
    assert pose.header.frame_id == 'odom_ned'
    np.testing.assert_allclose([pose.pose.position.x,pose.pose.position.y,pose.pose.position.z],pad_position,atol=.003)
    np.testing.assert_allclose([pose.pose.orientation.x,pose.pose.orientation.y,
                               pose.pose.orientation.z,pose.pose.orientation.w],fixed_q,atol=1e-12)
    node.position_publisher.publish.reset_mock();node.fixed_orientation=None
    module.TinsPadPoseEstimator.detections_callback(node,message,odom)
    node.position_publisher.publish.assert_not_called()


def test_static_latest_lookup_zero_stamp_does_not_reject_fresh_cluster():
    # Reproduce TF2 semantics rather than assuming lookup_transform(Time())
    # preserves the source header. The original /tf_static sample is fresh.
    node = reference_node()
    reference=TransformStamped()
    reference.header.frame_id='odom_ned';reference.child_frame_id='pad/clustered_0'
    reference.header.stamp=Time(seconds=99.5).to_msg()
    reference.transform.rotation.z=.6;reference.transform.rotation.w=.8
    node.tf_buffer.set_transform_static(reference,'test')
    latest=node.tf_buffer.lookup_transform('odom_ned','pad/clustered_0',Time())
    assert latest.header.stamp.sec==0 and latest.header.stamp.nanosec==0
    from tf2_msgs.msg import TFMessage
    module.TinsPadPoseEstimator._on_reference_transforms(node,TFMessage(transforms=[reference]))
    request=TrashToggleFrame.Request(enable=True,trash_frame_clustered='pad/clustered_0')
    result=module.TinsPadPoseEstimator.set_orientation_reference(node,request,TrashToggleFrame.Response())
    assert result.success
    np.testing.assert_allclose(node.fixed_orientation,[0,0,.6,.8])


def test_reference_orientation_composes_parent_rotation():
    node = reference_node()
    parent=TransformStamped();parent.header.frame_id='odom_ned';parent.child_frame_id='intermediate'
    parent.transform.rotation.z=parent.transform.rotation.w=2**-.5
    node.tf_buffer.set_transform_static(parent,'test')
    reference=TransformStamped();reference.header.frame_id='intermediate';reference.child_frame_id='pad'
    reference.header.stamp=Time(seconds=99.5).to_msg();reference.transform.rotation.w=1.
    node.reference_transforms['pad']=reference
    result=module.TinsPadPoseEstimator.set_orientation_reference(
        node,TrashToggleFrame.Request(enable=True,trash_frame_clustered='pad'),TrashToggleFrame.Response())
    assert result.success
    np.testing.assert_allclose(node.fixed_orientation,[0,0,2**-.5,2**-.5])


def test_sequential_camera_handoff_does_not_expire_between_cameras():
    reference=TransformStamped()
    reference.header.frame_id='odom_ned';reference.child_frame_id='pad'
    reference.header.stamp=Time(seconds=98.2).to_msg()
    reference.transform.rotation.z=.6;reference.transform.rotation.w=.8
    captured=[]
    # Reproduce the observed run: wide succeeds around age 1.8 s, while
    # narrow handles the same reference after the former 2 s deadline.
    for now in (100.,100.4):
        node = reference_node(now)
        node.reference_transforms['pad'] = reference
        result=module.TinsPadPoseEstimator.set_orientation_reference(
            node,TrashToggleFrame.Request(enable=True,trash_frame_clustered='pad'),TrashToggleFrame.Response())
        assert result.success
        captured.append(node.fixed_orientation)
    np.testing.assert_array_equal(captured[0],captured[1])
    node.get_clock=lambda:SimpleNamespace(now=lambda:Time(seconds=104))
    result=module.TinsPadPoseEstimator.set_orientation_reference(
        node,TrashToggleFrame.Request(enable=True,trash_frame_clustered='pad'),TrashToggleFrame.Response())
    assert not result.success and node.fixed_orientation is None
    assert '5.00' in result.message


@pytest.mark.parametrize('clock_time',[101.,160.,10000.])
def test_explicit_saved_orientation_survives_a_long_approach(clock_time):
    from bb_perception_msgs.srv import SetOrientationReference
    request=SetOrientationReference.Request(enable=True)
    request.orientation.header.frame_id='odom_ned'
    request.orientation.header.stamp=Time(seconds=100).to_msg()
    request.orientation.quaternion.z=.6;request.orientation.quaternion.w=.8
    node = reference_node(clock_time)
    response=module.TinsPadPoseEstimator.set_saved_orientation(node,request,SetOrientationReference.Response())
    assert response.success
    np.testing.assert_allclose(node.fixed_orientation,[0,0,.6,.8])
    request.orientation.quaternion.z=0.
    np.testing.assert_allclose(node.fixed_orientation,[0,0,.6,.8])
    request.enable=False
    assert module.TinsPadPoseEstimator.set_saved_orientation(node,request,SetOrientationReference.Response()).success
    assert node.fixed_orientation is None


@pytest.mark.parametrize('case',['wrong_frame','zero_quaternion','nonfinite_quaternion'])
def test_invalid_explicit_orientation_is_rejected(case):
    from bb_perception_msgs.srv import SetOrientationReference
    request=SetOrientationReference.Request(enable=True)
    request.orientation.header.frame_id='home_ned' if case=='wrong_frame' else 'odom_ned'
    request.orientation.quaternion.w=0. if case=='zero_quaternion' else 1.
    if case=='nonfinite_quaternion':request.orientation.quaternion.z=float('nan')
    node=SimpleNamespace(fixed_orientation=np.array([0,0,0,1]),orientation_frame='odom_ned')
    response=module.TinsPadPoseEstimator.set_saved_orientation(node,request,SetOrientationReference.Response())
    assert not response.success and node.fixed_orientation is None
