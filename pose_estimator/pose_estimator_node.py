#!/usr/bin/env python3

import copy
import glob
import json
import logging
import os
import threading
import traceback
from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import rclpy
import tf2_ros
from ament_index_python import get_package_share_directory
from bb_msgs.srv import (
    IMPoseEstimatorConfig,
    IMPoseEstimatorGetStatus,
    IMPoseEstimatorGetTemplates,
    IMPoseEstimatorToggleTemplate,
)
from cv_bridge import CvBridge, CvBridgeError
from feature_matcher.keypoints_match_producer import get_keypoints_match_producer
from geometry_msgs.msg import (
    Point,
    PoseStamped,
    PoseWithCovarianceStamped,
    Quaternion,
    TransformStamped,
    Vector3,
)
from nav_msgs.msg import Odometry
from pose_estimator.PinholeCamera import PinholeCamera
from pose_estimator.pose_weighted_average import get_kmeans_center
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.time import Time
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import CameraInfo, CompressedImage
from transforms3d.affines import compose, decompose
from transforms3d.euler import euler2quat, mat2euler, quat2euler
from transforms3d.quaternions import mat2quat, quat2mat

from pose_estimator.src import PoseEstimator

mutex = threading.Lock()


@dataclass
class Image:
    img: np.ndarray
    descriptor: Any
    timestamp: float
    pose: PoseStamped


@dataclass
class Template:
    name: str
    matcher: str
    min_matches: int
    reprojection_error_threshold: float
    object_name: str
    offset: Tuple[
        float, float
    ]  # x, y offset of template center from object_name frame in meters


@dataclass
class TemplateObject:
    name: str
    poses: pd.DataFrame
    computed_pose: Optional[PoseWithCovarianceStamped]
    min_buffer_size: int
    max_buffer_size: int
    max_history: float


def get_matcher(matcher):
    if matcher == "coarse_loftr":
        image_match_producer = get_keypoints_match_producer(
            None, "coarse_loftr", {"debug": True}, {"debug": True}
        )
    elif matcher == "sift_flann":
        image_match_producer = get_keypoints_match_producer(
            "sift", "flann", {"debug": True}, {"debug": True}
        )
    elif matcher == "sift_bf":
        image_match_producer = get_keypoints_match_producer(
            "sift", "bf", {"debug": True}, {"debug": True}
        )
    elif matcher == "superpoint_bf":
        image_match_producer = get_keypoints_match_producer(
            "superpoint", "bf", {"debug": True}, {"debug": True}
        )
    elif matcher == "superpoint_superglue":
        image_match_producer = get_keypoints_match_producer(
            "superpoint", "superglue", {"debug": True}, {"debug": True}
        )
    elif matcher == "superpoint_lightglue":
        image_match_producer = get_keypoints_match_producer(
            "superpoint", "lightglue", {"debug": True}, {"debug": True}
        )
    elif matcher == "fast_bf":
        image_match_producer = get_keypoints_match_producer(
            "fast", "bf", {"debug": True}, {"debug": True}
        )
    elif matcher == "orb_bf":
        image_match_producer = get_keypoints_match_producer(
            "orb", "bf", {"debug": True}, {"debug": True}
        )
    elif matcher == "orb_flann":
        image_match_producer = get_keypoints_match_producer(
            "orb", "flann", {"debug": True}, {"debug": True}
        )
    elif matcher == "alike_bf":
        image_match_producer = get_keypoints_match_producer(
            "alike", "bf", {"debug": True}, {"debug": True}
        )
    elif matcher == "dkm":
        image_match_producer = get_keypoints_match_producer(
            None, "dkm", {"debug": True}, {"debug": True}
        )
    elif matcher == "keyaffhard_flann":
        image_match_producer = get_keypoints_match_producer(
            "keyaffhard", "flann", {"debug": True}, {"debug": True}
        )
    elif matcher == "disk_lightglue":
        image_match_producer = get_keypoints_match_producer(
            "disk",
            "lightglue",
            {"debug": True},
            {"debug": True, "weights": "disk"},
        )
    elif matcher == "dalf_flann":
        image_match_producer = get_keypoints_match_producer(
            "dalf", "flann", {"debug": True}, {"debug": True}
        )
    elif matcher == "dalf_bf":
        image_match_producer = get_keypoints_match_producer(
            "dalf", "bf", {"debug": True}, {"debug": True}
        )
    elif matcher == "xfeat":
        image_match_producer = get_keypoints_match_producer(
            "xfeat", "xfeat", {"debug": True}, {"debug": True}
        )
    else:
        raise NotImplementedError(f"Matcher {matcher} unimplemented!")
    return image_match_producer


class BasicPoseEstimator(Node):
    def publish_img(self, img):
        try:
            self.visualization_pub.publish(
                self.bridge.cv2_to_compressed_imgmsg(img, "jpeg")
            )
        except Exception as e:
            self.get_logger().error(str(e))

    def filter_forward_facing(self, pose):
        """
        Given pose np.array([x,y,z,qw,qx,qy,qz])
        return True if valid pose else False
        """
        if pose[2] < -2 or pose[2] > 10:
            self.get_logger().warn("Rubbish z", throttle_duration_sec=1)
            return False
        r, p, y = np.rad2deg(quat2euler(pose[3:], "rzyx"))
        if abs((y % 180) - 90) > 30:
            self.get_logger().warn(f"Not vertical {r} {p} {y}")
            return False
        return True

    def filter_bottom_facing(self, pose):
        if pose[2] < 0 or pose[2] > 3:
            self.get_logger().warn("Rubbish z", throttle_duration_sec=1)
            return False
        y, p, r = np.rad2deg(quat2euler(pose[3:], "rzyx"))
        if r >= 45 or r <= -45 or p >= 45 or p <= -45:
            self.get_logger().warn(
                f">>>>>>>>> ignore: r:{r} p: {p}, y: {y}",
                throttle_duration_sec=1,
            )
            return False
        return True

    def __init__(
        self,
        # image_match_producers: Dict[str, KeypointsMatchProducer],
        # visualization_topic,
        # detected_objects_topic=None,
        # templates_dir="./",
        # debug=False,
        # map_ned_frame="world_ned"
    ):
        super().__init__("pose_estimator", allow_undeclared_parameters=False)

        # matchers,
        # visualization_topic,
        # detected_objects_topic,
        # templates_dir,
        # debug,
        # map_ned_frame,
        # rospy.init_node("pose_estimator_dev", anonymous=False, log_level=rospy.INFO)

        # self.callback_group = MutuallyExclusiveCallbackGroup()

        self.timer = self.create_timer(0.05, self.cropped_image_callback)

        self.declare_parameter("debug", rclpy.Parameter.Type.BOOL)
        self.declare_parameter("front_camera_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("front_camera_info_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("bottom_camera_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("bottom_camera_info_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("visualization_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("detected_objects_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("matcher", rclpy.Parameter.Type.STRING)
        self.declare_parameter("map_ned_frame", rclpy.Parameter.Type.STRING)

        self.debug = (
            self.get_parameter_or("debug", True).get_parameter_value().bool_value
        )
        if self.debug:
            self.debug_file = open("debug_poses.csv", "w")
            self.get_logger().info(
                f"Writing debug poses to {os.path.abspath(self.debug_file.name)}"
            )
            rclpy.get_default_context().on_shutdown(self.debug_file.close)
        front_camera_topic = (
            self.get_parameter_or(
                "front_camera_topic",
                "/auv4/front_cam/image_rect_color/compressed",
            )
            .get_parameter_value()
            .string_value
        )
        front_camera_info_topic = (
            self.get_parameter_or(
                "front_camera_info_topic", "/auv4/front_cam/camera_info"
            )
            .get_parameter_value()
            .string_value
        )
        bottom_camera_topic = (
            self.get_parameter_or(
                "bottom_camera_topic",
                "/auv4/bot_cam/image_rect_color/compressed",
            )
            .get_parameter_value()
            .string_value
        )
        bottom_camera_info_topic = (
            self.get_parameter_or(
                "bottom_camera_info_topic", "/auv4/bot_cam/camera_info"
            )
            .get_parameter_value()
            .string_value
        )
        visualization_topic = (
            self.get_parameter_or(
                "visualization_topic", "/pose_estimator_vis/compressed"
            )
            .get_parameter_value()
            .string_value
        )

        # detected_objects_topic = (
        #     self.get_parameter_or("detected_objects_topic", None)
        #     .get_parameter_value()
        #     .string_value
        # )

        self.matcher_name = (
            self.get_parameter_or("matcher", "sift_flann")
            .get_parameter_value()
            .string_value
        )
        map_ned_frame = (
            self.get_parameter_or("map_ned_frame", "map_ned")
            .get_parameter_value()
            .string_value
        )

        # Register templates, template dimensions from json file
        templates_dir = os.path.abspath(
            Path(get_package_share_directory("image_matching")) / "templates"
        )

        # define image_matchers
        image_match_producers = {}
        image_match_producers[self.matcher_name] = get_matcher(self.matcher_name)

        self.latest_msgs: Dict[str, cv2.Mat] = {}
        self.bridge = CvBridge()
        self.templates: Dict[str, Template] = {}
        self.template_objects: Dict[str, TemplateObject] = {}
        self.templates_dir = templates_dir

        self.image_match_producers = image_match_producers
        self.pose_estimator = PoseEstimator(image_match_producers)

        self.visualization_pub: rclpy.Publisher = self.create_publisher(
            CompressedImage, visualization_topic, qos_profile=1
        )
        self.topics: Dict[str, str] = {}

        if self.debug:
            self.pose_estimator.visualize_callbacks.append(self.publish_img)
        self.map_ned_frame = map_ned_frame

        self.active_templates: Set[Tuple[str, str]] = (
            set()
        )  # (template_name, camera_frame_id)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30), node=self)
        self.br = tf2_ros.StaticTransformBroadcaster(self)
        self.odom_pub = self.create_publisher(
            Odometry, "impose_estimates", qos_profile=1
        )

        self.get_templates_service = self.create_service(
            IMPoseEstimatorGetTemplates,
            "impose_get_templates",
            self.get_templates,
        )
        self.toggle_template_service = self.create_service(
            IMPoseEstimatorToggleTemplate,
            "impose_toggle_template",
            self.toggle_template,
        )
        self.get_status_service = self.create_service(
            IMPoseEstimatorGetStatus,
            "impose_get_status",
            self.get_status,
        )
        self.config_service = self.create_service(
            IMPoseEstimatorConfig, "impose_config", self.update_config
        )

        self.PADDING = 10
        self.clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))

        self.subscribers: Dict[str, rclpy.subscription.Subscription] = {}

        # NOTE: template.json values are real world dimensions corresponding to
        # width and height of image: [width, height] in meters.
        templates = json.loads(
            open(os.path.join(templates_dir, "templates.json")).read()
        )
        for template in templates.keys():
            if template.startswith("_"):
                continue
            template = os.path.splitext(template)[0]
            template_path = os.path.join(templates_dir, template)
            possible_templates = glob.glob(os.path.join(templates_dir, f"{template}.*"))
            if not possible_templates:
                self.get_logger().warn(
                    f"No template found for {template} in {templates_dir}",
                    once=True,
                )
                continue
            template_path = possible_templates[-1]  # take newest template
            self.get_logger().info(f"Registering template {template_path}")

            template_filename = template_path.split("/")[-1]

            template_img = cv2.imread(template_path)
            regions = {}
            regions[template] = [0, 0, 1, 1]
            if isinstance(templates[template_filename], list):
                template_width = templates[template_filename][0]
                template_height = templates[template_filename][1]
            else:
                template_width = templates[template_filename]["dimensions"][0]
                template_height = templates[template_filename]["dimensions"][1]
                for region_name, region in templates[template_filename][
                    "regions"
                ].items():
                    template_name = f"{template}_{region_name}"
                    if template_name in templates.keys():
                        self.get_logger().warn(
                            f"{template_name} already registered!", once=True
                        )
                        continue
                    regions[template_name] = region

            for region_name, region in regions.items():
                if not isinstance(region, list) or len(region) != 4:
                    continue
                x1, y1, x2, y2 = region
                x1, x2 = (
                    int(x1 * template_img.shape[1]),
                    int(x2 * template_img.shape[1]),
                )
                y1, y2 = (
                    int(y1 * template_img.shape[0]),
                    int(y2 * template_img.shape[0]),
                )
                template_img_width, template_img_height = x2 - x1, y2 - y1
                region_img = template_img[y1:y2, x1:x2]
                print(template_img.shape, region_img.shape)
                region_px_offset = (
                    (x1 + x2) / 2 - template_img.shape[1] / 2,
                    (y1 + y2) / 2 - template_img.shape[0] / 2,
                )
                region_offset = (
                    region_px_offset[0] / template_img.shape[1] * template_width,
                    region_px_offset[1] / template_img.shape[0] * template_height,
                )
                region_width, region_height = (
                    (template_img_width / template_img.shape[1]) * template_width,
                    (template_img_height / template_img.shape[0]) * template_height,
                )

                if region_img.shape[0] > 480 or region_img.shape[1] > 480:
                    self.get_logger().error(
                        f"Region {region_name} is too large! Resizing the image"
                    )
                    if template_img_height > template_img_width:
                        template_img_width, template_img_height = (
                            480,
                            int(480 * template_img_width / template_img_height),
                        )
                    else:
                        template_img_width, template_img_height = (
                            int(480 * region_img.shape[0] / region_img.shape[1]),
                            480,
                        )
                    region_img = cv2.resize(
                        region_img, (template_img_height, template_img_width)
                    )
                self.get_logger().info(
                    f"Using template dimensions {region_width}x{region_height} \
        for template of size {region_img.shape[:2]} with offset {region_offset}"
                )
                self.register_template(
                    region_img,
                    region_name,
                    (region_width, region_height),
                    template,
                    region_offset,
                )
                # self.get_clock().sleep_for(Duration(seconds=0.05))

        if front_camera_topic is not None and front_camera_info_topic is not None:
            valid, front_camera_info = wait_for_message(
                CameraInfo, self, front_camera_info_topic, time_to_wait=10
            )
            if not valid:
                raise ValueError("Failed to get camera info")
            self.register_camera(
                front_camera_topic,
                PinholeCamera.from_camera_info(
                    front_camera_info, "rect" in front_camera_topic
                ),
            )
        if bottom_camera_topic is not None and bottom_camera_info_topic is not None:
            valid, bottom_camera_info = wait_for_message(
                CameraInfo, self, bottom_camera_info_topic, time_to_wait=10
            )
            if not valid:
                raise ValueError("Failed to get camera info")
            self.register_camera(
                bottom_camera_topic,
                PinholeCamera.from_camera_info(
                    bottom_camera_info, "rect" in bottom_camera_topic
                ),
            )

    def teardown(self):
        for sub in self.subscribers.values():
            sub.unregister()

    def update_config(self, req, res):
        template_name = req.template_name
        matcher = req.matcher
        if template_name not in self.templates.keys():
            # self.templates[template_name] = self.create_default_template(template_name, template_name, (0, 0))
            self.get_logger().error(f"Template {template_name} not registered")
            res.success = False
            return res

        if matcher not in self.image_match_producers.keys():
            self.get_logger().error(f"Matcher {matcher} not loaded")
            res.success = False
            return res

        object_name = self.templates[template_name].object_name
        if object_name not in self.template_objects.keys():
            self.get_logger().error(f"Object {object_name} not registered")
            res.success = False
            return res

        setattr(
            self.template_objects[object_name],
            "min_buffer_size",
            req.min_buffer_size,
        )
        setattr(
            self.template_objects[object_name],
            "max_buffer_size",
            req.max_buffer_size,
        )
        setattr(self.template_objects[object_name], "max_history", req.max_history)

        if req.reset:
            self.template_objects[object_name].poses = pd.DataFrame(
                columns=["stamp", "x", "y", "z", "qw", "qx", "qy", "qz"]
            )
            self.template_objects[object_name].computed_pose = None

        setattr(
            self.templates[template_name],
            "reprojection_error_threshold",
            req.max_reprojection_threshold,
        )
        setattr(self.templates[template_name], "min_matches", req.min_matches)
        setattr(self.templates[template_name], "matcher", matcher)

        self.get_logger().info(
            f"Config updated: {req} {self.templates[template_name]} {self.template_objects[object_name]}"
        )
        res.success = True
        return res

    def get_templates(self, req, res):
        active_templates = list(
            set(
                [
                    template_name + ":" + frame_id
                    for (template_name, frame_id) in self.active_templates
                ]
            )
        )
        res.registered_templates = self.pose_estimator.available_templates
        res.active_templates = active_templates
        res.loaded_matchers = list(self.image_match_producers.keys())
        return res

    def toggle_template(self, req, res):
        if req.template_name == "" and req.enabled is False:
            print("Disabling all templates")
            self.active_templates.clear()
            res.new_state = False
            res.error_message = "All templates disabled"
            return res

        if req.template_name not in self.pose_estimator.available_templates:
            res.new_state = False
            res.error_message = f"Template {req.template_name} not registered"
            return res
        if req.camera_frame_id not in self.pose_estimator.cameras:
            res.new_state = False
            res.error_message = f"Camera {req.camera_frame_id} not registered"
            return res
        if req.enabled:
            self.active_templates.add((req.template_name, req.camera_frame_id))
        else:
            if (
                req.template_name,
                req.camera_frame_id,
            ) not in self.active_templates:
                res.new_state = False
                res.error_message = f"Template {req.template_name} not active"
                return res
            self.active_templates.remove((req.template_name, req.camera_frame_id))
        print("toggled template", self.subscribers)
        res.new_state = (
            req.template_name,
            req.camera_frame_id,
        ) in self.active_templates
        res.error_message = ""
        return res

    @staticmethod
    def create_default_object(name):
        return TemplateObject(
            name,
            pd.DataFrame(columns=["stamp", "x", "y", "z", "qw", "qx", "qy", "qz"]),
            None,
            1,  # min_buffer_size
            30,  # max_buffer_size
            10,  # max_history,
        )

    @staticmethod
    def create_default_template(
        name, matcher_name="sift_flann", object_name=None, offset=(0, 0)
    ):
        if object_name is None:
            object_name = name
        return Template(
            name,
            matcher_name,  # "sift_flann",
            4,  # min_matches
            2,  # reprojection_error_threshold
            object_name,
            offset,  # of template center from object center
        )

    def get_status(
        self,
        req: IMPoseEstimatorGetStatus.Request,
        res: IMPoseEstimatorGetStatus.Response,
    ):
        if req.template_name not in self.templates.keys():
            res.is_valid = False
            return res
        template = self.templates[req.template_name]
        template_object = self.template_objects[template.object_name]
        if template_object.computed_pose is None:
            res.is_valid = False
            return res
        time_since_last = (
            self.get_clock().now().seconds_nanoseconds()[0]
            - template_object.computed_pose.header.stamp.sec
        )
        if (
            template_object.max_history > 0
            and time_since_last > template_object.max_history
        ):
            res.is_valid = False
            res.num_poses = len(template_object.poses)
            return res
        res.pose = template_object.computed_pose
        res.is_valid = True
        res.num_poses = len(template_object.poses)
        return res

    def register_template(self, img, name, dimensions, object_name, offset=(0, 0)):
        # if self.pose_estimator is None:
        #     self.get_logger().error("pose_estimator not defined")
        #     return
        self.pose_estimator.register_template(name, dimensions, img)
        if object_name not in self.template_objects.keys():
            self.template_objects[object_name] = self.create_default_object(object_name)
        self.templates[name] = BasicPoseEstimator.create_default_template(
            name, self.matcher_name, object_name, offset
        )
        self.get_logger().error(f"registered template {name}")

    def check_subscribers(self):
        for subscriber in self.subscribers.values():
            if subscriber.get_num_connections() == 0:
                return False

    def register_camera(self, camera_topic: str, camera: PinholeCamera):
        if camera_topic not in self.subscribers:
            self.get_logger().info(f"registering camera {camera_topic}")
            self.subscribers[camera_topic] = self.create_subscription(
                CompressedImage,
                camera_topic,
                self.img_msg_callback(camera.frame_id),
                QoSProfile(depth=10),
            )
            self.pose_estimator.register_camera(camera)
            self.topics[camera.frame_id] = camera_topic

    def img_msg_callback(self, camera_frame_id):
        def callback(msg):
            self.get_logger().info(f"cb: {self.active_templates}")
            if len(self.active_templates) == 0:
                return
            if mutex.acquire(blocking=False):
                try:
                    self.get_logger().info(f"Saving to: {camera_frame_id}")
                    if camera_frame_id not in self.latest_msgs or Time.from_msg(
                        msg.header.stamp
                    ) > Time.from_msg(self.latest_msgs[camera_frame_id].header.stamp):
                        self.latest_msgs[camera_frame_id] = msg
                        self.get_logger().info(
                            f"Saving recent timestamp: {msg.header.stamp}"
                        )
                    # else:
                    # print("Received old timestamp! ", msg.header.stamp)
                except Exception:
                    # print("ee")
                    self.get_logger().info(str(traceback.format_exc()))
                finally:
                    # print("Mutex released")
                    mutex.release()
            else:
                self.get_logger().warn(
                    "Dropping message for %s",
                    camera_frame_id,
                    throttle_duration_sec=1,
                )

        return callback

    def cropped_image_callback(self, debug=True):
        self.get_logger().info(str(self.active_templates))
        valid = True
        with mutex:
            images = {}
            camera_stamp_poses: Dict[Tuple[float, np.ndarray]] = {}
            for camera_frame_id, msg in self.latest_msgs.items():
                try:
                    img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                except CvBridgeError as e:
                    self.get_logger().error(str(e))
                    valid = False
                # CLAHE to L in LAB space
                # lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                # lab_img[:, :, 0] = self.clahe.apply(lab_img[:, :, 0])
                # img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

                # Contrast Normalization
                # img = cv2.normalize(
                #     img,
                #     None,
                #     alpha=0,
                #     beta=1.0,
                #     norm_type=cv2.NORM_MINMAX,
                #     dtype=cv2.CV_32F,
                # )
                # img = (255 * img).astype(np.uint8)

                images[camera_frame_id] = img
                try:
                    camera_tf = self.tf_buffer.lookup_transform(
                        self.map_ned_frame,
                        camera_frame_id,
                        # msg.header.stamp,
                        Time(seconds=0.0),
                        Duration(seconds=5.0),
                    )
                    camera_stamp_poses[camera_frame_id] = (
                        msg.header.stamp,
                        compose(
                            attrgetter("x", "y", "z")(camera_tf.transform.translation),
                            quat2mat(
                                attrgetter("w", "x", "y", "z")(
                                    camera_tf.transform.rotation
                                )
                            ),
                            np.ones(3),
                        ),
                    )
                except Exception as e:
                    valid = False
                    self.get_logger().error(str(e))
            self.latest_msgs = {}
        if not valid:
            return
        active_templates = copy.deepcopy(self.active_templates)

        for active_template in active_templates:
            # self.get_clock().sleep_for(Duration(seconds=0.01))
            template_name, camera_frame_id = active_template
            if camera_frame_id not in self.pose_estimator.available_cameras:
                self.get_logger().error(f"Camera {camera_frame_id} not registered")
                continue
            if camera_frame_id not in images.keys() or images[camera_frame_id] is None:
                self.get_logger().error(
                    f"Camera {camera_frame_id} image not received, trying to restart",
                    throttle_duration_sec=1.0,
                )
                try:
                    self.get_logger().info(f"Registering Camera...")
                    self.register_camera(
                        self.topics[camera_frame_id],
                        self.pose_estimator.cameras[camera_frame_id],
                    )
                    self.get_logger().info(f"Camera registered!")
                except Exception:
                    self.get_logger().info(traceback.format_exc())
                continue
            if template_name not in self.templates:
                self.get_logger().error(
                    f"Template {template_name} not registered",
                    throttle_duration_sec=1.0,
                )
                continue
            if (
                camera_frame_id not in camera_stamp_poses
                or len(camera_stamp_poses[camera_frame_id]) == 0
            ):
                self.get_logger().error(f"No camera poses found for {camera_frame_id}")
                continue
            _s = camera_stamp_poses[camera_frame_id][0].sec
            _ns = camera_stamp_poses[camera_frame_id][0].nanosec
            if _s == 0 and _ns == 0:
                self.get_logger().error(
                    f"Camera {camera_frame_id} has no timestamp, skipping"
                )
                continue
            self.get_logger().debug(
                f"Processing {template_name}<->{camera_frame_id}: {_s}.{_ns}",
                throttle_duration_sec=10,
            )
            template = self.templates[template_name]
            self.get_logger().info("Computing pose...")
            rot, trans = self.pose_estimator.compute_pose(
                images[camera_frame_id],
                template_name,
                camera_frame_id,
                matcher=template.matcher,
                num_keypoints=300,
                lxtyrxby=None,
                debug=True,
                is_planar=False,  # Use homography to do rejection
                max_reprojection_error=template.reprojection_error_threshold,
                min_matches=template.min_matches,
            )
            self.get_logger().info("Pose computed")
            self.get_logger().info(f"Rot: {rot}")
            self.get_logger().info(f"Trans: {trans}")
            if rot is not None and trans is not None and trans[2] > 0:
                yaw, pitch, roll = mat2euler(rot, axes="szyx")
                yaw = np.rad2deg(yaw)
                pitch = np.rad2deg(pitch)
                roll = np.rad2deg(roll)

                self.get_logger().info(
                    f"YPR: {yaw:.2f}, {pitch:.2f}, {roll:.2f} {template_name} {trans}",
                    throttle_duration_sec=1,
                )

                if np.all(np.abs(trans) < 1e-10) or np.all(
                    np.abs([yaw, pitch, roll]) < 1e-10
                ):
                    continue

                self.update_pose(
                    rot,
                    trans,
                    camera_frame_id,
                    template,
                    *camera_stamp_poses[camera_frame_id],
                    debug,
                )

    def update_pose(
        self,
        rot,
        trans,
        frame_id,
        template: Template,
        stamp,
        camera_pose,
        debug=False,
    ):
        """
        Updates the pose estimate of the template in the world frame
        params:
            rot: 3x3 rotation matrix
            trans: 3x1 translation vector
            frame_id: camera frame id
            template: Template object
            camera_pose: 4x4 homogeneous transformation matrix from
            world to camera frame
        """
        template_object = self.template_objects[template.object_name]
        object_trans = np.array(
            [
                trans[0] - template.offset[0],
                trans[1] - template.offset[1],
                trans[2],
            ]
        )
        tfm_camera_to_frame = np.eye(4)
        tfm_camera_to_frame[:3, :3] = rot
        tfm_camera_to_frame[:3, -1] = object_trans

        tfm_world_to_frame = camera_pose @ tfm_camera_to_frame

        try:
            T, R, _, _ = decompose(tfm_world_to_frame)
            object_quat = mat2quat(R)
        except Exception as e:
            self.get_logger().warn(f"Failed to decompose {e}", throttle_duration_sec=1)
            return

        x, y, z = T

        pose = np.array([x, y, z, *object_quat])
        if any(np.isnan(pose)) or any(np.isinf(pose)) or any(np.abs([x, y, z]) > 1000):
            self.get_logger().warn(
                f"Invalid pose estimate for {template.object_name} in {frame_id}: \
{x:.2f}, {y:.2f}, {z:.2f}, {object_quat}",
                throttle_duration_sec=1,
            )
            return

        qw, qx, qy, qz = object_quat
        template_object.poses.loc[len(template_object.poses)] = [
            stamp.sec,
            *pose,
        ]
        if len(template_object.poses) > template_object.min_buffer_size:
            old_rows = template_object.poses.iloc[
                : len(template_object.poses) - template_object.min_buffer_size
            ]
            keep_rows = template_object.poses.iloc[
                len(template_object.poses) - template_object.min_buffer_size :
            ]
            if template_object.max_history > 0:
                template_object.poses = pd.concat(
                    [
                        old_rows.loc[
                            (
                                old_rows.stamp < stamp.sec - template_object.max_history
                            ).index
                        ],
                        keep_rows,
                    ]
                )
            else:
                template_object.poses = pd.concat(
                    [
                        old_rows,
                        keep_rows,
                    ]
                )
            template_object.poses = template_object.poses.iloc[
                -template_object.max_buffer_size :
            ]
        template_object.poses.reset_index(drop=True, inplace=True)
        if self.debug:
            self.debug_file.write(
                f"{template_object.name}, {stamp}, {x}, {y}, \
{z}, {qw}, {qx}, {qy}, {qz}\n"
            )

        if len(template_object.poses) < template_object.min_buffer_size:
            return
        poses = template_object.poses.to_numpy()[:, 1:]
        fused_pose = get_kmeans_center(poses)

        fused_pose_ang = quat2euler(fused_pose[3:])
        _fused_pose = np.hstack([fused_pose[:3], fused_pose_ang])
        _poses = np.hstack(
            [poses[:, :3], np.array([quat2euler(q) for q in poses[:, 3:]])]
        )
        _err = _poses - _fused_pose
        variance = np.maximum(np.var(_err, 0), 0.00001)

        transform_stamped = TransformStamped()
        transform_stamped.header.stamp = stamp
        transform_stamped.header.frame_id = self.map_ned_frame
        transform_stamped.child_frame_id = template.object_name + "_optical"

        transform_stamped.transform.translation = Vector3(
            x=fused_pose[0], y=fused_pose[1], z=fused_pose[2]
        )
        qw, qx, qy, qz = fused_pose[3:]
        transform_stamped.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        self.br.sendTransform(transform_stamped)
        transform_zeroed = transform_stamped

        rangle = np.pi / 2
        r, p, y = quat2euler(fused_pose[3:], axes="rzyx")
        new_r, new_p, new_y = (
            r,
            np.round(p / rangle) * rangle,
            np.round(y / rangle) * rangle,
        )
        qw, qx, qy, qz = euler2quat(new_r, new_p, new_y, axes="rzyx")

        transform_zeroed.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        transform_zeroed.child_frame_id = template.object_name + "_stabilized"
        self.br.sendTransform(transform_zeroed)

        fused_pose_covariance_stamped = PoseWithCovarianceStamped()
        fused_pose_covariance_stamped.header.stamp = stamp
        fused_pose_covariance_stamped.header.frame_id = self.map_ned_frame
        fused_pose_covariance_stamped.pose.pose.position = Point(
            x=fused_pose[0], y=fused_pose[1], z=fused_pose[2]
        )
        fused_pose_covariance_stamped.pose.pose.orientation = (
            transform_zeroed.transform.rotation
        )
        fused_pose_covariance_stamped.pose.covariance = (
            np.diag(variance).flatten().tolist()
        )

        template_object.computed_pose = fused_pose_covariance_stamped

        self.get_logger().info(
            f"Published transform {template.object_name}_stabilized:\
                {transform_stamped.transform.translation}",
        )

        odometry = Odometry()
        odometry.header = fused_pose_covariance_stamped.header
        odometry.pose.pose = fused_pose_covariance_stamped.pose.pose
        odometry.child_frame_id = template.object_name + "_stabilized"
        odometry.pose = fused_pose_covariance_stamped.pose
        self.odom_pub.publish(odometry)


def main(args=None):
    logging.basicConfig(level=logging.INFO)
    rclpy.init(args=args)

    executor = MultiThreadedExecutor()
    pose_estimator = BasicPoseEstimator()
    executor.add_node(pose_estimator)
    tf2_ros.TransformListener(
        pose_estimator.tf_buffer, pose_estimator, spin_thread=False
    )

    rclpy.get_default_context().on_shutdown(pose_estimator.teardown)

    rclpy.spin(pose_estimator)


if __name__ == "__main__":
    main()
