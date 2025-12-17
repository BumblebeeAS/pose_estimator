import cv2
import rclpy
import tf2_ros
from foxglove_msgs.msg import ImageAnnotations, PointsAnnotation
from geometry_msgs.msg import Transform
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.wait_for_message import wait_for_message
from scipy.spatial import ConvexHull
from tf_transformations import (
    quaternion_inverse,
    quaternion_matrix,
    quaternion_multiply,
)
from yolo_msgs.msg import DetectionArray

from image_processing.utils.image_annotations import get_image_annotations
from pose_estimator.config.robotx26_object_points import HELIPAD_RADIUS
from pose_estimator.utils.circles import (
    fit_ellipse_RANSAC,
    get_ellipse_point_correspondences,
    get_ellipse_points,
)
from pose_estimator.utils.detections import (
    filter_detections_by_num_points,
    get_best_detections_per_class,
    get_detection_polygon,
)
from pose_estimator.utils.pose_estimator import get_object_pose
from pose_estimator.utils.pose_estimator_node import PoseEstimatorNode


class HelipadPoseEstimator(PoseEstimatorNode):
    def __init__(self):
        super().__init__("helipad_pose_estimator_node")

        self.object_frame_id = (
            self.declare_parameter("object_frame_id", "helipad")
            .get_parameter_value()
            .string_value
        )
        input_detections_topic = (
            self.declare_parameter("input_detections_topic", "yolo/detections")
            .get_parameter_value()
            .string_value
        )
        self.odom_ned_topic = (
            self.declare_parameter("odom_ned", "/uav2/odom_ned")
            .get_parameter_value()
            .string_value
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_timeout_sec = 1.0

        # Get base_link frame from odom_ned
        valid, odom_ned_msg = wait_for_message(Odometry, self, self.odom_ned_topic)
        if not valid:
            raise ValueError("Failed to get camera info")
        else:
            odom_ned_msg: Odometry
        self.base_link_frame = odom_ned_msg.child_frame_id
        self.get_logger().info(
            f"Determined base_link frame as {self.base_link_frame} from odom_ned message."
        )

        # Creation of subscribers must happen after wait_for_message above
        self.detections_sub = Subscriber(
            self,
            DetectionArray,
            input_detections_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.odom_ned_sub = Subscriber(
            self,
            Odometry,
            self.odom_ned_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.time_sync = ApproximateTimeSynchronizer(
            fs=[self.detections_sub, self.odom_ned_sub],
            queue_size=200,
            slop=0.1,
        )
        self.time_sync.registerCallback(self.detections_callback)

        # Debug annotations publisher
        self.ellipse_annotation_publisher = self.create_publisher(
            ImageAnnotations,
            "ellipse_annotation",
            qos_profile=qos_profile_sensor_data,
        )
        self.inlier_points_publisher = self.create_publisher(
            ImageAnnotations,
            "inlier_points_annotation",
            qos_profile=qos_profile_sensor_data,
        )

        self.is_setup = False

    def get_base_link_to_camera_transform(self) -> tuple[bool, Transform | None]:
        camera_frame_id = self.camera.frame_id
        self.get_logger().info(
            f"Looking up transform from {self.base_link_frame} to {camera_frame_id}"
        )
        if not self.tf_buffer.can_transform(
            self.base_link_frame, camera_frame_id, Time()
        ):
            return False, None

        tf = self.tf_buffer.lookup_transform(
            self.base_link_frame, camera_frame_id, Time()
        )
        return True, tf.transform

    def detections_callback(
        self, detections_msg: DetectionArray, odom_ned_msg: Odometry
    ):
        if not self.is_setup:
            self.get_logger().info("Setting up helipad pose estimator.")
            success, base_link_to_camera_tf = self.get_base_link_to_camera_transform()
            if not success:
                self.get_logger().warn(
                    "Failed to get base_link to camera transform. Cannot proceed with pose estimation."
                )
                return
            self.base_link_to_camera_tf = base_link_to_camera_tf
            self.is_setup = True
            self.get_logger().info("Successfully set up helipad pose estimator.")

        # We require at least 3 points for polygon creation
        filtered_detections = filter_detections_by_num_points(detections_msg, 3)

        # Only include relevant classes
        relevant_classes = ["helipad"]
        required_objects = 1

        best_detections = get_best_detections_per_class(
            filtered_detections, relevant_classes
        )
        num_detected_objects = len(best_detections.keys())
        if num_detected_objects < required_objects:
            self.get_logger().warn(
                f"""Insufficient detected objects.
                Received: {num_detected_objects}, require: {required_objects}."""
            )
            return

        # Match image points and object points
        detection_polygon = get_detection_polygon(best_detections["helipad"])
        detection_convex_hull = ConvexHull(detection_polygon)
        detection_convex_hull_coords = detection_polygon[detection_convex_hull.vertices]

        ellipse, inlier_points = fit_ellipse_RANSAC(
            detection_convex_hull_coords, num_iter=1000, thresh=5.0
        )

        (fit_cx, fit_cy), (fit_axis1, fit_axis2), fit_angle = ellipse
        fit_r1, fit_r2 = fit_axis1 / 2, fit_axis2 / 2
        image_points, object_points = get_ellipse_point_correspondences(
            fit_cx,
            fit_cy,
            fit_r1,
            fit_r2,
            fit_angle,
            object_semi_major=HELIPAD_RADIUS,
            object_semi_minor=HELIPAD_RADIUS,
        )

        # Publish debug annotations
        vis_points = get_ellipse_points(fit_cx, fit_cy, fit_r1, fit_r2, fit_angle, 20)
        ellipse_annotation = get_image_annotations(
            detections_msg.header, [[vis_points]]
        )
        self.ellipse_annotation_publisher.publish(ellipse_annotation)
        inlier_points_annotation = get_image_annotations(
            detections_msg.header,
            [[inlier_points]],
            points_annotation_type=PointsAnnotation.POINTS,
        )
        self.inlier_points_publisher.publish(inlier_points_annotation)

        self.get_logger().info(
            f"Object points:\n{object_points}\nImage points:\n{image_points}"
        )

        assert (
            object_points.shape[0] == image_points.shape[0]
        ), "Number of object points and image points must match"

        try:
            # We set a large max re-projection error as the segmentation masks are noisy and
            # the matched points for one detection may be at an offset from the matched points
            # for another. Setting a lower max re-projection error may result in a more confident
            # pose estimator, but it may also result in no inliers being found.
            rvec, tvec, inliers = get_object_pose(
                self.camera, object_points, image_points, max_reprojection_error=100
            )

            if inliers is None:
                raise ValueError("No inliers found during pose estimation.")

            # We do not refine the pose as the points are likely to have large re-projection error.

        except Exception as e:
            self.get_logger().warn(f"Pose estimation failed: {e}")
            return

        odom_to_base_link_rotation = odom_ned_msg.pose.pose.orientation
        base_link_to_camera_rotation = self.base_link_to_camera_tf.rotation
        odom_to_camera_rotation = quaternion_multiply(
            odom_to_base_link_rotation, base_link_to_camera_rotation
        )
        camera_to_odom_rotation = quaternion_inverse(odom_to_camera_rotation)
        R = quaternion_matrix(camera_to_odom_rotation)[:3, :3]
        camera_to_odom_rvec, _ = cv2.Rodrigues(R)

        self.publish_transform(
            tvec, camera_to_odom_rvec, detections_msg.header, self.object_frame_id
        )


def main(args=None):
    rclpy.init(args=args)
    node = HelipadPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
