import cv2
import message_filters
import numpy as np
import rclpy
import tf2_ros
from bb_perception_msgs.srv import TrashToggleFrame
from bb_perception_msgs.msg import DetectedObject3D, DetectedObject3DArray, Shape
from geometry_msgs.msg import Point32, Pose, PoseArray, PoseStamped
from frames.utils.transform_ros_msgs import compose_transforms, get_base_to_world_tf_stamped
from tf_transformations import quaternion_matrix
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.wait_for_message import wait_for_message
from yolo_msgs.msg import DetectionArray

from pose_estimator.utils.circles import fit_circle_RANSAC, get_circle_points
from pose_estimator.utils.detections import (
    get_detection_obb,
    get_detection_polygon,
    get_top_k_detections_per_class,
)
from pose_estimator.utils.pose_estimator_node import PoseEstimatorPosePubNode
from pose_estimator.utils.odometry_transforms import add_odometry_transform


def transform_matrix(transform):
    q, p = transform.rotation, transform.translation
    values = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
    if not np.isfinite(values).all() or np.linalg.norm(values[3:]) < 1e-6:
        raise ValueError("Invalid camera/pad transform")
    matrix = quaternion_matrix(values[3:])
    matrix[:3, 3] = values[:3]
    return matrix


def project_pixels_to_plane(pixels, camera, world_camera, plane):
    """Intersect calibrated rays with the frozen pad plane, in camera metres."""
    pixels = np.asarray(pixels, dtype=np.float32).reshape(-1, 2)
    if not len(pixels) or not np.isfinite(pixels).all():
        raise ValueError("Invalid mask contour")
    rays = np.column_stack((cv2.undistortPoints(
        pixels.reshape(-1, 1, 2), camera.camera_matrix(), camera.dist_coeffs()).reshape(-1, 2),
        np.ones(len(pixels))))
    point, normal = plane
    normal_camera = world_camera[:3, :3].T @ normal
    distance = normal @ (point - world_camera[:3, 3])
    denominator = rays @ normal_camera
    if np.any(np.abs(denominator) < 1e-6):
        raise ValueError("Mask ray parallel to pad plane")
    depths = distance / denominator
    if not np.isfinite(depths).all() or np.any(depths <= 0):
        raise ValueError("Pad plane is behind the camera")
    return rays * depths[:, None]


def project_mask_to_plane(pixels, camera, world_camera, plane):
    polygon = project_pixels_to_plane(pixels, camera, world_camera, plane)
    if len(polygon) < 3:
        raise ValueError("Mask requires at least three points")
    local = polygon - polygon[0]
    area = float(np.linalg.norm(np.cross(local, np.roll(local, -1, axis=0)).sum(axis=0)) / 2)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Invalid metric mask area")
    return polygon, area


def fit_tin_circle(pixels, camera, world_camera, plane):
    """Return a fitted centre and full metric footprint from a supported visible arc.

    Discard image-edge contour points, then fit in the pad plane. Require five
    RANSAC inliers spanning at least 120 degrees. A failed fit supplies no size
    evidence; the cluster median handles variation between accepted estimates.
    """
    pixels = np.asarray(pixels, dtype=np.float32).reshape(-1, 2)
    visible = np.all(pixels > 1, axis=1) & np.all(
        pixels < [camera.width-2, camera.height-2], axis=1)
    pixels = np.unique(pixels[visible], axis=0)
    if len(pixels) < 5:
        raise ValueError("Insufficient visible tin boundary")
    # Bound RANSAC cost without inventing samples on the cropped closing edge.
    if len(pixels) > 100:
        pixels = pixels[np.linspace(0, len(pixels)-1, 100).astype(int)]
    points = project_pixels_to_plane(pixels, camera, world_camera, plane)
    origin = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points-origin, full_matrices=False)
    axes = axes[:2]
    xy = (points-origin) @ axes.T
    scale = float(np.linalg.norm(np.ptp(xy, axis=0)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Degenerate tin boundary")
    xy /= scale
    # Express image quantisation tolerance in the same metric plane as the fit.
    centre_pixel = pixels.mean(axis=0)
    steps = project_pixels_to_plane(
        [centre_pixel, centre_pixel+[1,0], centre_pixel+[0,1]], camera, world_camera, plane)
    pixel_size = float(np.max(np.linalg.norm(steps[1:]-steps[0], axis=1))) / scale
    tolerance = min(1.5 * pixel_size, .1)
    circle, _ = fit_circle_RANSAC(xy, num_iter=100, thresh=tolerance)
    cx, cy, radius = circle
    if not np.isfinite(circle).all() or radius <= 0:
        raise ValueError("Invalid tin circle")
    errors = np.abs(np.linalg.norm(xy-[cx, cy], axis=1)-radius)
    inliers = xy[errors <= tolerance]
    if len(inliers) < 5:
        raise ValueError("Insufficient tin circle consensus")
    angles = np.sort(np.arctan2(inliers[:,1]-cy, inliers[:,0]-cx))
    coverage = 2*np.pi - np.max(np.diff(np.r_[angles, angles[0]+2*np.pi]))
    if coverage < np.deg2rad(120):
        raise ValueError("Visible tin arc is too short")
    centre = origin + (np.array([cx,cy])*scale) @ axes
    footprint = origin + (get_circle_points(cx,cy,radius,32)*scale) @ axes
    if np.any(footprint[:,2] <= 0) or not np.isfinite(footprint).all():
        raise ValueError("Invalid fitted tin circle")
    return centre, footprint


class TinsPoseEstimatorDepthFromOdom(PoseEstimatorPosePubNode):
    def __init__(self):
        super().__init__("tins_pose_estimator_depth_from_odom_node")

        input_track_detections_topic = (
            self.declare_parameter(
                "input_track_detections_topic", rclpy.Parameter.Type.STRING
            )
            .get_parameter_value()
            .string_value
        )
        self.input_odom_topic = (
            self.declare_parameter("input_odom_topic", rclpy.Parameter.Type.STRING)
            .get_parameter_value()
            .string_value
        )
        self.odom_frame = (
            self.declare_parameter("odom_frame", rclpy.Parameter.Type.STRING)
            .get_parameter_value()
            .string_value
        )
        time_sync_slop = (
            self.declare_parameter("time_sync_slop", 0.2)
            .get_parameter_value()
            .double_value
        )

        self.camera_frame = self.camera.frame_id

        track_detections_subscription = message_filters.Subscriber(
            self,
            DetectionArray,
            input_track_detections_topic,
            qos_profile=qos_profile_sensor_data,
        )
        odom_subscription = message_filters.Subscriber(
            self, Odometry, self.input_odom_topic, qos_profile=qos_profile_sensor_data
        )
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [track_detections_subscription, odom_subscription],
            20,
            slop=time_sync_slop,
        )
        self.time_synchronizer.registerCallback(self.detections_callback)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30))
        # Clustered anchors and camera extrinsics are static. Vehicle motion is
        # supplied by odometry rather than by the global dynamic TF stream.
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, static_only=True)
        odom_subscription.registerCallback(
            lambda msg: add_odometry_transform(self.tf_buffer, msg))

        self.toggle_frame_service = self.create_service(
            TrashToggleFrame,
            "toggle_frame_clustered",
            self.toggle_frame_callback,
        )
        self.is_enabled = False

        self._red_tins_publisher = self.create_publisher(
            PoseStamped, "red_tin_0/from_odom/pose", qos_profile_sensor_data
        )
        self._green_tins_publisher = self.create_publisher(
            PoseStamped, "green_tin_0/from_odom/pose", qos_profile_sensor_data
        )
        self._blue_tins_publisher = self.create_publisher(
            PoseStamped, "blue_tin_0/from_odom/pose", qos_profile_sensor_data
        )
        self.all_tins_publishers = {
            color: self.create_publisher(PoseArray, f"{color}_tins/from_odom/poses", qos_profile_sensor_data)
            for color in ("red", "green", "blue")
        }
        self.metric_tins_publishers = {
            color: self.create_publisher(DetectedObject3DArray, f"{color}_tins/from_odom/objects", qos_profile_sensor_data)
            for color in self.all_tins_publishers
        }
        self.largest_tins_publishers = {
            color: self.create_publisher(PoseStamped, f"{color}_tin_largest/from_odom/pose", qos_profile_sensor_data)
            for color in self.all_tins_publishers
        }

    @property
    def pose_publishers(self):
        return {
            "red_tin_0": self._red_tins_publisher,
            "green_tin_0": self._green_tins_publisher,
            "blue_tin_0": self._blue_tins_publisher,
        }

    def toggle_frame_callback(
        self, request: TrashToggleFrame.Request, response: TrashToggleFrame.Response
    ):
        if not request.enable:
            self.is_enabled = False
            response.success = True
            return response

        self.is_enabled = False
        self.trash_frame_clustered = request.trash_frame_clustered

        is_setup_success = self.setup()

        if not is_setup_success:
            message = "Failed to setup tins pose estimator from odom."
            self.get_logger().error(message)
            response.success = False
            response.message = message
            return response

        self.is_enabled = True
        response.success = True
        return response

    def setup(self) -> bool:
        valid, odom_msg = wait_for_message(Odometry, self, self.input_odom_topic)
        if not valid:
            self.get_logger().warning("Cannot freeze pad plane without odometry")
            return False
        odom_msg: Odometry

        try:
            pad = self.tf_buffer.lookup_transform(
                odom_msg.header.frame_id, self.trash_frame_clustered, Time())
            matrix = transform_matrix(pad.transform)
            self.pad_plane = (matrix[:3, 3], matrix[:3, 2])
            self.plane_frame = odom_msg.header.frame_id
        except (tf2_ros.TransformException, ValueError) as exc:
            self.get_logger().warning(f"Cannot freeze tins pad plane: {exc}")
            return False
        return True

    def detections_callback(
        self,
        track_detection_array_msg: DetectionArray,
        odom_msg: Odometry,
    ):
        if not self.is_enabled:
            return

        start_time = self.get_clock().now()

        try:
            if track_detection_array_msg.header.frame_id != self.camera_frame:
                raise ValueError("Detection frame differs from calibrated camera")
            if odom_msg.header.frame_id != self.plane_frame:
                raise ValueError("Odometry frame differs from frozen pad plane")
            extrinsic = self.tf_buffer.lookup_transform(
                odom_msg.child_frame_id, self.camera_frame, Time())
            world_camera = transform_matrix(compose_transforms(
                extrinsic.transform, get_base_to_world_tf_stamped(odom_msg).transform))
            # Reject unusable geometry before publishing any empty-view evidence.
            project_mask_to_plane([[self.camera.cx, self.camera.cy],
                                   [self.camera.cx+1, self.camera.cy],
                                   [self.camera.cx, self.camera.cy+1]],
                                  self.camera, world_camera, self.pad_plane)
        except (tf2_ros.TransformException, ValueError, cv2.error) as exc:
            self.get_logger().warning(f"Cannot project tins onto pad: {exc}", throttle_duration_sec=5.)
            return

        all_detections = get_top_k_detections_per_class(
            track_detection_array_msg,
            {f"{color}_tin": len(track_detection_array_msg.detections)
             for color in self.all_tins_publishers},
        )
        header = track_detection_array_msg.header

        # We are unable to estimate orientation, so we set it to identity
        rvec = np.zeros((3, 1), dtype=np.float32)

        for color, publisher in self.all_tins_publishers.items():
            poses = PoseArray(header=header)
            objects = DetectedObject3DArray(header=header, name=f"{color}_tin")
            objects.source.frame_id = header.frame_id
            objects.sensor_pose.orientation.w = 1.
            invalid_mask = False
            largest_pose, largest_area = None, 0.
            class_name = f"{color}_tin"
            for index, detection in enumerate(all_detections[class_name]):
                try:
                    _, obb = get_detection_obb(detection)
                    pixels = get_detection_polygon(detection)
                    polygon, area = project_mask_to_plane(
                        pixels, self.camera, world_camera, self.pad_plane)
                    centre_pixels = np.mean(obb, axis=0)
                    # Project the same OBB centre used by the legacy pose output.
                    X, Y, Z = project_pixels_to_plane(
                        [centre_pixels], self.camera, world_camera, self.pad_plane)[0]
                except (ValueError, cv2.error) as error:
                    self.get_logger().warning(f"Skipping invalid {class_name} mask: {error}",
                                              throttle_duration_sec=5.0)
                    invalid_mask = True
                    continue
                if not np.isfinite(obb).all() or cv2.contourArea(obb.astype(np.float32)) <= 0:
                    invalid_mask = True
                    continue
                footprint = None
                try:
                    centre, footprint = fit_tin_circle(
                        pixels, self.camera, world_camera, self.pad_plane)
                    X, Y, Z = centre
                    local = footprint-centre
                    area = float(np.linalg.norm(np.cross(local, np.roll(local,-1,axis=0)).sum(axis=0))/2)
                except (ValueError, np.linalg.LinAlgError, cv2.error):
                    # Keep the measured position/count when the arc cannot
                    # establish a reliable complete circle and size.
                    pass
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = float(X), float(Y), float(Z)
                pose.orientation.w = 1.  # No measured tin orientation.
                poses.poses.append(pose)
                obj = DetectedObject3D()
                obj.hypothesis.kinematics.header = header
                obj.hypothesis.kinematics.pose_with_covariance.pose = pose
                obj.hypothesis.probability = float(detection.score)
                obj.hypothesis.shape.category = Shape.POLYGON
                if footprint is not None:
                    obj.hypothesis.shape.footprint.points = [
                        Point32(x=float(p[0]-X), y=float(p[1]-Y), z=float(p[2]-Z)) for p in footprint]
                objects.objects.append(obj)
                if footprint is not None and area > largest_area:
                    largest_pose, largest_area = pose, area
                if index == 0:
                    tvec = np.array([X, Y, Z]).reshape((3, 1))
                    self.publish_data(tvec, rvec, obb, header, f"{class_name}_0")
            # Empty arrays distinguish a valid empty view from a missing feed.
            if not invalid_mask:
                publisher.publish(poses)
                self.metric_tins_publishers[color].publish(objects)
            if largest_pose is not None:
                self.largest_tins_publishers[color].publish(PoseStamped(header=header, pose=largest_pose))

        end_time = self.get_clock().now()
        elapsed_time = (end_time - start_time).nanoseconds / 1e6
        # self.get_logger().info(
        #     f"Processed {len(detection_array_msg.detections)} detections in {elapsed_time:.2f} ms"
        # )


def main(args=None):
    rclpy.init(args=args)
    node = TinsPoseEstimatorDepthFromOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
