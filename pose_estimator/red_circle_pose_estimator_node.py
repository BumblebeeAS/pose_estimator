import rclpy
from foxglove_msgs.msg import ImageAnnotations, PointsAnnotation
from geometry_msgs.msg import PoseStamped
from image_processing.utils.image_annotations import get_image_annotations
from rclpy.publisher import Publisher
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial import ConvexHull
from yolo_msgs.msg import DetectionArray

from pose_estimator.config.object_points import RED_CIRCLE_RADII
from pose_estimator.utils.circles import (
    fit_ellipse_RANSAC,
    get_ellipse_point_correspondences,
    get_ellipse_points,
)
from pose_estimator.utils.detections import (
    filter_detections_by_num_points,
    get_detection_polygon,
    get_top_k_detections_per_class,
)
from pose_estimator.utils.pose_estimator import get_object_pose
from pose_estimator.utils.pose_estimator_node import PoseEstimatorPosePubNode

RED_CIRCLE_ANNOTATION_COLORS = {
    "small": "#3cb44b",
    "large": "#e6194b",
}


class RedCirclePoseEstimator(PoseEstimatorPosePubNode):
    def __init__(self):
        super().__init__("red_circle_pose_estimator_node")

        self.small_hole_frame_id = (
            self.declare_parameter("small_hole_frame_id", "torpedo_small_hole")
            .get_parameter_value()
            .string_value
        )
        self.large_hole_frame_id = (
            self.declare_parameter("large_hole_frame_id", "torpedo_large_hole")
            .get_parameter_value()
            .string_value
        )
        # Used when only one red_circle is detected. The nav stack sets this
        # via `ros2 param set ... target_hole small|large`.
        self.declare_parameter("target_hole", "large")

        detections_topic = (
            self.declare_parameter("input_detections_topic", "yolo/detections")
            .get_parameter_value()
            .string_value
        )

        self.detections_sub = self.create_subscription(
            DetectionArray,
            detections_topic,
            self.detections_callback,
            qos_profile_sensor_data,
        )

        self._small_pose_publisher = self.create_publisher(
            PoseStamped,
            f"{self.small_hole_frame_id}/pose",
            qos_profile_sensor_data,
        )
        self._large_pose_publisher = self.create_publisher(
            PoseStamped,
            f"{self.large_hole_frame_id}/pose",
            qos_profile_sensor_data,
        )

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

        self._frame_id_by_label = {
            "small": self.small_hole_frame_id,
            "large": self.large_hole_frame_id,
        }

        self.get_logger().info("RedCirclePoseEstimator node initialized.")

    @property
    def pose_publishers(self) -> dict[str, Publisher]:
        return {
            self.small_hole_frame_id: self._small_pose_publisher,
            self.large_hole_frame_id: self._large_pose_publisher,
        }

    def detections_callback(self, msg: DetectionArray):
        # Need >=5 points to fit an ellipse.
        filtered_detections = filter_detections_by_num_points(msg, 5)

        top_detections = get_top_k_detections_per_class(
            filtered_detections, {"red_circle": 2}
        ).get("red_circle", [])

        # Fit an ellipse to each retained detection.
        fits = []  # list of (ellipse, inlier_points, max_axis)
        for detection in top_detections:
            detection_polygon = get_detection_polygon(detection)
            if len(detection_polygon) < 5:
                continue
            try:
                convex_hull_idxs = ConvexHull(detection_polygon).vertices
            except Exception as e:
                self.get_logger().warn(f"ConvexHull failed: {e}")
                continue
            convex_hull_coords = detection_polygon[convex_hull_idxs]
            if len(convex_hull_coords) < 5:
                self.get_logger().warn(
                    f"Insufficient convex hull points: {len(convex_hull_coords)} < 5."
                )
                continue

            try:
                ellipse, inlier_points = fit_ellipse_RANSAC(
                    convex_hull_coords, num_iter=200, thresh=5.0
                )
            except ValueError as e:
                self.get_logger().warn(f"Ellipse fit failed: {e}")
                continue

            (_, _), (axis1, axis2), _ = ellipse
            max_axis = max(axis1, axis2)
            fits.append((ellipse, inlier_points, max_axis))

        if not fits:
            self.get_logger().warn("No red_circle detections produced an ellipse fit.")
            return

        # Assign size labels.
        if len(fits) >= 2:
            fits.sort(key=lambda f: f[2])
            labelled = [("small", fits[0]), ("large", fits[1])]
        else:
            target = self.get_parameter("target_hole").get_parameter_value().string_value
            if target not in RED_CIRCLE_RADII:
                self.get_logger().warn(
                    f"Invalid target_hole parameter '{target}'; expected 'small' or 'large'."
                )
                return
            labelled = [(target, fits[0])]

        # Per-circle PnP + publish.
        vis_point_sets = []
        inlier_point_sets = []
        annotation_colors = []
        for label, (ellipse, inlier_points, _) in labelled:
            (fit_cx, fit_cy), (fit_axis1, fit_axis2), fit_angle = ellipse
            object_radius = RED_CIRCLE_RADII[label]

            image_points, object_points = get_ellipse_point_correspondences(
                fit_cx,
                fit_cy,
                inlier_points,
                object_semi_major=object_radius,
                object_semi_minor=object_radius,
            )

            try:
                rvec, tvec, inliers = get_object_pose(
                    self.camera,
                    object_points,
                    image_points,
                    max_reprojection_error=100,
                )
                if inliers is None:
                    raise ValueError("No inliers found during pose estimation.")
            except Exception as e:
                self.get_logger().warn(f"Pose estimation failed for {label}: {e}")
                continue

            self.publish_data(
                tvec, rvec, object_points, msg.header, self._frame_id_by_label[label]
            )

            vis_points = get_ellipse_points(
                fit_cx, fit_cy, fit_axis1 / 2, fit_axis2 / 2, fit_angle, 20
            )
            vis_point_sets.append([vis_points])
            inlier_point_sets.append([inlier_points])
            annotation_colors.append(RED_CIRCLE_ANNOTATION_COLORS[label])

        if vis_point_sets:
            self.ellipse_annotation_publisher.publish(
                get_image_annotations(
                    msg.header, vis_point_sets, colors=annotation_colors
                )
            )
            self.inlier_points_publisher.publish(
                get_image_annotations(
                    msg.header,
                    inlier_point_sets,
                    colors=annotation_colors,
                    points_annotation_type=PointsAnnotation.POINTS,
                )
            )


def main(args=None):
    rclpy.init(args=args)
    node = RedCirclePoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
