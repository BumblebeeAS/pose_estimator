import rclpy
from foxglove_msgs.msg import ImageAnnotations, PointsAnnotation
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial import ConvexHull
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

        self.detections_sub = self.create_subscription(
            DetectionArray,
            input_detections_topic,
            self.detections_callback,
            qos_profile_sensor_data,
        )

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

    def detections_callback(self, detections_msg: DetectionArray):
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

        self.publish_transform(tvec, rvec, detections_msg.header, self.object_frame_id)


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
