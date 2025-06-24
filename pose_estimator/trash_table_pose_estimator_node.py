from typing import Dict, List

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from yolo_msgs.msg import Detection, DetectionArray

from pose_estimator.config.object_points import TRASH_TABLE_OBJECT_POINTS
from pose_estimator.utils.detections import (
    filter_detections_by_num_points,
    get_best_detections_per_class,
    get_detection_centroid,
    get_detection_obb,
    match_polygon_points,
    order_points_clockwise,
)
from pose_estimator.utils.pose_estimator import get_object_pose
from pose_estimator.utils.pose_estimator_node import PoseEstimatorNode


def match_bucket_points(table_detection: Detection, bucket_detections: List[Detection]):
    table_centroid = get_detection_centroid(table_detection)
    object_points = []
    image_points = []

    for bucket_detection in bucket_detections:
        object_polygon = TRASH_TABLE_OBJECT_POINTS[bucket_detection.class_name]
        angle, detected_polygon = get_detection_obb(bucket_detection)

        # Match bucket points by least squares method (using their aspect ratio).
        bucket_object_points, bucket_image_points = match_polygon_points(
            object_polygon, detected_polygon, A_angle=angle
        )

        # This leaves a 180 degree ambiguity, where either the 1st or 3rd point in
        # the list is the correct starting point. For the yellow bucket, we select
        # from the 1st and 3rd points the point further from the table as the starting
        # point. For the pink bucket, we pick the point closer to the table.
        dist_0 = np.linalg.norm(bucket_image_points[0] - table_centroid)
        dist_2 = np.linalg.norm(bucket_image_points[2] - table_centroid)
        if bucket_detection.class_name == "yellow_bucket" and dist_2 > dist_0:
            bucket_image_points = np.roll(bucket_image_points, -2, axis=0)
        elif bucket_detection.class_name == "pink_bucket" and dist_2 < dist_0:
            bucket_image_points = np.roll(bucket_image_points, -2, axis=0)

        object_points.extend(bucket_object_points)
        image_points.extend(bucket_image_points)

    return object_points, image_points


def match_table_points(
    table_detection: Detection, bucket_detections_dict: Dict[str, Detection]
):
    table_object_points = TRASH_TABLE_OBJECT_POINTS["table"]
    table_object_points = order_points_clockwise(table_object_points)
    table_detected_polygon = get_detection_obb(table_detection)[1]
    table_detected_polygon = order_points_clockwise(table_detected_polygon)

    # The table is square so there is a 90 degree ambiguity in the starting point.
    # At this line, we are guaranteed to have at least one bucket detection.

    # If the yellow bucket is present, the starting point is one of the two points
    # closest to the yellow bucket. Of these two, we choose the point such that the
    # next point in the sequence (with loop around) is further from the yellow bucket.
    if "yellow_bucket" in bucket_detections_dict:
        yellow_bucket_detection = bucket_detections_dict["yellow_bucket"]
        yellow_bucket_centroid = get_detection_centroid(yellow_bucket_detection)

        norms = np.linalg.norm(table_detected_polygon - yellow_bucket_centroid, axis=1)
        closest_points = np.argsort(norms)[:2]
        next_points = (closest_points + 1) % 4
        if norms[next_points[0]] > norms[next_points[1]]:
            table_detected_polygon = np.roll(
                table_detected_polygon, -closest_points[0], axis=0
            )
        else:
            table_detected_polygon = np.roll(
                table_detected_polygon, -closest_points[1], axis=0
            )

    # Otherwise, the pink bucket is present. The starting point is one of the two points
    # furthest from the pink bucket. Of these two, we choose the point such that the
    # next point in the sequence (with loop around) is closer to the pink bucket.
    else:
        pink_bucket_detection = bucket_detections_dict["pink_bucket"]
        pink_bucket_centroid = get_detection_centroid(pink_bucket_detection)

        norms = np.linalg.norm(table_detected_polygon - pink_bucket_centroid, axis=1)
        furthest_points = np.argsort(norms)[-2:]
        next_points = (furthest_points + 1) % 4
        if norms[next_points[0]] < norms[next_points[1]]:
            table_detected_polygon = np.roll(
                table_detected_polygon, -furthest_points[0], axis=0
            )
        else:
            table_detected_polygon = np.roll(
                table_detected_polygon, -furthest_points[1], axis=0
            )

    return table_object_points, table_detected_polygon


class TrashTablePoseEstimator(PoseEstimatorNode):
    def __init__(self):
        super().__init__("trash_table_pose_estimator_node")

        self.object_frame_id = (
            self.declare_parameter("object_frame_id", "table/center")
            .get_parameter_value()
            .string_value
        )
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

    def detections_callback(self, msg: DetectionArray):
        # We require at least 3 points for polygon creation
        filtered_detections = filter_detections_by_num_points(msg, 3)

        # Get required detections
        table_detection_dict = get_best_detections_per_class(
            filtered_detections, ["table"]
        )
        bucket_detections_dict = get_best_detections_per_class(
            filtered_detections, ["yellow_bucket", "pink_bucket"]
        )
        if len(table_detection_dict) == 0 or len(bucket_detections_dict) == 0:
            self.get_logger().warn(
                f"""Insufficient detected objects. Require 1 table and at least 1 bucket detection.
                Found {len(table_detection_dict)} tables and {len(bucket_detections_dict)} buckets."""
            )
            return

        table_detection = table_detection_dict["table"]
        bucket_detections = list(bucket_detections_dict.values())

        # Match image points and object points
        # From left: yellow bucket, table, pink bucket.
        bucket_object_points, bucket_image_points = match_bucket_points(
            table_detection, bucket_detections
        )
        table_object_points, table_detected_polygon = match_table_points(
            table_detection, bucket_detections_dict
        )

        object_points = bucket_object_points + list(table_object_points)
        image_points = bucket_image_points + list(table_detected_polygon)

        object_points = np.array(object_points)
        image_points = np.array(image_points)
        object_points = np.hstack(
            [object_points, np.zeros((object_points.shape[0], 1))]
        )

        assert (
            object_points.shape[0] == image_points.shape[0]
        ), "Number of object points and image points must match"

        try:
            # We set a large max re-projection error as the segmentation masks are noisy and
            # the matched points for one detection may be at an offset from the matched points
            # for another. Settiall_object_pointsng a lower max re-projection error may result in a more confident
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

        self.publish_data(tvec, rvec, object_points, msg.header, self.object_frame_id)


def main(args=None):
    rclpy.init(args=args)
    node = TrashTablePoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
