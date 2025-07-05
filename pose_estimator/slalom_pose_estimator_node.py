from typing import List

import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from foxglove_msgs.msg import ImageAnnotations
from message_filters import TimeSynchronizer
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from supervision.detection.utils import polygon_to_mask
from yolo_msgs.msg import Detection, DetectionArray

from pose_estimator.config.object_points import SLALOM_GATE_OBJECT_POINTS_DICT
from pose_estimator.utils.detections import (
    assign_to_centroids,
    filter_detections_by_num_points,
    get_detection_obb,
    get_object_depth,
    get_top_k_detections_per_class,
    match_polygon_points_sequence,
)
from pose_estimator.utils.image_annotations import get_image_annotations
from pose_estimator.utils.pose_estimator import get_object_pose
from pose_estimator.utils.pose_estimator_node import PoseEstimatorNode


class SlalomPoseEstimator(PoseEstimatorNode):
    def __init__(self):
        super().__init__("slalom_pose_estimator_node")

        input_detections_topic = (
            self.declare_parameter("input_detections_topic", "yolo/detections")
            .get_parameter_value()
            .string_value
        )
        input_depth_image_topic = (
            self.declare_parameter("input_depth_image_topic", "depth/image_raw")
            .get_parameter_value()
            .string_value
        )

        self.cv_bridge = CvBridge()
        detections_subscription = message_filters.Subscriber(
            self,
            DetectionArray,
            input_detections_topic,
            qos_profile=qos_profile_sensor_data,
        )
        depth_image_subscription = message_filters.Subscriber(
            self, Image, input_depth_image_topic, qos_profile=qos_profile_sensor_data
        )
        self.time_synchronizer = TimeSynchronizer(
            [detections_subscription, depth_image_subscription], 10
        )
        self.time_synchronizer.registerCallback(self.detections_callback)

        # Debug annotations publisher
        self.debug_annotations_publisher = self.create_publisher(
            ImageAnnotations,
            "debug_annotations",
            qos_profile=qos_profile_sensor_data,
        )

    def detections_callback(self, detections_msg: DetectionArray, depth_msg: Image):
        start_time = self.get_clock().now()

        # We require at least 3 points for polygon creation
        filtered_detections = filter_detections_by_num_points(detections_msg, 3)

        # Parse detections
        depth_img = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")

        def process_pole_detections(detections: List[Detection]):
            """Get pole OBBs and pole depths."""
            pole_obbs = [get_detection_obb(det)[1] for det in detections]
            resolution_wh_list = [
                (det.mask.width, det.mask.height) for det in detections
            ]
            pole_depths = [
                get_object_depth(depth_img, polygon_to_mask(obb, resolution_wh))
                for obb, resolution_wh in zip(pole_obbs, resolution_wh_list)
            ]
            return pole_obbs, pole_depths

        red_pole_detections = get_top_k_detections_per_class(
            filtered_detections, {"red_pole": 3}
        )["red_pole"]

        # Instead of keeping only the best 6 white poles, we keep the best 2 white poles in
        # each slalom gate layer. This is because we may see less than 3 layers and this method
        # is more robust to outliers (assuming red pole detections is reliable).
        white_pole_detections = get_top_k_detections_per_class(
            filtered_detections, {"white_pole": 100}
        )["white_pole"]

        num_red = len(red_pole_detections)
        num_white = len(white_pole_detections)
        if num_red == 0 or num_white == 0:
            self.get_logger().warn(
                f"""Insufficient detected objects.
                Received {num_red} red poles and {num_white} white poles.
                Require at least 1 red pole and 1 white pole."""
            )
            return

        red_pole_obbs, red_pole_depths = process_pole_detections(red_pole_detections)
        white_pole_obbs, white_pole_depths = process_pole_detections(
            white_pole_detections
        )
        white_pole_scores = [det.score for det in white_pole_detections]

        # Sort red poles by decreasing depth (nearest -> furthest)
        red_pole_depths, red_pole_obbs = zip(
            *sorted(
                zip(red_pole_depths, red_pole_obbs), key=lambda x: x[0], reverse=True
            )
        )

        # Assign white poles to red poles based on difference in depth
        white_pole_depths = np.array(white_pole_depths)[:, np.newaxis]
        red_pole_depths = np.array(red_pole_depths)[:, np.newaxis]
        white_pole_labels = assign_to_centroids(white_pole_depths, red_pole_depths)

        self.get_logger().info(f"red pole depths: {red_pole_depths}")
        self.get_logger().info(f"white pole depths: {white_pole_depths}")

        self.get_logger().info(f"Assigned {white_pole_labels}.")

        # Partition white poles into layers based on their assigned red pole
        white_pole_obb_score_pairs = list(zip(white_pole_obbs, white_pole_scores))
        white_pole_layers = {}
        for i, label in enumerate(white_pole_labels):
            if label not in white_pole_layers:
                white_pole_layers[label] = []
            white_pole_layers[label].append(white_pole_obb_score_pairs[i])

        # Select the best matching white poles for each red pole
        unique_white_pole_labels = np.unique(white_pole_labels)
        detected_polygons_dict = {label: [] for label in unique_white_pole_labels}
        object_polygons_dict = {label: [] for label in unique_white_pole_labels}
        for label, white_pole_obb_score_pairs in white_pole_layers.items():
            red_pole_obb = red_pole_obbs[label]
            red_pole_centroid = np.mean(red_pole_obb, axis=0)
            best_left_white_pole_obb = None
            best_right_white_pole_obb = None
            best_left_score = 0
            best_right_score = 0
            for white_pole_obb, white_pole_score in white_pole_obb_score_pairs:
                white_pole_centroid = np.mean(white_pole_obb, axis=0)
                if white_pole_centroid[0] < red_pole_centroid[0]:
                    if white_pole_score > best_left_score:
                        best_left_score = white_pole_score
                        best_left_white_pole_obb = white_pole_obb
                else:
                    if white_pole_score > best_right_score:
                        best_right_score = white_pole_score
                        best_right_white_pole_obb = white_pole_obb

            if best_left_white_pole_obb is None and best_right_white_pole_obb is None:
                self.get_logger().warn(
                    f"No valid white poles found for red pole {label}. "
                    "Skipping this layer."
                )
                continue

            if best_left_white_pole_obb is not None:
                detected_polygons_dict[label].append(best_left_white_pole_obb)
                object_polygons_dict[label].append(
                    SLALOM_GATE_OBJECT_POINTS_DICT["white_pole_left"]
                )

            detected_polygons_dict[label].append(red_pole_obb)
            object_polygons_dict[label].append(
                SLALOM_GATE_OBJECT_POINTS_DICT["red_pole"]
            )

            if best_right_white_pole_obb is not None:
                detected_polygons_dict[label].append(best_right_white_pole_obb)
                object_polygons_dict[label].append(
                    SLALOM_GATE_OBJECT_POINTS_DICT["white_pole_right"]
                )

        image_annotations = get_image_annotations(
            detections_msg.header, detected_polygons_dict.values()
        )
        self.debug_annotations_publisher.publish(image_annotations)

        # Estimate pose for each layer
        for layer in detected_polygons_dict.keys():
            detected_polygons = detected_polygons_dict[layer]
            object_polygons = object_polygons_dict[layer]

            object_points, image_points = match_polygon_points_sequence(
                object_polygons, detected_polygons
            )
            object_points = np.hstack(
                [object_points, np.zeros((object_points.shape[0], 1))]
            )

            assert (
                object_points.shape[0] == image_points.shape[0]
            ), "Number of object points and image points must match"

            try:
                # We set a large max re-projection error as the segmentation masks are noisy and
                # the matched points for one detection may be at an offset from the matched points
                # for another. Setting a lower max re-projection error may result in a more confident
                # pose estimator, but it may also result in no inliers being found.

                # However, the re-projection error cannot be too large here. Otherwise, the slalom
                # poses from multiple views may be too dispersed and different slalom layers may be
                # counted as one cluster

                rvec, tvec, inliers = get_object_pose(
                    self.camera, object_points, image_points, max_reprojection_error=10
                )

                if inliers is None:
                    raise ValueError("No inliers found during pose estimation.")

                # We do not refine the pose as the points are likely to have large re-projection error.

            except Exception as e:
                self.get_logger().warn(f"Pose estimation failed: {e}")
                continue

            self.publish_data(
                tvec,
                rvec,
                object_points,
                detections_msg.header,
                f"slalom_layer_{layer}",
            )

        end_time = self.get_clock().now()
        elapsed_time = (end_time - start_time).nanoseconds / 1e6
        self.get_logger().info(f"Processed detections in {elapsed_time:.2f} ms")


def main(args=None):
    rclpy.init(args=args)
    node = SlalomPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
