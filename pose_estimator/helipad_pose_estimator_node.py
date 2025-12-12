import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial import ConvexHull
from shapely.geometry import LineString, Polygon
from yolo_msgs.msg import DetectionArray

from pose_estimator.config.robotx26_object_points import HELIPAD_RADIUS
from pose_estimator.utils.detections import (
    filter_detections_by_num_points,
    get_best_detections_per_class,
    get_detection_polygon,
)
from pose_estimator.utils.pose_estimator import get_object_pose, refine_object_pose
from pose_estimator.utils.pose_estimator_node import PoseEstimatorNode


def sample_polygon_by_angle(coords: np.ndarray, N: int) -> np.ndarray:
    """Sample N boundary points of a possibly concave polygon defined
    by coords, equally spaced in angle.

    Args:
        coords (np.ndarray): Array of (x, y) vertices.
        N (int): number of angular samples.

    Returns:
        np.ndarray: list of N boundary points (x, y), equally spaced in angle

    Raises:
        ValueError: If the polygon is not valid.
        RuntimeError: If there is no intersection between ray and polygon.
    """
    poly = Polygon(coords)
    if not poly.is_valid:
        raise ValueError("Polygon is not valid (self-intersections, etc.)")

    # Guaranteed interior point, even for concave polygons
    center_geom = poly.representative_point()
    cx, cy = center_geom.x, center_geom.y

    # Big radius: bigger than polygon's bounding box diagonal
    minx, miny, maxx, maxy = poly.bounds
    R = 2.0 * np.hypot(maxx - minx, maxy - miny)

    boundary = poly.boundary
    points = []

    for k in range(N):
        theta = 2.0 * np.pi * k / N

        # Ray: center -> far point in direction theta
        far_x = cx + R * np.cos(theta)
        far_y = cy + R * np.sin(theta)
        ray = LineString([(cx, cy), (far_x, far_y)])

        intersection = boundary.intersection(ray)
        intersection_pts = np.array(intersection.coords)

        if len(intersection_pts) == 0:
            raise RuntimeError("Ray missed polygon; center may not be inside.")

        dists = np.linalg.norm(intersection_pts - np.array([cx, cy]), axis=1)
        nearest_pt = intersection_pts[np.argmin(dists)]
        points.append(nearest_pt)

    return np.array(points)


def match_circle_points(
    detection_polygon: np.ndarray, radius: float, num_points: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Match points on a detected circular helipad to 3D object points.

    Args:
        detection_polygon (np.ndarray): _description_
        radius (float): _description_
        num_points (int, optional): _description_. Defaults to 8.

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    # Sample points on the detected polygon boundary
    image_points = sample_polygon_by_angle(detection_polygon, num_points)

    # Define 3D object points on the helipad circle in its local frame
    angles = np.linspace(0.0, 2 * np.pi, num_points, endpoint=False)
    object_points = np.array(
        [[radius * np.cos(angle), radius * np.sin(angle), 0.0] for angle in angles]
    )

    return image_points, object_points


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

    def detections_callback(self, msg: DetectionArray):
        # We require at least 3 points for polygon creation
        filtered_detections = filter_detections_by_num_points(msg, 3)

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
        image_points, object_points = match_circle_points(
            detection_convex_hull_coords, radius=HELIPAD_RADIUS, num_points=8
        )

        self.get_logger().info(
            f"Object points:\n{object_points}\nImage points:\n{image_points}"
        )

        assert (
            object_points.shape[0] == image_points.shape[0]
        ), "Number of object points and image points must match"

        try:
            # We can set a low max re-projection error and refine pose even though
            # the segmentation mask is noisy because we only have 4 points

            rvec, tvec, inliers = get_object_pose(
                self.camera, object_points, image_points
            )

            if inliers is None:
                raise ValueError("No inliers found during pose estimation.")

            rvec, tvec = refine_object_pose(
                self.camera, object_points, image_points, rvec, tvec
            )

        except Exception as e:
            self.get_logger().warn(f"Pose estimation failed: {e}")
            return

        self.publish_transform(tvec, rvec, msg.header, self.object_frame_id)


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
