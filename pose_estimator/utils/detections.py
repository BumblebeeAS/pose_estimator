from operator import attrgetter
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import rclpy
import shapely
from cv2.typing import MatLike
from numpy.typing import ArrayLike
from rclpy.impl.rcutils_logger import RcutilsLogger
from yolo_msgs.msg import Detection, DetectionArray


def order_points_clockwise(pts: ArrayLike) -> np.ndarray:
    """Order points in a clockwise manner starting from the
    point with the smallest angle.

    Args:
        pts (ArrayLike): Exterior points of a polygon.

    Returns:
        np.ndarray: Exterior points of the polygon ordered in a clockwise
        manner.
    """
    pts = np.array(pts)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    return pts[np.argsort(angles)]


def normalize_polygon(pts: ArrayLike) -> np.ndarray:
    """Translate the polygon's centroid to the origin and scale the polygon so
    that the average distance from the centroid to the points is 1. This operation
    preserves the aspect ratio of the polygon.

    Args:
        pts (ArrayLike): Exterior points of a polygon.

    Raises:
        ValueError: If the average distance from the centroid to the points is zero.

    Returns:
        np.ndarray: Normalized polygon.
    """
    pts = np.array(pts).copy()
    centroid = np.mean(pts, axis=0)
    pts -= centroid
    avg_distance = np.mean(np.linalg.norm(pts, axis=1))

    if avg_distance == 0:
        raise ValueError("Cannot normalize a polygon with zero average distance.")

    pts /= avg_distance
    return pts


def rotate_polygon(pts: ArrayLike, angle: float) -> np.ndarray:
    """Rotate a polygon by a given angle around its centroid.

    Args:
        pts (ArrayLike): Exterior points of a polygon.
        angle (float): Angle in degrees to rotate the polygon anti-clockwise.

    Returns:
        np.ndarray: Rotated polygon.
    """
    pts = np.array(pts, dtype=np.float64).copy()
    centroid = np.mean(pts, axis=0)
    angle = np.deg2rad(angle)
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    pts -= centroid
    pts = pts @ rotation_matrix.T
    pts += centroid
    return pts


def match_polygon_points(
    A: ArrayLike, B: ArrayLike, A_angle: float = 0, B_angle: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Match two polygons by ordering their points in a clockwise manner and
    finding the best order of points of B that minimizes the normalized
    distance to A.

    Args:
        A (ArrayLike): Points of a polygon to match against.
        B (ArrayLike): Points of a polygon to be matched.
        A_angle (float, optional): Angle in degrees to rotate A anti-clockwise
        about its centroid before matching. Defaults to 0.
        B_angle (float, optional): Angle in degrees to rotate B anti-clockwise
        about its centroid before matching. Defaults to 0.

    Raises:
        AssertionError: If the number of points in A and B are not equal.
        ValueError: If the average distance from the centroid to the any set
        of points is zero.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Pairs of ordered points from A and B
        that minimize the distance between them.
    """
    assert len(A) == len(B), "Polygons must have the same number of points."

    A_ordered = order_points_clockwise(A)
    B_ordered = order_points_clockwise(B)

    A_norm = normalize_polygon(A_ordered)
    B_norm = normalize_polygon(B_ordered)

    A_rotated = rotate_polygon(A_norm, A_angle)
    B_rotated = rotate_polygon(B_norm, B_angle)

    # Try all permutations of B and find the one that matches A the best
    # by minimizing the sum of squared distances between corresponding points.
    min_cost = float("inf")
    best_permutation = np.arange(len(B))
    identity_permutation = np.arange(len(B))
    for i in range(len(A)):
        permutation = np.roll(identity_permutation, i, axis=0)
        cost = np.sum(np.linalg.norm(A_rotated - B_rotated[permutation], axis=1) ** 2)
        if cost < min_cost:
            min_cost = cost
            best_permutation = permutation

    return A_ordered, B_ordered[best_permutation]


def match_polygon_points_sequence(
    polygons_A: Sequence[ArrayLike],
    polygons_B: Sequence[ArrayLike],
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized version of `match_polygon_points` for matching multiple
    polygons."""
    matched_A, matched_B = [], []

    for A, B in zip(polygons_A, polygons_B):
        A, B = match_polygon_points(A, B)

        matched_A.extend(A)
        matched_B.extend(B)

    matched_A = np.array(matched_A)
    matched_B = np.array(matched_B)
    return matched_A, matched_B


def polygon_to_obb(points: ArrayLike) -> Tuple[float, np.ndarray]:
    """Compute the Oriented Bounding Box (OBB) from a NumPy coordinate array
    of a polygon. The angle returned is between the longer edge of the rectangle
    and the vertical, in the range (0, 180], increasing anti-clockwise.

    Args:
        points (np.ndarray): N x 2 array of points representing the polygon.

    Raises:
        ValueError: If there are fewer than 3 points.

    Returns:
        Tuple[float, np.ndarray]: Angle in degrees and OBB points.
    """
    if len(points) < 3:
        raise ValueError("At least 3 points are required to compute an OBB.")

    points = np.array(points, dtype=np.float32)
    rect = cv2.minAreaRect(points)
    _, (width, height), angle = rect
    box_points = cv2.boxPoints(rect)

    # NOTE: OpenCV’s angle can be confusing and the conventions differ between
    # OpenCV versions.

    # The following returns the angle between the longer edge of the rectangle
    # and the vertical, in the range (0, 180], increasing anti-clockwise.
    # Tested for OpenCV 4.10 and 4.11, MAY break for OpenCV 4.12+.
    # See: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga3d476a3417130ae5154aea421ca7ead9

    if width < height:
        corrected_angle = angle
    else:
        corrected_angle = angle + 90

    return corrected_angle, box_points


def get_best_fit_polygon(points: ArrayLike, n: int) -> np.ndarray:
    """Get the NumPy coordinate array forming the best-fit convex polygon
    for a collection of (unordered) points.

    This finds a polygon formed by a subset of points best approximating the
    boundary of the original set, which may not be a bounding polygon. For a
    bounding polygon with n sides, see https://stackoverflow.com/a/74620309.
    But this latter method may produce a polygon with corners far from the
    original points.

    Args:
        points (ArrayLike):  N x 2 array of points.
        n (int, optional): Number of sides of the best-fit polygon.

    Raises:
        ValueError: If n is less than 3 or if the number of points is less than n.
        ValueError: If the best-fit polygon cannot be found.

    Returns:
        np.ndarray: N x 2 array of points representing the polygon.
    """

    def get_triangle_area(a, b, c):
        return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))

    if n < 3 or len(points) < n:
        error_msg = f"""\
            Cannot find best-fit polygon with {n} sides for {len(points)} points.
            Polygons must have at least 3 points and the number of starting points must
            be greater than or equal to n."""
        one_line = " ".join(error_msg.split())
        raise ValueError(one_line)

    points = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(points)
    hull = np.array(hull).reshape((len(hull), 2)).astype(np.float32)

    while len(hull) > n:
        best_candidate = None

        for pt_idx_2 in range(len(hull)):
            pt_idx_1 = (pt_idx_2 - 1) % len(hull)
            pt_idx_3 = (pt_idx_2 + 1) % len(hull)

            # Remove the point that forms the triangle with the smallest area
            # with its two adjacent points.
            pt_1 = hull[pt_idx_1]
            pt_2 = hull[pt_idx_2]
            pt_3 = hull[pt_idx_3]
            area = get_triangle_area(pt_1, pt_2, pt_3)

            if best_candidate and best_candidate[1] < area:
                continue

            better_hull = np.delete(hull, pt_idx_2, axis=0)
            best_candidate = (better_hull, area)

        hull = best_candidate[0]

    return hull


def get_detection_best_fit_quad(
    detection: Detection, logger: RcutilsLogger
) -> np.ndarray:
    mask = detection.mask
    mask_points = [attrgetter("x", "y")(point) for point in mask.data]

    try:
        polygon_points = get_best_fit_polygon(mask_points, n=4)
    except ValueError as e:
        logger.warn(f"Failed to get best-fit polygon: {e}, using OBB instead")
        _, polygon_points = polygon_to_obb(mask_points)

    return polygon_points


def filter_detections_by_num_points(
    detection_array_msg: DetectionArray, min_num_points: int
) -> DetectionArray:
    """Filters detections based on the number of points in their masks.

    Args:
        detection_array_msg (DetectionArray): Array of Detections.
        min_num_points (int): Minimum number of points required in the mask.

    Returns:
        DetectionArray: Filtered array of detections.
    """
    filtered_detections = [
        detection
        for detection in detection_array_msg.detections
        if len(detection.mask.data) >= min_num_points
    ]
    filtered_detections_msg = DetectionArray(
        header=detection_array_msg.header, detections=filtered_detections
    )
    return filtered_detections_msg


def get_best_detections_per_class(
    detection_array_msg: DetectionArray, classes: List[str]
) -> Dict[str, Detection]:
    """Returns the best detection (by score) per class.

    Note: If there are no detections for a class,
    the returned dictionary would not contain an entry for that class.

    Args:
        detection_array_msg (DetectionArray): Array of Detections
        classes (List[str]): List of classes to get best detections for.

    Returns:
        Dict[str, Detection]: Dictionary of detections with class names as key.
    """
    best_detections = {}
    best_scores = {}
    classes = set(classes)
    for detection in detection_array_msg.detections:
        class_name = detection.class_name
        score = detection.score
        if class_name in classes and score > best_scores.get(class_name, -1):
            best_scores[class_name] = score
            best_detections[class_name] = detection
    return best_detections


def get_top_k_detections_per_class(
    detection_array_msg: DetectionArray, class_k_dict: Dict[str, int]
) -> Dict[str, List[Detection]]:
    """Returns the top k detections (by score) per class.

    Note: If there are no detections for a class,
    the returned dictionary would not contain an entry for that class.

    Args:
        detection_array_msg (DetectionArray): Array of Detections
        class_k_dict (Dict[str, int]): Dictionary mapping class names to k values.

    Returns:
        Dict[str, List[Detection]]: Dictionary of lists of detections with class names as key.
    """
    top_k_detections = {class_name: [] for class_name in class_k_dict.keys()}
    for detection in detection_array_msg.detections:
        class_name = detection.class_name
        if class_name in class_k_dict:
            top_k_detections[class_name].append(detection)

    for class_name in class_k_dict.keys():
        top_k_detections[class_name].sort(key=lambda x: x.score, reverse=True)
        k = class_k_dict[class_name]
        top_k_detections[class_name] = top_k_detections[class_name][:k]
    return top_k_detections


def get_detection_polygon(detection: Detection) -> np.ndarray:
    """Get the polygon points of a detection mask.

    Args:
        detection (Detection): Detection object containing the mask.

    Returns:
        np.ndarray: Array of points representing the polygon.
    """
    mask = detection.mask
    mask_points = [attrgetter("x", "y")(point) for point in mask.data]
    return np.array(mask_points, dtype=np.float32)


def get_detection_centroid(detection: Detection) -> np.ndarray:
    """Get the centroid of a detection using its mask.

    Args:
        detection (Detection): Detection object containing the mask.

    Returns:
        np.ndarray: Centroid coordinates as a NumPy array.
    """
    mask_polygon_points = get_detection_polygon(detection)

    # If there are fewer than 3 points, shapely.Polygon cannot be created.
    if len(mask_polygon_points) < 3:
        return mask_polygon_points.mean(axis=0)

    centroid = shapely.Polygon(mask_polygon_points).centroid
    return np.array([centroid.x, centroid.y], dtype=np.float32)


def get_detection_obb(detection: Detection) -> Tuple[float, np.ndarray]:
    """Get the Oriented Bounding Box (OBB) of a detection.

    Args:
        detection (Detection): Detection object containing the mask.

    Raises:
        ValueError: If the number of points in the mask is less than 3.

    Returns:
        Tuple[float, np.ndarray]: Angle in degrees and OBB points.
    """
    mask_polygon_points = get_detection_polygon(detection)
    angle, obb_points = polygon_to_obb(mask_polygon_points)
    return angle, obb_points


def assign_to_centroids(data: ArrayLike, centroids: ArrayLike) -> np.ndarray:
    """
    data: (n_samples, n_features)
    centroids: (k, n_features)
    returns: (n_samples,) array of cluster indices
    """
    data = np.asarray(data)
    centroids = np.asarray(centroids)
    dists = np.linalg.norm(data[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    return np.argmin(dists, axis=1)


def get_object_depth(depth_image: MatLike, object_mask: MatLike) -> float:
    """Get the median depth of an object in a depth image.

    Args:
        depth_image (MatLike): Depth image.
        object_mask (MatLike): Binary mask of the object.

    Returns:
        float: Median depth of the object.
    """
    # Resize mask to match depth image size
    # cv2.imshow("depth_image", depth_image)
    # cv2.imshow("object_mask", object_mask * 255)
    # cv2.waitKey(0)

    if depth_image.shape[:2] != object_mask.shape[:2]:
        object_mask = cv2.resize(
            object_mask, (depth_image.shape[1], depth_image.shape[0])
        )

    masked_depth = depth_image[object_mask > 0]

    if masked_depth.size == 0:
        return float("nan")

    return np.median(masked_depth)


def get_IoA(contained_polygon: np.ndarray, containing_polygon: np.ndarray) -> float:
    """Calculate the Intersection over Area (IoA) of two polygons.

    Args:
        contained_polygon (np.ndarray): Exterior points of the contained polygon.
        containing_polygon (np.ndarray): Exterior points of the containing polygon.

    Returns:
        float: IoA value.
    """
    containing_poly = shapely.Polygon(containing_polygon)
    contained_poly = shapely.Polygon(contained_polygon)

    if not containing_poly.is_valid or not contained_poly.is_valid:
        return 0.0

    intersection_area = containing_poly.intersection(contained_poly).area
    return intersection_area / contained_poly.area if contained_poly.area > 0 else 0.0


def get_detection_perimeter(detection: Detection) -> float:
    """Calculate the perimeter covered by the polygon of a Detection

    Args:
        detection (Detection): Incoming detection to get perimeter of

    Returns:
        float: Perimeter of the polygon of detection
    """
    detection_points = get_detection_polygon(detection)
    return shapely.length(shapely.Polygon(detection_points))

def is_in_polygon(detection: Detection, polygon: np.ndarray):
    detection_polygon = shapely.Polygon(get_detection_polygon(detection))
    containing_polygon = shapely.Polygon(polygon)

    return shapely.contains(containing_polygon, detection_polygon)