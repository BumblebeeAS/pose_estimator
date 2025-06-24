from operator import attrgetter
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import shapely
import sympy
from cv2.typing import MatLike
from numpy.typing import ArrayLike
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


# Source: https://stackoverflow.com/a/74620309
def get_best_fit_ngon(points: np.ndarray, n: int = 4) -> np.ndarray:
    """Get the NumPy coordinate array forming the best fit convex n-gon
    for a collection of (unordered) points.

    Args:
        points (np.ndarray):  N x 2 array of points.
        n (int, optional): Number of sides of best-fit polygon. Defaults to 4.

    Raises:
        ValueError: If the best fit n-gon cannot be found.

    Returns:
        np.ndarray: N x 2 array of points representing the polygon.
    """
    hull = cv2.convexHull(points)
    hull = np.array(hull).reshape((len(hull), 2))
    hull = [sympy.Point(*pt) for pt in hull]

    # run until we cut down to n vertices
    while len(hull) > n:
        best_candidate = None

        # for all edges in hull ( <edge_idx_1>, <edge_idx_2> ) ->
        for edge_idx_1 in range(len(hull)):
            edge_idx_2 = (edge_idx_1 + 1) % len(hull)

            adj_idx_1 = (edge_idx_1 - 1) % len(hull)
            adj_idx_2 = (edge_idx_1 + 2) % len(hull)

            edge_pt_1 = sympy.Point(*hull[edge_idx_1])
            edge_pt_2 = sympy.Point(*hull[edge_idx_2])
            adj_pt_1 = sympy.Point(*hull[adj_idx_1])
            adj_pt_2 = sympy.Point(*hull[adj_idx_2])

            subpoly = sympy.Polygon(adj_pt_1, edge_pt_1, edge_pt_2, adj_pt_2)
            angle1 = subpoly.angles[edge_pt_1]
            angle2 = subpoly.angles[edge_pt_2]

            # we need to first make sure that the sum of the interior angles the edge
            # makes with the two adjacent edges is more than 180°
            if sympy.N(angle1 + angle2) <= sympy.pi:
                continue

            # find the new vertex if we delete this edge
            adj_edge_1 = sympy.Line(adj_pt_1, edge_pt_1)
            adj_edge_2 = sympy.Line(edge_pt_2, adj_pt_2)
            intersect = adj_edge_1.intersection(adj_edge_2)[0]

            # the area of the triangle we'll be adding
            area = sympy.N(sympy.Triangle(edge_pt_1, intersect, edge_pt_2).area)
            # should be the lowest
            if best_candidate and best_candidate[1] < area:
                continue

            # delete the edge and add the intersection of adjacent edges to the hull
            better_hull = list(hull)
            better_hull[edge_idx_1] = intersect
            del better_hull[edge_idx_2]
            best_candidate = (better_hull, area)

        if not best_candidate:
            raise ValueError("Could not find the best fit n-gon!")

        hull = best_candidate[0]

    hull = [(int(x), int(y)) for x, y in hull]
    hull = np.array(hull, dtype=np.float32)

    return hull


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
    mask_points = get_detection_polygon(detection)
    angle, obb_points = polygon_to_obb(mask_points)
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
