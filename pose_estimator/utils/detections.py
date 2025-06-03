from operator import attrgetter
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import shapely
import sympy
from numpy.typing import ArrayLike
from yolo_msgs.msg import Detection, DetectionArray

# Order: top-left, bottom-left, bottom-right, top-right
# x-coordinates increase rightwards, y-coordinates increase downwards.
# Origin at the top-left corner of the gate.
_object_points_mm_dict = {
    "gate_sides_left": [
        (0, 152.40),
        (0, 152.40 + 1219.20),
        (76.20, 152.40 + 1219.20),
        (76.20, 152.40),
    ],
    "gate_sides_right": [
        (3048.00 - 76.20, 152.40),
        (3048.00 - 76.20, 152.40 + 1219.20),
        (3048.00, 152.40 + 1219.20),
        (3048.00, 152.4),
    ],
    "gate_center": [
        ((3048.00 - 50.80) / 2, 0),
        ((3048.00 - 50.80) / 2, 609.60),
        ((3048.00 + 50.80) / 2, 609.60),
        ((3048.00 + 50.80) / 2, 0),
    ],
}
OBJECT_POINTS_DICT = {}
for key, object_points_mm in _object_points_mm_dict.items():
    object_points = np.array(object_points_mm, dtype=np.float32) / 1000.0
    OBJECT_POINTS_DICT[key] = np.array(object_points, dtype=np.float32)

# Order: top-left, bottom-left, bottom-right, top-right
# x-coordinates increase rightwards, y-coordinates increase downwards.
# Origin at the top-left corner of the bin.
BIN_OBJECT_POINTS = np.array(
    [(0, 0), (0, 609.60), (304.80, 609.60), (304.80, 0)],
    dtype=np.float32,
)
BIN_OBJECT_POINTS = BIN_OBJECT_POINTS / 1000.0


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


def match_polygon_points(A: ArrayLike, B: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Match two polygons by ordering their points in a clockwise manner and
    finding the best rotation of B that minimizes the distance to A.

    Args:
        A (ArrayLike): Points of a polygon to match against.
        B (ArrayLike): Points of a polygon to be matched.

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

    # Try all permutations of B and find the one that matches A the best
    # by minimizing the sum of squared distances between corresponding points.
    A_norm = normalize_polygon(A_ordered)
    B_norm = normalize_polygon(B_ordered)

    min_cost = float("inf")
    best_permutation = np.arange(len(B))
    for i in range(len(A)):
        permutation = np.roll(best_permutation, i, axis=0)
        cost = np.sum(np.linalg.norm(A_norm - B_norm[permutation], axis=1) ** 2)
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


def get_normalized_coords_array(polygon: shapely.Polygon | Sequence) -> np.ndarray:
    """Get the normalized NumPy coordinates array of a polygon in
    strict canonical form.

    Args:
        array (shapely.Polygon | Sequence): The polygon to normalize. It can be a
        shapely.Polygon or a sequence of points (e.g., list of tuples).

    Returns:
        np.ndarray: Coordinates of the polygon in strict canonical form. The
        points are ordered anti-clockwise from top-left (because the y-coordinate
        increases downwards in the image coordinate system).
    """
    polygon = shapely.Polygon(polygon)
    polygon = polygon.normalize()

    # Exclude the the last point as `exterior.coords` repeats the starting point
    # at the end of the list
    coords_array = np.array(polygon.exterior.coords[:-1])

    return coords_array


def polygon_to_obb(points: np.ndarray) -> shapely.Polygon:
    """Compute the Oriented Bounding Box (OBB) from a NumPy coordinate array
    of a polygon.

    Args:
        points (np.ndarray): N x 2 array of points representing the polygon.

    Returns:
        shapely.Polygon: Oriented bounding box of the polygon.
    """
    polygon = shapely.Polygon(points)
    min_area_rect = polygon.minimum_rotated_rectangle

    return min_area_rect


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
        msg (DetectionArray): Array of Detections
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


def get_detection_centroid(detection: Detection) -> np.ndarray:
    mask = detection.mask
    mask_points = [attrgetter("x", "y")(point) for point in mask.data]
    return np.mean(mask_points, axis=0)


def get_detection_obb(detection: Detection) -> np.ndarray:
    mask = detection.mask
    mask_points = [attrgetter("x", "y")(point) for point in mask.data]
    mask_obb = polygon_to_obb(mask_points)
    obb_points = np.array(mask_obb.exterior.coords[:-1])
    return obb_points
