from typing import Dict, List, Sequence

import cv2
import numpy as np
import shapely
import sympy
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
    object_points_2d = np.array(object_points_mm, dtype=np.float32) / 1000.0
    object_points = np.hstack(
        [object_points_2d, np.zeros((object_points_2d.shape[0], 1))]
    )
    OBJECT_POINTS_DICT[key] = np.array(object_points, dtype=np.float32)

BIN_OBJECT_POINTS = np.array(
    [(0, 0), (0, 609.60), (304.80, 609.60), (304.80, 0)],
    dtype=np.float32,
)


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
