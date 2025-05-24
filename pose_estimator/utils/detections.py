from typing import Dict, List

import numpy as np
import shapely
from yolo_msgs.msg import Detection, DetectionArray

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


def polygon_to_obb(points: np.ndarray) -> np.ndarray:
    """Compute the Oriented Bounding Box (OBB) of a polygon in strict canonical form.

    Args:
        points (np.ndarray): N x 2 array of points representing the polygon.

    Returns:
        np.ndarray: Oriented bounding box represented as a 4 x 2 array of points.
        The points are ordered from top-left to top-right in an anti-clockwise
        manner. Note the vertical direction is different since the y-coordinate
        increases downwards in the image coordinate system.
    """
    polygon = shapely.Polygon(points)
    min_area_rect = polygon.minimum_rotated_rectangle.normalize()

    # Exclude the the last point as `exterior.coords` repeats the starting point
    # at the end of the list
    coords_array = np.array(min_area_rect.exterior.coords[:4])

    return coords_array


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
