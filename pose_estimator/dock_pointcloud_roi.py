"""Point-cloud ROI filters used by dock pose estimation."""

import numpy as np


def crop_points_in_dock_boxes(
    points: np.ndarray,
    dock_position: np.ndarray,
    dock_yaw: float,
    box_centers: np.ndarray,
    box_size: float,
    padding: float,
) -> np.ndarray:
    """Keep points inside dock-aligned cubic crop boxes."""
    if len(points) == 0:
        return points

    delta = points - dock_position
    cosine = np.cos(dock_yaw)
    sine = np.sin(dock_yaw)
    local_points = np.column_stack(
        (
            cosine * delta[:, 0] + sine * delta[:, 1],
            -sine * delta[:, 0] + cosine * delta[:, 1],
            delta[:, 2],
        )
    )
    half_extent = box_size / 2.0 + padding
    inside = np.any(
        np.all(
            np.abs(local_points[:, np.newaxis, :] - box_centers) <= half_extent,
            axis=2,
        ),
        axis=1,
    )
    return points[inside]


def points_in_oriented_boxes(
    image_points: np.ndarray,
    boxes: np.ndarray,
    padding: float,
) -> np.ndarray:
    """Return mask for pixels inside any ``[cx, cy, width, height, theta]`` OBB."""
    inside = np.zeros(len(image_points), dtype=bool)
    for center_x, center_y, width, height, theta in boxes:
        delta_x = image_points[:, 0] - center_x
        delta_y = image_points[:, 1] - center_y
        cosine = np.cos(theta)
        sine = np.sin(theta)
        local_x = cosine * delta_x + sine * delta_y
        local_y = -sine * delta_x + cosine * delta_y
        inside |= (
            (np.abs(local_x) <= width / 2.0 + padding)
            & (np.abs(local_y) <= height / 2.0 + padding)
        )
    return inside
