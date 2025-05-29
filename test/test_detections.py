import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from pose_estimator.utils.detections import OBJECT_POINTS_DICT, match_polygon_points


def plot_quadrilaterals(ax: Axes, points: ArrayLike) -> None:
    for i in range(0, len(points), 4):
        quad = points[i : i + 4]
        quad = np.vstack([quad, quad[0]])  # Close the quadrilateral
        ax.plot(quad[:, 0], quad[:, 1])
        ax.plot(quad[:, 0], quad[:, 1], "bo")

    points = np.array(points)
    label_offset = (max(points[:, 0]) - min(points[:, 0])) * 0.02

    for i, (x, y) in enumerate(points, start=1):
        ax.text(x + label_offset, y, str(i), fontsize=12, color="black")

    ax.set_xlabel("X")
    ax.set_xlabel("Y")
    ax.grid(True)


object_points = np.concatenate(
    [OBJECT_POINTS_DICT["gate_center"], OBJECT_POINTS_DICT["gate_sides_left"]]
)
object_points_2d = object_points[:, :2]


def test_match_points(image_points: ArrayLike) -> None:
    matched_image_points_list, matched_object_points_list = [], []
    for i in range(0, len(object_points_2d), 4):
        matched_image_points, matched_object_points = match_polygon_points(
            image_points[i : i + 4], object_points_2d[i : i + 4]
        )
        matched_image_points_list.extend(matched_image_points)
        matched_object_points_list.extend(matched_object_points)

    _, axes = plt.subplots(1, 2, figsize=(8, 4))
    plot_quadrilaterals(axes[0], object_points_2d)
    plot_quadrilaterals(axes[1], matched_object_points_list)
    plt.show()

    _, axes = plt.subplots(1, 2, figsize=(8, 4))
    plot_quadrilaterals(axes[0], image_points)
    plot_quadrilaterals(axes[1], matched_image_points_list)
    plt.show()
