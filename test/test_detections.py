import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from pose_estimator.utils.detections import (
    OBJECT_POINTS_DICT,
    match_polygon_points,
    polygon_to_obb,
    rotate_polygon,
)


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


def format_plot_axis(ax: Axes) -> None:
    ax.invert_yaxis()
    ax.set_aspect("equal")


def test_obb(points):
    angle, obb = polygon_to_obb(points)

    obb = np.vstack((obb, obb[0]))  # Close the polygon for plotting
    points = np.vstack((points, points[0]))  # Close the polygon for plotting

    plt.plot(*points.T, "b-", label="Polygon Points")
    plt.plot(*obb.T, "r-", label="Minimum Rotated Rectangle")
    plt.gca().set_aspect("equal")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()

    print(angle)


def test_match_gate(image_points: ArrayLike) -> None:
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
    format_plot_axis(axes[0])
    format_plot_axis(axes[1])
    axes[0].set_title("Original order of object points")
    axes[1].set_title("Matched order of object points")
    plt.show()

    _, axes = plt.subplots(1, 2, figsize=(8, 4))
    plot_quadrilaterals(axes[0], image_points)
    plot_quadrilaterals(axes[1], matched_image_points_list)
    format_plot_axis(axes[0])
    format_plot_axis(axes[1])
    axes[0].set_title("Original order of image points")
    axes[1].set_title("Matched order of image points")
    plt.show()


def test_match_bin(image_points: ArrayLike) -> None:
    object_points = np.array(
        [[0.0, 0.0], [0.30479997, 0.0], [0.30479997, 0.60959995], [0.0, 0.60959995]]
    )

    # Match image points and object points
    angle, detected_points = polygon_to_obb(image_points)

    # Rotate object points before matching by point distances because bin yaw can
    # be large and we assume perspective does not change imaged aspect ratio by much.
    matched_object_points, matched_image_points = match_polygon_points(
        object_points, detected_points, A_angle=angle
    )

    # Rotate just for visualization
    rotated_object_points = rotate_polygon(matched_object_points, angle)

    _, axes = plt.subplots(1, 2, figsize=(8, 4))
    plot_quadrilaterals(axes[0], object_points)
    plot_quadrilaterals(axes[1], rotated_object_points)
    format_plot_axis(axes[0])
    format_plot_axis(axes[1])
    axes[0].set_title("Original order of object points")
    axes[1].set_title("Matched order of object points")
    plt.show()

    _, axes = plt.subplots(1, 2, figsize=(8, 4))
    plot_quadrilaterals(axes[0], detected_points)
    plot_quadrilaterals(axes[1], matched_image_points)
    format_plot_axis(axes[0])
    format_plot_axis(axes[1])
    axes[0].set_title("Original order of image points")
    axes[1].set_title("Matched order of image points")
    plt.show()
