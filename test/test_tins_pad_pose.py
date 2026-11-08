"""Recover the whole platform frame from non-concentric colored circles."""

import cv2
import numpy as np
import pytest

from pose_estimator.utils.PinholeCamera import PinholeCamera
from pose_estimator.utils.tins_pad import estimate_tins_pad_pose

CENTERS = {
    "red_circle": [-0.5, -0.35],
    "green_circle": [0.5, -0.35],
    "blue_circle": [0.0, 0.5],
}


def projected_contours(camera, rotation, translation, centers=CENTERS, radius=0.35):
    angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    contours = {}
    for label, (x, y) in centers.items():
        points = np.column_stack([
            x + radius * np.cos(angles), y + radius * np.sin(angles), np.zeros_like(angles)
        ])
        pixels, _ = cv2.projectPoints(
            points, cv2.Rodrigues(rotation)[0], translation,
            camera.camera_matrix(), camera.dist_coeffs(),
        )
        contours[label] = pixels[:, 0]
    return contours


@pytest.mark.parametrize("yaw, tilt, depth", [
    (0.0, [0.0, 0.0, 0.0], 4.0),
    (0.9, [0.35, -0.2, 0.1], 4.0),
    (-1.8, [-0.5, 0.3, -0.2], 8.0),
])
@pytest.mark.parametrize("visible_labels", [
    tuple(CENTERS),
    ("red_circle", "green_circle"),
    ("red_circle", "blue_circle"),
    ("green_circle", "blue_circle"),
])
def test_platform_center_and_circle_children(yaw, tilt, depth, visible_labels):
    camera = PinholeCamera("camera", 1280, 720, 900, 700, 640, 360, 0.03, -0.01, 0, 0, 0)
    level = cv2.Rodrigues(np.asarray(tilt, dtype=float))[0]
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = level @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    translation = np.array([0.3, -0.2, depth])
    contours = projected_contours(camera, rotation, translation)
    contours = {label: contours[label] for label in visible_labels}
    np.random.seed(0)
    rvec, tvec, error = estimate_tins_pad_pose(camera, contours, CENTERS, 0.35, level)
    fitted_rotation = cv2.Rodrigues(rvec)[0]
    np.testing.assert_allclose(tvec.ravel(), translation, atol=0.002)
    np.testing.assert_allclose(fitted_rotation, rotation, atol=0.002)
    assert error < 0.1
    # Compose root -> circle: the fixed offsets must recover each true center.
    for xy in CENTERS.values():
        offset = np.array([*xy, 0.0])
        np.testing.assert_allclose(
            fitted_rotation @ offset + tvec.ravel(), rotation @ offset + translation,
            atol=0.003,
        )


def test_incompatible_circle_geometry_is_rejected():
    camera = PinholeCamera("camera", 1280, 720, 800, 800, 640, 360)
    contours = projected_contours(camera, np.eye(3), np.array([0.0, 0.0, 4.0]))
    contours["blue_circle"] += np.array([200.0, 50.0])
    np.random.seed(0)
    with pytest.raises(ValueError, match="consensus"):
        estimate_tins_pad_pose(camera, contours, CENTERS, 0.35, np.eye(3))


def test_degenerate_circle_is_rejected():
    camera = PinholeCamera("camera", 1280, 720, 800, 800, 640, 360)
    contours = {label: np.column_stack([np.arange(10), np.zeros(10)]) for label in CENTERS}
    with pytest.raises(ValueError, match="Degenerate"):
        estimate_tins_pad_pose(camera, contours, CENTERS, 0.35, np.eye(3))


@pytest.mark.parametrize("visible_labels", [(), ("red_circle",)])
def test_fewer_than_two_circles_is_rejected(visible_labels):
    camera = PinholeCamera("camera", 1280, 720, 800, 800, 640, 360)
    contours = projected_contours(camera, np.eye(3), np.array([0.0, 0.0, 4.0]))
    with pytest.raises(ValueError, match="At least two"):
        estimate_tins_pad_pose(
            camera, {label: contours[label] for label in visible_labels},
            CENTERS, 0.35, np.eye(3),
        )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("inlier_threshold", [3.0, 8.0])
@pytest.mark.parametrize("clip_fraction", [0.2, 0.4])
@pytest.mark.parametrize("visible_labels", [
    ("red_circle", "green_circle"),
    ("red_circle", "blue_circle"),
    ("green_circle", "blue_circle"),
])
def test_clipped_sides_and_noisy_arcs(seed, inlier_threshold, clip_fraction, visible_labels):
    camera = PinholeCamera("camera", 1280, 720, 900, 700, 640, 360)
    level = cv2.Rodrigues(np.array([0.25, -0.15, 0.1]))[0]
    yaw = 0.8
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = level @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    translation = np.array([0.3, -0.2, 4.0])
    contours = projected_contours(camera, rotation, translation)
    rng = np.random.default_rng(seed)
    clipped = {}
    for label in visible_labels:
        points = contours[label].copy()
        # Clamp a closed segmentation contour at a straight clipping boundary.
        # This retains artificial straight-side points, not just a clean arc.
        boundary = points[:, 0].min() + clip_fraction * np.ptp(points[:, 0])
        points[:, 0] = np.maximum(points[:, 0], boundary)
        points += rng.normal(0, 0.2, points.shape)
        clipped[label] = points
    np.random.seed(seed)
    rvec, tvec, error = estimate_tins_pad_pose(
        camera, clipped, CENTERS, 0.35, level, inlier_threshold=inlier_threshold)
    np.testing.assert_allclose(tvec.ravel(), translation, atol=0.04)
    np.testing.assert_allclose(cv2.Rodrigues(rvec)[0], rotation, atol=0.025)
    # A wider consensus includes larger residuals; both settings must still
    # retain low synthetic residuals and the same pose accuracy bounds.
    assert error < (1.0 if inlier_threshold == 3.0 else 5.0)


@pytest.mark.parametrize("label", list(CENTERS))
@pytest.mark.parametrize("tilt,yaw", [([0., 0., 0.], 0.), ([.35, -.2, .1], .9)])
def test_single_circle_translation_with_retained_orientation(label, tilt, yaw):
    camera = PinholeCamera("camera", 1280, 720, 900, 700, 640, 360)
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = cv2.Rodrigues(np.asarray(tilt))[0] @ np.array([[c,-s,0], [s,c,0], [0,0,1]])
    translation = np.array([.3, -.2, 4.])
    contours = projected_contours(camera, rotation, translation)
    np.random.seed(0)
    rvec, tvec, error = estimate_tins_pad_pose(
        camera, {label:contours[label]}, CENTERS, .35, rotation, fixed_yaw=0.)
    np.testing.assert_allclose(tvec.ravel(), translation, atol=.002)
    np.testing.assert_allclose(cv2.Rodrigues(rvec)[0], rotation, atol=1e-10)
    assert error < .1


def test_multiple_circles_cannot_change_retained_orientation():
    camera = PinholeCamera("camera", 1280, 720, 800, 800, 640, 360)
    rotation = cv2.Rodrigues(np.array([.1, -.15, .8]))[0]
    translation = np.array([.3,-.2,4.])
    contours = projected_contours(camera, rotation, translation)
    # Small independent contour noise must change translation, not orientation.
    rng = np.random.default_rng(4)
    contours = {k:v+rng.normal(0,.2,v.shape) for k,v in contours.items()}
    np.random.seed(0)
    rvec, tvec, _ = estimate_tins_pad_pose(
        camera, contours, CENTERS, .35, rotation, fixed_yaw=0.)
    np.testing.assert_allclose(cv2.Rodrigues(rvec)[0], rotation, atol=1e-10)
    np.testing.assert_allclose(tvec.ravel(), translation, atol=.01)


def test_empty_contours_still_fail_with_fixed_orientation():
    camera = PinholeCamera("camera",1280,720,800,800,640,360)
    with pytest.raises(ValueError, match="At least one"):
        estimate_tins_pad_pose(camera, {}, CENTERS, .35, np.eye(3), fixed_yaw=0.)
