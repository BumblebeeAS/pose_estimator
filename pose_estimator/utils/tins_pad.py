"""Fit platform translation and optional yaw from colored circle contours."""

import cv2
import numpy as np
from scipy.optimize import least_squares

from pose_estimator.utils.circles import _get_ellipse_inliers, fit_ellipse_RANSAC


def estimate_tins_pad_pose(camera, contours, centers, radius, level_rotation,
                           inlier_threshold=8.0, fixed_yaw=None):
    """Return platform (rvec, tvec, RMS pixel error).

    ``level_rotation`` maps NED-oriented axes to camera coordinates. The fit
    estimates translation and platform yaw jointly, using the known circle
    radius and configurable circle centers in platform coordinates. At least two
    colors are required unless fixed_yaw is supplied. With fixed_yaw, only
    translation is fitted and one circle is sufficient. level_rotation can then
    carry the complete retained pad orientation in camera coordinates.
    Circle contours have no angular point correspondence.
    Ellipse RANSAC removes contour outliers before the joint fit. Joint-fit
    consensus is then refined and scored using only pixel inliers, requiring
    support from every supplied color rather than silently dropping a circle.
    """
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("Circle radius must be positive")
    if inlier_threshold <= 0 or not np.isfinite(inlier_threshold):
        raise ValueError("Inlier threshold must be positive")
    if fixed_yaw is not None and not np.isfinite(fixed_yaw):
        raise ValueError("Fixed yaw must be finite")
    if len(contours) < (2 if fixed_yaw is None else 1):
        raise ValueError("At least two colored circles are required" if fixed_yaw is None
                         else "At least one colored circle is required")
    model = np.asarray([centers[label] for label in contours], dtype=float)
    if (model.shape != (len(contours), 2) or not np.isfinite(model).all()
            or (fixed_yaw is None and np.linalg.matrix_rank(model - model.mean(0)) < 1)):
        raise ValueError("Circle centers must include two distinct finite points")
    pixels, model_centers, initial_centers = [], [], []
    group_sizes = []
    for label, center in zip(contours, model):
        points = np.asarray(contours[label], dtype=np.float32).reshape(-1, 2)
        if len(points) < 5 or not np.isfinite(points).all():
            raise ValueError("Circle contour must have at least five finite points")
        hull = cv2.convexHull(points).reshape(-1, 2)
        if len(hull) < 5:
            raise ValueError("Degenerate circle contour")
        ellipse, inliers = fit_ellipse_RANSAC(hull, num_iter=200, thresh=inlier_threshold)
        if len(inliers) < 5:
            raise ValueError("Insufficient circle inliers")
        # Reclassify after the RANSAC ellipse has been refined. Its original
        # consensus can contain points that no longer agree with the fit.
        inliers = hull[_get_ellipse_inliers(hull, ellipse, inlier_threshold)]
        if len(inliers) < max(5, int(np.ceil(0.5 * len(hull)))):
            raise ValueError("Insufficient circle consensus")
        # Bound optimization cost without duplicating short visible arcs.
        points = inliers[np.linspace(0, len(inliers) - 1, min(40, len(inliers))).astype(int)]
        normalized = cv2.undistortPoints(
            points[:, None], camera.camera_matrix(), camera.dist_coeffs()
        ).reshape(-1, 2)
        (cx, cy), axes, _ = cv2.fitEllipse(normalized.astype(np.float32))
        depth = 2 * radius / max(axes)
        initial_centers.append((np.array([cx, cy, 1.0]) * depth) @ level_rotation)
        pixels.append(points)
        group_sizes.append(len(points))
        model_centers.append(np.tile(center, (len(points), 1)))

    pixels = np.concatenate(pixels)
    model_centers = np.concatenate(model_centers)
    normalized = cv2.undistortPoints(
        pixels[:, None], camera.camera_matrix(), camera.dist_coeffs()
    ).reshape(-1, 2)
    rays = np.column_stack([normalized, np.ones(len(normalized))]) @ level_rotation
    if np.min(np.abs(rays[:, 2])) < 1e-4:
        raise ValueError("Platform is too close to the horizon")
    initial_centers = np.asarray(initial_centers)
    distance = np.median(initial_centers[:, 2])
    sign = np.sign(distance)
    if not sign or np.any(sign * rays[:, 2] <= 0):
        raise ValueError("Circle rays do not intersect a common visible plane")
    # Labeled center offsets give a starting yaw without assuming image axes.
    observed = initial_centers[:, :2]
    if fixed_yaw is None:
        u, _, vt = np.linalg.svd((model - model.mean(0)).T @ (observed - observed.mean(0)))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        yaw = float(fixed_yaw)
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s], [s, c]])
    translation_xy = (observed - model @ rotation.T).mean(0)
    level_rvec, _ = cv2.Rodrigues(level_rotation)

    def unpack(params):
        x, y, log_distance = params[:3]
        angle = params[3] if fixed_yaw is None else fixed_yaw
        c, s = np.cos(angle), np.sin(angle)
        return np.array([x, y]), sign * np.exp(log_distance), np.array([[c, -s], [s, c]])

    def residual(params):
        translation_xy, distance, rotation = unpack(params)
        observed_xy = distance * rays[:, :2] / rays[:, 2, None]
        circle_xy = model_centers @ rotation.T + translation_xy
        radial = observed_xy - circle_xy
        radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
        nearest = np.column_stack([
            circle_xy + radius * radial, np.full(len(pixels), distance)
        ])
        projected, _ = cv2.projectPoints(
            nearest, level_rvec, np.zeros(3), camera.camera_matrix(), camera.dist_coeffs()
        )
        return (projected[:, 0] - pixels).ravel()

    initial = [*translation_xy, np.log(abs(distance))]
    lower, upper = [-np.inf, -np.inf, -8], [np.inf, np.inf, 10]
    if fixed_yaw is None:
        initial.append(yaw)
        lower.append(-np.inf)
        upper.append(np.inf)
    bounds = (lower, upper)
    result = least_squares(
        residual, initial,
        bounds=bounds,
        loss="soft_l1", f_scale=2.0, max_nfev=100,
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("Platform fit did not converge")
    # Like RANSAC, classify whole 2D points by pixel error and refit only the
    # consensus. A robust loss alone still lets clipped edges bias the pose.
    slices = []
    start = 0
    for count in group_sizes:
        slices.append(slice(start, start + count))
        start += count
    def consensus(params):
        errors = np.linalg.norm(residual(params).reshape(-1, 2), axis=1)
        mask = errors <= inlier_threshold
        if any(np.count_nonzero(mask[group]) < max(5, int(np.ceil(0.5 * len(errors[group]))))
               for group in slices):
            raise ValueError("Insufficient platform consensus for a circle")
        return errors, mask

    for _ in range(3):
        _, inlier_mask = consensus(result.x)
        result = least_squares(
            lambda params: residual(params).reshape(-1, 2)[inlier_mask].ravel(),
            result.x,
            bounds=bounds,
            max_nfev=100,
        )
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError("Platform consensus fit did not converge")
    translation_xy, distance, rotation = unpack(result.x)
    translation = level_rotation @ np.array([*translation_xy, distance])
    root_rotation = level_rotation @ np.array([
        [rotation[0, 0], rotation[0, 1], 0],
        [rotation[1, 0], rotation[1, 1], 0], [0, 0, 1],
    ])
    rvec, _ = cv2.Rodrigues(root_rotation)
    if translation[2] <= 0:
        raise ValueError("Platform is behind the camera")
    final_errors, inlier_mask = consensus(result.x)
    error = np.sqrt(np.mean(final_errors[inlier_mask] ** 2))
    return rvec, translation.reshape(3, 1), float(error)
