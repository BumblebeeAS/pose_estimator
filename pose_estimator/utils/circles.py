import cv2
import numpy as np
from shapely.geometry import LineString, Polygon


def sample_polygon_by_angle(coords: np.ndarray, N: int) -> np.ndarray:
    """Sample N boundary points of a possibly concave polygon defined
    by coords, equally spaced in angle.

    Args:
        coords (np.ndarray): Array of (x, y) vertices.
        N (int): number of angular samples.

    Returns:
        np.ndarray: list of N boundary points (x, y), equally spaced in angle

    Raises:
        ValueError: If the polygon is not valid.
        RuntimeError: If there is no intersection between ray and polygon.
    """
    poly = Polygon(coords)
    if not poly.is_valid:
        raise ValueError("Polygon is not valid (self-intersections, etc.)")

    # Guaranteed interior point, even for concave polygons
    center_geom = poly.representative_point()
    cx, cy = center_geom.x, center_geom.y

    # Big radius: bigger than polygon's bounding box diagonal
    minx, miny, maxx, maxy = poly.bounds
    R = 2.0 * np.hypot(maxx - minx, maxy - miny)

    boundary = poly.boundary
    points = []

    for k in range(N):
        theta = 2.0 * np.pi * k / N

        # Ray: center -> far point in direction theta
        far_x = cx + R * np.cos(theta)
        far_y = cy + R * np.sin(theta)
        ray = LineString([(cx, cy), (far_x, far_y)])

        intersection = boundary.intersection(ray)
        intersection_pts = np.array(intersection.coords)

        if len(intersection_pts) == 0:
            raise RuntimeError("Ray missed polygon; center may not be inside.")

        dists = np.linalg.norm(intersection_pts - np.array([cx, cy]), axis=1)
        nearest_pt = intersection_pts[np.argmin(dists)]
        points.append(nearest_pt)

    return np.array(points)


def get_circle_from_3_pts(p1, p2, p3):
    """Return (cx, cy, r) from 3 non-collinear points."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    x1_2y1_2 = x1 * x1 + y1 * y1
    x2_2y2_2 = x2 * x2 + y2 * y2
    x3_2y3_2 = x3 * x3 + y3 * y3

    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    cx = (x1_2y1_2 * (y2 - y3) + x2_2y2_2 * (y3 - y1) + x3_2y3_2 * (y1 - y2)) / d
    cy = (x1_2y1_2 * (x3 - x2) + x2_2y2_2 * (x1 - x3) + x3_2y3_2 * (x2 - x1)) / d
    r = np.hypot(cx - x1, cy - y1)
    return cx, cy, r


def is_points_collinear(p1, p2, p3):
    """Check if points are collinear using cross product."""
    x1, y1 = p2[0] - p1[0], p2[1] - p1[1]
    x2, y2 = p3[0] - p1[0], p3[1] - p1[1]
    return abs(x1 * y2 - x2 * y1) < 1e-6


def fit_circle_least_squares(points: np.ndarray):
    """Algebraic least squares circle fit:
    x^2 + y^2 + ax + by + c = 0
    center = (-a/2, -b/2), r = sqrt((a^2+b^2)/4 - c)
    """
    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    a, bcoef, c = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = -a / 2.0, -bcoef / 2.0
    r2 = cx * cx + cy * cy - c
    r = np.sqrt(max(r2, 0.0))
    return cx, cy, r


def fit_circle_RANSAC(
    points: np.ndarray, num_iter: int, thresh: float
) -> tuple[float, float, float]:
    """Fit a circle to a set of noisy points using RANSAC.

    Args:
        points (np.ndarray): N x 2
        num_iter (int): Number of iterations to run the RANSAC loop for
        thresh (float): Inlier threshold in pixels (abs(distance_to_center - r))

    Raises:
        ValueError: If <3 points are supplied.
        np.linalg.LinAlgError: If fit_circle_least_squares fails in the refinement step.

    Returns:
        tuple[float, float, float]: (cx, cy, r)
    """
    if len(points) < 3:
        raise ValueError("We need at least 3 points to fit a circle.")

    # RANSAC
    best_inlier_mask = np.zeros(len(points), dtype=bool)

    for _ in range(num_iter):
        while True:
            sample_idxs = np.random.choice(
                np.arange(len(points)), size=3, replace=False
            )
            p1, p2, p3 = points[sample_idxs]
            if not is_points_collinear(p1, p2, p3):
                break

        cx, cy, r = get_circle_from_3_pts(p1, p2, p3)
        d = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
        res = np.abs(d - r)
        inlier_mask = res < thresh
        count = np.count_nonzero(inlier_mask)

        if count > np.count_nonzero(best_inlier_mask):
            best_inlier_mask = inlier_mask

    # Refinement
    inlier_points = points[best_inlier_mask]
    cx, cy, r = fit_circle_least_squares(inlier_points)

    return cx, cy, r


def get_circle_points(cx: float, cy: float, r: float, num_points: int) -> np.ndarray:
    """Get points on a circle.

    Args:
        cx (float): x-coordinate of circle center.
        cy (float): y-coordinate of circle center.
        r (float): radius of circle.
        num_points (int): Number of points to generate.

    Returns:
        np.ndarray: num_points x 2 array of (x, y) points on the circle.
    """
    angles = np.linspace(0.0, 2 * np.pi, num_points, endpoint=False)
    x_points = cx + r * np.cos(angles)
    y_points = cy + r * np.sin(angles)
    return np.column_stack([x_points, y_points])


def match_circle_points(
    detection_polygon: np.ndarray, radius: float, num_points: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Match points on a detected circle to 3D object points by sampling
    points on the polygon spaced equally by angle. This method is more
    efficient but not robust to partial occlusion.

    Args:
        detection_polygon (np.ndarray): N x 2 array.
        radius (float): Radius of the circular object in real-world units.
        num_points (int, optional): Number of points to sample. Defaults to 8.

    Returns:
        tuple[np.ndarray, np.ndarray]: image_points, object_points
    """
    # Sample points on the detected polygon boundary
    image_points = sample_polygon_by_angle(detection_polygon, num_points)

    # Define 3D object points on the helipad circle in its local frame
    object_points = get_circle_points(0.0, 0.0, radius, num_points)
    object_points = np.column_stack([object_points, np.zeros(num_points)])

    return image_points, object_points


def get_circle_point_correspondences(
    cx: float, cy: float, image_r: float, object_r: float
):
    """Get image-object point correspondences for a detected circle with known radius.

    Args:
        cx (float): x-coordinate of the detected circle's center.
        cy (float): y-coordinate of the detected circle's center.
        image_r (float): Radius of the detected circle.
        object_r (float): Radius of the circle in real-world units.

    Returns:
        tuple[np.ndarray, np.ndarray]: image_points, object_points
    """
    image_points = get_circle_points(cx, cy, image_r, 3)
    object_points = get_circle_points(0.0, 0.0, object_r, 3)
    object_points = np.column_stack([object_points, np.zeros(3)])
    return image_points, object_points
