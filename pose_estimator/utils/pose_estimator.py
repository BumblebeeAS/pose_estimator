import cv2
import numpy as np

from pose_estimator.utils.detections import get_detection_obb, match_polygon_points
from pose_estimator.utils.PinholeCamera import PinholeCamera


def get_object_pose(
    camera: PinholeCamera,
    object_points: np.ndarray,
    image_points: np.ndarray,
    max_reprojection_error: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the object pose from the camera and point correspondences.

    Args:
        camera (PinholeCamera):
        object_points (np.ndarray): N x 3
        image_points (np.ndarray): N x 2
        max_reprojection_error (float): Maximum reprojection error for RANSAC.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (R, t, inliers)

    Raises:
        ValueError: If the number of object points is less than 4.
        Exception: If cv2.solvePnPRansac fails.
    """
    # TODO: Account for equidistant distortion
    if len(object_points) < 4:
        raise ValueError(
            f"At least 4 points needed to estimate pose, only {len(object_points)} given"
        )

    # RANSAC accounts for outliers
    # A small max reprojection error is used to get a good pose estimate
    # SQPnP is more robust than EPnP
    _, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera.camera_matrix(),
        camera.dist_coeffs(),
        useExtrinsicGuess=False,
        reprojectionError=max_reprojection_error,
        flags=cv2.SOLVEPNP_SQPNP,
    )

    return rvec, tvec, inliers


def refine_object_pose(
    camera: PinholeCamera,
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    term_eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine the object pose using the initial pose estimate.

    Args:
        camera (PinholeCamera): Camera object.
        object_points (np.ndarray): N x 3 array of object points.
        image_points (np.ndarray): N x 2 array of image points.
        rvec (np.ndarray): Initial rotation vector.
        tvec (np.ndarray): Initial translation vector.
        term_eps (float): Maximum error for refinement.

    Returns:
        tuple[np.ndarray, np.ndarray]: Refined rotation and translation vectors.

    Raises:
        Exception: If cv2.solvePnPRefineVVS fails.
    """
    # TODO: For the planar case, init cv2.solvePnPRefineLM directly with homography
    # TODO: Split into planar and non-planar cases.
    # TODO: Uncomment the following lines to use refineVVS after filtering outliers.
    # TODO: See if refinement even helps.
    rvec, tvec = cv2.solvePnPRefineVVS(
        object_points,
        image_points,
        camera.camera_matrix(),
        camera.dist_coeffs(),
        rvec,
        tvec,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 1000, term_eps),
    )

    return rvec, tvec


def filter_by_homography(object_points: np.ndarray, image_points: np.ndarray) -> tuple:
    """Filter the object points and image points by homography,
    assuming that the object points are in the same plane.

    Note: The last coordinate of the object points are discarded.
    The object points should first be transformed so that the last
    coordinates are zero.

    Args:
        object_points (np.ndarray): N x 3
        image_points (np.ndarray): N x 2

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (R, t, inliers)

    Raises:
        ValueError: If the number of object points is less than 4.
        ValueError: If the number of object points and image points are not equal.
    """
    if len(object_points) < 4:
        raise ValueError(
            f"At least 4 points needed to estimate homography, only {len(object_points)} given"
        )

    if len(object_points) != len(image_points):
        raise ValueError(
            f"Number of object points and image points must be equal, "
            f"but got {len(object_points)} and {len(image_points)}"
        )

    object_points_2d = object_points[:, :2]
    _, mask = cv2.findHomography(
        object_points_2d,
        image_points,
        cv2.USAC_MAGSAC,
        3.5,
        maxIters=1_000,
        confidence=0.999,
    )
    mask = mask.flatten().astype(bool)
    object_points = object_points[mask]
    image_points = image_points[mask]
    return object_points, image_points


def estimate_covariance(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera: PinholeCamera,
) -> np.ndarray:
    """Get covariance of pose estimate from reprojection.

    Args:
        object_points (np.ndarray): N x 3
        rvec (np.ndarray): Rotation vector
        tvec (np.ndarray): Translation vector
        camera (PinholeCamera): Camera object

    Returns:
        np.ndarray: 6 x 6 covariance matrix

    Raises:
        np.linalg.LinAlgError: If the inverse of the Jacobian cannot be computed.
    """
    # Jacobian is a 2N x 15 matrix
    # See https://github.com/opencv/opencv/blob/16a3d37dc159dbcaaf8ee74cf63669f0203f9655/modules/calib3d/src/calibration_base.cpp#L1508-L1512
    _, jacobian = cv2.projectPoints(
        object_points, rvec, tvec, camera.camera_matrix(), camera.dist_coeffs()
    )

    # Get jacobian of rotation and translation
    # Interchange rotation and translation covariance
    jacobian = jacobian[:, :6]
    jacobian[:, :3], jacobian[:, 3:] = (
        jacobian[:, 3:].copy(),
        jacobian[:, :3].copy(),
    )

    # Fisher information matrix
    return np.linalg.inv(jacobian.T @ jacobian)


def get_object_pose_from_detection_using_obb(
    camera: PinholeCamera, object_polygon: np.ndarray, detection: object
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimates pose of a detection using the oriented bounding box of its mask.
    The most naive method of pose estimation, but works well when the following
    conditions are met:

    - The object is not rotated more than 45 degrees from its upright orientation.
    - The object is rectangular.
    - Perspective does not significantly affect the object's aspect ratio.

    Args:
        camera (PinholeCamera):
        object_polygon (np.ndarray): N x 2 array representing the polygon of the object in world coordinates.
        detection (object): Detection object containing the mask and class name.

    Raises:
        ValueError: If the number of object points is less than 4.
        Exception: If cv2.solvePnPRansac fails.
        ValueError: If no inliers are found during pose estimation.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (rvec, tvec)
    """
    _, detected_polygon = get_detection_obb(detection)
    object_points, image_points = match_polygon_points(object_polygon, detected_polygon)
    object_points = np.hstack([object_points, np.zeros((object_points.shape[0], 1))])

    # We can set a low max re-projection error and refine pose even though
    # the segmentation mask is noisy because we only have 4 points
    rvec, tvec, inliers = get_object_pose(camera, object_points, image_points)

    if inliers is None:
        raise ValueError("No inliers found during pose estimation.")

    rvec, tvec = refine_object_pose(camera, object_points, image_points, rvec, tvec)

    return rvec, tvec
