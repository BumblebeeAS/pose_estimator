import cv2
import numpy as np

from pose_estimator.utils.circles import get_circle_points, get_ellipse_points
from pose_estimator.utils.detections import match_polygon_points
from pose_estimator.utils.PinholeCamera import PinholeCamera


def estimate_circle_pose_from_ellipse(
    camera: PinholeCamera,
    ellipse: tuple,
    R: float,
    rvec: np.ndarray,
    tvec: np.ndarray,
    N: int = 100,
    iters: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (rvec, tvec) pose of circle object frame w.r.t camera frame.

    Args:
        camera (PinholeCamera):
        ellipse (tuple): ((cx,cy), (MA,ma), angle_deg), same format as cv2.fitEllipse
        R (float): circle radius in real world units
        N (int, optional): number of points to sample on the ellipse and circle. Defaults to 200.
        iters (int, optional): number of iterations for the iterative pose refinement. Defaults to 10.

    Raises:
        RuntimeError: If solvePnP fails.

    Returns:
        tuple[np.ndarray, np.ndarray]: (rvec, tvec) pose of circle object frame w.r.t camera frame
    """
    camera_matrix = camera.camera_matrix()
    dist_coeffs = camera.dist_coeffs()

    (cx, cy), (MA, ma), angle_deg = ellipse
    img_obs = get_ellipse_points(cx, cy, MA / 2, ma / 2, angle_deg, N)
    obj_pts_2d = get_circle_points(0.0, 0.0, R, N)
    obj_pts = np.hstack([obj_pts_2d, np.zeros((obj_pts_2d.shape[0], 1))]).astype(
        np.float32
    )

    # Iterative re-matching + PnP
    for _ in range(iters):
        # project with current pose
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
        proj = proj.reshape(-1, 2).astype(np.float32)

        _, img_matched = match_polygon_points(proj, img_obs)

        # SolvePnP using these correspondences
        ok, rvec_new, tvec_new = cv2.solvePnP(
            objectPoints=obj_pts,
            imagePoints=img_matched,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("solvePnP failed")

        rvec, tvec = rvec_new, tvec_new

    return rvec, tvec
