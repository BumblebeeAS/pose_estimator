import logging
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
from feature_matcher.keypoints_match_producer import (
    KeypointsMatchProducer,
    get_keypoints_match_producer,
)
from feature_matcher.tools import create_show_image, plot_matches
from utils.PinholeCamera import PinholeCamera


def get_object_pose(
    camera: PinholeCamera,
    object_points: np.ndarray,
    image_points: np.ndarray,
    max_reprojection_error: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the object pose from the camera and point correspondences.

    Args:
        camera (PinholeCamera):
        object_points (np.ndarray): N x 3
        image_points (np.ndarray): N x 2
        max_reprojection_error (float): Maximum reprojection error for RANSAC.

    Returns:
        tuple[np.ndarray, np.ndarray]: (R, t)

    Raises:
        ValueError: If the number of object points is less than 4.
        ValueError: If no inliers are found.
        Exception: If cv2.solvePnPRansac or cv2.solvePnPRefineLM fails.
    """
    # TODO: Account for equidistant distortion
    # TODO: For the planar case, init cv2.solvePnPRefineLM directly with homography
    if len(object_points) < 4:
        raise ValueError(
            f"At least 4 points needed to estimate pose, only {len(object_points)} given"
        )

    # This step gives a rough estimate of the pose for solvePnPRefineLM and
    # allows for quick termination if no inliers are found. This is useful
    # when there are few point correspondences and homography estimation
    # cannot determine if the points are inliers or not.

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

    if inliers is None:
        raise ValueError("No inliers found")

    # TODO: Split into planar and non-planar cases.
    # Case 1: Object points are non-planar.
    # Use only the inliers from RANSAC as homography estimation does not apply.
    # Case 2: Use all points for the planar case.
    # Homography estimation filters well. RANSAC filtering is too strict, resulting
    # in too few point correspondences and a noisy pose estimate.
    rvec, tvec = cv2.solvePnPRefineVVS(
        object_points,
        image_points,
        camera.camera_matrix(),
        camera.dist_coeffs(),
        rvec,
        tvec,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 1000, 1e-6),
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


################
#   OLD CODE   #
################


def homography_based_filter(
    kp1,
    kp2,
    camera_matrix,
    dist_coeffs,
    min_inliers=6,
):
    if (kp1.shape[0] < min_inliers) or (kp2.shape[0] < min_inliers):
        print("NOT ENOUGH POINTS -> Object not in view? ")
        return None, None, None, None

    kp1_undistort = cv2.undistortPoints(kp1.reshape(-1, 1, 2), np.eye(3), dist_coeffs)
    kp2_undistort = cv2.undistortPoints(kp2.reshape(-1, 1, 2), np.eye(3), dist_coeffs)

    H, mask = cv2.findHomography(kp1_undistort, kp2_undistort, cv2.RANSAC)
    if H is None or sum(mask) < min_inliers:
        print("NO INLIERS")
        return None, None, None, None

    _, Rs, ts, ns = cv2.decomposeHomographyMat(H, camera_matrix)
    for R, tvec, n in zip(Rs, ts, ns):
        if n[2] > 0:  # Wrong direction -> skip
            continue
        if abs(n[2] + 1) > 0.1:  # Wrong normal -> skip
            continue

        # yaw, pitch, roll = mat2euler(R, axes="szyx")
        # print("Yaw (H): ", np.rad2deg(yaw))
        # print("Pitch (H): ", np.rad2deg(pitch))
        # print("Roll (H): ", np.rad2deg(roll))

        if mask is not None:
            # bef = len(kp1)
            kp1 = kp1[mask.ravel() == 1]
            kp2 = kp2[mask.ravel() == 1]

            # rospy.loginfo("Inliers (H): ", len(kp1), " / ", bef)
            return kp1, kp2, R, tvec
    return None, None, None, None


class PoseEstimator:
    def __init__(
        self,
        keypoints_match_producers: Dict[str, KeypointsMatchProducer],
        debug=False,
    ):
        self.keypoints_match_producers = keypoints_match_producers
        self.visualize_callbacks: List[Callable[[np.ndarray], None]] = []
        self.cameras: Dict[str, PinholeCamera] = {}
        self.templates: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.min_inliers = 4
        self.debug = debug

    def register_camera(self, camera: PinholeCamera):
        self.cameras[camera.frame_id] = camera
        return camera.frame_id

    def register_template(self, name, world_dimensions, template_img):
        if "_seg" in name:
            print("Will not perform matching for segmentation templates")
        else:
            for producer in self.keypoints_match_producers.values():
                producer.register_template(name, template_img)
        self.templates[name] = (
            np.array(world_dimensions),
            np.array(template_img.shape[1::-1]),
        )  # width, height

    @property
    def available_cameras(self):
        return self.cameras.keys()

    @property
    def available_templates(self):
        return self.templates.keys()

    @staticmethod
    def draw_object_points(img, object_points, R, t, camera):
        imgpts, _ = cv2.projectPoints(
            object_points, R, t, camera.camera_matrix(), camera.dist_coeffs()
        )
        return cv2.polylines(img, [np.int32(imgpts)], True, (0, 255, 0), 3)

    def visualize(self, image):
        for callback in self.visualize_callbacks:
            callback(image)

    def compute_pose_from_keypoints(
        self,
        template,
        camera_frame,
        keypoints1,  # np.ndarray (x, y) * N
        keypoints2,  # np.ndarray (x, y) * N
        is_planar=False,  # If true, we assume object is planar => homography is used first
        max_reprojection_error=2.0,  # Maximum reprojection error for pose to be accepted
        min_matches=4,
        debug=False,
    ):
        if camera_frame not in self.cameras:
            raise Exception(f"Camera {camera_frame} not registered.")
        else:
            camera = self.cameras[camera_frame]

        R_H = None
        t_H = None
        if is_planar:
            # Homography-based filtering
            kp1, kp2, R_H, t_H = homography_based_filter(
                keypoints1,
                keypoints2,
                camera.camera_matrix(),
                camera.dist_coeffs(),
                min_inliers=min_matches,
            )
            if kp1 is None or kp2 is None:
                return None, None, None

        source_dimensions, source_image_size = self.templates[template]
        object_coord = (
            (keypoints1 - source_image_size / 2) * source_dimensions / source_image_size
        )
        object_coord = np.hstack((object_coord, np.zeros((len(object_coord), 1))))

        if len(object_coord) < max(min_matches, 4):
            print(
                "NOT ENOUGH POINTS " + str(len(object_coord)),
                str(max(min_matches, 4)),
            )
            return None, None, None
        else:
            print("Enough points", str(len(object_coord)))
        try:
            _, rvec, t, inliers = cv2.solvePnPRansac(
                object_coord.astype(np.float64),
                keypoints2.astype(np.float32),
                camera.camera_matrix(),
                camera.dist_coeffs(),
                useExtrinsicGuess=False,
                reprojectionError=max_reprojection_error,
                flags=cv2.SOLVEPNP_SQPNP,
            )
        except Exception as e:
            print(e)
            return None, None, None
        R = cv2.Rodrigues(rvec)[0]
        t = t.squeeze()

        if R_H is not None:
            r_diff = np.arccos((np.trace(R @ R_H.T) - 1) / 2)
            if r_diff > 0.2:  # Radian -> 11 degrees
                print(f"Homography vs PnP: {r_diff}, skipping")
                return None, None, None

        if debug and R is not None and t is not None:
            img = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
            img = plot_matches(
                np.zeros(
                    (source_image_size[1], source_image_size[0], 3),
                    dtype=np.uint8,
                ),
                img,
                keypoints1,
                keypoints2,
            )
            self.visualize(img)
        return R, t, inliers

    def compute_pose(
        self,
        img,
        template,
        camera_frame,
        *,
        matcher="superpoint_superglue",
        num_keypoints=20,
        lxtyrxby=None,
        debug=False,
        logger=None,
        is_planar=True,  # If true, we assume object is planar => homography is used first
        max_reprojection_error=2.0,  # Maximum reprojection error for pose to be accepted
        min_matches=4,
    ):
        if template is None:
            raise Exception("Template has to be specified.")
        if camera_frame not in self.cameras:
            raise Exception(f"Camera {camera_frame} not registered.")
        else:
            camera = self.cameras[camera_frame]
        if matcher not in self.keypoints_match_producers.keys():
            raise Exception(f"Matcher {matcher} not registered.")
        keypoints_match_producer = self.keypoints_match_producers[matcher]
        print("Processing Image!")
        keypoints1, keypoints2 = keypoints_match_producer.process_image(
            img,
            template,
            debug,
            num_keypoints=num_keypoints,
            lxtyrxby=lxtyrxby,
            logger=logger,
        )

        if keypoints1 is None or keypoints2 is None:
            print("No keypoints generated!")
            return None, None
        print(len(keypoints1))

        R, t, inliers = self.compute_pose_from_keypoints(
            template,
            camera_frame,
            keypoints1.keypoints,
            keypoints2.keypoints,
            is_planar=is_planar,
            min_matches=min_matches,
            max_reprojection_error=max_reprojection_error,
        )

        if debug and R is not None and t is not None:
            source_dimensions, _ = self.templates[template]
            _x = source_dimensions[0] / 2
            _y = source_dimensions[1] / 2
            object_rect = np.array(
                [[-_x, -_y, 0], [_x, -_y, 0], [_x, _y, 0], [-_x, _y, 0]]
            )
            # Draw object bbox
            img = self.draw_object_points(img, object_rect, R, t, camera)

            # Draw axes
            img = cv2.drawFrameAxes(
                img,
                camera.camera_matrix(),
                camera.dist_coeffs(),
                R,
                t,
                length=0.1,
            )

            mask = np.zeros(keypoints1.keypoints.shape[0], dtype=np.uint8)
            if inliers is not None:
                mask[inliers.squeeze()] = 1
            else:
                mask = np.ones(keypoints1.keypoints.shape[0], dtype=np.uint8)

            img = plot_matches(
                keypoints_match_producer.get_template(template).img,
                img,
                keypoints1.keypoints,
                keypoints2.keypoints,
                scores=mask,
            )
            # create_save_image("/home/nvidia/catkin_ws/src/image_matching/debug.png")(img)
            # exit(1)
            self.visualize(img)
        return R, t


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    import os

    # folder_path = "/home/developer/workspace/src/rosbags/tommy_gun_sim3_2022-12-26-05-16-08/_auv4_front_cam_image_rect_color"
    # bboxes_file = "/home/developer/workspace/src/rosbags/tommy_gun_sim3_2022-12-26-05-16-08/tommygun_gt.csv"
    folder_path = "/home/developer/workspace/src/rosbags/bootlegger_torpedo_sim1_2022-12-26-18-25-31/Images"
    bboxes_file = "/home/developer/workspace/src/rosbags/bootlegger_torpedo_sim1_2022-12-26-18-25-31/bootlegger1.csv"

    # image_match_producers = {
    #     # 0.0103288540s
    #     "sift": get_keypoints_match_producer("sift", "bf", {"debug": True}, {"debug": True}),
    #     # 0.0042234153s
    #     "orb": get_keypoints_match_producer("orb", "bf", {"debug": True}, {"debug": True}),
    #     # 0.0026472004s
    #     "fast": get_keypoints_match_producer("fast", "bf", {"debug": True}, {"debug": True}),
    #     # 0.2101211615s
    #     "superpoint": get_keypoints_match_producer("superpoint", "superglue", {"debug": True}, {"debug": True}),
    #     # 0.0883604178s
    #     "coarse_loftr": get_keypoints_match_producer(None, "coarse_loftr", {"debug": True}, {"debug": True}),
    #     # "loftr": get_keypoints_match_producer(None, "loftr", {"debug": True}, {"debug": True}),
    # }

    # image_match_producer = get_keypoints_match_producer("superpoint", "superglue", {"debug": True}, {"debug": True}) # 0.6256848859s
    # image_match_producer = get_keypoints_match_producer(None, "coarse_loftr", {"debug": True}, {"debug": True}) # 0.1275215585s
    # image_match_producer = get_keypoints_match_producer("sift", "flann", {"debug": True}, {"debug": True}) # 0.0294936401s
    image_match_producer = get_keypoints_match_producer(
        "sift", "bf", {"debug": True}, {"debug": True}
    )  # 0.0318265118s
    # image_match_producer = get_keypoints_match_producer("superpoint", "bf", {"debug": True}, {"debug": True})

    pose_estimator_1 = PoseEstimator(image_match_producer)
    camera = PinholeCamera(
        "auv4/front_cam",
        1024,
        768,
        1104.9647584119537,
        1103.3380651945358,
        1031.4081561220519,
        752.7821761752537,
        0,
        0,
        0,
        0,
        0,
    )
    # camera = PinholeCamera(
    #     "auv4/front_cam",
    #     768,
    #     492,
    #     407.0646129842357,
    #     407.0646129842357,
    #     384.5,
    #     246.5,
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    # )
    pose_estimator_1.register_camera(camera)

    pose_estimator_1.visualize_callbacks.append(
        create_show_image(pose_estimator_1.__class__.__name__)
    )
    # pose_estimator_1.visualize_callbacks.append(create_save_image())

    templates = {
        "Tommy Gun": (
            (0.6096, 1.2192),
            "/home/developer/workspace/src/image_matching/templates/Tommy Gun.jpeg",
        ),
        "Bootlegger": (
            (0.6096, 1.2192),
            "/home/developer/workspace/src/image_matching/templates/Bootlegger.jpeg",
        ),
    }
    for key, value in templates.items():
        pose_estimator_1.register_template(key, value[0], cv2.imread(value[1]))

    bboxes = np.loadtxt(bboxes_file, delimiter=",")
    PADDING = 10
    CROP_IMAGES = True
    for i, file in enumerate(os.listdir(folder_path)):
        try:
            img = cv2.imread(f"{folder_path}/{file}")
            left, top, w, h = [int(_) for _ in bboxes[i]]
            left = max(0, left - PADDING)
            top = max(0, top - PADDING)
            r = min(img.shape[1], left + w + PADDING)
            b = min(img.shape[0], top + h + PADDING)

            # 0.48s
            rot, trans = pose_estimator_1.compute_pose(
                img,
                "Bootlegger",
                "auv4/front_cam",
                lxtyrxby=(left, top, r, b) if CROP_IMAGES else None,
                debug=True,
            )
            # print(", ".join(map(str,np.rad2deg(rot.squeeze()))), ", ".join(map(str, trans.squeeze())))

        except Exception as e:
            logging.error(e)
            continue
