import logging

import cv2
import numpy as np
import rclpy
import tf2_ros
from bb_perception_msgs.msg import PointCorrespondencesStamped
from geometry_msgs.msg import (
    Point,
    PoseWithCovarianceStamped,
    Quaternion,
    TransformStamped,
    Vector3,
)
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import CameraInfo
from src.PinholeCamera import PinholeCamera
from transforms3d.quaternions import mat2quat
from utils.ros_np_multiarray import to_numpy_f64


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


class SimplePoseEstimator(Node):

    def __init__(self):
        super().__init__("pose_estimator")

        self.declare_parameter("camera_info_topic", "camera_info")
        camera_info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        valid, camera_info = wait_for_message(CameraInfo, self, camera_info_topic)
        if not valid:
            raise ValueError("Failed to get camera info")
        else:
            camera_info: CameraInfo
            self.camera = PinholeCamera.from_camera_info(camera_info, rectified=False)
            self.camera_frame_id = camera_info.header.frame_id

        self.br = tf2_ros.TransformBroadcaster(self)

        self.point_subscriber = self.create_subscription(
            PointCorrespondencesStamped,
            "image_matching/point_correspondences",
            self.point_correspondences_callback,
            1,
        )

        self.pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, "image_matching/pose", 1
        )

    def point_correspondences_callback(self, msg: PointCorrespondencesStamped):
        # TODO: Filter using clustering or Kalman
        object_points = to_numpy_f64(msg.object_points)
        image_points = to_numpy_f64(msg.image_points)

        if object_points.shape[0] < 4 or image_points.shape[0] < 4:
            self.get_logger().warn(
                f"""Not enough point correspondences:
                {object_points.shape[0]} object points and {image_points.shape[0]} image points"""
            )
            return

        object_points, image_points = filter_by_homography(object_points, image_points)

        try:
            rvec, tvec = get_object_pose(self.camera, object_points, image_points)
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.squeeze()
        except Exception as e:
            self.get_logger().warn(f"Pose estimation failed: {e}")
            return

        try:
            covariance = estimate_covariance(object_points, rvec, tvec, self.camera)
        except np.linalg.LinAlgError as e:
            self.get_logger().warn(
                f"Covariance estimation failed, inversion for FIM matrix failed: {e}"
            )
            return

        # self.get_logger().info(
        #     f"Pose estimation std dev: {np.sqrt(covariance.diagonal())}"
        # )

        try:
            qx, qy, qz, qw = mat2quat(R)
        except np.linalg.LinAlgError as e:
            self.get_logger().warn(f"Error in mat2quat, failed to convert R: {e}")
            return

        pose = PoseWithCovarianceStamped()
        pose.header = msg.header
        pose.header.frame_id = self.camera_frame_id
        pose.pose.pose.position = Point(x=t[0], y=t[1], z=t[2])
        pose.pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        pose.pose.covariance = covariance.flatten().tolist()

        self.pose_publisher.publish(pose)

        transform_stamped = TransformStamped()
        transform_stamped.header = msg.header
        transform_stamped.child_frame_id = msg.object_frame_id
        transform_stamped.transform.translation = Vector3(x=t[0], y=t[1], z=t[2])
        transform_stamped.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        self.br.sendTransform(transform_stamped)


def main(args=None):
    logging.basicConfig(level=logging.INFO)
    rclpy.init(args=args)
    node = SimplePoseEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
