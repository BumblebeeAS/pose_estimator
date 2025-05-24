import numpy as np
from sensor_msgs.msg import CameraInfo


class PinholeCamera(object):
    def __init__(self, frame_id, width, height, fx, fy, cx, cy, *distortion_params):
        self.frame_id = frame_id
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        if len(distortion_params) == 0:
            distortion_params = [0, 0, 0, 0, 0]
        self.distortion = abs(distortion_params[0]) > 0.0000001
        self.d = np.array(distortion_params, dtype=np.float32)

    def from_camera_info(camera_info: CameraInfo, rectified=False):
        if rectified:
            return PinholeCamera(
                camera_info.header.frame_id,
                camera_info.width,
                camera_info.height,
                camera_info.p[0],
                camera_info.p[5],
                camera_info.p[2],
                camera_info.p[6],
            )
        else:
            return PinholeCamera(
                camera_info.header.frame_id,
                camera_info.width,
                camera_info.height,
                camera_info.k[0],
                camera_info.k[4],
                camera_info.k[2],
                camera_info.k[5],
                *camera_info.d,
            )

    def camera_matrix(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    def dist_coeffs(self):
        return self.d


PINHOLE_CAMERAS = {
    "sim": PinholeCamera(
        768,
        492,
        407.0646129842357,
        407.0646129842357,
        384.5,
        246.5,
        0,
        0,
        0,
        0,
        0,
    )
}
