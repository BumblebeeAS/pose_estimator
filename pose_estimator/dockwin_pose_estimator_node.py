import rclpy

from pose_estimator.config.robotx26_object_points import WINDOW_OBJECT_POINTS_DICT
from pose_estimator.utils.best_fit_quad_pose_estimator_node import (
    BestFitQuadPoseEstimator,
)


def main(args=None):
    rclpy.init(args=args)
    node = BestFitQuadPoseEstimator(
        WINDOW_OBJECT_POINTS_DICT,
        {"left_window": "dockwin/yolo", "right_window": "dockwin/yolo"},
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
