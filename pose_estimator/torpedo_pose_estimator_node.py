import rclpy

from pose_estimator.config.object_points import TORPEDO_OBJECT_POINTS_DICT
from pose_estimator.utils.best_fit_quad_pose_estimator_node import (
    BestFitQuadPoseEstimator,
)


def main(args=None):
    rclpy.init(args=args)
    node = BestFitQuadPoseEstimator(
        TORPEDO_OBJECT_POINTS_DICT,
        {"torpedo_1": "torpedo/yolo", "torpedo_2": "torpedo/yolo"},
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
