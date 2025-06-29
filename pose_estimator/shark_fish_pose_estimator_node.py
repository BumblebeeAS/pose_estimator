import rclpy

from pose_estimator.config.object_points import SYMBOL_OBJECT_POINTS_DICT
from pose_estimator.utils.best_fit_quad_pose_estimator_node import (
    BestFitQuadPoseEstimator,
)


def main(args=None):
    rclpy.init(args=args)
    node = BestFitQuadPoseEstimator(
        SYMBOL_OBJECT_POINTS_DICT,
        {"reef_shark": "trash/shark", "sawfish": "trash/fish"},
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
