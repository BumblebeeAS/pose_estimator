from typing import List

import message_filters
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import  Float32MultiArray # TODO: use custom message
from tf_transformations import euler_from_quaternion
from yolo_msgs.msg import Detection, DetectionArray

from pose_estimator.utils.detections import (
    filter_detections_by_num_points,
    get_top_k_detections_per_class,
)


class SlalomYawScore(Node):
    def __init__(self):
        super().__init__("slalom_yaw_score")

        detections_topic = (
            self.declare_parameter("detections_topic", "slalom/yolo/detections")
            .get_parameter_value()
            .string_value
        )
        odom_topic = (
            self.declare_parameter("odom_topic", "nav/odom_ned")
            .get_parameter_value()
            .string_value
        )

        self.det_sub = message_filters.Subscriber(detections_topic, DetectionArray)
        self.odom_sub = message_filters.Subscriber(odom_topic, Odometry)
        self.time_sync = message_filters.ApproximateTimeSynchronizer(
            fs=[self.det_sub, self.odom_sub],
            queue_size=10,
            slop=0.001,
        )
        self.time_sync.registerCallback(self.callback)

        self.publisher = self.create_publisher(Float32MultiArray, "slalom/yaw_score", 10)

    def callback(self, detections: DetectionArray, odom: Odometry):
        filtered_detections = filter_detections_by_num_points(detections, 3)
        red_pole_detections = get_top_k_detections_per_class(
            filtered_detections, {"red_pole": 3}
        )["red_pole"]
        yaw = self._get_yaw_from_odom(odom)
        score = self._compute_score(red_pole_detections)

        yaw_score_msg = Float32MultiArray()
        yaw_score_msg.data = [float(yaw), float(score)]
        self.publisher.publish(yaw_score_msg)

    def _get_yaw_from_odom(self, odom: Odometry) -> float:
        _, _, yaw = euler_from_quaternion(
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
        )
        return yaw

    def _compute_score(self, detections: List[Detection]) -> float:
        if len(detections) == 0:
            return -1e6

        # we compare the center of the bounding box to the center of the image based on the img res
        score = 0
        for det in detections:
            bbox_center_position_x = det.bbox.center.position.x
            img_center_x = det.mask.width / 2.0
            dist = abs(bbox_center_position_x - img_center_x)
            score += det.score - dist / img_center_x # should be between 0 and 1 for score and dist penalty
        return score


def main(args=None):
    rclpy.init(args=args)
    slalom_yaw_score_node = SlalomYawScore()
    try:
        rclpy.spin(slalom_yaw_score_node)
    except KeyboardInterrupt:
        pass
    finally:
        slalom_yaw_score_node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
