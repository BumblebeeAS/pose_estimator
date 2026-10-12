"""Unit tests for detection score filtering and pose estimator confidence gating."""

import sys
import unittest
from unittest.mock import MagicMock

# Mock third-party C/ROS dependencies if not present in the environment
for mod in [
    "cv2",
    "cv2.typing",
    "numpy",
    "numpy.typing",
    "scipy",
    "scipy.spatial",
    "shapely",
    "rclpy",
    "rclpy.impl",
    "rclpy.impl.rcutils_logger",
    "yolo_msgs",
    "yolo_msgs.msg",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()


class MockDetection:
    def __init__(self, class_name: str = "", score: float = 0.0, num_points: int = 4):
        self.class_name = class_name
        self.score = float(score)
        mock_mask = MagicMock()
        mock_mask.data = [MagicMock() for _ in range(num_points)]
        self.mask = mock_mask


class MockDetectionArray:
    def __init__(self, header=None, detections=None):
        self.header = header
        self.detections = detections or []


sys.modules["yolo_msgs.msg"].Detection = MockDetection
sys.modules["yolo_msgs.msg"].DetectionArray = MockDetectionArray

from pose_estimator.utils.detections import (
    filter_detections_by_num_points,
    filter_detections_by_score,
    get_best_detections_per_class,
)


class TestDetectionFiltering(unittest.TestCase):
    def test_filter_by_score_drops_low_confidence(self):
        detections = [
            MockDetection("red", score=0.15),
            MockDetection("red", score=0.45),
            MockDetection("blue", score=0.20),
            MockDetection("green", score=0.85),
        ]
        arr = MockDetectionArray(detections=detections)
        filtered = filter_detections_by_score(arr, min_score=0.35)

        self.assertEqual(len(filtered.detections), 2)
        scores = [d.score for d in filtered.detections]
        self.assertIn(0.45, scores)
        self.assertIn(0.85, scores)
        self.assertNotIn(0.15, scores)
        self.assertNotIn(0.20, scores)

    def test_filter_by_score_all_rejected(self):
        detections = [
            MockDetection("red", score=0.10),
            MockDetection("blue", score=0.25),
        ]
        arr = MockDetectionArray(detections=detections)
        filtered = filter_detections_by_score(arr, min_score=0.50)
        self.assertEqual(len(filtered.detections), 0)

    def test_filter_by_score_all_accepted(self):
        detections = [
            MockDetection("red", score=0.60),
            MockDetection("blue", score=0.75),
        ]
        arr = MockDetectionArray(detections=detections)
        filtered = filter_detections_by_score(arr, min_score=0.30)
        self.assertEqual(len(filtered.detections), 2)

    def test_filter_by_score_boundary_condition(self):
        detections = [
            MockDetection("red", score=0.35),
            MockDetection("red", score=0.3499),
        ]
        arr = MockDetectionArray(detections=detections)
        filtered = filter_detections_by_score(arr, min_score=0.35)
        self.assertEqual(len(filtered.detections), 1)
        self.assertEqual(filtered.detections[0].score, 0.35)

    def test_integration_with_best_detections_per_class(self):
        # Multiple red detections: low confidence and medium confidence
        detections = [
            MockDetection("red", score=0.18, num_points=4),
            MockDetection("red", score=0.40, num_points=4),
            MockDetection("blue", score=0.22, num_points=4),
        ]
        arr = MockDetectionArray(detections=detections)

        # Apply num_points filter followed by score filter (matching dockwin_pose_estimator_node)
        filtered = filter_detections_by_num_points(arr, min_num_points=3)
        filtered = filter_detections_by_score(filtered, min_score=0.35)
        best = get_best_detections_per_class(filtered, ["red", "blue"])

        self.assertIn("red", best)
        self.assertEqual(best["red"].score, 0.40)
        self.assertNotIn("blue", best)


if __name__ == "__main__":
    unittest.main()
