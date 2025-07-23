from dataclasses import dataclass

import numpy as np
from yolo_msgs.msg import Detection

from pose_estimator.utils.detections import get_detection_centroid
from pose_estimator.utils.pose_estimator import backproject_pixel
from pose_estimator.utils.pose_estimator_node import PoseEstimatorNode


@dataclass
class TrackObject:
    track_id: int = -1
    other_id: int = -1
    past_positions: np.ndarray = np.zeros((1, 3))
    life: int = -1


NUM_LAST_POSITIONS = 100


class TrashPoseEstimator(PoseEstimatorNode):
    def __init__(self, name):
        super().__init__(name)

        self.track_dict: dict[str, TrackObject] = {
            "bottle": TrackObject(),
            "ladle": TrackObject(),
        }

    def get_detection_position(
        self, detection: Detection, object_depth: float
    ) -> np.ndarray:
        """Get the position of the detection in the camera frame."""
        detection_centroid = get_detection_centroid(detection)
        X, Y = backproject_pixel(
            detection_centroid[0],
            detection_centroid[1],
            object_depth,
            self.camera.camera_matrix(),
        )
        return np.array([X, Y, object_depth])

    def get_track_detections(
        self,
        best_detections: dict[str, list[Detection]],
        object_depth: float,
        keep_track_alive_counter: int = 12,
    ) -> dict[str, Detection]:
        """Given not more than two objects per class, select the detection to track.

        We keep track of two track ids in each frame. The primary track id identifies the target
        the robot homes towards. The "other" track id exploits the fact that there are only two
        detections to track the primary target. In the current frame, if there is a detection
        with the primary track id from the previous frame, select that detection. Otherwise, if
        one of the detections has the "other" track id from the previous frame, skip the target
        for that frame and decrement a counter. Finally, if both track ids are not present in the
        current frame, select the detection with the closest centroid to the target in the
        previous frame. Note that the distances are computed in the camera frame.

        The idea of the "other" track id is that even if we lose track of the primary target, we
        are able to know that it is not the other object if the "other" track id is still valid.
        This allows us to re-track the primary target in subsequent frames if it reappears.

        Args:
            best_detections (dict[str, list[Detection]]): Dictionary of best detections per class.
            object_depth (float): Depth of the object in the camera frame.

        Returns:
            dict[str, Detection]: Dictionary of tracked detections.
        """
        track_detections = {}

        for class_name, detections in best_detections.items():
            if len(detections) == 0:
                continue

            if class_name in self.track_dict:
                # If the class is to be tracked,
                detections_by_id = {d.id: d for d in detections}
                curr_id = self.track_dict[class_name].track_id
                other_id = self.track_dict[class_name].other_id
                past_positions = self.track_dict[class_name].past_positions
                life = self.track_dict[class_name].life

                if curr_id in detections_by_id:
                    # Either select the detection with the previous frame's track_id
                    selected_detection = detections_by_id[curr_id]
                    life = keep_track_alive_counter

                elif len(detections) == 2 and other_id in detections_by_id:
                    # Or if there are two detections, and one has the previous frame's
                    # other_id, the detection with the new id
                    if detections[0].id != other_id:
                        selected_detection = detections[0]
                    else:
                        selected_detection = detections[1]
                    life = keep_track_alive_counter

                elif len(detections) == 1 and other_id in detections_by_id:
                    # Or if there is only one detection, and it has the previous frame's
                    # other_id, don't publish anything and decrement the life counter
                    if life > 0:
                        self.track_dict[class_name].life = life - 1
                        continue
                    else:
                        # If the life counter is zero, select the only available detection
                        selected_detection = detections[0]
                        life = keep_track_alive_counter

                else:
                    # Or the detection with the closest centroid to the tracked object
                    track_position = past_positions.mean(axis=0)
                    selected_dist = float("inf")
                    selected_detection = None
                    for detection in detections:
                        detection_position = self.get_detection_position(
                            detection, object_depth
                        )
                        dist = np.linalg.norm(track_position - detection_position)
                        if dist < selected_dist:
                            selected_dist = dist
                            selected_detection = detection

                # Update the track object properties
                other_detection = list(
                    filter(lambda d: d.id != selected_detection.id, detections)
                )
                if len(other_detection) > 0:
                    other_id = other_detection[0].id
                # Else, leave other_id as is
                detection_position = self.get_detection_position(
                    selected_detection, object_depth
                )
                past_positions = np.vstack((past_positions, detection_position))
                past_positions = past_positions[-NUM_LAST_POSITIONS:]
                self.track_dict[class_name] = TrackObject(
                    selected_detection.id, other_id, past_positions
                )

            else:
                selected_detection = detections[0]

            track_detections[class_name] = selected_detection

        return track_detections
