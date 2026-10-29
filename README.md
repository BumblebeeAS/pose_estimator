# Vision-based Pose Estimation

Pose estimation using PnP.

## Overview

Task estimators subscribe to detections and / or point correspondences plus camera intrinsics, estimate a 3D pose with PnP / back-projection, and broadcast the result through TFs / poses.

There are generally two approaches:

Case 1: Object points are non-planar or homography cannot be well-estimated. Use only the inliers from RANSAC as homography estimation does not apply. Pose refinement, if applicable, is done on RANSAC inlier points.

Case 2: Homography estimation filters well. We use RANSAC to get an initial pose estimate but do not filter by RANSAC inliers because RANSAC filtering is too strict, resulting in too few point correspondences and a noisy pose estimate. Pose refinement is then done on points filtered by the homography.

## RobotX 2026

### Helipad

The helipad estimator finds the pad's center from its three concentric circles. It fits an ellipse to each detected circle and uses the known radius to estimate the pad's position. One usable circle is enough; when more are visible, their points are combined in the same estimate.

The detection labels run from `circle_1` on the outside to `circle_3` on the inside. The circles cannot tell us which way the pad is facing, so the published orientation is aligned with `odom_ned`.

### Buoy poses and gate tracking

`buoy_pose_estimator_node` consumes classified `yolo_msgs/DetectionArray` messages on `buoy/classified_detections` and rectified CameraInfo on `rect/camera_info`. Accepted labels are `buoy_red`, `buoy_green`, `buoy_blue`, `buoy_off`, and `buoy_unknown`. Classification runs upstream; this node does not load a classifier or subscribe to images. Use the separately configured classifier in `yolo_ros_trt` to produce these labels, or provide an equivalent detection stream.

Every matching detection is processed, including multiple buoys of the same color. Pose estimation reuses `get_object_pose_from_detection_using_best_fit_quad()` for quad fitting, PnP, and refinement. The modeled bottom footprint is a centered 0.4318 m square, excluding floats, from the 17-inch panel bottom edge in the [official RoboBuoy drawing](https://robonation.gitbook.io/robotx-2026-team-handbook/robotx-2026-or-task-build-guides/robobuoy-build-guide#robobuoy-side-panels). Its points are in `config/robotx26_object_points.py`. This assumes a near top-down body silhouette whose fitted corners represent that footprint; oblique views or masks containing floats can bias range. The shared quad helper falls back to an oriented bounding rectangle if fitting raises an error. There are no image-border or quad-coverage rejection thresholds.

Outputs are `buoy/detections` (`bb_robotx_msgs/BuoyDetectionArray`), `buoy/poses` (`geometry_msgs/PoseArray`), and `buoy/pose_labels` (`visualization_msgs/MarkerArray`). All poses and markers retain the source camera frame and detection timestamp, and positions refer to the bottom footprint center. Orientation is identity because unique yaw is unobserved. Color values are 0 off/BLACK, 1 red, 2 green, 3 blue, and 255 unknown. Failed estimates and invalid, behind-camera, or out-of-range positions are skipped; `max_range` defaults to 100 m. Empty frames publish empty pose arrays and a stamped marker clear. Text markers have per-frame IDs, matching colors, 0.15 m height, and a one-second lifetime.

Configure `input_detections_topic`, `camera_info_topic`, `output_detections_topic`, `output_poses_topic`, and `output_pose_labels_topic` as needed. Common-frame conversion belongs downstream. When using `buoy_gate_manager_node`, set its `buoy_detections_topic` to the estimator's actual output topic and provide the TF tree needed for the manager's world-frame conversion. The gate manager itself is unchanged.

## RoboSub 2025

### Slalom

**Idea**: Assign poles to slalom gate layers by assigning them to the closest red pole by depth.

**Assumptions**:

- YOLO detections detect red poles reliably.
- The camera looks at the slalom gate from the front (looking from the side would lead to inaccurate pairing of poles to layers).
