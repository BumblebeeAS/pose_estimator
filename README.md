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

## RoboSub 2025

### Slalom

**Idea**: Assign poles to slalom gate layers by assigning them to the closest red pole by depth.

**Assumptions**:

- YOLO detections detect red poles reliably.
- The camera looks at the slalom gate from the front (looking from the side would lead to inaccurate pairing of poles to layers).
