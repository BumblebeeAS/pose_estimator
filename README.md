# Vision-based Pose Estimation

Pose estimation using PnP.

## Overview

There are generally two approaches:

Case 1: Object points are non-planar or homography cannot be well-estimated. Use only the inliers from RANSAC as homography estimation does not apply. Pose refinement, if applicable, is done on RANSAC inlier points.

Case 2: Homography estimation filters well. We use RANSAC to get an initial pose estimate but do not filter by RANSAC inliers because RANSAC filtering is too strict, resulting in too few point correspondences and a noisy pose estimate. Pose refinement is then done on points filtered by the homography.

## RoboSub 2025

### Slalom

**Idea**: Assign poles to slalom gate layers by assigning them to the closest red pole by depth.

**Assumptions**:

- YOLO detections detect red poles reliably.
- The camera looks at the slalom gate from the front (looking from the side would lead to inaccurate pairing of poles to layers).
