# Unless otherwise stated, object points follow this convention:
# Order: top-left, bottom-left, bottom-right, top-right
# x-coordinates increase rightwards, y-coordinates increase downwards.
# Origin at the top-left corner.

import numpy as np

# https://robonation.gitbook.io/robotx-2026-team-handbook/section-3-autonomy-challenge/3.5-task-build-specifications

# Section 3.5.8: diameters 1.5 m, 1.0 m, 0.5 m.
HELIPAD_RADII = {
    "circle_1": 0.75,
    "circle_2": 0.50,
    "circle_3": 0.25,
}

# WINDOW_OBJECT_POINTS_DICT = {
#     "red_window": [(0.0, 0.0), (0.0, 0.29), (0.21, 0.29), (0.21, 0.0)],
#     "green_window": [(0.0, 0.0), (0.0, 0.29), (0.21, 0.29), (0.21, 0.0)],
#     "blue_window": [(0.0, 0.0), (0.0, 0.29), (0.21, 0.29), (0.21, 0.0)],
# }

WINDOW_OBJECT_POINTS_DICT = {  # includes the black borders
    "red_window": [(0.0, 0.0), (0.0, 0.36), (0.28, 0.36), (0.28, 0.0)],
    "green_window": [(0.0, 0.0), (0.0, 0.36), (0.28, 0.36), (0.28, 0.0)],
    "blue_window": [(0.0, 0.0), (0.0, 0.36), (0.28, 0.36), (0.28, 0.0)],
    "null_window": [(0.0, 0.0), (0.0, 0.36), (0.28, 0.36), (0.28, 0.0)],
}

# 1m x 1m square
WINDOW_WHITE_PLANE_DICT = {
    "dock_window_plane": np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]),
    
}


WINDOW_FRAME_REMAP = {
    "red_window": "dockwin/red",
    "green_window": "dockwin/green",
    "blue_window": "dockwin/blue",
    "null_window": "dockwin/null",
}

# RoboBuoy body bottom footprint, excluding floats and beacon.
# Source: official RoboBuoy Build Guide, "RoboBuoy Side Panels" drawing:
# https://robonation.gitbook.io/robotx-2026-team-handbook/robotx-2026-or-task-build-guides/robobuoy-build-guide#robobuoy-side-panels
# Each panel's bottom edge is 17 inches; the four edges form the square base.
BUOY_FOOTPRINT_WIDTH = 17.0 * 0.0254  # 0.4318 m
# Centered origin at bottom footprint; +x right, +y down in a top-down view,
# +z downward into the water. Corner order: TL, BL, BR, TR.
BUOY_FOOTPRINT_OBJECT_POINTS = np.array([
    [-BUOY_FOOTPRINT_WIDTH / 2, -BUOY_FOOTPRINT_WIDTH / 2, 0.0],
    [-BUOY_FOOTPRINT_WIDTH / 2, BUOY_FOOTPRINT_WIDTH / 2, 0.0],
    [BUOY_FOOTPRINT_WIDTH / 2, BUOY_FOOTPRINT_WIDTH / 2, 0.0],
    [BUOY_FOOTPRINT_WIDTH / 2, -BUOY_FOOTPRINT_WIDTH / 2, 0.0],
], dtype=np.float64)
