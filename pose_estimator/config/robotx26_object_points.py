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

WINDOW_FRAME_REMAP = {
    "red_window": "dockwin/red",
    "green_window": "dockwin/green",
    "blue_window": "dockwin/blue",
    "null_window": "dockwin/null",
}
