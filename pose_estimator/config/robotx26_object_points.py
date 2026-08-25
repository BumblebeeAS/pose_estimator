# Unless otherwise stated, object points follow this convention:
# Order: top-left, bottom-left, bottom-right, top-right
# x-coordinates increase rightwards, y-coordinates increase downwards.
# Origin at the top-left corner.

import numpy as np

HELIPAD_RADII = {
    "inner_circle": 0.3048,
    "outer_circle": 0.762,
}

WINDOW_OBJECT_POINTS_DICT = {
    "red_window": [(0.0, 0.0), (0.0, 0.29), (0.21, 0.29), (0.21, 0.0)],
    "green_window": [(0.0, 0.0), (0.0, 0.29), (0.21, 0.29), (0.21, 0.0)],
    "blue_window": [(0.0, 0.0), (0.0, 0.29), (0.21, 0.29), (0.21, 0.0)],
}

WINDOW_FRAME_REMAP = {
    "red_window": "dockwin/red",
    "green_window": "dockwin/green",
    "blue_window": "dockwin/blue",
}
