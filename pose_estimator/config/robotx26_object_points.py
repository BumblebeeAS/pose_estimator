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
    "left_window": [(0.0, 0.0), (0.21, 0.0), (0.21, 0.21), (0.0, 0.21)],
    "right_window": [(0.0, 0.0), (0.21, 0.0), (0.21, 0.21), (0.0, 0.21)],
}