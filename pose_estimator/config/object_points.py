# Order: top-left, bottom-left, bottom-right, top-right
# x-coordinates increase rightwards, y-coordinates increase downwards.
# Origin at the top-left corner of the gate.

import numpy as np

# Gate when passing from the front.
GATE_FRONT_OBJECT_POINTS_DICT = {
    "gate_sides_left": [
        (0, 152.40),
        (0, 152.40 + 1219.20),
        (76.20, 152.40 + 1219.20),
        (76.20, 152.40),
    ],
    "gate_sides_right": [
        (3048.00 - 76.20, 152.40),
        (3048.00 - 76.20, 152.40 + 1219.20),
        (3048.00, 152.40 + 1219.20),
        (3048.00, 152.4),
    ],
    "gate_center": [
        ((3048.00 - 50.80) / 2, 0),
        ((3048.00 - 50.80) / 2, 609.60),
        ((3048.00 + 50.80) / 2, 609.60),
        ((3048.00 + 50.80) / 2, 0),
    ],
}
for key, object_points_mm in GATE_FRONT_OBJECT_POINTS_DICT.items():
    object_points = np.array(object_points_mm, dtype=np.float32) / 1000.0
    GATE_FRONT_OBJECT_POINTS_DICT[key] = object_points

# Gate when passing from the back.
# Same as passing from the front but left and right sides are swapped.
# NOTE: This is DIFFERENT from flipping the points as the matching of polygon points
# is done in a specific order.
GATE_BACK_OBJECT_POINTS_DICT = {
    "gate_sides_left": GATE_FRONT_OBJECT_POINTS_DICT["gate_sides_right"],
    "gate_sides_right": GATE_FRONT_OBJECT_POINTS_DICT["gate_sides_left"],
    "gate_center": GATE_FRONT_OBJECT_POINTS_DICT["gate_center"],
}

BIN_OBJECT_POINTS = np.array(
    [(0, 0), (0, 609.60), (304.80, 609.60), (304.80, 0)],
    dtype=np.float32,
)
BIN_OBJECT_POINTS = BIN_OBJECT_POINTS / 1000.0


# Planar object points for each gate (or layer) of the slalom.
# Each gate (white-red-white triplet0) is treated as a separate object with its own points.
SLALOM_GATE_OBJECT_POINTS_DICT = {
    "white_pole_left": [(0, 0), (0, 0.9144), (0.0254, 0.9144), (0.0254, 0)],
    "red_pole": [
        (0.0254 + 1.524, 0),
        (0.0254 + 1.524, 0.9144),
        (0.0254 + 1.524 + 0.0254, 0.9144),
        (0.0254 + 1.524 + 0.0254, 0),
    ],
    "white_pole_right": [
        (0.0254 + 1.524 + 0.0254 + 1.524, 0),
        (0.0254 + 1.524 + 0.0254 + 1.524, 0.9144),
        (0.0254 + 1.524 + 0.0254 + 1.524 + 0.0254, 0.9144),
        (0.0254 + 1.524 + 0.0254 + 1.524 + 0.0254, 0),
    ],
}
for key, object_points in SLALOM_GATE_OBJECT_POINTS_DICT.items():
    object_points = np.array(object_points, dtype=np.float32)
