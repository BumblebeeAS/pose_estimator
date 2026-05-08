# Unless otherwise stated, object points follow this convention:
# Order: top-left, bottom-left, bottom-right, top-right
# x-coordinates increase rightwards, y-coordinates increase downwards.
# Origin at the top-left corner.

import numpy as np

# Gate when passing from the front.
GATE_FRONT_OBJECT_POINTS_DICT = {
    "left_gate": [
        (0, 152.40),
        (0, 152.40 + 1219.20),
        (76.20, 152.40 + 1219.20),
        (76.20, 152.40),
    ],
    "right_gate": [
        (3048.00 - 76.20, 152.40),
        (3048.00 - 76.20, 152.40 + 1219.20),
        (3048.00, 152.40 + 1219.20),
        (3048.00, 152.4),
    ],
    "mid_gate": [
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
    "left_gate": GATE_FRONT_OBJECT_POINTS_DICT["right_gate"],
    "right_gate": GATE_FRONT_OBJECT_POINTS_DICT["left_gate"],
    "mid_gate": GATE_FRONT_OBJECT_POINTS_DICT["mid_gate"],
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
    SLALOM_GATE_OBJECT_POINTS_DICT[key] = object_points

# Order: top-left, bottom-left, bottom-right, top-right
# x-coordinates increase rightwards, y-coordinates increase downwards.
# Origin at the CENTER. Yellow bucket on the left, pink bucket on the right.
# Buckets are 184 mm (x) by 270 mm (y). Table is 609.60 mm by 609.60 mm.
TRASH_TABLE_OBJECT_POINTS_DICT = {
    "table": [
        (-304.80, -304.80),
        (-304.80, 304.80),
        (304.80, 304.80),
        (304.80, -304.80),
    ],
    "yellow_bucket": [
        (-184 - 304.80, -270 / 2),
        (-184 - 304.80, 270 / 2),
        (-304.80, 270 / 2),
        (-304.80, -270 / 2),
    ],
    "pink_bucket": [
        (304.80, -270 / 2),
        (304.80, 270 / 2),
        (304.80 + 184, 270 / 2),
        (304.80 + 184, -270 / 2),
    ],
}

for key, object_points in TRASH_TABLE_OBJECT_POINTS_DICT.items():
    object_points = np.array(object_points, dtype=np.float32) / 1000.0
    TRASH_TABLE_OBJECT_POINTS_DICT[key] = object_points

SYMBOL_OBJECT_POINTS_DICT = {
    "reef_shark": [(0.0, 0.0), (0.0, 0.305), (0.305, 0.305), (0.305, 0.0)],
    "sawfish": [(0.0, 0.0), (0.0, 0.305), (0.305, 0.305), (0.305, 0.0)],
}

TORPEDO_OBJECT_POINTS_DICT = {
    "torpedo_1": [(0.0, 0.0), (0.0, 0.6096), (0.6096, 0.6096), (0.6096, 0.0)],
    "torpedo_2": [(0.0, 0.0), (0.0, 0.6096), (0.6096, 0.6096), (0.6096, 0.0)],
}
