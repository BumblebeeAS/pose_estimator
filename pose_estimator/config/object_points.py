import numpy as np

# Order: top-left, bottom-left, bottom-right, top-right
# x-coordinates increase rightwards, y-coordinates increase downwards.
# Origin at the top-left corner of the gate.
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
GATE_FRONT_OBJECT_POINTS_DICT = {}
for key, object_points_mm in GATE_FRONT_OBJECT_POINTS_DICT.items():
    object_points = np.array(object_points_mm, dtype=np.float32) / 1000.0
    GATE_FRONT_OBJECT_POINTS_DICT[key] = object_points

# Order: top-left, bottom-left, bottom-right, top-right
# x-coordinates increase rightwards, y-coordinates increase downwards.
# Origin at the top-left corner of the bin.
BIN_OBJECT_POINTS = np.array(
    [(0, 0), (0, 609.60), (304.80, 609.60), (304.80, 0)],
    dtype=np.float32,
)
BIN_OBJECT_POINTS = BIN_OBJECT_POINTS / 1000.0
