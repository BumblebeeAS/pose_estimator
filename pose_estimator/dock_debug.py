"""Debug visualisation for the dock lidar pose estimator: RViz markers and
the annotated BEV image."""

import cv2
import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


def dock_pose_marker(pose):
    marker = Marker()
    marker.header = pose.header
    marker.ns = "dock_pose"
    marker.id = 0
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    marker.pose = pose.pose
    marker.scale.z = 0.5
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 1.0
    marker.color.a = 1.0
    marker.text = "Dock Pose"
    return marker


def template_marker(header, candidate, match, pixel_to_asv, bev_scale,
                    bev_z_min, bev_z_max, stride):
    """Matched template as a CUBE_LIST in the BEV's base frame.

    Each non-empty template pixel (at the matched scale and rotation) is
    placed at its BEV position and lifted to the height its grey level
    encodes, coloured low (blue) to high (red), so the template can be
    compared against the point cloud in 3D.
    """
    sampled = candidate[::stride, ::stride]
    rows, cols = np.nonzero(sampled)
    grey = sampled[rows, cols].astype(np.float32)

    # Pixel centres in full-resolution BEV pixels, same placement as
    # draw_debug_image (candidate centred on the match centre).
    u = match["center_u"] - candidate.shape[1] / 2.0 + cols * stride + 0.5
    v = match["center_v"] - candidate.shape[0] / 2.0 + rows * stride + 0.5
    x, y = pixel_to_asv(u, v)
    z = bev_z_min + (grey - 1.0) / 254.0 * (bev_z_max - bev_z_min)
    bgr = cv2.applyColorMap(
        grey.astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET
    ).reshape(-1, 3) / 255.0

    marker = Marker()
    marker.header = header
    marker.ns = "dock_template"
    marker.id = 0
    marker.type = Marker.CUBE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    cell = bev_scale * stride
    marker.scale.x = cell
    marker.scale.y = cell
    marker.scale.z = cell
    marker.points = [
        Point(x=float(px), y=float(py), z=float(pz)) for px, py, pz in zip(x, y, z)
    ]
    marker.colors = [
        ColorRGBA(r=float(r), g=float(g), b=float(b), a=0.8) for b, g, r in bgr
    ]
    return marker


def clear_template_marker(header):
    marker = Marker()
    marker.header = header
    marker.ns = "dock_template"
    marker.id = 0
    marker.action = Marker.DELETE
    return marker


def draw_debug_image(bev_image, image_center, elapsed_ms, match=None,
                     candidate=None, accepted=False, position=None):
    """BEV in colour with the image centre, matched template outline and a
    status label. ``position`` is the published (x_forward, y_left), if any."""
    display = cv2.cvtColor(bev_image, cv2.COLOR_GRAY2BGR)
    cv2.drawMarker(
        display,
        (int(round(image_center[0])), int(round(image_center[1]))),
        (255, 255, 0),
        cv2.MARKER_CROSS,
        16,
        2,
    )

    label = "%.0fms" % elapsed_ms
    if match is not None:
        center_u, center_v = match["center_u"], match["center_v"]
        top_left = np.array(
            [[[int(round(center_u - candidate.shape[1] / 2.0)),
               int(round(center_v - candidate.shape[0] / 2.0))]]],
            dtype=np.int32,
        )
        # Green: matched dock template.
        contours, _ = cv2.findContours(
            candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            display, [c.astype(np.int32) + top_left for c in contours], -1,
            (0, 255, 0), 2,
        )
        cv2.circle(
            display, (int(round(center_u)), int(round(center_v))), 4,
            (0, 0, 255), -1,
        )
        label += " score=%.3f angle=%.1f scale=%.2f" % (
            match["score"], match["angle_deg"], match["scale"],
        )
        if position is not None:
            x_forward, y_left = position
            label += " x=%.2f y=%.2f bearing=%.1f" % (
                x_forward, y_left, np.degrees(np.arctan2(y_left, x_forward)),
            )
        elif not accepted:
            label += " below threshold"

    cv2.putText(
        display, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (0, 255, 255), 1, cv2.LINE_AA,
    )
    return display
