#!/usr/bin/env python3
"""PX4 local-origin reference and NED/ENU conversion for the buoy manager."""

from copy import deepcopy
import math

from geographic_msgs.msg import GeoPointStamped
from geometry_msgs.msg import PoseStamped
from pygeodesy.ltp import LocalCartesian
from px4_msgs.msg import VehicleGlobalPosition
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix, NavSatStatus


def pose_to_fix(message, converter, axes, gps_frame, altitude_valid=True):
    p = message.pose.position
    if not all(math.isfinite(v) for v in (p.x, p.y, p.z)):
        raise ValueError('Nonfinite local position')
    east, north, up = (p.y, p.x, -p.z) if axes == 'ned' else (p.x, p.y, p.z)
    point = converter.reverse(east, north, up)
    fix = NavSatFix(header=deepcopy(message.header))
    fix.header.frame_id = gps_frame
    fix.latitude, fix.longitude = point.lat, point.lon
    fix.altitude = point.height if altitude_valid else float('nan')
    fix.status.status = NavSatStatus.STATUS_FIX
    fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
    return fix


class Px4GeographicReference:
    """Shared PX4 reference and AMSL-to-ellipsoid correction for ROS nodes."""

    def setup_geographic_reference(self, reference_topic, global_position_topic):
        self.converter = None
        self.reference = None
        self.altitude_offset = None
        self.create_subscription(GeoPointStamped, reference_topic,
                                 self._reference, qos_profile_sensor_data)
        self.create_subscription(VehicleGlobalPosition, global_position_topic,
                                 self._global_position, qos_profile_sensor_data)

    def _update_converter(self):
        if self.reference is not None:
            lat, lon, altitude = self.reference
            self.converter = LocalCartesian(lat, lon, altitude + (self.altitude_offset or 0.))

    def _global_position(self, message):
        # Both heights describe the same sample, so their difference is
        # the local AMSL-to-ellipsoid correction, independent of UAV motion.
        offset = (message.alt_ellipsoid - message.alt
                  if message.alt_valid and message.timestamp
                  and math.isfinite(message.alt) and math.isfinite(message.alt_ellipsoid)
                  else None)
        if offset != self.altitude_offset:
            self.altitude_offset = offset
            self._update_converter()

    def _reference(self, message):
        p = message.position
        reference = (p.latitude, p.longitude, p.altitude)
        if (not all(math.isfinite(v) for v in reference)
                or not -90 <= p.latitude <= 90 or not -180 <= p.longitude <= 180):
            self.converter = None
            self.reference = None
            return
        if reference != self.reference:
            self.reference = reference
            self._update_converter()
            self.get_logger().info(f'GPS conversion reference: {reference}')
