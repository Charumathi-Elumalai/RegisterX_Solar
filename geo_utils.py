"""
RegistraX Solar — geolocation helpers.
Real GPS math (haversine great-circle distance), not a rough approximation.
"""
import math


def haversine_distance_meters(lat1, lon1, lat2, lon2):
    """Distance in meters between two lat/lon points on Earth's surface."""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (math.sin(d_phi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def is_within_geofence(emp_lat, emp_lon, site_lat, site_lon, radius_meters):
    """
    Returns (within: bool, distance_meters: float).
    Called as is_within_geofence(employee_lat, employee_lon, site_lat, site_lon, radius).
    """
    distance = haversine_distance_meters(
        float(emp_lat), float(emp_lon), float(site_lat), float(site_lon)
    )
    return distance <= float(radius_meters), distance
