"""Map projections of the Earth, Tissot's indicatrix, and their singular values.

Coastlines and countries: Natural Earth (public domain), downloaded at runtime.
All angles in radians; the globe has unit radius.
"""
import json

import numpy as np

from .meshes import _download

NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"


# ---------------------------------------------------------------- projections
def mercator(lon, lat):
    """Conformal: sigma_1 = sigma_2 = sec(lat)."""
    return lon, np.log(np.tan(np.pi / 4 + lat / 2))


def lambert(lon, lat):
    """Lambert cylindrical equal-area: sigma = (sec(lat), cos(lat)), product 1."""
    return lon, np.sin(lat)


def gall_peters(lon, lat):
    """Gall-Peters equal-area: sigma = (sec(lat) / sqrt2, sqrt2 cos(lat)), product 1; true at 45 deg."""
    return lon / np.sqrt(2), np.sqrt(2) * np.sin(lat)


def equirectangular(lon, lat):
    """Plate carree: sigma = (sec(lat), 1)."""
    return lon, lat


SIGMAS = {   # principal stretches (along the parallel, along the meridian) at latitude lat
    mercator: lambda lat: (1 / np.cos(lat), 1 / np.cos(lat)),
    lambert: lambda lat: (1 / np.cos(lat), np.cos(lat)),
    gall_peters: lambda lat: (1 / (np.sqrt(2) * np.cos(lat)), np.sqrt(2) * np.cos(lat)),
    equirectangular: lambda lat: (1 / np.cos(lat), np.ones_like(lat)),
}


# ---------------------------------------------------------------- data
def _features(name):
    with open(_download(NE + name), encoding="utf8") as f:
        return json.load(f)["features"]


def _rings(geom):
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    for poly in polys:
        for k, ring in enumerate(poly):
            yield k == 0, np.radians(np.asarray(ring))   # (is_outer, [[lon, lat], ...])


def land():
    """Outer rings of all land polygons, as (lon, lat) arrays in radians."""
    return [r for f in _features("ne_50m_land.geojson") for outer, r in _rings(f["geometry"]) if outer]


def countries(where):
    """Rings of the countries whose properties satisfy `where(props)`."""
    out = []
    for f in _features("ne_50m_admin_0_countries.geojson"):
        if where(f["properties"]):
            out += [(outer, r) for outer, r in _rings(f["geometry"])]
    return out


def area_km2(rings, R=6371.0):
    """Area on the sphere: shoelace in the equal-area projection (exact up to the polygon's
    straight edges in that projection)."""
    total = 0.0
    for outer, r in rings:
        x, y = lambert(r[:, 0], r[:, 1])
        a = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        total += a if outer else -a
    return total * R**2


# ---------------------------------------------------------------- Tissot's indicatrix
def small_circle(lon0, lat0, radius, n=120):
    """Points at angular distance `radius` from (lon0, lat0): a true circle on the globe."""
    b = np.linspace(0, 2 * np.pi, n)
    lat = np.arcsin(np.sin(lat0) * np.cos(radius) + np.cos(lat0) * np.sin(radius) * np.cos(b))
    lon = lon0 + np.arctan2(np.sin(b) * np.sin(radius) * np.cos(lat0),
                            np.cos(radius) - np.sin(lat0) * np.sin(lat))
    return lon, lat
