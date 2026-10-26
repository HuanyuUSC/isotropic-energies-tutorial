"""Procedural 2D shapes for the planar demos: letters from a font outline, and a disk.

    letter(ch, height, max_area)   triangle mesh of a glyph (holes kept, e.g. the A), base at y = 0, centered in x
    disk(radius, max_area)         triangle mesh of a disk centered at the origin
"""
import numpy as np


def _triangulate(polygon, max_area):
    """Triangulate a shapely polygon (with holes) using `triangle`, quality mesh with area bound."""
    import triangle
    verts, segs, holes = [], [], []

    def ring(coords):
        P = np.asarray(coords)[:-1]
        k = len(verts)
        verts.extend(P.tolist())
        segs.extend([[k + i, k + (i + 1) % len(P)] for i in range(len(P))])

    ring(polygon.exterior.coords)
    for r in polygon.interiors:
        ring(r.coords)
        from shapely.geometry import Polygon
        holes.append(np.asarray(Polygon(r).representative_point().coords)[0])
    data = dict(vertices=np.array(verts), segments=np.array(segs))
    if holes:
        data["holes"] = np.array(holes)
    m = triangle.triangulate(data, f"pq30a{max_area}")
    return m["vertices"], m["triangles"]


def _resample(polygon, step):
    """Evenly resample every ring of a polygon so boundary edges have length ~step."""
    from shapely.geometry import Polygon

    def ring(coords):
        B = np.asarray(coords)[:-1]
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.vstack([B, B[:1]]), axis=0), axis=1))]
        t = np.linspace(0, L[-1], max(int(L[-1] / step), 8), endpoint=False)
        return np.c_[np.interp(t, L, np.r_[B[:, 0], B[0, 0]]), np.interp(t, L, np.r_[B[:, 1], B[0, 1]])]

    return Polygon(ring(polygon.exterior.coords), [ring(r.coords) for r in polygon.interiors])


def letter(ch, height=1.0, max_area=0.002, font="DejaVu Sans", weight="bold"):
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    path = TextPath((0, 0), ch, size=1.0, prop=FontProperties(family=font, weight=weight))
    rings = [Polygon(p) for p in path.to_polygons() if len(p) > 3]
    # the outer ring is the one with the largest area; the others are holes inside it
    rings.sort(key=lambda r: -r.area)
    outer = rings[0]
    shape = outer
    for r in rings[1:]:
        shape = shape.difference(r) if outer.contains(r) else unary_union([shape, r])
    minx, miny, maxx, maxy = shape.bounds
    from shapely import affinity
    s = height / (maxy - miny)
    shape = affinity.scale(affinity.translate(shape, -(minx + maxx) / 2, -miny), s, s, origin=(0, 0))
    shape = _resample(shape.buffer(0), np.sqrt(2.2 * max_area))
    V, F = _triangulate(shape, max_area)
    return _orient(V, F)


def disk(radius=0.4, max_area=0.002):
    from shapely.geometry import Point
    shape = _resample(Point(0, 0).buffer(radius, quad_segs=32), np.sqrt(2.2 * max_area))
    V, F = _triangulate(shape, max_area)
    return _orient(V, F)


def _orient(V, F):
    D = np.stack([V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]], 2)
    neg = np.linalg.det(D) < 0
    F = F.copy()
    F[neg] = F[neg][:, [0, 2, 1]]
    return np.asarray(V, float), np.asarray(F, np.int64)
