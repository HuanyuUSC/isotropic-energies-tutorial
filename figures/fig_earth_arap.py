"""The Earth flattened by minimizing ARAP: the globe as a latitude-longitude triangle mesh, cut along one meridian
(the cut longitude: argv[1], default 0 = Greenwich), Tutte start, projected Newton (distortion.atlas.flatten).
Rendered with the Earth texture and Tissot circles (radius 4 deg at latitudes 0, +-30, +-60, every 30 deg of longitude).
    python figures/fig_earth_arap.py [cut_lon_deg]      -> assets/fig/d1_earth_arap[_cut<lon>].png"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from distortion import style  # noqa: E402
from distortion.atlas import flatten  # noqa: E402
from distortion.eigensystem import ARAP  # noqa: E402
from distortion.planar import dets  # noqa: E402
from distortion.render import earth_texture  # noqa: E402

style.use()
CUT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
HIGHLIGHT = "--highlight" in sys.argv                  # fill Africa (blue) and Greenland (orange), as on the Mercator maps
NLAT, NLON = 90, 180                                   # 2-degree grid


def globe(cut):
    """Vertices: two poles + an (NLAT - 1) x (NLON + 1) grid; the seam column is duplicated so the mesh is a disk."""
    lat = np.radians(np.linspace(-90, 90, NLAT + 1))[1:-1]
    lon = np.radians(cut + np.linspace(0, 360, NLON + 1))       # starts and ends at the cut longitude
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    ll = np.c_[LA.ravel(), LO.ravel()]
    ll = np.vstack([[-np.pi / 2, np.radians(cut + 180)], ll, [np.pi / 2, np.radians(cut + 180)]])
    V = np.c_[np.cos(ll[:, 0]) * np.cos(ll[:, 1]), np.cos(ll[:, 0]) * np.sin(ll[:, 1]), np.sin(ll[:, 0])]
    idx = lambda i, j: 1 + i * (NLON + 1) + j
    S, N = 0, len(V) - 1
    F = []
    for j in range(NLON):
        F.append([S, idx(0, j + 1), idx(0, j)])
        F.append([N, idx(NLAT - 2, j), idx(NLAT - 2, j + 1)])
        for i in range(NLAT - 2):
            a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j), idx(i + 1, j + 1)
            F += [[a, b, d], [a, d, c]]
    return V, np.array(F), ll


def main():
    V, F, ll = globe(CUT)
    hist = []
    X, E = flatten(V, F, iters=200, energy=ARAP(), tol=1e-10, history=hist)
    s = np.linalg.svd(E.deformation(X), compute_uv=False)
    print(f"cut at {CUT:g} deg: {len(F)} triangles, {len(hist)} Newton iterations, flips {(dets(X, F) <= 0).sum()}, "
          f"sigma1/sigma2 max {np.max(s[:, 0] / s[:, 1]):.2f}, area ratio range {np.min(s.prod(1)):.2f}..{np.max(s.prod(1)):.2f}")
    rest_area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    np.savez(os.path.join(HERE, "..", "..", "assets", "fig", f"d1_earth_arap_cut{CUT:g}_sigma.npz"),
             sigma=s, area=rest_area, lat=np.degrees(ll[F][:, :, 0].mean(1)),
             lon=np.degrees(np.angle(np.exp(1j * ll[F][:, :, 1]).sum(1))))      # circular mean, in (-180, 180]
    # orient: north up, west left (rotate so the north pole is above the south pole; mirror if needed)
    d = X[-1] - X[0]
    ang = np.arctan2(d[0], d[1])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    X = X @ R.T
    east = X[1 + (NLAT // 2) * (NLON + 1) + NLON // 2 + 10] - X[1 + (NLAT // 2) * (NLON + 1) + NLON // 2]
    if east[0] < 0:
        X[:, 0] *= -1
    # per-pixel texture: each output pixel -> its triangle -> barycentric -> (lat, lon) on the globe -> Earth texture
    import matplotlib.tri as mtri
    tex = earth_texture()
    th, tw = tex.shape[:2]
    tri = mtri.Triangulation(X[:, 0], X[:, 1], F)
    finder = tri.get_trifinder()
    lo, hi = X.min(0), X.max(0)
    W = 1800
    H = int(W * (hi[1] - lo[1]) / (hi[0] - lo[0]))
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], W), np.linspace(hi[1], lo[1], H))
    t = finder(gx, gy)
    img = np.ones((H, W, 3))
    m = t >= 0
    tt = t[m]
    P0, P1, P2 = X[F[tt, 0]], X[F[tt, 1]], X[F[tt, 2]]
    q = np.c_[gx[m], gy[m]] - P0
    e1, e2 = P1 - P0, P2 - P0
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    b1 = (q[:, 0] * e2[:, 1] - q[:, 1] * e2[:, 0]) / det
    b2 = (e1[:, 0] * q[:, 1] - e1[:, 1] * q[:, 0]) / det
    p3 = (1 - b1 - b2)[:, None] * V[F[tt, 0]] + b1[:, None] * V[F[tt, 1]] + b2[:, None] * V[F[tt, 2]]
    la = np.arcsin(np.clip(p3[:, 2] / np.linalg.norm(p3, axis=1), -1, 1))
    lg = np.arctan2(p3[:, 1], p3[:, 0])
    u = ((lg + np.pi) / (2 * np.pi) * tw).astype(int).clip(0, tw - 1)
    v = ((np.pi / 2 - la) / np.pi * th).astype(int).clip(0, th - 1)
    img[m] = 0.55 * tex[v, u] / 255.0 + 0.45                         # lighter, so the Tissot circles stand out
    fig, ax = plt.subplots(figsize=(9, 9 * H / W))
    ax.imshow(img, extent=(lo[0], hi[0], lo[1], hi[1]), interpolation="bilinear")
    # Tissot circles: map small circles on the sphere through the per-face affine maps (barycentric lookup)
    from scipy.spatial import cKDTree
    from distortion.maps import small_circle
    ctr = cKDTree(V[F].mean(1))
    for la in np.radians([-60, -30, 0, 30, 60]):
        for lon0 in np.radians(np.arange(-165, 181, 30)):
            clo, cla = small_circle(lon0, la, np.radians(4))
            P3 = np.c_[np.cos(cla) * np.cos(clo), np.cos(cla) * np.sin(clo), np.sin(cla)]
            _, fi = ctr.query(P3)
            pts = []
            for p, f in zip(P3, fi):
                A = V[F[f]]
                M = np.c_[A[1] - A[0], A[2] - A[0], np.cross(A[1] - A[0], A[2] - A[0])]
                b = np.linalg.solve(M, p - A[0])[:2]
                pts.append(X[F[f, 0]] + b[0] * (X[F[f, 1]] - X[F[f, 0]]) + b[1] * (X[F[f, 2]] - X[F[f, 0]]))
            ax.fill(*np.array(pts).T, color=style.ACCENT, alpha=0.55, lw=0)
    # the map's edge: mesh boundary edges (used by one triangle only), drawn as a dark outline
    E2 = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1)
    u_e, cnt = np.unique(E2, axis=0, return_counts=True)
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(X[u_e[cnt == 1]], colors="#333333", linewidths=1.6, zorder=8))

    def to_map(lon, lat):                     # a point on the globe -> its place on the flattened map
        P3 = np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        fi = ctr.query(P3)[1]
        A = V[F[fi]]
        e1, e2 = A[:, 1] - A[:, 0], A[:, 2] - A[:, 0]
        M = np.stack([e1, e2, np.cross(e1, e2)], axis=2)
        b = np.linalg.solve(M, (P3 - A[:, 0])[..., None])[:, :2, 0]
        Q = X[F[fi]]
        return Q[:, 0] + b[:, :1] * (Q[:, 1] - Q[:, 0]) + b[:, 1:] * (Q[:, 2] - Q[:, 0])

    # graticule: parallels every 30 deg, meridians every 30 deg (the 180 deg meridian is the outline itself)
    import matplotlib.patheffects as pe
    halo = [pe.Stroke(linewidth=2.6, foreground="#4a4a4a", alpha=0.55), pe.Normal()]
    t = np.radians(np.linspace(-179.8, 179.8, 721))
    for la in np.radians([-60, -30, 0, 30, 60]):
        Q = to_map(t, np.full_like(t, la))
        ax.plot(*Q.T, color="white", lw=1.0 if la else 1.6, alpha=0.9, zorder=7, path_effects=halo)
    t = np.radians(np.linspace(-89.5, 89.5, 361))
    for lon0 in np.radians(np.arange(-150, 151, 30)):
        Q = to_map(np.full_like(t, lon0), t)
        ax.plot(*Q.T, color="white", lw=1.6 if lon0 == 0 else 1.0, alpha=0.9, zorder=7, path_effects=halo)
    from distortion import maps
    GRE = maps.countries(lambda q: q["NAME"] == "Greenland")
    AFR = maps.countries(lambda q: q["CONTINENT"] == "Africa")

    def area(rings, outline=None):
        a = 0.0
        for outer, r in rings:
            Q = to_map(r[:, 0], r[:, 1])
            sa = 0.5 * abs(np.dot(Q[:, 0], np.roll(Q[:, 1], -1)) - np.dot(Q[:, 1], np.roll(Q[:, 0], -1)))
            a += sa if outer else -sa
            if outer and outline:
                ax.fill(*Q.T, color=outline, alpha=0.75, lw=0, zorder=6)
        return a
    ratio = area(AFR, style.BLUE if HIGHLIGHT else None) / area(GRE, style.ORANGE if HIGHLIGHT else None)
    print(f"Africa / Greenland apparent area on the ARAP map = {ratio:.2f} (true {maps.area_km2(AFR) / maps.area_km2(GRE):.1f})")
    pad = 0.01 * (hi[0] - lo[0]); ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal"); ax.axis("off")
    # stretch of every Tissot circle along its parallel (east) and meridian (north), from the face's 3D -> 2D map
    st = []
    for la in np.radians([-60, -30, 0, 30, 60]):
        for lon0 in np.radians(np.arange(-165, 181, 30)):
            p = np.array([np.cos(la) * np.cos(lon0), np.cos(la) * np.sin(lon0), np.sin(la)])
            f = ctr.query(p)[1]
            A3 = V[F[f]]; A2 = X[F[f]]
            M = np.c_[A2[1] - A2[0], A2[2] - A2[0]] @ np.linalg.pinv(np.c_[A3[1] - A3[0], A3[2] - A3[0]])
            east = np.array([-np.sin(lon0), np.cos(lon0), 0.0])
            north = np.array([-np.sin(la) * np.cos(lon0), -np.sin(la) * np.sin(lon0), np.cos(la)])
            st.append((np.degrees(la), np.linalg.norm(M @ east), np.linalg.norm(M @ north)))
    st = np.array(st)
    # st[:, 1:] /= np.sqrt(np.mean(st[:, 1] * st[:, 2]))           # remove the overall scale (ARAP's is free up to the mesh's)
    np.save(os.path.join(HERE, "..", "..", "assets", "fig", f"d1_earth_arap_cut{CUT:g}_tissot.npy"), st)
    print("Tissot stretches (parallel, meridian):", np.round(st[:, 1].min(), 2), np.round(st[:, 1].max(), 2),
          np.round(st[:, 2].min(), 2), np.round(st[:, 2].max(), 2))
    fig.tight_layout(pad=0)
    out = os.path.join(HERE, "..", "..", "assets", "fig", f"d1_earth_arap_cut{CUT:g}{'_hl' if HIGHLIGHT else ''}.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
