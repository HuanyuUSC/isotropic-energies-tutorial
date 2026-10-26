"""Rendering helpers: paint an equirectangular texture (e.g. the Earth) onto a mesh through
its spherical parameterization."""
import numpy as np

BLUE_MARBLE = "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57730/land_ocean_ice_2048.jpg"


def earth_texture(url=BLUE_MARBLE):
    """NASA Blue Marble (public domain), equirectangular, as an RGB uint8 array."""
    from PIL import Image
    from .meshes import _download
    return np.asarray(Image.open(_download(url)).convert("RGB"))


def subdivide(V, S, F, levels=1):
    """Midpoint subdivision of the mesh and of its spherical map (midpoints re-projected to
    the sphere = the piecewise-gnomonic map evaluated at edge midpoints)."""
    for _ in range(levels):
        E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
        E, inv = np.unique(E, axis=0, return_inverse=True)
        m = len(V) + np.arange(len(E))
        V = np.vstack([V, (V[E[:, 0]] + V[E[:, 1]]) / 2])
        Sm = S[E[:, 0]] + S[E[:, 1]]
        S = np.vstack([S, Sm / np.linalg.norm(Sm, axis=1, keepdims=True)])
        a, b, c = m[inv[:len(F)]], m[inv[len(F):2 * len(F)]], m[inv[2 * len(F):]]
        F = np.concatenate([np.c_[F[:, 0], a, c], np.c_[a, F[:, 1], b], np.c_[c, b, F[:, 2]], np.c_[a, b, c]])
    return V, S, F


def sphere_uv(S):
    lon = np.arctan2(S[:, 1], S[:, 0])
    lat = np.arcsin(np.clip(S[:, 2], -1, 1))
    return np.c_[(lon + np.pi) / (2 * np.pi), (lat + np.pi / 2) / np.pi]


def corner_uv_mesh(V, S, F):
    """Unshared-corner mesh with per-corner UVs; triangles crossing the date line are unwrapped."""
    uv = sphere_uv(S)[F]                       # (t, 3, 2)
    u = uv[..., 0]
    wrap = u.max(1) - u.min(1) > 0.5
    u[wrap] = np.where(u[wrap] < 0.5, u[wrap] + 1, u[wrap])
    return V[F].reshape(-1, 3), np.arange(3 * len(F)).reshape(-1, 3), uv.reshape(-1, 2)


def render_textured(V, S, F, texture, camera=None, size=(1200, 1200), filename=None, rotate=None,
                    blend=0.0, shape=False):
    """Render the mesh painted with an equirectangular texture via its spherical map S.

    blend: mix the texture toward light gray (0 = full texture, 0.5 = "50% opacity" over a plaster model).
    shape: shape-revealing look (smooth normals, key + fill light, low ambient, ambient occlusion),
           so the geometry stays readable under a dark texture."""
    import pyvista as pv
    if rotate is not None:
        S = S @ rotate.T
    Vc, Fc, uv = corner_uv_mesh(V, S, F)
    faces = np.hstack([np.full((len(Fc), 1), 3), Fc]).ravel()
    mesh = pv.PolyData(Vc, faces)
    mesh.active_texture_coordinates = uv.astype(np.float32)
    if blend:
        texture = (texture * (1 - blend) + 235 * blend).astype(np.uint8)
    tex = pv.Texture(np.ascontiguousarray(texture))      # row 0 = north (pv.Texture puts row 0 at the top)
    tex.repeat = True
    tex.interpolate = True
    if shape:
        mesh = mesh.compute_normals(split_vertices=False)
        p = pv.Plotter(off_screen=True, window_size=size, lighting="none")
        # lights move with the camera: key from upper right, weak fill from the left
        p.add_light(pv.Light(position=(0.6, 0.8, 1.0), light_type="camera light", intensity=1.0))
        p.add_light(pv.Light(position=(-1.0, 0.1, 0.6), light_type="camera light", intensity=0.35))
        p.add_mesh(mesh, texture=tex, smooth_shading=True, ambient=0.12, diffuse=0.95, specular=0.25, specular_power=20)
        p.enable_ssao(radius=0.15, bias=0.005, kernel_size=128)
    else:
        p = pv.Plotter(off_screen=True, window_size=size)
        p.add_mesh(mesh, texture=tex, smooth_shading=False, ambient=0.35, diffuse=0.75, specular=0.1)
    p.set_background("white")
    if camera is not None:
        p.camera_position = camera
    img = p.screenshot(filename, return_img=True)
    p.close()
    return img
