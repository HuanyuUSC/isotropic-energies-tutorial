"""Shade a flattened (2D) mesh with the lighting of its 3D surface: each 2D triangle gets the color the 3D
triangle would have under a headlight from `cam` (ambient + Lambert diffuse, smooth vertex normals)."""
import numpy as np
from matplotlib.colors import to_rgb


def vertex_normals(V3, F):
    n = np.cross(V3[F[:, 1]] - V3[F[:, 0]], V3[F[:, 2]] - V3[F[:, 0]])          # area-weighted
    N = np.zeros_like(V3)
    for k in range(3):
        np.add.at(N, F[:, k], n)
    return N / np.linalg.norm(N, axis=1, keepdims=True)


def face_colors(V3, F, cam, target=(0, 0, 0), base="#e7e5df", ambient=0.35, diffuse=0.7):
    N = vertex_normals(V3, F)[F].mean(1)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    c = V3[F].mean(1)
    view = np.asarray(cam, float) - c
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    if np.mean(np.einsum("ij,ij->i", N, view)) < 0:                              # normals toward the camera
        N = -N
    L = np.asarray(cam, float) - np.asarray(target, float)
    L /= np.linalg.norm(L)
    I = ambient + diffuse * np.abs(N @ L)
    return np.clip(np.asarray(to_rgb(base))[None] * I[:, None], 0, 1)
