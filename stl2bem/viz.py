"""Visualization helpers: pressure on a plane through the body (field maps), body outlines, wave animations.

    from stl2bem.viz import plane_grid, field_on_points, section_segments, animate_field
    pts, shape = plane_grid(center=(0, 0, 0), u=(1, 0, 0), v=(0, 0, 1), half=(0.3, 0.3), n=(181, 181))
    p = field_on_points(mesh, 2000.0, [1, 0, 0], pts)          # NaN inside the body
    animate_field(p.reshape(shape), extent, "wave.gif", outline=section_segments(mesh, (0, 0, 0), (0, 1, 0)))
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from .analysis import load_table
from .bem_core import solve_table
from .chief import chief_points, points_inside


def plane_grid(center, u, v, half, n):
    """Points of a rectangular grid on the plane center + a u + b v, a in [-half[0], half[0]], b in [-half[1], half[1]].
    Returns (points (n_v * n_u, 3), (n_v, n_u)); row index along v, column along u."""
    center, u, v = (np.asarray(x, np.float64) for x in (center, u, v))
    a = np.linspace(-half[0], half[0], n[0])
    b = np.linspace(-half[1], half[1], n[1])
    A, B = np.meshgrid(a, b)
    pts = center + A[..., None] * u + B[..., None] * v
    return pts.reshape(-1, 3), A.shape


def field_on_points(mesh, freq, doa, points, chief=None, backend="auto", precision="single",
                    min_clearance: float = 0.0, budget_mb: float = 512.0, verbose: bool = False):
    """Total pressure (P,) at field points for one frequency and one direction; NaN inside the body (and within
    min_clearance meters of an element centroid, if given)."""
    pts = np.asarray(points, np.float64)
    inside = points_inside(pts, mesh.verts, mesh.tris, use_gpu=(backend != "numpy"))
    if min_clearance > 0:
        from scipy.spatial import cKDTree
        inside |= cKDTree(mesh.centroids).query(pts)[0] < min_clearance
    out = np.full(len(pts), np.nan + 1j * np.nan, np.complex64)
    sel = np.nonzero(~inside)[0]
    if chief is None:
        chief = chief_points(mesh, n=20, seed=0, use_gpu=(backend != "numpy"), verbose=False)
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "field.h5")
        solve_table(mesh.verts, mesh.tris, mesh.centroids, mesh.normals, mesh.areas, [float(freq)],
                    np.atleast_2d(np.asarray(doa, np.float64)), fn, receivers=pts[sel], chief_points=chief,
                    backend=backend, precision=precision, budget_mb=budget_mb, verbose=verbose)
        out[sel] = load_table(fn).p[:, 0, 0]
    return out


def surface_on_elements(mesh, freqs, doa, chief=None, backend="auto", verbose: bool = False):
    """Element (centroid) values (N, D, F) of the total surface pressure, for coloring the surface."""
    if chief is None:
        chief = chief_points(mesh, n=20, seed=0, use_gpu=(backend != "numpy"), verbose=False)
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "surf.h5")
        solve_table(mesh.verts, mesh.tris, mesh.centroids, mesh.normals, mesh.areas, np.atleast_1d(freqs),
                    np.atleast_2d(np.asarray(doa, np.float64)), fn, panel_indices=np.arange(len(mesh.tris)),
                    chief_points=chief, backend=backend, verbose=verbose)
        return load_table(fn).p


def section_segments(mesh, point, normal, u=None, v=None):
    """Intersection of the surface with a plane: list of 2D segments in the plane coordinates (u, v)."""
    point, normal = np.asarray(point, np.float64), np.asarray(normal, np.float64)
    normal = normal / np.linalg.norm(normal)
    if u is None:
        u = np.cross(normal, [0, 0, 1.0] if abs(normal[2]) < 0.9 else [1.0, 0, 0])
        u /= np.linalg.norm(u)
    if v is None:
        v = np.cross(normal, u)
    tri = mesh.verts[mesh.tris] - point
    d = tri @ normal
    segs = []
    for t, dd in zip(tri, d):
        if dd.min() >= 0 or dd.max() <= 0:
            continue
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if (dd[i] < 0) != (dd[j] < 0):
                s = dd[i] / (dd[i] - dd[j])
                q = t[i] + s * (t[j] - t[i])
                pts.append((q @ np.asarray(u), q @ np.asarray(v)))
        if len(pts) == 2:
            segs.append(pts)
    return segs


def animate_field(field2d, extent, out_path, outline=None, frames: int = 24, fps: int = 12, title: str = "",
                  cmap: str = "RdBu_r", vmax: float | None = None, dpi: int = 110):
    """GIF of Re{p e^{j w t}} over one period. field2d (n_v, n_u) complex with NaN inside the body;
    extent = (u_min, u_max, v_min, v_max) in meters; outline = section_segments(...)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection

    f = np.ma.masked_invalid(field2d)
    vmax = vmax or float(np.nanpercentile(np.abs(field2d), 99))
    fig, ax = plt.subplots(figsize=(5, 5 * (extent[3] - extent[2]) / (extent[1] - extent[0])), dpi=dpi)
    im = ax.imshow(np.real(f), origin="lower", extent=extent, cmap=cmap, vmin=-vmax, vmax=vmax,
                   interpolation="bilinear")
    if outline:
        ax.add_collection(LineCollection(outline, colors="k", linewidths=1.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    fig.tight_layout()

    def frame(i):
        im.set_data(np.real(f * np.exp(2j * np.pi * i / frames)))
        return (im,)

    FuncAnimation(fig, frame, frames=frames, blit=True).save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
