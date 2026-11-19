"""Validation, step 1 (compute): rigid sphere (r = 10 cm) against the exact series solution.

Stores in data/:
  field.npz        pressure on the plane y = 0 (exact and stl2bem), plane wave from +x, ka = 3.66
  ring.npz         surface pressure on a great circle at ka = 1, 3, 6 (exact, stl2bem, original code)
  convergence.json error versus element size, original code and stl2bem
Pictures: export.py (rasters, animation) and docs/figures/*.tex. The sphere goes through an STL file in mm.

    python examples/validation_sphere/run.py [--backend cupy|numpy]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from stl2bem import build_mesh, chief_points, map_to_elements, rigid_sphere, solve_table, surface_receivers, write_stl
from stl2bem.analysis import load_table
from stl2bem.shapes import icosphere
from stl2bem.viz import field_on_points, plane_grid

A, C = 0.10, 343.0


def sphere_mesh(sub):
    """Write the icosphere as an STL in mm and read it back through the normal STL path."""
    v, t = icosphere(A * 1000.0, sub)
    fn = os.path.join(tempfile.gettempdir(), f"stl2bem_sphere_sub{sub}.stl")
    write_stl(fn, v[t], header="stl2bem validation sphere, mm")
    return build_mesh(fn, units="mm", recenter="none", verbose=False)


def table(mesh, freqs, doa, mics, method, backend):
    pts = chief_points(mesh, n=20, seed=0, use_gpu=(backend != "numpy"), verbose=False)
    if method == "original":           # unreleased code this package grew from
        kw = dict(panel_indices=map_to_elements(mics, mesh, verbose=False), integration="centroid")
    else:
        rec, elem, _ = surface_receivers(mics, mesh, verbose=False)
        kw = dict(receivers=rec, receiver_elements=elem)
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.h5")
        solve_table(mesh.verts, mesh.tris, mesh.centroids, mesh.normals, mesh.areas, freqs, doa, fn,
                    chief_points=pts, backend=backend, verbose=False, **kw)
        return load_table(fn).p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--n", type=int, default=241)
    args = ap.parse_args()
    out = os.path.join(HERE, "data")
    os.makedirs(out, exist_ok=True)

    # field on y = 0
    f_field, doa = 2000.0, np.array([1.0, 0.0, 0.0])
    mesh = sphere_mesh(4)
    half = 0.3
    pts, shape = plane_grid((0, 0, 0), (1, 0, 0), (0, 0, 1), (half, half), (args.n, args.n))
    p_bem = field_on_points(mesh, f_field, doa, pts, backend=args.backend, budget_mb=512)
    outside = np.linalg.norm(pts, axis=1) >= A
    p_ex = np.full(len(pts), np.nan + 1j * np.nan, np.complex128)
    p_ex[outside] = rigid_sphere(2 * np.pi * f_field / C, A, pts[outside], doa[None])[:, 0]
    np.savez_compressed(os.path.join(out, "field.npz"), bem=p_bem.reshape(shape), exact=p_ex.reshape(shape),
                        half=half, freq=f_field, ka=2 * np.pi * f_field / C * A, elements=len(mesh.tris))

    # surface ring at ka = 1, 3, 6
    th = np.radians(np.arange(0, 360, 2.0))
    ring = A * np.column_stack([np.cos(th), np.zeros_like(th), np.sin(th)])
    kas = np.array([1.0, 3.0, 6.0])
    freqs = kas * C / (2 * np.pi * A)
    m3 = sphere_mesh(3)
    old_idx = np.arange(2, len(th), 10)            # 20 deg apart: each point snaps to its own element
    exact = np.stack([rigid_sphere(2 * np.pi * f / C, A, ring, doa[None])[:, 0] for f in freqs], 1)
    np.savez_compressed(os.path.join(out, "ring.npz"), deg=np.degrees(th), ka=kas, exact=exact,
                        new=table(m3, freqs, doa[None], ring, "stl2bem", args.backend)[:, 0, :],
                        old_deg=np.degrees(th[old_idx]),
                        old=table(m3, freqs, doa[None], ring[old_idx], "original", args.backend)[:, 0, :],
                        elements=len(m3.tris))

    # convergence
    rng = np.random.default_rng(1)
    mics = rng.normal(size=(24, 3))
    mics = A * mics / np.linalg.norm(mics, axis=1, keepdims=True)
    i = np.arange(26) + 0.5
    y = 1 - 2 * i / 26
    r = np.sqrt(1 - y * y)
    phi = np.pi * (1 + 5 ** 0.5) * i
    dirs = np.column_stack([np.cos(phi) * r, y, np.sin(phi) * r])
    fc = [500.0, 2000.0]
    ref = [rigid_sphere(2 * np.pi * f / C, A, mics, dirs) for f in fc]
    rows = []
    for sub in (2, 3, 4):
        m = sphere_mesh(sub)
        for meth in ("original", "stl2bem"):
            p = table(m, fc, dirs, mics, meth, args.backend)
            rows.append({"subdivision": sub, "elements": len(m.tris), "edge_mm": m.info["edge_mean_m"] * 1e3,
                         "method": meth, "rel_error": [float(np.abs(p[:, :, k] - ref[k]).mean() / np.abs(ref[k]).mean())
                                                        for k in range(len(fc))]})
            print(rows[-1], flush=True)
    json.dump({"freqs": fc, "rows": rows}, open(os.path.join(out, "convergence.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
