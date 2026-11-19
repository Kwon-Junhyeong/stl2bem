"""Cross-check of the gallery against bempp-cl: same STL mesh, same frequency and direction, independent code and
formulation (see bempp_solve.py). Reads examples/gallery/data/<name>.npz (stl2bem results) and stores
data/<name>.npz with the bempp-cl field on the same plane and the surface values at the element centroids.

    python examples/crosscheck_bempp/run.py [--only star,mug]      # in an environment with bempp-cl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from stl2bem import build_mesh
from stl2bem.viz import plane_grid
from bempp_solve import Solver, make_grid

NAMES = ["star", "boat", "mug", "bottle", "bell", "pipe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    names = [s for s in args.only.split(",") if s] or NAMES
    path = os.path.join(HERE, "results.json")
    summary = json.load(open(path)) if os.path.exists(path) else {}
    for name in names:
        own = os.path.join(HERE, "data", f"ours_{name}.npz")     # resonators: off-resonance field (ours.py)
        g = np.load(own if os.path.exists(own) else os.path.join(ROOT, "examples", "gallery", "data", f"{name}.npz"))
        mesh = build_mesh(os.path.join(ROOT, "examples", "gallery", "models", f"{name}.stl"), units="mm",
                          recenter="none", verbose=False)
        ours_field = g["field"]
        pts, shape = plane_grid(g["center"], g["u"], g["v"], g["half"], ours_field.shape)
        ok = np.isfinite(ours_field.ravel())
        t0 = time.time()
        s = Solver(make_grid(mesh.verts, mesh.tris))
        sol, k, d, info = s.total_surface(float(g["freq"]), g["doa"])
        field = np.full(len(pts), np.nan + 1j * np.nan, np.complex128)
        field[ok] = s.field(sol, k, d, pts[ok])
        surf = s.at_centroids(sol)
        info["seconds_total"] = time.time() - t0
        np.savez_compressed(os.path.join(HERE, "data", f"{name}.npz"), field=field.reshape(shape).astype(np.complex64),
                            surface=surf.astype(np.complex64), freq=g["freq"])
        df = np.abs(ours_field.ravel()[ok] - field[ok])                # incident amplitude is 1
        ds = np.abs(g["surface"] - surf) / np.abs(surf).mean()
        rec = dict(freq_hz=float(g["freq"]), elements=len(mesh.tris), p1_nodes=int(s.space.global_dof_count),
                   field_median=float(np.median(df)), field_p95=float(np.percentile(df, 95)),
                   field_rel_mean=float(df.mean() / np.abs(field[ok]).mean()),
                   surface_rel_mean=float(ds.mean()), surface_p95=float(np.percentile(ds, 95)), **info)
        summary[name] = rec
        json.dump(summary, open(path, "w"), indent=1)
        print(name, json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
