"""Gallery, step 1 (compute): six everyday objects (CAD models exported from Fusion as STL, mm).

For each object it stores in data/<name>.npz:
  field     total pressure on a plane through the object (NaN inside the body), one frequency and direction
  surface   element values of the surface pressure, same frequency and direction
  spec_f, spec_p   (hollow objects) pressure at a point in the air inside, over frequency
Pictures are made by export.py (rasters, animations) and the TeX files in docs/figures.

    python examples/gallery/run.py [--backend cupy] [--only star,boat] [--n 241]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from stl2bem import build_mesh, chief_points, solve_table
from stl2bem.analysis import load_table
from stl2bem.viz import field_on_points, plane_grid, surface_on_elements

C = 343.0
X, Y, Z = np.eye(3)
_D = np.array([1.0, 0, 0.35]) / np.linalg.norm([1.0, 0, 0.35])

# plane = (center, u, v) in meters; doa = unit vector towards the source; probe = a point in the air inside
MODELS = {
    "star":   dict(freq=4000.0, doa=X, plane=((0, 0, 0.002), X, Y), half=(0.16, 0.16)),
    "boat":   dict(freq=3000.0, doa=_D, plane=((0, 0, 0.03), X, Z), half=(0.2, 0.12)),
    "mug":    dict(freq=2500.0, doa=-X, plane=((0.015, 0, 0.05), X, Z), half=(0.14, 0.11),
                   probe=(0, 0, 0.03), spectrum=(150.0, 2000.0, 64),
                   estimate=("quarter wave", C / (4 * (0.089 + 0.6 * 0.036)))),
    "bottle": dict(freq=None, doa=X, plane=((0, 0, 0.125), X, Z), half=(0.16, 0.19),
                   probe=(0, 0, 0.08), spectrum=(60.0, 400.0, 69),
                   estimate=("Helmholtz", 160.0)),
    "bell":   dict(freq=2500.0, doa=-Z, plane=((0, 0, 0.06), X, Z), half=(0.16, 0.14),
                   probe=(0, 0, 0.08), spectrum=(150.0, 3000.0, 58)),
    "pipe":   dict(freq=None, doa=-Z, plane=((0.06, 0, 0.065), X, Z), half=(0.16, 0.16),
                   probe=(0.04, 0, 0.12), spectrum=(200.0, 3000.0, 57),
                   estimate=("open pipe", C / (2 * (0.2228 + 1.2 * 0.011)))),
}


def spectrum(mesh, probe, doa, freqs, chief, backend):
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "s.h5")
        solve_table(mesh.verts, mesh.tris, mesh.centroids, mesh.normals, mesh.areas, freqs, np.atleast_2d(doa), fn,
                    receivers=np.atleast_2d(probe), chief_points=chief, backend=backend, verbose=False)
        return load_table(fn).p[0, 0, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--only", default="")
    ap.add_argument("--n", type=int, default=241, help="grid points per side of the field maps")
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    names = [s for s in args.only.split(",") if s] or list(MODELS)
    summary = {}
    for name in names:
        cfg = MODELS[name]
        t0 = time.time()
        mesh = build_mesh(os.path.join(HERE, "models", f"{name}.stl"), units="mm", recenter="none", verbose=False)
        chief = chief_points(mesh, n=24, seed=0, oversample=200, use_gpu=(args.backend != "numpy"), verbose=False)
        doa = np.asarray(cfg["doa"], float)
        rec = {"elements": len(mesh.tris), "edge_mm": mesh.info["edge_mean_m"] * 1e3}
        extra = {}
        if "spectrum" in cfg:
            f = np.linspace(*cfg["spectrum"])
            p = spectrum(mesh, cfg["probe"], doa, f, chief, args.backend)
            k = int(np.argmax(np.abs(p)))                  # refine around the highest peak (sharp resonances)
            df = f[1] - f[0]
            f2 = np.linspace(f[k] - 1.5 * df, f[k] + 1.5 * df, 31)
            p2 = spectrum(mesh, cfg["probe"], doa, f2, chief, args.backend)
            order = np.argsort(np.concatenate([f, f2]))
            f, p = np.concatenate([f, f2])[order], np.concatenate([p, p2])[order]
            k = int(np.argmax(np.abs(p)))
            rec.update(peak_hz=float(f[k]), peak_db=float(20 * np.log10(np.abs(p[k]))))
            extra.update(spec_f=f, spec_p=p.astype(np.complex64), probe=np.asarray(cfg["probe"], float))
            if "estimate" in cfg:
                rec.update(estimate=cfg["estimate"][0], estimate_hz=float(cfg["estimate"][1]))
        freq = cfg["freq"] or rec["peak_hz"]
        rec["field_hz"] = float(freq)
        c0, u, v = (np.asarray(x, float) for x in cfg["plane"])
        pts, shape = plane_grid(c0, u, v, cfg["half"], (args.n, args.n))
        field = field_on_points(mesh, freq, doa, pts, chief=chief, backend=args.backend, budget_mb=512)
        surf = surface_on_elements(mesh, freq, doa, chief=chief, backend=args.backend)[:, 0, 0]
        np.savez_compressed(os.path.join(HERE, "data", f"{name}.npz"), field=field.reshape(shape).astype(np.complex64),
                            surface=surf.astype(np.complex64), freq=freq, doa=doa, center=c0, u=u, v=v,
                            half=np.asarray(cfg["half"], float), **extra)
        rec["seconds"] = time.time() - t0
        summary[name] = rec
        print(name, json.dumps(rec), flush=True)
    path = os.path.join(HERE, "results.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(summary)
    json.dump(old, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
