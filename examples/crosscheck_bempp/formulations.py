"""CBIE + CHIEF (default) against Burton-Miller (--formulation burton-miller), both measured against bempp-cl.

Uses the data of run.py / ours.py / resonance.py in this folder:
  * field on the gallery slices (same frequency and direction as run.py; the resonators off resonance), difference
    to bempp-cl relative to max(|p|, 1), and the solve time;
  * the two resonances, swept around the peaks with Burton-Miller and fitted like resonance.py.
Writes results_formulations.json.

    python examples/crosscheck_bempp/formulations.py [--backend cupy]
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
ROOT = os.path.join(HERE, "..", "..")
GAL = os.path.join(ROOT, "examples", "gallery")
sys.path.insert(0, ROOT)
from stl2bem import build_mesh, solve_table
from stl2bem.analysis import load_table
from stl2bem.viz import plane_grid


def fit_resonance(f, p):
    """p(f) = a + b / (f - z) by least squares (same as resonance.py, kept here so no bempp-cl is needed)."""
    A = np.stack([p, f.astype(complex), np.ones_like(p)], axis=1)
    z, a, c = np.linalg.lstsq(A, p * f, rcond=None)[0]
    return z, a, c + a * z


def solve(mesh, freqs, doa, pts, form, backend):
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "t.h5")
        solve_table(mesh.verts, mesh.tris, mesh.centroids, mesh.normals, mesh.areas, freqs, np.atleast_2d(doa), fn,
                    receivers=pts, backend=backend, verbose=False, formulation=form, budget_mb=512)
        return load_table(fn).p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()
    out = {}
    for name in ["star", "boat", "mug", "bottle", "bell", "pipe"]:
        own = os.path.join(HERE, "data", f"ours_{name}.npz")
        g = np.load(own if os.path.exists(own) else os.path.join(GAL, "data", f"{name}.npz"))
        ref = np.load(os.path.join(HERE, "data", f"{name}.npz"))
        mesh = build_mesh(os.path.join(GAL, "models", f"{name}.stl"), units="mm", recenter="none", verbose=False)
        pts, _ = plane_grid(g["center"], g["u"], g["v"], g["half"], g["field"].shape[::-1])
        ok = np.isfinite(g["field"].ravel())
        rec = {}
        for form in ("chief", "burton-miller"):
            t0 = time.time()
            field = solve(mesh, [float(g["freq"])], g["doa"], pts[ok], form, args.backend)[:, 0, 0]
            rf = ref["field"].ravel()[ok]
            d = np.abs(field - rf) / np.maximum(np.abs(rf), 1.0)
            rec[form] = dict(median=float(np.median(d)), p95=float(np.percentile(d, 95)), seconds=time.time() - t0)
        out[name] = rec
        print(name, json.dumps(rec), flush=True)
    res = json.load(open(os.path.join(HERE, "results.json")))
    for name, lo, hi, n in (("bottle", 144.3, 145.6, 27), ("pipe", 1420.0, 1475.0, 23)):
        o = np.load(os.path.join(HERE, "data", f"ours_{name}.npz"))
        mesh = build_mesh(os.path.join(GAL, "models", f"{name}.stl"), units="mm", recenter="none", verbose=False)
        f = np.linspace(lo, hi, n)
        p = solve(mesh, f, o["doa"], np.atleast_2d(o["probe"]), "burton-miller", args.backend)[0, 0, :].astype(complex)
        k = int(np.argmax(np.abs(p)))
        z, _, b = fit_resonance(f[max(0, k - 4):k + 5], p[max(0, k - 4):k + 5])
        q = res[f"{name}_resonance"]
        out[f"{name}_resonance"] = dict(bm_f0=float(z.real), bm_halfwidth=float(abs(z.imag)),
                                        bm_peak_db=float(20 * np.log10(abs(b / z.imag))),
                                        chief_or_cbie_f0=q["ours_f0"], bempp_f0=q["bempp_f0"])
        print(name, json.dumps(out[f"{name}_resonance"]), flush=True)
    json.dump(out, open(os.path.join(HERE, "results_formulations.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
