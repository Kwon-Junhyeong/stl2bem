"""Cross-check of the resonances with bempp-cl: pressure at the probe point at a few frequencies around the peak,
then the resonance a + b / (f - z) fitted to them (Re z = resonance frequency, Im z = half width).
Reads data/ours_<name>.npz (ours.py) for the probe, direction and sweep window; writes data/res_<name>.npz.

    python examples/crosscheck_bempp/resonance.py          # in an environment with bempp-cl
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from stl2bem import build_mesh
from bempp_solve import Solver, make_grid

N_POINTS = 6


def fit_resonance(f, p):
    """Least-squares fit of p(f) = a + b / (f - z); linear in (z, a, c) through p f = p z + a f + c."""
    A = np.stack([p, f.astype(complex), np.ones_like(p)], axis=1)
    z, a, c = np.linalg.lstsq(A, p * f, rcond=None)[0]
    b = c + a * z
    return z, a, b


def main():
    path = os.path.join(HERE, "results.json")
    summary = json.load(open(path)) if os.path.exists(path) else {}
    for name in ["bottle", "pipe"]:
        o = np.load(os.path.join(HERE, "data", f"ours_{name}.npz"))
        mesh = build_mesh(os.path.join(ROOT, "examples", "gallery", "models", f"{name}.stl"), units="mm",
                          recenter="none", verbose=False)
        s = Solver(make_grid(mesh.verts, mesh.tris))
        f_ours, p_ours = o["sweep_f"], o["sweep_p"].astype(complex)
        k = int(np.argmax(np.abs(p_ours)))
        z_o, _, b_o = fit_resonance(f_ours[max(0, k - 6):k + 7], p_ours[max(0, k - 6):k + 7])
        span = 6 * abs(z_o.imag) + 1.5 * (f_ours[1] - f_ours[0])         # a few half widths around our peak
        freqs = np.linspace(z_o.real - span, z_o.real + span, N_POINTS)
        p = []
        for fr in freqs:
            sol, kk, d, info = s.total_surface(float(fr), o["doa"])
            p.append(s.field(sol, kk, d, o["probe"][None])[0])
            print(name, f"{fr:.3f} Hz", abs(p[-1]), info, flush=True)
        p = np.array(p)
        z_b, _, b_b = fit_resonance(freqs, p)
        rec = dict(ours_f0=float(z_o.real), ours_halfwidth=float(abs(z_o.imag)),
                   ours_peak_db=float(20 * np.log10(abs(b_o / z_o.imag))),
                   bempp_f0=float(z_b.real), bempp_halfwidth=float(abs(z_b.imag)),
                   bempp_peak_db=float(20 * np.log10(abs(b_b / z_b.imag))))
        summary[f"{name}_resonance"] = rec
        json.dump(summary, open(path, "w"), indent=1)
        np.savez(os.path.join(HERE, "data", f"res_{name}.npz"), f=freqs, p=p, z=z_b, b=b_b)
        print(name, json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
