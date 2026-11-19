"""Cross-check, stl2bem side for the two resonators (wine bottle, elbow pipe).

At a sharp resonance a shift of the resonance frequency by a fraction of its width changes the amplitude many
times, so two correct codes need not agree at one frequency on the peak. The resonators are therefore compared
  * on the gallery plane at a frequency off resonance (FIELD_HZ), and
  * by the shape of the resonance itself: pressure at the probe point over a fine sweep around the peak.
Writes data/ours_<name>.npz. Run with stl2bem's environment (GPU if available).

    python examples/crosscheck_bempp/ours.py [--backend cupy]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples", "gallery"))
from stl2bem import build_mesh, chief_points
from stl2bem.viz import field_on_points, plane_grid, surface_on_elements
from run import MODELS, spectrum  # examples/gallery/run.py

FIELD_HZ = {"bottle": 300.0, "pipe": 1100.0}             # off resonance (smooth parts of the gallery spectra)
SWEEP_HZ = {"bottle": (139.0, 150.0, 221), "pipe": (1375.0, 1520.0, 291)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--n", type=int, default=241)
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    for name, freq in FIELD_HZ.items():
        cfg = MODELS[name]
        mesh = build_mesh(os.path.join(ROOT, "examples", "gallery", "models", f"{name}.stl"), units="mm",
                          recenter="none", verbose=False)
        chief = chief_points(mesh, n=24, seed=0, oversample=200, use_gpu=(args.backend != "numpy"), verbose=False)
        doa = np.asarray(cfg["doa"], float)
        c0, u, v = (np.asarray(x, float) for x in cfg["plane"])
        pts, shape = plane_grid(c0, u, v, cfg["half"], (args.n, args.n))
        field = field_on_points(mesh, freq, doa, pts, chief=chief, backend=args.backend, budget_mb=512)
        surf = surface_on_elements(mesh, freq, doa, chief=chief, backend=args.backend)[:, 0, 0]
        f = np.linspace(*SWEEP_HZ[name])
        p = spectrum(mesh, cfg["probe"], doa, f, chief, args.backend)
        np.savez_compressed(os.path.join(HERE, "data", f"ours_{name}.npz"),
                            field=field.reshape(shape).astype(np.complex64), surface=surf.astype(np.complex64),
                            freq=freq, doa=doa, center=c0, u=u, v=v, half=np.asarray(cfg["half"], float),
                            probe=np.asarray(cfg["probe"], float), sweep_f=f, sweep_p=p.astype(np.complex64))
        k = int(np.argmax(np.abs(p)))
        print(name, f"field at {freq:g} Hz done; peak {f[k]:.2f} Hz, {20 * np.log10(abs(p[k])):.1f} dB", flush=True)


if __name__ == "__main__":
    main()
