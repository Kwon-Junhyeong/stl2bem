"""Gallery, extra: the boat's field on a horizontal slice at mid-hull height ("water level"), same frequency and
direction as in run.py. Used for the picture at the top of the README (export.py).

    python examples/gallery/water_slice.py [--backend cupy]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from stl2bem import build_mesh, chief_points
from stl2bem.viz import field_on_points, plane_grid
from run import MODELS  # examples/gallery/run.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()
    cfg = MODELS["boat"]
    mesh = build_mesh(os.path.join(HERE, "models", "boat.stl"), units="mm", recenter="none", verbose=False)
    chief = chief_points(mesh, n=24, seed=0, oversample=200, use_gpu=(args.backend != "numpy"), verbose=False)
    center, u, v = np.array([0.0025, 0.0, 0.02]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    half, n = (0.2, 0.13), (241, 157)                     # square cells of 1.67 mm
    pts, shape = plane_grid(center, u, v, half, n)
    field = field_on_points(mesh, cfg["freq"], np.asarray(cfg["doa"], float), pts, chief=chief,
                            backend=args.backend, budget_mb=512)
    np.savez_compressed(os.path.join(HERE, "data", "boat_water.npz"), field=field.reshape(shape).astype(np.complex64),
                        center=center, u=u, v=v, half=np.asarray(half), freq=cfg["freq"])
    print("boat water slice", shape)


if __name__ == "__main__":
    main()
