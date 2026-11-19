"""Benchmark: time and peak memory per frequency against the number of elements, CPU and GPU.

Body: a 30 x 20 x 15 cm box with n x n squares per face (12 n^2 triangles), 8 microphones on the surface,
6 840 plane-wave directions (1 degree in azimuth x 10 degrees in elevation), 20 CHIEF points. Every case runs in
its own process so that peak memory can be read: resident memory on the CPU, device memory sampled during the
solve on the GPU (above what the process held before).
Times are per frequency (two frequencies solved after a warm-up on a small mesh), assembly included.

    python examples/benchmark/run.py --backend numpy --n 10,15,20,25,30
    python examples/benchmark/run.py --backend cupy  --n 10,15,20,25,30,35
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")


def case(n: int, backend: str) -> dict:
    import numpy as np
    sys.path.insert(0, ROOT)
    from stl2bem import chief_points, mesh_from_arrays, solve_table, spherical_doa_grid, surface_receivers
    from stl2bem.shapes import box

    def solve(mesh, freqs, fn):
        rng = np.random.default_rng(0)
        mics = mesh.verts[rng.choice(len(mesh.verts), 8, replace=False)] * 1.0
        rec, w, _ = surface_receivers(mics, mesh, verbose=False)
        doa, _, _ = spherical_doa_grid(360, 19)
        pts = chief_points(mesh, n=20, seed=0, use_gpu=(backend == "cupy"), verbose=False)
        t0 = time.time()
        solve_table(mesh.verts, mesh.tris, mesh.centroids, mesh.normals, mesh.areas, freqs, doa, fn,
                    receivers=rec, receiver_elements=w, chief_points=pts, backend=backend, verbose=False)
        return time.time() - t0, len(pts)

    sampler = None
    if backend == "cupy":                     # device memory in use, sampled while solving (the pool is freed after)
        import threading
        import cupy
        state = {"base": None, "peak": 0, "run": True}

        def sample():
            while state["run"]:
                free, total = cupy.cuda.runtime.memGetInfo()
                state["peak"] = max(state["peak"], total - free)
                time.sleep(0.05)

    with tempfile.TemporaryDirectory() as tmp:
        solve(mesh_from_arrays(*box(0.3, 0.2, 0.15, 4)), [500.0], os.path.join(tmp, "w.h5"))     # warm-up
        mesh = mesh_from_arrays(*box(0.3, 0.2, 0.15, n))
        if backend == "cupy":
            cupy.get_default_memory_pool().free_all_blocks()
            free, total = cupy.cuda.runtime.memGetInfo()
            state["base"] = total - free
            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
        sec, n_chief = solve(mesh, [800.0, 1000.0], os.path.join(tmp, "b.h5"))
    out = dict(n=n, elements=len(mesh.tris), backend=backend, seconds_per_freq=sec / 2, chief=n_chief)
    if sampler is not None:
        state["run"] = False
        sampler.join()
        out["gpu_peak_gb"] = (state["peak"] - state["base"]) / 2**30
    try:
        import resource
        out["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20   # kB on Linux
    except ImportError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--n", default="10,15,20,25,30")
    ap.add_argument("--case", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.case is not None:
        print(json.dumps(case(args.case, args.backend)))
        return
    path = os.path.join(HERE, "results.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for n in [int(x) for x in args.n.split(",")]:
        r = subprocess.run([sys.executable, __file__, "--backend", args.backend, "--case", str(n)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"n={n} {args.backend}: failed\n{r.stderr[-800:]}", flush=True)
            break
        rec = json.loads(r.stdout.strip().splitlines()[-1])
        res[f"{args.backend}_{rec['elements']}"] = rec
        json.dump(res, open(path, "w"), indent=1)
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
