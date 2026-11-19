"""Benchmark, step 2: data files for docs/figures/fig_benchmark.tex from results.json."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "..", "docs", "figures")
res = json.load(open(os.path.join(HERE, "results.json")))
for backend in ("numpy", "cupy"):
    rows = sorted((r for r in res.values() if r["backend"] == backend), key=lambda r: r["elements"])
    mem_key = "gpu_peak_gb" if backend == "cupy" else "peak_rss_gb"
    with open(os.path.join(FIG, "data", f"bench_{backend}.dat"), "w") as f:
        f.write("elements seconds memory\n")
        for r in rows:
            f.write(f"{r['elements']} {r['seconds_per_freq']:.4f} {r.get(mem_key, float('nan')):.4f}\n")
            print(backend, r["elements"], f"{r['seconds_per_freq']:.2f} s", f"{r.get(mem_key, float('nan')):.2f} GB")
