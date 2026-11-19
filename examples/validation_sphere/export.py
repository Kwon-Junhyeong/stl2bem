"""Validation, step 2 (pictures): rasters and data files for docs/figures/fig_sphere_*.tex, and the animation."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "docs", "figures"))
from rasters import animation, save_png, seq_rgb, signed_rgb  # noqa: E402

FIG = os.path.join(ROOT, "docs", "figures")
os.makedirs(os.path.join(FIG, "raster"), exist_ok=True)
os.makedirs(os.path.join(FIG, "data"), exist_ok=True)
D = os.path.join(HERE, "data")
RADIUS = 0.1                                       # sphere of run.py

# field maps
f = np.load(os.path.join(D, "field.npz"))
bem, ex = f["bem"], f["exact"]
vmax = float(np.nanpercentile(np.abs(np.real(ex)), 99.5))
save_png(signed_rgb(ex, vmax, gamma=1.0), os.path.join(FIG, "raster", "sphere_exact.png"))
save_png(signed_rgb(bem, vmax, gamma=1.0), os.path.join(FIG, "raster", "sphere_bem.png"))
err = np.log10(np.abs(bem - ex) + 1e-12)
save_png(seq_rgb(err, -4.0, -1.0), os.path.join(FIG, "raster", "sphere_error.png"))
animation(bem, vmax, os.path.join(FIG, "..", "img", "sphere_wave.gif"), width=400, gamma=0.8, frames=24, ms=75,
          dither=True)
ok = np.isfinite(bem) & np.isfinite(ex)
d = np.abs(bem - ex)
half, n = float(f["half"]), bem.shape[0]
g = np.linspace(-half, half, n)
gap_mm = (np.hypot(*np.meshgrid(g, g, indexing="xy")) - RADIUS) * 1e3      # distance of each point from the sphere
near = ok & (gap_mm < 1.0)
vals = {"vmax": vmax, "ka": float(f["ka"]), "elements": int(f["elements"]), "points": int(ok.sum()),
        "median_err": float(np.median(d[ok])), "max_err": float(d[ok].max()),
        "frac_below_1e-2": float((d[ok] < 1e-2).mean()),
        "points_within_1mm": int(near.sum()), "max_err_from_2mm": float(d[ok & (gap_mm >= 2.0)].max())}

# surface ring
r = np.load(os.path.join(D, "ring.npz"))
db = lambda p: 20 * np.log10(np.abs(p))
cols = ["deg"] + [f"ex{i}" for i in range(3)] + [f"new{i}" for i in range(3)]
np.savetxt(os.path.join(FIG, "data", "sphere_ring.dat"),
           np.column_stack([r["deg"], db(r["exact"]), db(r["new"])]), header=" ".join(cols), comments="", fmt="%.4f")
np.savetxt(os.path.join(FIG, "data", "sphere_ring_old.dat"), np.column_stack([r["old_deg"], db(r["old"])]),
           header="deg old0 old1 old2", comments="", fmt="%.4f")
idx_old = [int(np.argmin(np.abs(r["deg"] - d))) for d in r["old_deg"]]
vals["ring_err_new"] = [float(np.abs(r["new"][:, i] - r["exact"][:, i]).mean() / np.abs(r["exact"][:, i]).mean())
                        for i in range(3)]
vals["ring_err_old"] = [float(np.abs(r["old"][:, i] - r["exact"][idx_old, i]).mean()
                              / np.abs(r["exact"][idx_old, i]).mean()) for i in range(3)]

# convergence
c = json.load(open(os.path.join(D, "convergence.json")))
for meth in ("original", "stl2bem"):
    rows = [x for x in c["rows"] if x["method"] == meth]
    np.savetxt(os.path.join(FIG, "data", f"sphere_conv_{meth}.dat"),
               np.array([[x["edge_mm"], 100 * x["rel_error"][0], 100 * x["rel_error"][1]] for x in rows]),
               header="edge e500 e2000", comments="", fmt="%.5f")
vals["convergence"] = c
json.dump(vals, open(os.path.join(FIG, "data", "sphere_values.json"), "w"), indent=1)
json.dump(vals, open(os.path.join(HERE, "results.json"), "w"), indent=1)    # the numbers quoted in the README
with open(os.path.join(HERE, "results.md"), "w") as fm:
    fm.write("# Validation: rigid sphere\n\n"
             f"Field map ({vals['elements']} elements, ka = {vals['ka']:.2f}, {vals['points']} points): median "
             f"|p - p_exact| {vals['median_err']:.1e}, {100 * vals['frac_below_1e-2']:.1f} % below 1e-2; maximum "
             f"{vals['max_err']:.2f} on the surface itself ({vals['points_within_1mm']} points within 1 mm), "
             f"{vals['max_err_from_2mm']:.3f} from 2 mm outward.\n\n"
             "Surface ring (1280 elements), mean relative error: "
             + ", ".join(f"ka {ka}: stl2bem {100 * a:.2f} % / original {100 * b:.2f} %"
                         for ka, a, b in zip((1, 3, 6), vals["ring_err_new"], vals["ring_err_old"])) + "\n\n"
             "| elements | edge (mm) | method | 500 Hz | 2000 Hz |\n|---|---|---|---|---|\n"
             + "".join(f"| {x['elements']} | {x['edge_mm']:.1f} | {x['method']} | {100 * x['rel_error'][0]:.2f} % | "
                       f"{100 * x['rel_error'][1]:.2f} % |\n" for x in c["rows"]))
with open(os.path.join(FIG, "data", "sphere_values.tex"), "w") as ft:
    ft.write("\\def\\sphVmax{%.4f}\n\\def\\sphKa{%.2f}\n\\def\\sphElements{%d}\n"
             % (vals["vmax"], vals["ka"], vals["elements"]))
print(json.dumps(vals, indent=1))
