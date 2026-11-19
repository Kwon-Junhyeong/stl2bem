"""Cross-check, step 2 (pictures): difference maps and surface-level pairs for docs/figures/fig_crosscheck.tex."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "docs", "figures"))
from rasters import save_png, seq_rgb  # noqa: E402

FIG = os.path.join(ROOT, "docs", "figures")
NAMES = ["star", "boat", "mug", "bottle", "bell", "pipe"]
LO, HI = -4.0, -1.0                    # log10 of the difference, as in fig_sphere_field
PLOT_HZ = {"bottle": (143.8, 146.2), "pipe": (1380.0, 1515.0)}      # frequency axes of the resonance panels

macros, rows = [], []
for n in NAMES:
    fb = os.path.join(HERE, "data", f"{n}.npz")
    if not os.path.exists(fb):
        continue
    own = os.path.join(HERE, "data", f"ours_{n}.npz")              # resonators: compared off resonance
    ours = np.load(own if os.path.exists(own) else os.path.join(ROOT, "examples", "gallery", "data", f"{n}.npz"))
    ref = np.load(fb)
    assert abs(float(ours["freq"]) - float(ref["freq"])) < 1e-6, (n, ours["freq"], ref["freq"])
    f = float(ref["freq"])
    macros.append(f"\\expandafter\\def\\csname cc{n}f\\endcsname{{{f / 1000:.1f} kHz}}" if f >= 1000 else
                  f"\\expandafter\\def\\csname cc{n}f\\endcsname{{{f:.0f} Hz}}")
    # difference relative to the local level or to the incident wave (amplitude 1), whichever is larger, so that
    # a resonance inside a cavity is compared on its own scale
    scale = np.maximum(np.abs(ref["field"]), 1.0)
    diff = np.abs(ours["field"] - ref["field"]) / scale
    aspect = float(ours["half"][1] / ours["half"][0])
    save_png(seq_rgb(np.log10(np.maximum(diff, 10 ** LO)), LO, HI), os.path.join(FIG, "raster", f"cc_{n}_diff.png"),
             width=720, aspect=aspect)
    s_ours = 20 * np.log10(np.abs(ours["surface"]))
    s_ref = 20 * np.log10(np.abs(ref["surface"]))
    step = max(1, len(s_ref) // 700)
    for a, b in zip(s_ref[::step], s_ours[::step]):
        rows.append(f"{a:.3f} {b:.3f} {n}")
    ok = np.isfinite(diff)
    med = float(np.median(diff[ok]))
    macros.append(f"\\expandafter\\def\\csname cc{n}\\endcsname{{{100 * med:.2f}}}")
    print(n, "median", med, "p95", np.percentile(diff[ok], 95),
          "surface dB diff median", np.median(np.abs(s_ours - s_ref)))

res = json.load(open(os.path.join(HERE, "results.json")))
for n in ["bottle", "pipe"]:
    fr = os.path.join(HERE, "data", f"res_{n}.npz")
    if not os.path.exists(fr):
        continue
    o = np.load(os.path.join(HERE, "data", f"ours_{n}.npz"))
    r = np.load(fr)
    np.savetxt(os.path.join(FIG, "data", f"cc_res_{n}_ours.dat"),
               np.c_[o["sweep_f"], 20 * np.log10(np.abs(o["sweep_p"]))], header="f level", comments="", fmt="%.4f")
    fb_, pb_ = r["f"], r["p"]
    if "check_f" in r:                                              # solved at the fitted peak (verify_peak.py)
        fb_, pb_ = np.append(fb_, r["check_f"]), np.append(pb_, r["check_p"])
    np.savetxt(os.path.join(FIG, "data", f"cc_res_{n}_bempp.dat"),
               np.c_[fb_, 20 * np.log10(np.abs(pb_))], header="f level", comments="", fmt="%.4f")
    ff = o["sweep_f"]
    a = np.mean(r["p"] - r["b"] / (r["f"] - r["z"]))                # constant term of the fit
    fit = a + r["b"] / (ff - r["z"])
    np.savetxt(os.path.join(FIG, "data", f"cc_res_{n}_fit.dat"),
               np.c_[ff, 20 * np.log10(np.abs(fit))], header="f level", comments="", fmt="%.4f")
    q = res[f"{n}_resonance"]
    macros += [f"\\expandafter\\def\\csname cc{n}Ours\\endcsname{{{q['ours_f0']:.1f}}}",
               f"\\expandafter\\def\\csname cc{n}Ref\\endcsname{{{q['bempp_f0']:.1f}}}",
               f"\\expandafter\\def\\csname cc{n}Lo\\endcsname{{{PLOT_HZ[n][0]:g}}}",
               f"\\expandafter\\def\\csname cc{n}Hi\\endcsname{{{PLOT_HZ[n][1]:g}}}"]
    print(n, json.dumps(q))

with open(os.path.join(FIG, "data", "cc_surface.dat"), "w") as f:
    f.write("ref ours name\n" + "\n".join(rows) + "\n")
with open(os.path.join(FIG, "data", "cc_values.tex"), "w") as f:
    f.write("\n".join(macros) + "\n")
