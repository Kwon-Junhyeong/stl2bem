"""Gallery, step 2 (pictures): rasters, animations and data files for docs/figures/fig_gallery*.tex."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "figures"))
from rasters import animation, render_surface, render_with_plane, save_png, signed_rgb  # noqa: E402
from stl2bem import build_mesh  # noqa: E402
from stl2bem.viz import plane_grid  # noqa: E402

FIG = os.path.join(ROOT, "docs", "figures")
for d in ("raster", "data"):
    os.makedirs(os.path.join(FIG, d), exist_ok=True)
NAMES = ["star", "boat", "mug", "bottle", "bell", "pipe"]
VIEW = {"star": (35, -60), "boat": (22, -50), "mug": (25, -60), "bottle": (15, -60), "bell": (-20, -60),
        "pipe": (20, -70)}

data = {n: np.load(os.path.join(HERE, "data", f"{n}.npz")) for n in NAMES
        if os.path.exists(os.path.join(HERE, "data", f"{n}.npz"))}
lo, hi = -8.0, 8.0          # surface level (dB re incident): gray in the shadow, white = incident, navy = louder
macros = [f"\\def\\galLo{{{lo:g}}}", f"\\def\\galHi{{{hi:g}}}"]

for n, d in data.items():
    field = d["field"]
    aspect = float(d["half"][1] / d["half"][0])
    # color range from the field around the object, so that a resonance inside saturates instead of washing out
    vmax = float(min(np.nanpercentile(np.abs(field), 99), 2.5 * np.nanmedian(np.abs(field))))
    save_png(signed_rgb(field, vmax, gamma=0.8), os.path.join(FIG, "raster", f"gal_{n}_field.png"),
             width=720, aspect=aspect)
    animation(field, vmax, os.path.join(FIG, "..", "img", f"gallery_{n}.gif"), width=800, aspect=aspect, gamma=0.8,
              canvas=400, frames=24, ms=75, dither=True)
    mesh = build_mesh(os.path.join(HERE, "models", f"{n}.stl"), units="mm", recenter="none", verbose=False)
    render_surface(mesh, 20 * np.log10(np.abs(d["surface"])), lo, hi, os.path.join(FIG, "raster", f"gal_{n}_surface.png"),
                   elev=VIEW[n][0], azim=VIEW[n][1])
    f = float(d["freq"])
    label = f"{f / 1000:.1f} kHz" if f >= 1000 else f"{f:.0f} Hz"
    macros.append(f"\\expandafter\\def\\csname gal{n}f\\endcsname{{{label}}}")
    if "spec_f" in d:
        lev = 20 * np.log10(np.abs(d["spec_p"]))
        np.savetxt(os.path.join(FIG, "data", f"gal_{n}_spectrum.dat"), np.column_stack([d["spec_f"], lev]),
                   header="f level", comments="", fmt="%.4f")
    print(n, "field", field.shape, "vmax %.2f" % vmax, "freq", label)

# README diagram: the boat mesh with its triangle edges, the same view colored by the surface pressure, and the
# boat on its sound field for the social preview
boat = build_mesh(os.path.join(HERE, "models", "boat.stl"), units="mm", recenter="none", verbose=False)
p = os.path.join(FIG, "raster", "stl_boat_mesh.png")
render_surface(boat, np.zeros(len(boat.tris)), -1.0, 1.0, p, elev=22, azim=-50, size=1200, edges="#1F4E79",
               edge_width=0.3)
from PIL import Image  # noqa: E402
im = Image.open(p)
im.crop(im.getbbox()).save(p)
macros.append(f"\\def\\galBoatTriangles{{{len(boat.tris):,}}}".replace(",", "\\,"))
if "boat" in data:
    p = os.path.join(FIG, "raster", "stl_boat_pressure.png")
    render_surface(boat, 20 * np.log10(np.abs(data["boat"]["surface"])), lo, hi, p, elev=22, azim=-50, size=1200)
    im = Image.open(p)
    im.crop(im.getbbox()).save(p)
water = os.path.join(HERE, "data", "boat_water.npz")        # from water_slice.py
if "boat" in data and os.path.exists(water):
    w = np.load(water)
    field = w["field"]
    nv, nu = field.shape
    pts, _ = plane_grid(w["center"], w["u"], w["v"], tuple(w["half"]), (nu, nv))
    P = pts.reshape(nv, nu, 3).copy()
    P[~np.isfinite(field)] = np.nan
    keep = (np.abs(P[..., 0] - w["center"][0]) <= 0.16) & (np.abs(P[..., 1] - w["center"][1]) <= 0.11)
    rows, cols = np.nonzero(keep.any(axis=1))[0], np.nonzero(keep.any(axis=0))[0]
    sub = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    vmax = float(min(np.nanpercentile(np.abs(field), 99), 2.5 * np.nanmedian(np.abs(field))))
    p = os.path.join(FIG, "raster", "stl_boat_field.png")
    render_with_plane(boat, signed_rgb(data["boat"]["surface"], vmax, gamma=0.8), P[sub],
                      signed_rgb(field[sub], vmax, gamma=0.8), p, elev=22, azim=-50, size=1200, step=1)
    im = Image.open(p)
    im.crop(im.getbbox()).save(p)

res = json.load(open(os.path.join(HERE, "results.json")))
for n, key in (("mug", "Mug"), ("bottle", "Bottle"), ("pipe", "Pipe")):
    if n in res and "estimate_hz" in res[n]:
        macros.append(f"\\def\\gal{key}Est{{{res[n]['estimate_hz']:.1f}}}")
with open(os.path.join(FIG, "data", "gallery_values.tex"), "w") as ft:
    ft.write("\n".join(macros) + "\n")
print("surface level range", lo, hi)
