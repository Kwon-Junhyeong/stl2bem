"""Check the fitted bempp-cl resonance of the bottle by solving at the fitted peak frequency itself."""
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

path = os.path.join(HERE, "results.json")
summary = json.load(open(path))
q = summary["bottle_resonance"]
o = np.load(os.path.join(HERE, "data", "ours_bottle.npz"))
mesh = build_mesh(os.path.join(ROOT, "examples", "gallery", "models", "bottle.stl"), units="mm", recenter="none",
                  verbose=False)
s = Solver(make_grid(mesh.verts, mesh.tris))
sol, k, d, info = s.total_surface(q["bempp_f0"], o["doa"])
p = s.field(sol, k, d, o["probe"][None])[0]
q["bempp_at_f0_db"] = float(20 * np.log10(abs(p)))
json.dump(summary, open(path, "w"), indent=1)
r = dict(np.load(os.path.join(HERE, "data", "res_bottle.npz")))
r["check_f"], r["check_p"] = q["bempp_f0"], p
np.savez(os.path.join(HERE, "data", "res_bottle.npz"), **r)
print("bottle check", json.dumps(q), flush=True)
