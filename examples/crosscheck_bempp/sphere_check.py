"""Check the bempp-cl reference and the convention mapping on a rigid sphere (exact series solution)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from stl2bem import mesh_from_arrays, rigid_sphere
from stl2bem.shapes import icosphere
from bempp_solve import Solver, make_grid

a, f = 0.1, 2000.0
v, t = icosphere(a, 4)
m = mesh_from_arrays(v, t)
s = Solver(make_grid(m.verts, m.tris))
doa = np.array([1.0, 0, 0])
sol, k, d, info = s.total_surface(f, doa)
pts = np.array([[0.2, 0, 0], [-0.2, 0, 0], [0, 0.25, 0.05], [0.1, -0.15, 0.1]])
fb = s.field(sol, k, d, pts)
ex = rigid_sphere(k, a, pts, doa[None])[:, 0]
print("field  rel err", np.abs(fb - ex).max() / np.abs(ex).mean(), info)
cen = a * m.centroids / np.linalg.norm(m.centroids, axis=1, keepdims=True)
sb = s.at_centroids(sol)
se = rigid_sphere(k, a, cen, doa[None])[:, 0]
print("surface rel err", np.abs(sb - se).mean() / np.abs(se).mean())
