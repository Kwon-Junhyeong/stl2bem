"""Reference solutions with bempp-cl (https://github.com/bempp/bempp-cl) on the same triangle mesh as stl2bem.

bempp-cl is an independent, established BEM library. It is solved here with a different formulation from stl2bem:
Burton-Miller (OSRC-coupled hypersingular operator) with continuous piecewise-linear (P1) elements, instead of
CBIE + CHIEF with piecewise-constant elements. Agreement of the two on the same CAD mesh is therefore a check of
the implementation, not of a shared assumption.

Conventions: bempp uses e^{-i w t}. stl2bem's field for a wave from d (incident exp(+jk d.x), e^{+jwt}) is the
complex conjugate of bempp's field for the incident wave exp(i k (-d).x).

Needs a Python environment with bempp-cl (pip install bempp-cl); stl2bem itself only needs numpy/scipy/h5py.
"""
from __future__ import annotations

import time

import numpy as np

import bempp_cl.api as bempp
from bempp_cl.api.linalg import gmres
from bempp_cl.api.operators.boundary import helmholtz, sparse
from bempp_cl.api.operators.potential import helmholtz as helmholtz_potential


def make_grid(verts, tris):
    return bempp.Grid(np.ascontiguousarray(np.asarray(verts, np.float64).T),
                      np.ascontiguousarray(np.asarray(tris, np.uint32).T))


class Solver:
    """Burton-Miller solver for one grid; one call per (frequency, direction)."""

    def __init__(self, grid, c=343.0, quad_regular=6, quad_singular=6):
        bempp.GLOBAL_PARAMETERS.quadrature.regular = quad_regular
        bempp.GLOBAL_PARAMETERS.quadrature.singular = quad_singular
        self.grid, self.c = grid, c
        self.space = bempp.function_space(grid, "P", 1)

    def total_surface(self, freq, doa, tol=1e-8):
        """P1 grid function of the total surface pressure, stl2bem convention not yet applied."""
        k = 2 * np.pi * freq / self.c
        d = -np.asarray(doa, np.float64)                    # bempp wave exp(i k d.x) travels along d
        sp = self.space
        ident = sparse.identity(sp, sp, sp)
        dlp = helmholtz.double_layer(sp, sp, sp, k)
        hyp = helmholtz.hypersingular(sp, sp, sp, k)
        ntd = helmholtz.osrc_ntd(sp, k)
        lhs = 0.5 * ident - dlp - ntd * hyp

        @bempp.complex_callable
        def dirichlet(x, n, domain_index, result):
            result[0] = np.exp(1j * k * (x[0] * d[0] + x[1] * d[1] + x[2] * d[2]))

        @bempp.complex_callable
        def neumann(x, n, domain_index, result):
            result[0] = 1j * k * (n[0] * d[0] + n[1] * d[1] + n[2] * d[2]) * \
                np.exp(1j * k * (x[0] * d[0] + x[1] * d[1] + x[2] * d[2]))

        rhs = bempp.GridFunction(sp, fun=dirichlet) - ntd * bempp.GridFunction(sp, fun=neumann)
        t0 = time.time()
        sol, info, it = gmres(lhs, rhs, tol=tol, return_iteration_count=True)
        return sol, k, d, dict(gmres_info=int(info), iterations=int(it), seconds=time.time() - t0)

    def field(self, sol, k, d, points, chunk=4000):
        """Total pressure at exterior points (P, 3), stl2bem convention."""
        points = np.asarray(points, np.float64)
        out = np.empty(len(points), np.complex128)
        for s in range(0, len(points), chunk):
            p = points[s:s + chunk]
            pot = helmholtz_potential.double_layer(self.space, np.ascontiguousarray(p.T), k)
            out[s:s + chunk] = np.exp(1j * k * (p @ d)) + (pot * sol).ravel()
        return np.conj(out)

    def at_centroids(self, sol):
        """Surface pressure at element centroids (P1 function evaluated there), stl2bem convention."""
        return np.conj(sol.evaluate_on_element_centers()[0])
