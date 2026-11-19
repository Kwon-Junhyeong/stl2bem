"""Closed test shapes (vertices in meters, triangles with outward winding) for validation and examples."""
from __future__ import annotations

import numpy as np


def icosphere(radius: float = 0.1, subdivision: int = 3, center=(0.0, 0.0, 0.0)):
    """Geodesic sphere: 20 * 4**subdivision triangles."""
    phi = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                  [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
                  [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]], float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    t = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], int)
    verts = list(v)
    for _ in range(subdivision):
        cache, new = {}, []

        def mid(i, j):
            key = (min(i, j), max(i, j))
            if key not in cache:
                m = (verts[i] + verts[j]) / 2
                cache[key] = len(verts)
                verts.append(m / np.linalg.norm(m))
            return cache[key]

        for a, b, c in t:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        t = np.asarray(new)
    return np.asarray(verts) * radius + np.asarray(center), _outward(np.asarray(verts), t)


def box(lx: float, ly: float, lz: float, n_per_edge: int | tuple = 10, center=(0.0, 0.0, 0.0)):
    """Rectangular box; each face is an n x n grid of squares split into two triangles (n per face side, or
    (nx, ny, nz) divisions along x, y, z)."""
    nx, ny, nz = (n_per_edge,) * 3 if np.isscalar(n_per_edge) else n_per_edge
    hx, hy, hz = lx / 2, ly / 2, lz / 2
    verts, tris = [], []

    def face(origin, u, v, nu, nv):
        base = len(verts)
        for i in range(nu + 1):
            for j in range(nv + 1):
                verts.append(origin + u * (i / nu) + v * (j / nv))
        for i in range(nu):
            for j in range(nv):
                a = base + i * (nv + 1) + j
                b, c, d = a + (nv + 1), a + (nv + 1) + 1, a + 1
                tris.extend([[a, b, c], [a, c, d]])

    X, Y, Z = np.eye(3)
    face(np.array([-hx, -hy, -hz]), 2 * hy * Y, 2 * hx * X, ny, nx)      # z = -hz (normal -z)
    face(np.array([-hx, -hy, hz]), 2 * hx * X, 2 * hy * Y, nx, ny)       # z = +hz
    face(np.array([-hx, -hy, -hz]), 2 * hx * X, 2 * hz * Z, nx, nz)      # y = -hy
    face(np.array([-hx, hy, -hz]), 2 * hz * Z, 2 * hx * X, nz, nx)       # y = +hy
    face(np.array([-hx, -hy, -hz]), 2 * hz * Z, 2 * hy * Y, nz, ny)      # x = -hx
    face(np.array([hx, -hy, -hz]), 2 * hy * Y, 2 * hz * Z, ny, nz)       # x = +hx
    verts, tris = _weld(np.asarray(verts), np.asarray(tris))
    return verts + np.asarray(center), _outward(verts, tris)


def box_with_edge(lx, ly, lz, edge: float, center=(0.0, 0.0, 0.0)):
    """Box whose grid spacing is about `edge` on every side."""
    n = tuple(max(1, int(np.ceil(L / edge))) for L in (lx, ly, lz))
    return box(lx, ly, lz, n, center)


def cylinder(radius: float, height: float, edge: float, center=(0.0, 0.0, 0.0)):
    """Closed circular cylinder along z (side, top and bottom discs) with element size about `edge`."""
    n_around = max(8, int(np.ceil(2 * np.pi * radius / edge)))
    n_h = max(1, int(np.ceil(height / edge)))
    n_r = max(1, int(np.ceil(radius / edge)))
    th = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
    verts, tris = [], []
    zs = np.linspace(-height / 2, height / 2, n_h + 1)
    for z in zs:                                                          # side rings
        verts += [[radius * np.cos(t), radius * np.sin(t), z] for t in th]
    for i in range(n_h):
        for j in range(n_around):
            a, b = i * n_around + j, i * n_around + (j + 1) % n_around
            tris += [[a, b, b + n_around], [a, b + n_around, a + n_around]]
    for z, ring0 in ((zs[0], 0), (zs[-1], n_h * n_around)):              # caps: concentric rings
        prev = list(range(ring0, ring0 + n_around))
        for m in range(n_r - 1, 0, -1):
            rr = radius * m / n_r
            nm = max(6, int(np.ceil(2 * np.pi * rr / edge)))
            cur = list(range(len(verts), len(verts) + nm))
            verts += [[rr * np.cos(2 * np.pi * q / nm), rr * np.sin(2 * np.pi * q / nm), z] for q in range(nm)]
            tris += _strip(prev, cur, verts)
            prev = cur
        c = len(verts)
        verts.append([0.0, 0.0, z])
        tris += [[prev[q], prev[(q + 1) % len(prev)], c] for q in range(len(prev))]
    verts = np.asarray(verts, float)
    tris = np.asarray(tris)
    return verts + np.asarray(center), _outward(verts, tris)


def plate(lx: float, ly: float, thickness: float, edge: float, center=(0.0, 0.0, 0.0)):
    """Thin rectangular plate (a box of the given thickness), element size about `edge` on the faces."""
    nx, ny = max(1, int(np.ceil(lx / edge))), max(1, int(np.ceil(ly / edge)))
    nz = max(1, int(np.ceil(thickness / edge)))
    return box(lx, ly, thickness, (nx, ny, nz), center)


# Helpers
def _strip(outer, inner, verts):
    """Triangles between two closed rings of possibly different vertex counts (by angle)."""
    ang = lambda i: np.arctan2(verts[i][1], verts[i][0]) % (2 * np.pi)
    o = sorted(outer, key=ang)
    n = sorted(inner, key=ang)
    tris, i, j = [], 0, 0
    while i < len(o) or j < len(n):
        a_o = ang(o[(i + 1) % len(o)]) + (2 * np.pi if i + 1 >= len(o) else 0)
        a_n = ang(n[(j + 1) % len(n)]) + (2 * np.pi if j + 1 >= len(n) else 0)
        if j >= len(n) or (i < len(o) and a_o <= a_n):
            tris.append([o[i % len(o)], o[(i + 1) % len(o)], n[j % len(n)]])
            i += 1
        else:
            tris.append([o[i % len(o)], n[(j + 1) % len(n)], n[j % len(n)]])
            j += 1
    return tris


def _weld(verts, tris, tol=1e-9):
    key = np.round(verts / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return verts[first], inv.ravel()[tris]


def _outward(verts, tris):
    """Flip each triangle whose normal points towards the shape's centroid side (star-shaped bodies) and make the
    signed volume positive."""
    tris = np.asarray(tris).copy()
    v = verts[tris]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    c = v.mean(1) - verts.mean(0)
    flip = (n * c).sum(1) < 0
    tris[flip] = tris[flip][:, ::-1]
    v = verts[tris]
    vol = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum()
    return tris if vol > 0 else tris[:, ::-1]
