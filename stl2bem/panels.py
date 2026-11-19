"""Receivers (microphones): reading positions, placing them on the surface, legacy snapping to elements.

Receiver modes used by the solver
  surface : each microphone is projected onto the closest point of the surface; the pressure there is the corrected
            boundary-integral representation 2 (p_inc - M p)(x) - p_j (j: the element under the point), which is
            second-order accurate at any surface point (bem_core.solve_table). Default.
  field   : the pressure is evaluated exactly at the given points, which must lie outside the body
            (microphones held away from the surface, or any field point).
  snap    : the original behavior: the pressure of the element whose centroid is closest (constant element value).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_mic_positions(path: str | Path, scale: float = 1.0,
                       shift: np.ndarray | None = None) -> np.ndarray:
    """(N, 3) microphone positions. csv/txt: one 'x, y, z' row per microphone ('#' comments and a header row are
    skipped); json: [[x, y, z], ...] or {"mics": [[x, y, z], ...]}. Scaled to meters by `scale`, then `shift` is
    subtracted (the origin shift applied to the mesh)."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text())
        pts = np.asarray(obj["mics"] if isinstance(obj, dict) else obj, dtype=np.float64)
    else:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip().replace(",", " ")
            if not line or line.startswith("#"):
                continue
            try:
                vals = [float(x) for x in line.split()]
            except ValueError:                       # header row
                continue
            if len(vals) >= 3:
                rows.append(vals[:3])
        pts = np.asarray(rows, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise ValueError(f"could not read microphone positions from {path}")
    pts = pts * scale
    return pts - shift if shift is not None else pts


def auto_panel_targets(mesh, n: int = 24) -> np.ndarray:
    """n points spread over the surface: farthest-point sampling of the element centroids (deterministic)."""
    c = mesh.centroids
    start = int(np.argmax(np.linalg.norm(c - c.mean(0), axis=1)))
    idx = [start]
    d = np.linalg.norm(c - c[start], axis=1)
    for _ in range(n - 1):
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(c - c[i], axis=1))
    return c[idx]


def map_to_elements(targets: np.ndarray, mesh, max_dist: float = 0.05,
                    verbose: bool = True) -> np.ndarray:
    """Legacy 'snap' mode: index of the element whose centroid is closest to each target (no element twice)."""
    targets = np.atleast_2d(targets)
    d = np.linalg.norm(targets[:, None, :] - mesh.centroids[None, :, :], axis=2)
    used, out = set(), np.zeros(len(targets), dtype=np.int64)
    for t in range(len(targets)):
        for idx in np.argsort(d[t]):
            if int(idx) in used:
                continue
            if d[t, idx] > max_dist:
                raise ValueError(
                    f"receiver[{t}] {np.round(targets[t], 4)}: closest element centroid is "
                    f"{d[t, idx] * 1000:.1f} mm away (limit {max_dist * 1000:.0f} mm). "
                    f"Check the coordinates and units, or --max_receiver_dist")
            out[t] = idx
            used.add(int(idx))
            break
    if verbose:
        snap = np.linalg.norm(targets - mesh.centroids[out], axis=1)
        print(f"  receivers: {len(out)} snapped to element centroids, mean distance {snap.mean() * 1000:.2f} mm, "
              f"maximum {snap.max() * 1000:.2f} mm")
    return out


def closest_points_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Closest point to p (3,) on each triangle (a, b, c each (T, 3)); Ericson, Real-Time Collision Detection 5.1.5."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp_ = p - c
    d5, d6 = (ab * cp_).sum(1), (ac * cp_).sum(1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    out = np.empty_like(a)
    done = np.zeros(len(a), bool)

    def put(mask, val):
        m = mask & ~done
        out[m] = val[m] if val.ndim == 2 else val
        done[m] = True

    with np.errstate(divide="ignore", invalid="ignore"):
        put((d1 <= 0) & (d2 <= 0), a)                                                      # vertex a
        put((d3 >= 0) & (d4 <= d3), b)                                                     # vertex b
        v = d1 / (d1 - d3)
        put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v[:, None] * ab)                       # edge ab
        put((d6 >= 0) & (d5 <= d6), c)                                                     # vertex c
        w = d2 / (d2 - d6)
        put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w[:, None] * ac)                       # edge ac
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + w[:, None] * (c - b))    # edge bc
        den = 1.0 / (va + vb + vc)
        v, w = vb * den, vc * den
        put(np.ones(len(a), bool), a + v[:, None] * ab + w[:, None] * ac)                  # face interior
    return out


def solid_angle(x: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Signed solid angle of triangles (a, b, c each (T, 3)) seen from x (Van Oosterom & Strackee 1983);
    negative when x is on the side the right-hand normal points to."""
    ra, rb, rc = a - x, b - x, c - x
    la, lb, lc = (np.linalg.norm(r, axis=1) for r in (ra, rb, rc))
    trip = (ra * np.cross(b - a, c - a)).sum(1)
    den = la * lb * lc + (ra * rb).sum(1) * lc + (ra * rc).sum(1) * lb + (rb * rc).sum(1) * la
    return 2 * np.arctan2(trip, den)


def surface_receivers(points: np.ndarray, mesh, max_dist: float = 0.05, offset: float = 1e-6,
                      verbose: bool = True):
    """Microphones on the surface: evaluation points and the elements they touch, for the corrected surface value.

    Returns (evaluation points (R, 3), (index (R, K), weight (R, K)), distance to the surface (R,)).
    A point inside an element touches that element only (weight 1). A point on an edge or a vertex touches all
    elements meeting there; the jump of the double layer is shared between them in proportion to the solid angle
    each subtends from the evaluation point, and the correction subtracts that weighted mean
    (bem_core.solve_table)."""
    pts, elem, dist = project_to_surface(points, mesh, max_dist=max_dist, offset=offset, verbose=verbose)
    v = mesh.verts[mesh.tris]
    a, b, c = v[:, 0], v[:, 1], v[:, 2]
    tol = 1e-7 * float(np.linalg.norm(mesh.verts.max(0) - mesh.verts.min(0)))
    rows = []
    for x, j in zip(pts, elem):
        q = x - offset * mesh.normals[j]                                            # the point on the surface
        d = np.linalg.norm(closest_points_on_triangles(q, a, b, c) - q, axis=1)
        touch = np.nonzero(d <= tol)[0]
        if j not in touch:
            touch = np.append(touch, j)
        om = solid_angle(x, a[touch], b[touch], c[touch])
        w = om / om.sum() if abs(om.sum()) > 1e-9 else (touch == j).astype(float)
        rows.append((touch, w))
    K = max(len(t) for t, _ in rows)
    idx = np.zeros((len(rows), K), np.int64)
    wt = np.zeros((len(rows), K))
    for r, (t, w) in enumerate(rows):
        idx[r, :len(t)], wt[r, :len(t)] = t, w
        idx[r, len(t):] = t[0]
    if verbose and K > 1:
        n_shared = sum(len(t) > 1 for t, _ in rows)
        print(f"  receivers: {n_shared} of {len(rows)} lie on an edge or vertex (up to {K} elements shared)")
    return pts, (idx, wt), dist


def project_to_surface(points: np.ndarray, mesh, max_dist: float = 0.05, offset: float = 1e-6,
                       verbose: bool = True):
    """Closest surface point of each microphone, moved `offset` meters outwards along that element's normal.

    Returns (evaluation points (R, 3), element index (R,), distance of the given point to the surface (R,)).
    Raises if a point is farther than max_dist from the surface (wrong units or coordinates)."""
    v = mesh.verts[mesh.tris]
    a, b, c = v[:, 0], v[:, 1], v[:, 2]
    pts, elem, dist = [], [], []
    for p in np.atleast_2d(points):
        q = closest_points_on_triangles(p, a, b, c)
        d = np.linalg.norm(q - p, axis=1)
        j = int(np.argmin(d))
        if d[j] > max_dist:
            raise ValueError(f"receiver {np.round(p, 4)} is {d[j] * 1000:.1f} mm from the surface "
                             f"(limit {max_dist * 1000:.0f} mm). Check the coordinates and units, or use --receivers field")
        pts.append(q[j] + offset * mesh.normals[j])
        elem.append(j)
        dist.append(d[j])
    pts, elem, dist = np.asarray(pts), np.asarray(elem, np.int64), np.asarray(dist)
    if verbose:
        print(f"  receivers: {len(pts)} projected onto the surface, mean distance {dist.mean() * 1000:.2f} mm, "
              f"maximum {dist.max() * 1000:.2f} mm (evaluated {offset * 1e6:g} um outside)")
    return pts, elem, dist
