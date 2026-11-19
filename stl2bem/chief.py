"""CHIEF interior points for an arbitrary closed surface (ray casting + farthest-point sampling).

The CBIE of the exterior Neumann problem is not unique at the interior Dirichlet eigenfrequencies of the body
(irregular frequencies). CHIEF (Schenck 1968) adds the interior condition at points inside the body; the augmented
system is solved in the least-squares sense. Points are drawn at random in the bounding box, kept if inside
(majority vote of three rays), preferred far from the surface, and spread by farthest-point sampling.
"""
from __future__ import annotations

import numpy as np

from .backend import gpu_available

try:
    import cupy as cp
except Exception:
    cp = None


# three ray directions away from axes and diagonals (majority vote guards against hits on edges)
_DIRS = np.array([[0.5121, 0.7043, 0.4919],
                  [-0.3131, 0.8090, 0.4988],
                  [0.7071, -0.1961, 0.6797]])
_DIRS /= np.linalg.norm(_DIRS, axis=1, keepdims=True)


def _hit_count(origins, d, v0, e1, e2, xp):
    """Moeller-Trumbore: number of triangles crossed by the ray from each origin along d."""
    pv = xp.cross(xp.broadcast_to(d, e2.shape), e2)
    det = xp.einsum("tk,tk->t", e1, pv)
    ok = xp.abs(det) > 1e-16
    inv = xp.where(ok, 1.0 / xp.where(ok, det, 1.0), 0.0)

    cnt = xp.zeros(len(origins), dtype=xp.int32)
    step = max(1, int(1e7 // max(len(v0), 1)))
    for i in range(0, len(origins), step):
        tv = origins[i:i + step, None, :] - v0[None, :, :]
        u = xp.einsum("ptk,tk->pt", tv, pv) * inv
        qv = xp.cross(tv, xp.broadcast_to(e1, tv.shape))
        w = (qv @ d) * inv
        t = xp.einsum("ptk,tk->pt", qv, e2) * inv
        hit = ok[None, :] & (u >= 0) & (w >= 0) & (u + w <= 1) & (t > 1e-9)
        cnt[i:i + step] = hit.sum(axis=1)
    return cnt


def points_inside(pts, verts, tris, use_gpu: bool = True):
    """(P,) bool: inside the closed triangle surface (majority of three rays with an odd hit count)."""
    xp = cp if (use_gpu and gpu_available()[0]) else np
    v = xp.asarray(verts[tris])
    v0, e1, e2 = v[:, 0], v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
    P = xp.asarray(pts)
    odd = xp.zeros(len(pts), dtype=xp.int32)
    for d in _DIRS:
        odd += _hit_count(P, xp.asarray(d), v0, e1, e2, xp) % 2
    out = odd >= 2
    return cp.asnumpy(out) if xp is not np else out


def _nearest_surface_dist(pts, centroids):
    try:
        from scipy.spatial import cKDTree
        return cKDTree(centroids).query(pts)[0]
    except ImportError:
        out = np.empty(len(pts))
        for i in range(0, len(pts), 256):
            d = np.linalg.norm(pts[i:i + 256, None, :] - centroids[None, :, :], axis=2)
            out[i:i + 256] = d.min(axis=1)
        return out


def _fps(pts, n, rng):
    """Farthest-point sampling of n points."""
    idx = [int(rng.integers(len(pts)))]
    d = np.linalg.norm(pts - pts[idx[0]], axis=1)
    for _ in range(n - 1):
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return pts[idx]


def chief_points(mesh, n: int = 20, seed: int = 0, oversample: int = 60,
                 clear_frac: float = 0.35, min_clearance: float | None = None, use_gpu: bool = True,
                 verbose: bool = True):
    """(m, 3) CHIEF points inside the body (m <= n): farthest-point sampling among candidates far from the surface.

    A CHIEF row is the boundary-integral representation evaluated inside the material. Constant elements make
    that representation inaccurate within about one element of the surface, and inaccurate CHIEF rows pull the
    least-squares solution away from the correct one (seen at the Helmholtz resonance of a thin-walled bottle).
    Points closer than `min_clearance` (default 0.75 x the mean element edge) to the nearest element centroid are
    therefore rejected; this distance stands in for the distance to the surface. Thin-walled
    bodies may have no admissible point; they also do not need any, because the irregular frequencies of a wall
    of thickness t start near c / (2 t), far above the band a mesh resolves. An empty array then means CBIE only."""
    rng = np.random.default_rng(seed)
    lo, hi = mesh.verts.min(0), mesh.verts.max(0)
    cand = rng.uniform(lo, hi, size=(max(n * oversample, 200), 3))
    ins = points_inside(cand, mesh.verts, mesh.tris, use_gpu=use_gpu)
    cand = cand[ins]
    if min_clearance is None:
        min_clearance = 0.75 * mesh.info.get("edge_mean_m", 0.0)
    clear = _nearest_surface_dist(cand, mesh.centroids) if len(cand) else np.zeros(0)
    ok = clear >= min_clearance
    if ok.sum() == 0:
        if verbose:
            print(f"  chief    : no interior point at least {min_clearance * 1000:.1f} mm from the element centroids "
                  f"(thin-walled body; deepest {clear.max() * 1000 if len(clear) else 0:.1f} mm): CBIE only")
        return np.zeros((0, 3))
    cand, clear = cand[ok], clear[ok]
    keep = clear >= clear_frac * clear.max()
    pool = cand[keep] if keep.sum() >= n else cand[np.argsort(-clear)[:max(n, 10)]]
    pts = _fps(pool, min(n, len(pool)), rng)
    if verbose:
        cl = _nearest_surface_dist(pts, mesh.centroids)
        print(f"  chief    : {len(pts)} points ({len(cand)}/{len(ins)} candidates deep enough), "
              f"closest element centroid {cl.min() * 1000:.1f} mm away")
    return pts
