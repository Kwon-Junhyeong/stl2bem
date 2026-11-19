"""Double-layer operator of the rigid-body (sound-hard) Helmholtz problem on flat triangles, constant elements.

Convention (as in the rest of the package): time dependence e^{+jwt}, G(r) = exp(-jkr) / (4 pi r),
incident plane wave p_inc(x) = exp(+jk d.x) with d the unit vector towards the source, normals pointing out of
the body. For a rigid body the total pressure p satisfies

    on the surface (smooth point):   p(x)/2 + (M p)(x) = p_inc(x)
    outside the body (receivers):    p(x)   = p_inc(x) - (M p)(x)
    inside the body (CHIEF points):  0      = p_inc(x) - (M p)(x)

with  (M p)(x) = - sum_j p_j  int_{T_j} dG/dn_y (x, y) dS_y.

The kernel is split as G = G0 + (G - G0), G0 = 1 / (4 pi r):

* static part, exactly: int_T dG0/dn_y dS = -Omega / (4 pi), Omega the signed solid angle of the triangle seen from x
  (Van Oosterom & Strackee, IEEE Trans. Biomed. Eng. 30(2), 1983). This is the singular / near-singular part, so
  elements close to the evaluation point (neighbors, the two faces of a thin plate, a receiver just above the
  surface) are integrated exactly.
* dynamic part (G - G0): bounded and smooth, d/dr[(e^{-jkr} - 1) / (4 pi r)] -> -k^2 / (8 pi) as r -> 0,
  integrated with a symmetric triangle rule (1, 3 or 7 points).

``integration="centroid"`` reproduces the original one-point rule (whole kernel at the element centroid), kept for
comparison and backward compatibility.
"""
from __future__ import annotations

import math
import os

import numpy as np

# symmetric triangle rules (barycentric coordinates, weights summing to 1); Dunavant (1985)
_A7, _B7 = 0.059715871789770, 0.470142064105115
_C7, _D7 = 0.797426985353087, 0.101286507323456
QUAD_RULES = {
    1: (np.array([[1 / 3, 1 / 3, 1 / 3]]), np.array([1.0])),
    3: (np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]]), np.full(3, 1 / 3)),
    7: (np.array([[1 / 3, 1 / 3, 1 / 3],
                  [_A7, _B7, _B7], [_B7, _A7, _B7], [_B7, _B7, _A7],
                  [_C7, _D7, _D7], [_D7, _C7, _D7], [_D7, _D7, _C7]]),
        np.array([0.225] + [0.132394152788506] * 3 + [0.125939180544827] * 3)),
}


class Elements:
    """Per-element geometry on the backend (vertices, double-area normals, unit normals, areas, quadrature points)."""

    def __init__(self, bk, verts, tris, centroids, normals, areas, dyn_quad: int = 1, real_dtype=np.float32):
        if dyn_quad not in QUAD_RULES:
            raise ValueError(f"dyn_quad must be one of {sorted(QUAD_RULES)}")
        v = np.asarray(verts, np.float64)[np.asarray(tris)]                          # (N, 3, 3)
        bary, w = QUAD_RULES[dyn_quad]
        qp = np.einsum("qa,nak->qnk", bary, v)                                         # (Q, N, 3)
        asr = lambda a: bk.asarray(np.ascontiguousarray(a), dtype=real_dtype)
        self.bk, self.n = bk, len(tris)
        self.v0, self.v1, self.v2 = asr(v[:, 0]), asr(v[:, 1]), asr(v[:, 2])
        self.n2 = asr(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]))                 # 2 * area * normal
        self.centroids, self.normals, self.areas = asr(centroids), asr(normals), asr(areas)
        self.qp, self.qw = asr(qp), [float(x) for x in w]
        self.real_dtype = real_dtype


def _dynamic_factor(xp, k, r, cdtype):
    """D(r) = d/dr[(e^{-jkr} - 1) / (4 pi r)] = ((z - 1) e^z + 1) / (4 pi r^2), z = -jkr; series for small kr."""
    kr = (k * r).astype(r.dtype)
    z = (-1j * kr).astype(cdtype)
    num_direct = (z - 1) * xp.exp(z) + 1
    z2 = z * z
    num_series = z2 * (0.5 + z * (1 / 3 + z * (1 / 8 + z * (1 / 30 + z / 144))))   # sum_{n>=2} z^n (n-1)/n!
    num = xp.where(kr < 0.1, num_series, num_direct)
    rs = xp.where(r > 0, r, 1)
    return (num / (4 * math.pi * rs * rs)).astype(cdtype), rs


def double_layer(k: float, targets, el: Elements, cdtype, integration: str = "analytic",
                 exclude_self: bool = False, tile: int | None = None, budget_mb: float = 256.0,
                 threads: int | None = None):
    """M (T x N) on the backend: M[t, j] = - int_{T_j} dG/dn_y (x_t, y) dS_y.

    targets: (T, 3) evaluation points (backend array or NumPy). exclude_self: targets are the element centroids in
    element order; the diagonal is then the principal value over the flat element itself, which is zero.
    On the CPU, row tiles are filled by `threads` worker threads (NumPy releases the GIL inside its loops);
    the memory budget is shared between them.
    """
    bk, xp = el.bk, el.bk.xp
    x_all = bk.asarray(targets, dtype=el.real_dtype)
    T, N = x_all.shape[0], el.n
    if threads is None:
        threads = 1 if bk.gpu else max(1, min(16, os.cpu_count() or 1))
    if tile is None:
        per_pair = 160 if el.real_dtype == np.float32 else 320                       # bytes of temporaries
        tile = int(max(8, min(T, budget_mb * 2 ** 20 / threads // max(1, N * per_pair))))
    M = xp.empty((T, N), dtype=cdtype)
    starts = list(range(0, T, tile))
    if threads > 1 and len(starts) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(lambda t0: _fill_rows(k, x_all, el, M, t0, min(t0 + tile, T), cdtype, integration),
                          starts))
    else:
        for t0 in starts:
            _fill_rows(k, x_all, el, M, t0, min(t0 + tile, T), cdtype, integration)
    if exclude_self:
        if T != N:
            raise ValueError("exclude_self needs the element centroids as targets")
        xp.fill_diagonal(M, 0)
    return M


def _fill_rows(k, x_all, el, M, t0, t1, cdtype, integration):
    """Rows t0:t1 of the double-layer matrix (written in place into M)."""
    xp = el.bk.xp
    two_pi, four_pi = 2 * math.pi, 4 * math.pi
    x = x_all[t0:t1, None, :]
    if integration == "centroid":                                                      # original one-point rule
        R = x - el.centroids[None]
        r = xp.sqrt((R * R).sum(-1))
        rs = xp.where(r > 0, r, 1)
        Rn = (R * el.normals[None]).sum(-1)
        G = xp.exp((-1j * k * rs).astype(cdtype)) / (four_pi * rs)
        blk = G * (-1j * k - 1.0 / rs) * Rn / rs * el.areas[None]
        blk = xp.where(r > 0, blk, 0)
    elif integration == "analytic":
        a = el.v0[None] - x
        b = el.v1[None] - x
        c = el.v2[None] - x
        la, lb, lc = (xp.sqrt((u * u).sum(-1)) for u in (a, b, c))
        trip = (a * el.n2[None]).sum(-1)                                               # (v0 - x) . n2
        den = (la * lb * lc + (a * b).sum(-1) * lc + (a * c).sum(-1) * lb + (b * c).sum(-1) * la)
        del a, b, c, la, lb, lc
        blk = (xp.arctan2(trip, den) / two_pi).astype(cdtype)                          # Omega / (4 pi)
        del trip, den
        dyn = None
        for q, w in enumerate(el.qw):
            R = x - el.qp[q][None]
            r = xp.sqrt((R * R).sum(-1))
            Rn = (R * el.normals[None]).sum(-1)
            D, rs = _dynamic_factor(xp, k, r, cdtype)
            term = w * D * (Rn / rs)
            dyn = term if dyn is None else dyn + term
            del R, r, Rn, D, rs
        blk += dyn * el.areas[None]
        del dyn
    else:
        raise ValueError(f"unknown integration {integration!r} (analytic or centroid)")
    M[t0:t1] = blk


def plane_waves(k: float, doa, points, cdtype, xp):
    """(P, D) incident field exp(+jk d.x) at points (P, 3) for unit directions doa (D, 3), both backend arrays."""
    return xp.exp((1j * k) * (points @ doa.T)).astype(cdtype)


def point_sources(k: float, sources, points, cdtype, xp):
    """(P, S) incident field of point sources (monopoles) at sources (S, 3), normalized to 1 at the origin:
    (r0 / r) exp(-jk (r - r0)), r = |x - s|, r0 = |s|. Far away (r0 >> |x|) this tends to the plane wave
    exp(+jk d.x) with d = s / r0, so point-source and plane-wave tables share one normalization."""
    x = points.astype(xp.float64)                     # double precision: r - r0 is a small difference of
    s = sources.astype(xp.float64)                    # large numbers when the source is far away
    diff = x[:, None, :] - s[None, :, :]
    r = xp.sqrt((diff * diff).sum(axis=2))
    r0 = xp.sqrt((s * s).sum(axis=1))[None, :]
    dr = ((x * x).sum(axis=1)[:, None] - 2 * (x @ s.T)) / (r + r0)      # r - r0 without cancellation
    return ((r0 / r) * xp.exp((-1j * k) * dr)).astype(cdtype)


# Burton-Miller: hypersingular operator and normal derivatives of the incident field
#
# Normal derivative of the representation at a surface point x with normal n_x:
#     (N p)(x) = d/dn_x (M p)(x),    and for a rigid body  N p = dp_inc/dn.
# With G = G0 + g, g(r) = (e^{-jkr} - 1) / (4 pi r), R = x - y:
#   static part, exactly: d/dn_x [Omega_j(x) / (4 pi)] = n_x . grad Omega_j / (4 pi), where the gradient of the solid
#     angle of a triangle is a sum over its edges (the Biot-Savart field of the edge loop), continuous across the
#     element, so the self term is the finite-part value directly;
#   dynamic part: int_Tj [ B(r) (n_x . n_y) - A(r) (R.n_x)(R.n_y) / r^2 ] dS_y,
#     B = g'/r = f / (4 pi r^3),  A = g'/r - g'' = (3 f - z^2 e^z) / (4 pi r^3),  f = (z - 1) e^z + 1,  z = -jkr.
#     A and B both behave like -k^2 / (8 pi r): on the element itself (R.n = 0, n_x = n_y) the 1/r part is
#     integrated analytically and the bounded rest with the 7-point rule.

def _hyper_factors(xp, k, r, cdtype):
    """A(r), B(r) of the dynamic hypersingular kernel; series for small kr (r = 0 allowed: returns finite junk)."""
    kr = (k * r).astype(r.dtype)
    z = (-1j * kr).astype(cdtype)
    ez = xp.exp(z)
    f_direct = (z - 1) * ez + 1
    a_direct = 3 * f_direct - z * z * ez
    z2 = z * z
    f_series = z2 * (0.5 + z * (1 / 3 + z * (1 / 8 + z * (1 / 30 + z / 144))))
    a_series = z2 * (0.5 + z * z * (-1 / 8 + z * (-1 / 15 - z / 48)))
    small = kr < 0.1
    rs = xp.where(r > 0, r, 1)
    den = 4 * math.pi * rs * rs * rs
    A = xp.where(small, a_series, a_direct) / den
    B = xp.where(small, f_series, f_direct) / den
    return A.astype(cdtype), B.astype(cdtype), rs


def _b_regular(xp, k, r, cdtype):
    """B(r) + k^2 / (8 pi r): the bounded part of B, also at r = 0."""
    kr = (k * r).astype(r.dtype)
    z = (-1j * kr).astype(cdtype)
    jk3 = (-1j * k) ** 3
    series = jk3 * (1 / 3 + z * (1 / 8 + z * (1 / 30 + z / 144))) / (4 * math.pi)
    rs = xp.where(r > 0, r, 1)
    direct = ((z - 1) * xp.exp(z) + 1 - z * z / 2) / (4 * math.pi * rs * rs * rs)
    return xp.where(kr < 0.1, series, direct).astype(cdtype)


def _grad_solid_angle(xp, x, v0, v1, v2):
    """Gradient of the solid angle of triangles (v0, v1, v2) seen from x: sum over the edges of
    (u1 x u2)(|u1| + |u2|) / (|u1| |u2| (|u1| |u2| + u1.u2)), u = vertex - x. Broadcasts like its arguments."""
    out = None
    for p, q in ((v0, v1), (v1, v2), (v2, v0)):
        u1, u2 = p - x, q - x
        l1 = xp.sqrt((u1 * u1).sum(-1))
        l2 = xp.sqrt((u2 * u2).sum(-1))
        cr = xp.cross(u1, u2)
        ll, dot = l1 * l2, (u1 * u2).sum(-1)
        # l1 l2 + u1.u2 cancels when x is near the edge itself (u1, u2 opposite); there use the equal form
        # |u1 x u2|^2 / (l1 l2 - u1.u2), which does not (single precision, sliver triangles)
        c2 = (cr * cr).sum(-1)
        neg = dot < 0
        den = xp.where(neg, c2 / xp.where(neg, ll - dot, 1), ll + dot)
        term = cr * ((l1 + l2) / (ll * den))[..., None]
        out = term if out is None else out + term
    return out


def _inv_r_self(xp, v0, v1, v2, c):
    """int_T dS / r from the centroid c of every flat triangle (in its plane), analytically: sum over the edges of
    d ln((l2 + s2) / (l1 + s1)), d the distance from c to the edge line, s the edge coordinates of its ends."""
    tot = 0
    for A, B in ((v0, v1), (v1, v2), (v2, v0)):
        e = B - A
        t = e / xp.sqrt((e * e).sum(-1))[:, None]
        pa, pb = A - c, B - c
        s1, s2 = (pa * t).sum(-1), (pb * t).sum(-1)
        l1, l2 = xp.sqrt((pa * pa).sum(-1)), xp.sqrt((pb * pb).sum(-1))
        perp = pa - s1[:, None] * t
        d2 = (perp * perp).sum(-1)
        d = xp.sqrt(d2)
        # l + s cancels for an end behind the foot of the perpendicular (s < 0); l + s = d^2 / (l - s) there
        e1 = xp.where(s1 < 0, d2 / xp.where(s1 < 0, l1 - s1, 1), l1 + s1)
        e2 = xp.where(s2 < 0, d2 / xp.where(s2 < 0, l2 - s2, 1), l2 + s2)
        tot = tot + d * xp.log(e2 / e1)
    return tot


def hypersingular(k: float, el: Elements, cdtype, tile: int | None = None, budget_mb: float = 256.0,
                  threads: int | None = None):
    """N (N x N) on the backend: N[i, j] = d/dn_x of M[i, j] at the centroid x of element i, normal n_i."""
    bk, xp = el.bk, el.bk.xp
    N = el.n
    bary, w = QUAD_RULES[7]
    v0, v1, v2 = el.v0, el.v1, el.v2
    qp = [bk.asarray(b[0] * bk.asnumpy(v0) + b[1] * bk.asnumpy(v1) + b[2] * bk.asnumpy(v2),
                     dtype=el.real_dtype) for b in bary]                                   # 7 x (N, 3)
    if threads is None:
        threads = 1 if bk.gpu else max(1, min(16, os.cpu_count() or 1))
    if tile is None:
        per_pair = 480 if el.real_dtype == np.float32 else 960
        tile = int(max(8, min(N, budget_mb * 2 ** 20 / threads // max(1, N * per_pair))))
    H = xp.empty((N, N), dtype=cdtype)

    def rows(t0):
        t1 = min(t0 + tile, N)
        x = el.centroids[t0:t1, None, :]
        nx = el.normals[t0:t1, None, :]
        g = _grad_solid_angle(xp, x, v0[None], v1[None], v2[None])                      # (T, N, 3)
        blk = ((g * nx).sum(-1) / (4 * math.pi)).astype(cdtype)
        del g
        nxny = (nx * el.normals[None]).sum(-1)
        for q, wq in enumerate(w):
            R = x - qp[q][None]
            r = xp.sqrt((R * R).sum(-1))
            A, B, rs = _hyper_factors(xp, k, r, cdtype)
            Rnx = (R * nx).sum(-1)
            Rny = (R * el.normals[None]).sum(-1)
            blk += float(wq) * (B * nxny - A * (Rnx * Rny) / (rs * rs)) * el.areas[None]
            del R, r, A, B, rs, Rnx, Rny
        H[t0:t1] = blk

    starts = list(range(0, N, tile))
    if threads > 1 and len(starts) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(rows, starts))
    else:
        for t0 in starts:
            rows(t0)

    # diagonal: static finite part + dynamic part with the 1/r singularity integrated analytically
    c = el.centroids
    g = _grad_solid_angle(xp, c, v0, v1, v2)                                              # (N, 3)
    static = ((g * el.normals).sum(-1) / (4 * math.pi)).astype(cdtype)
    dyn = 0
    for q, wq in enumerate(w):
        R = c - qp[q]
        r = xp.sqrt((R * R).sum(-1))
        dyn = dyn + float(wq) * _b_regular(xp, k, r, cdtype)
    dyn = dyn * el.areas - (k * k / (8 * math.pi)) * _inv_r_self(xp, v0, v1, v2, c)
    idx = xp.arange(N)
    H[idx, idx] = static + dyn.astype(cdtype)
    return H


def plane_waves_dn(k: float, doa, points, normals, cdtype, xp):
    """(P, D) normal derivative of exp(+jk d.x): jk (n.d) exp(+jk d.x)."""
    return ((1j * k) * (normals @ doa.T) * xp.exp((1j * k) * (points @ doa.T))).astype(cdtype)


def point_sources_dn(k: float, sources, points, normals, cdtype, xp):
    """(P, S) normal derivative of the point-source field (r0 / r) exp(-jk (r - r0)):
    p_inc (-1/r - jk) ((x - s).n) / r."""
    x = points.astype(xp.float64)
    n = normals.astype(xp.float64)
    s = sources.astype(xp.float64)
    diff = x[:, None, :] - s[None, :, :]
    r = xp.sqrt((diff * diff).sum(axis=2))
    rn = (diff * n[:, None, :]).sum(axis=2)
    p = point_sources(k, sources, points, np.complex128, xp)
    return (p * (-1.0 / r - 1j * k) * rn / r).astype(cdtype)
