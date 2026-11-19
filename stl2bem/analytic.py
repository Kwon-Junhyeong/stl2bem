"""Reference solution: plane wave scattered by a rigid sphere (in the package convention).

Physics form (e^{-iwt}, incident exp(i k u.r), u the propagation direction):
    p = sum_n (2n + 1) i^n [ j_n(kr) - j_n'(ka) / h_n'(ka) h_n(kr) ] P_n(cos gamma),   cos gamma = u . r_hat
and on the surface r = a (Wronskian):  p = sum_n (2n + 1) i^n (i / (ka)^2) / h_n'(ka) P_n(cos gamma).
The package convention (e^{+jwt}, incident exp(+jk d.r), d towards the source) is the complex conjugate with u = -d.
"""
from __future__ import annotations

import numpy as np
from scipy.special import eval_legendre, spherical_jn, spherical_yn


def _n_terms(x_max: float) -> int:
    return int(x_max + 10 * max(x_max, 1e-3) ** (1 / 3) + 12)


def rigid_sphere(k: float, a: float, points, doa) -> np.ndarray:
    """Total pressure (P, D) at points (P, 3) on or outside a rigid sphere of radius a centered at the origin, for
    plane waves from directions doa (D, 3) (unit vectors towards the source). Package convention."""
    points = np.atleast_2d(np.asarray(points, np.float64))
    doa = np.atleast_2d(np.asarray(doa, np.float64))
    r = np.linalg.norm(points, axis=1)
    if (r < a * (1 - 1e-6)).any():
        raise ValueError("points inside the sphere")
    rhat = points / r[:, None]
    cos_g = -(rhat @ doa.T)                                          # u = -d
    ka = k * a
    n = np.arange(_n_terms(k * r.max()) + 1)
    hp = spherical_jn(n, ka, True) + 1j * spherical_yn(n, ka, True)   # h_n^(1)'(ka)
    P = np.stack([eval_legendre(int(m), cos_g) for m in n])           # (n, P, D)
    on = np.isclose(r, a, rtol=1e-6)
    out = np.empty(cos_g.shape, np.complex128)
    if on.any():
        coef = (2 * n + 1) * (1j ** n) * (1j / ka ** 2) / hp
        out[on] = np.einsum("n,npd->pd", coef, P[:, on])
    if (~on).any():
        kr = k * r[~on]
        jn = spherical_jn(n[:, None], kr[None])
        hn = jn + 1j * spherical_yn(n[:, None], kr[None])
        radial = jn - (spherical_jn(n, ka, True) / hp)[:, None] * hn         # (n, P')
        coef = (2 * n + 1) * (1j ** n)
        out[~on] = np.einsum("n,np,npd->pd", coef, radial, P[:, ~on])
    return np.conj(out)


def rigid_sphere_point(k: float, a: float, points, sources) -> np.ndarray:
    """Total pressure (P, S) at points (P, 3) on or outside a rigid sphere of radius a centered at the origin, for
    point sources at positions sources (S, 3), each farther from the center than every point. Normalized like
    stl2bem tables: divided by the free-field pressure the source produces at the origin. Package convention.

    Physics form (e^{-iwt}), source at distance r_s, a <= r < r_s, cos gamma = r_hat . s_hat:
        p = (ik / 4 pi) sum_n (2n + 1) h_n(k r_s) [ j_n(kr) - j_n'(ka) / h_n'(ka) h_n(kr) ] P_n(cos gamma),
    on the surface the bracket is i / ((ka)^2 h_n'(ka)) (Wronskian); free field at the origin exp(i k r_s) / (4 pi r_s).
    """
    points = np.atleast_2d(np.asarray(points, np.float64))
    sources = np.atleast_2d(np.asarray(sources, np.float64))
    r = np.linalg.norm(points, axis=1)
    rs = np.linalg.norm(sources, axis=1)
    if (r < a * (1 - 1e-6)).any():
        raise ValueError("points inside the sphere")
    if r.max() >= rs.min():
        raise ValueError("every point must be closer to the center than every source")
    cos_g = (points / r[:, None]) @ (sources / rs[:, None]).T                    # (P, S)
    ka = k * a
    ratio = r.max() / rs.min()                                                    # terms decay like ratio^n
    n_max = min(_n_terms(k * rs.max()) + int(np.ceil(np.log(1e-12) / np.log(ratio))), 120)
    n = np.arange(n_max + 1)
    hp = spherical_jn(n, ka, True) + 1j * spherical_yn(n, ka, True)               # h_n'(ka)
    hs = spherical_jn(n[:, None], k * rs[None]) + 1j * spherical_yn(n[:, None], k * rs[None])   # (n, S)
    P = np.stack([eval_legendre(int(m), cos_g) for m in n])                       # (n, P, S)
    on = np.isclose(r, a, rtol=1e-6)
    out = np.empty(cos_g.shape, np.complex128)
    if on.any():
        coef = (2 * n + 1) * (1j / ka ** 2) / hp                                  # (n,)
        out[on] = np.einsum("n,ns,nps->ps", coef, hs, P[:, on])
    if (~on).any():
        kr = k * r[~on]
        jn = spherical_jn(n[:, None], kr[None])
        hn = jn + 1j * spherical_yn(n[:, None], kr[None])
        radial = jn - (spherical_jn(n, ka, True) / hp)[:, None] * hn               # (n, P')
        out[~on] = np.einsum("n,np,ns,nps->ps", 2 * n + 1, radial, hs, P[:, ~on])
    out *= 1j * k / (4 * np.pi)
    if not np.isfinite(out).all():
        raise FloatingPointError("series overflow; move the sources farther from the points")
    free0 = np.exp(1j * k * rs) / (4 * np.pi * rs)                                # free field at the origin
    return np.conj(out / free0[None, :])


def interior_dirichlet_ka(n_max: int = 3, count: int = 3) -> list[tuple[float, int]]:
    """Irregular frequencies of the CBIE for a sphere: zeros ka of j_n (interior Dirichlet eigenvalues)."""
    from scipy.optimize import brentq
    out = []
    for n in range(n_max + 1):
        f = lambda x: spherical_jn(n, x)
        xs = np.linspace(0.5, 20, 4000)
        v = f(xs)
        roots = [brentq(f, xs[i], xs[i + 1]) for i in range(len(xs) - 1) if v[i] * v[i + 1] < 0][:count]
        out += [(float(x), n) for x in roots]
    return sorted(out)
