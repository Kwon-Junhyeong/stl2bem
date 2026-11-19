"""Analysis of response tables: loading, free-field references, direction-of-arrival error caused by the body.

Typical question: an array is mounted on a device; the localization algorithm uses free-field steering vectors
(microphone positions only). How far off are its estimates because of the body? `doa_error` feeds the table of the
body (the "truth") to an estimator steered with another model (free field, or another table) and returns the angular
error per source direction.

    from stl2bem.analysis import load_table, free_field, doa_error
    body = load_table("p_total_robot_..._4panel.h5")
    err = doa_error(body, free_field(body), band=(300, 2600), method="srp-phat", search="azimuth")
"""
from __future__ import annotations

from dataclasses import dataclass, field

import h5py
import numpy as np


@dataclass
class Table:
    p: np.ndarray            # (R, D, F) complex, package convention (see bem_core.CONVENTION)
    doa: np.ndarray          # (D, 3) unit vectors towards the source
    freqs: np.ndarray        # (F,)
    positions: np.ndarray    # (R, 3) receiver positions [m]
    attrs: dict = field(default_factory=dict)
    sources: np.ndarray | None = None   # (D, 3) point-source positions [m] (near field); None for plane waves

    @property
    def az_el(self):
        """(azimuth, elevation) in degrees of every direction."""
        az = np.degrees(np.arctan2(self.doa[:, 1], self.doa[:, 0])) % 360
        el = np.degrees(np.arcsin(np.clip(self.doa[:, 2], -1, 1)))
        return az, el

    def select(self, receivers=None, doa=None, freqs=None) -> "Table":
        r = slice(None) if receivers is None else receivers
        d = slice(None) if doa is None else doa
        f = slice(None) if freqs is None else freqs
        return Table(self.p[r][:, d][:, :, f], self.doa[d], self.freqs[f], self.positions[r], dict(self.attrs),
                     None if self.sources is None else self.sources[d])


def load_table(path, receivers=None, elevations=None, band=None) -> Table:
    """Read a table; optionally only some receivers, the directions at given elevations (degrees, nearest grid
    row of an az x el grid) and the frequencies within band=(lo, hi) Hz. Reads only what is needed."""
    with h5py.File(path, "r") as f:
        doa = f["doa_grid"][:].astype(np.float64)
        freqs = f["freqs"][:].astype(np.float64)
        pos = (f["positions"] if "positions" in f else f["centroids"])[:].astype(np.float64)
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        r_idx = np.arange(len(pos)) if receivers is None else np.asarray(receivers)
        if elevations is not None:
            el = np.degrees(np.arcsin(np.clip(doa[:, 2], -1, 1)))
            levels = np.unique(np.round(el, 6))
            d_idx = np.concatenate([np.nonzero(np.isclose(el, levels[np.argmin(np.abs(levels - e))]))[0]
                                    for e in np.atleast_1d(elevations)])
        else:
            d_idx = np.arange(len(doa))
        f_idx = (np.arange(len(freqs)) if band is None
                 else np.nonzero((freqs >= band[0]) & (freqs <= band[1]))[0])
        p = f["p"][np.sort(r_idx)][:, np.sort(d_idx)][:, :, np.sort(f_idx)]
        src = f["source_positions"][:][np.sort(d_idx)] if "source_positions" in f else None
    return Table(p, doa[np.sort(d_idx)], freqs[np.sort(f_idx)], pos[np.sort(r_idx)], attrs, src)


def free_field(t: Table, positions=None, doa=None, c: float = 343.0) -> Table:
    """Free-field table (no body) at the given (or the table's) receiver positions and directions; point sources
    if the table has them (same normalization as the solver: 1 at the origin)."""
    pos = t.positions if positions is None else np.asarray(positions, np.float64)
    k = 2 * np.pi * t.freqs / c
    if t.sources is not None and doa is None:
        s = t.sources
        r = np.linalg.norm(pos[:, None] - s[None], axis=2)                       # (R, D)
        r0 = np.linalg.norm(s, axis=1)[None]
        dr = ((pos * pos).sum(1)[:, None] - 2 * pos @ s.T) / (r + r0)
        p = (r0 / r)[:, :, None] * np.exp(-1j * dr[:, :, None] * k[None, None, :])
        return Table(p, t.doa.copy(), t.freqs.copy(), pos, {"mesh_type": "free_field"}, s.copy())
    d = t.doa if doa is None else np.asarray(doa, np.float64)
    p = np.exp(1j * (pos @ d.T)[:, :, None] * k[None, None, :])
    return Table(p, d, t.freqs.copy(), pos, {"mesh_type": "free_field"})


def azimuth_ring(n: int = 360, elevation_deg: float = 0.0) -> np.ndarray:
    a = np.radians(np.arange(n) * 360.0 / n)
    e = np.radians(elevation_deg)
    return np.column_stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.full(n, np.sin(e))])


def _srp_phat(X, S):
    """X (R, F) one source; S (R, G, F) steering. Sum over microphone pairs and bins of PHAT-weighted cross-spectra
    projected on the steering phases (DiBiase 2000)."""
    R = X.shape[0]
    out = np.zeros(S.shape[1])
    for i in range(R):
        for j in range(i + 1, R):
            g = X[i] * np.conj(X[j])
            g = g / (np.abs(g) + 1e-30)
            s = S[i] * np.conj(S[j])
            s = s / (np.abs(s) + 1e-30)
            out += np.real(np.conj(s) @ g)
    return out


def _music(X, S, n_src: int = 1):
    """Narrowband MUSIC per bin with the rank-one covariance of X, pseudo-spectra normalized and averaged."""
    R, F = X.shape
    out = np.zeros(S.shape[1])
    for fi in range(F):
        x = X[:, fi:fi + 1]
        C = x @ x.conj().T + 1e-9 * np.real(np.vdot(x, x)) * np.eye(R)
        w, V = np.linalg.eigh(C)
        En = V[:, : R - n_src]
        a = S[:, :, fi]
        a = a / (np.linalg.norm(a, axis=0, keepdims=True) + 1e-30)
        ps = 1.0 / (np.sum(np.abs(En.conj().T @ a) ** 2, axis=0) + 1e-12)
        out += ps / ps.max()
    return out


def angle_between(u, v) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.sum(u * v, -1), -1, 1)))


def doa_error(truth: Table, model: Table, band=None, method: str = "srp-phat", search: str = "azimuth",
              sources=None) -> dict:
    """Localization error when the observations come from `truth` and the estimator is steered with `model`.

    truth, model: tables with the same receivers and frequencies; model.doa is the search grid.
    search='azimuth': estimate the azimuth only (planar arrays, sources near the horizon); the error is the signed
    azimuth difference. search='full': search all model directions; the error is the great-circle angle.
    sources: indices of truth directions to test (default all). Returns a dict of arrays."""
    if truth.p.shape[0] != model.p.shape[0] or not np.allclose(truth.freqs, model.freqs):
        raise ValueError("truth and model need the same receivers and frequencies")
    f_sel = (np.ones(len(truth.freqs), bool) if band is None
             else (truth.freqs >= band[0]) & (truth.freqs <= band[1]))
    S = model.p[:, :, f_sel]
    est_fn = {"srp-phat": _srp_phat, "music": _music}[method]
    src = np.arange(len(truth.doa)) if sources is None else np.asarray(sources)
    est = np.empty((len(src), 3))
    for n, s in enumerate(src):
        est[n] = model.doa[int(np.argmax(est_fn(truth.p[:, s, f_sel], S)))]
    true = truth.doa[src]
    out = {"true": true, "estimate": est, "angle": angle_between(true, est)}
    if search == "azimuth":
        az_t = np.degrees(np.arctan2(true[:, 1], true[:, 0]))
        az_e = np.degrees(np.arctan2(est[:, 1], est[:, 0]))
        out["azimuth_error"] = (az_e - az_t + 180) % 360 - 180
    return out


def summarize(err: dict, key: str = None) -> str:
    e = err[key or ("azimuth_error" if "azimuth_error" in err else "angle")]
    a = np.abs(e)
    return (f"mean |err| {a.mean():.2f} deg, 95th pct {np.percentile(a, 95):.2f}, max {a.max():.2f}, "
            f"mean signed {e.mean():+.2f}")
