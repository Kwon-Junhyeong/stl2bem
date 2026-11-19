"""BEM solver: rigid body, plane-wave or point-source incidence, CBIE + CHIEF, tables of the total pressure at
receivers.

Convention: engineering time dependence e^{+jwt}; p_inc = exp(+jk d.r) with d the unit vector towards the source;
G = exp(-jkr) / (4 pi r); normals point out of the body. An FFT / STFT with the usual e^{-jwt} kernel returns the same
phase as the table for a positive-frequency bin (no conjugation needed); a time signal x_m(t) = s(t - tau_m) has
tau_m = -(d . r_m) / c.
Point sources (near field): p_inc = (r0 / r) exp(-jk (r - r0)), r = |x - s|, r0 = |s|, i.e. relative to the
free-field pressure the source produces at the origin (the normalization of near-field HRTFs); far away it tends to
the plane wave from d = s / r0.

For every frequency the surface equation (1/2 I + M) p = p_inc is collocated at the element centroids, augmented with
CHIEF rows M_c p = p_inc at interior points (least squares by QR) to remove the irregular frequencies of the CBIE.
Receivers are then evaluated with p(x) = p_inc(x) - (M p)(x) (see kernels.py), so they need not coincide with element
centroids. Only receiver rows are stored; the full surface solution never leaves the device.
"""
from __future__ import annotations

import os
import time

import h5py
import numpy as np

from .backend import Backend
from .kernels import (Elements, double_layer, hypersingular, plane_waves, plane_waves_dn, point_sources,
                      point_sources_dn)

C_SOUND = 343.0
CONVENTION = ("engineering e^{+jwt}; p_inc = exp(+jk d.r), d = unit vector towards the source; "
              "G = exp(-jkr)/(4 pi r); normals out of the body. Positive-frequency STFT bins (e^{-jwt} kernel) "
              "have the same phase as this table. Point sources: p_inc = (r0/r) exp(-jk(r - r0)), relative to the "
              "free field at the origin.")


# Grids and resource estimate

def spherical_doa_grid(n_az: int = 360, n_el: int = 181):
    """Uniform azimuth x elevation grid: az in [0, 360), el in [-90, 90] degrees.

    Returns (doa (n_az * n_el, 3), az, el); row index = i_az * n_el + i_el."""
    az = np.linspace(0, 360, n_az, endpoint=False)
    el = np.linspace(-90, 90, n_el, endpoint=True)
    a, e = np.meshgrid(np.deg2rad(az), np.deg2rad(el), indexing="ij")
    a, e = a.ravel(), e.ravel()
    doa = np.column_stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return doa, az, el


def estimate_cost(n_elem, n_chief, n_doa, n_freq, n_panel, doa_chunk, itemsize=8, tile=1024, gpu=True):
    """Rough peak memory (GB), solve time (s) and file size (MB).

    Time per frequency = assembly and factorization + one solve per direction, fitted to examples/benchmark and
    to runs with 65,160 directions (box, 8 receivers, single precision): GPU RTX 4060 Ti, CPU i5-13500 (20 threads).
    Other machines scale accordingly; many receivers (field maps) add to it."""
    N, M = n_elem, n_elem + n_chief
    peak = ((2 * M * N + N * N) * itemsize            # A + Q + R
            + (M + N) * doa_chunk * itemsize          # right-hand sides / solutions
            + 256 * 2 ** 20)                          # assembly temporaries (bounded by the tile budget)
    if gpu:
        per_freq = 9.8e-9 * N ** 2.15 + 1.45e-12 * N ** 2 * n_doa
    else:
        per_freq = 2.9e-11 * N ** 3 + 6.1e-10 * N ** 1.75 * n_doa
    return {"gpu_gb": peak / 1024 ** 3, "seconds": per_freq * max(n_freq - 1, 1),   # the 0 Hz bin is free
            "file_mb": n_panel * n_doa * n_freq * 8 / 1024 ** 2}


# Free field (reference without the body)

def write_free_field(panel_pos, doa_grid, freqs, out_h5_path, c=C_SOUND,
                     extra_metadata=None, freq_chunk=16, verbose=True, sources=None):
    """Table of the incident field alone (no scattering) at the same receiver positions: plane waves from doa_grid,
    or point sources at `sources` (doa_grid=None). No GPU needed."""
    panel_pos = np.asarray(panel_pos, dtype=np.float64)
    if sources is not None:
        sources = np.atleast_2d(np.asarray(sources, np.float64))
        doa_grid = sources / np.linalg.norm(sources, axis=1, keepdims=True)
    doa_grid = np.asarray(doa_grid, np.float64)
    N, G, F = len(panel_pos), len(doa_grid), len(freqs)
    kk = 2.0 * np.pi * np.asarray(freqs, np.float64) / c
    with h5py.File(out_h5_path, "w") as f:
        p = f.create_dataset("p", shape=(N, G, F), dtype="complex64")
        for f0 in range(0, F, freq_chunk):
            f1 = min(f0 + freq_chunk, F)
            if sources is None:
                p[:, :, f0:f1] = np.exp(1j * (panel_pos @ doa_grid.T)[:, :, None]
                                        * kk[None, None, f0:f1]).astype(np.complex64)
            else:
                p[:, :, f0:f1] = np.stack([point_sources(k, sources, panel_pos, np.complex128, np)
                                           for k in kk[f0:f1]], axis=2).astype(np.complex64)
        f.create_dataset("doa_grid", data=doa_grid.astype("float32"))
        if sources is not None:
            f.create_dataset("source_positions", data=sources)
        f.create_dataset("freqs", data=np.asarray(freqs, "float32"))
        f.create_dataset("positions", data=panel_pos.astype("float32"))
        f.create_dataset("centroids", data=panel_pos.astype("float32"))
        f.create_dataset("areas", data=np.ones(N, "float32"))
        f.create_dataset("panel_indices_in_full_mesh", data=np.arange(N, dtype=np.int64))
        f.attrs.update({"mesh_type": "free_field", "layout": "panel_major", "use_chief": False,
                        "incidence": "plane" if sources is None else "point",
                        "n_panels": N, "n_doa": G, "n_freq": F, "c_sound": c,
                        "convention": CONVENTION + " Free field: p = p_inc (no body)."})
        for key, val in (extra_metadata or {}).items():
            f.attrs[key] = val
    if verbose:
        print(f"[FF ] saved free-field table: {out_h5_path} ({os.path.getsize(out_h5_path) / 1024 ** 2:.1f} MB)")


# Solver

def solve_table(verts, tris, centroids, normals, areas, freqs, doa_grid, out_h5_path,
                receivers=None, panel_indices=None, receiver_elements=None,
                chief_points=None, use_chief=True,
                c=C_SOUND, doa_chunk=4096, precision="single", backend="auto",
                integration="analytic", dyn_quad=1, budget_mb=256.0,
                extra_metadata=None, resume=False, verbose=True, sources=None, formulation="chief"):
    """Total pressure table p[receiver, doa, freq] written to an HDF5 file (panel-major layout).

    Incidence: plane waves from the unit directions doa_grid (D, 3), towards the source (far field); or, with
    sources (S, 3) [m] and doa_grid=None, point sources at these positions (near field), normalized to the
    free-field pressure each produces at the origin. The table then stores the direction of every source in
    doa_grid and its position in source_positions.

    Geometry in meters (verts (V, 3), tris (N, 3), centroids / normals / areas per element, normals outward).
    Receivers, one of:
      receivers (R, 3) alone                  : exterior points, p = p_inc - M p ('field');
      receivers + receiver_elements (R,)      : points on the surface (projected, 1 um outside) and the element
                                                each lies on ('surface', see below);
      panel_indices (R,)                      : the constant value of these elements (legacy 'snap').

    Surface receivers. Just outside element j the representation formula returns, up to second-order terms, the
    mean of the true pressure p(x) and the element's own constant value p_j (its own solid angle contributes
    -1/2 p_j, the jump of the double layer). The difference p(x) - p_j is first order in the element size, so the
    plain representation, like snapping, is only first-order accurate on the surface. The corrected value
    2 (p_inc - M p)(x) - p_j removes that term: second-order accuracy at any point of the surface
    (see examples/validation_sphere). On an edge or a vertex the jump is shared by the elements meeting there, and
    p_j becomes their mean weighted by the solid angle each subtends (receiver_elements = (index, weight)).
    Formulation. "chief" (default): CBIE + CHIEF, least squares. "burton-miller": CBIE + alpha * (normal derivative
    equation), alpha = -j/k, square system (LU); unique at every frequency without interior points (CHIEF settings are
    then ignored). See docs/method.md.
    CHIEF (irregular frequencies). Without chief_points, interior points are chosen automatically (20 points,
    chief.chief_points); a thin-walled body that has no point deep enough is solved with the CBIE alone, which is
    safe there (its irregular frequencies lie far above the band the mesh resolves). Pass chief_points to choose
    them yourself, or use_chief=False to switch CHIEF off.
    `resume=True` continues an interrupted file.
    """
    if (receivers is None) == (panel_indices is None):
        raise ValueError("give exactly one of receivers= or panel_indices=")
    if formulation not in ("chief", "burton-miller"):
        raise ValueError(f"unknown formulation {formulation!r} (chief or burton-miller)")
    bm = formulation == "burton-miller"
    if bm:
        use_chief = False                             # Burton-Miller needs no interior points
    if use_chief and chief_points is None:            # automatic, as on the command line
        from .chief import chief_points as _find_chief_points
        from .stl_io import mesh_from_arrays
        chief_points = _find_chief_points(mesh_from_arrays(verts, tris), n=20, seed=0,
                                          use_gpu=(backend != "numpy"), verbose=verbose)
    if use_chief and len(chief_points) == 0:          # thin-walled body, see chief.chief_points
        use_chief = False
    from tqdm import tqdm

    bk = Backend(backend)
    xp = bk.xp
    rdt, cdt = (np.float32, np.complex64) if precision == "single" else (np.float64, np.complex128)
    freqs = np.asarray(freqs, np.float64)
    if sources is not None:
        sources = np.atleast_2d(np.asarray(sources, np.float64))
        r0 = np.linalg.norm(sources, axis=1)
        if (r0 <= 0).any():
            raise ValueError("a point source at the origin cannot be normalized (move the origin or the source)")
        from .chief import points_inside
        inside = points_inside(sources, np.asarray(verts, np.float64), np.asarray(tris), use_gpu=bk.gpu)
        if inside.any():
            raise ValueError(f"point sources inside the body: indices {np.nonzero(inside)[0][:10].tolist()}")
        doa_grid = sources / r0[:, None]
        incidence = "point"
    elif doa_grid is None:
        raise ValueError("give doa_grid (plane waves) or sources (point sources)")
    else:
        incidence = "plane"
    doa_grid = np.asarray(doa_grid, np.float64)
    N, n_doa, n_freq = len(tris), len(doa_grid), len(freqs)
    if receivers is not None:
        positions = np.asarray(receivers, np.float64)
        if receiver_elements is not None:
            mode = "surface"                    # corrected representation, see the docstring of this function
            if isinstance(receiver_elements, tuple):            # (index (R, K), weight (R, K)): panels.surface_receivers
                r_idx = np.asarray(receiver_elements[0], np.int64)
                r_w = np.asarray(receiver_elements[1], np.float64)
            else:                                               # one element per receiver
                r_idx = np.asarray(receiver_elements, np.int64)[:, None]
                r_w = np.ones(r_idx.shape)
            if r_idx.shape[0] != len(positions) or r_idx.shape != r_w.shape:
                raise ValueError("receiver_elements: one element index (or index/weight row) per receiver")
        else:
            mode = "representation"
    else:
        panel_indices = np.asarray(panel_indices, np.int64)
        if (panel_indices < 0).any() or (panel_indices >= N).any():
            raise ValueError(f"panel_indices out of range [0, {N})")
        positions = np.asarray(centroids, np.float64)[panel_indices]
        mode = "element"
    R = len(positions)

    scheme = "BM" if bm else ("CHIEF" if use_chief else "CBIE")
    if verbose:
        print(f"[BEM-{scheme}] {n_freq} freq x {n_doa} {'point sources' if incidence == 'point' else 'DOA'}, "
              f"mesh {N} elements -> {R} receivers "
              f"({mode}){f' + {len(chief_points)} CHIEF points' if use_chief else ''}")
        if bk.fallback_reason:
            print(f"          GPU not used: {bk.fallback_reason}")
        print(f"          backend={bk.name}, precision={precision}, integration={integration}"
              f"{f' (dynamic part {dyn_quad}-point)' if integration == 'analytic' else ''}, "
              f"file ~ {R * n_doa * n_freq * 8 / 1024 ** 2:.0f} MB")

    el = Elements(bk, verts, tris, centroids, normals, areas, dyn_quad=dyn_quad, real_dtype=rdt)
    doa_g = bk.asarray(doa_grid, dtype=rdt)
    src_g = bk.asarray(sources, dtype=np.float64) if incidence == "point" else None
    if incidence == "point" and verbose:
        edge = float(np.sqrt(2 * np.mean(areas)))
        near = np.min(np.linalg.norm(np.asarray(centroids)[None] - sources[:, None], axis=2), axis=1)
        if (near < 2 * edge).any():
            print(f"          warning: {(near < 2 * edge).sum()} source(s) within two element sizes of the surface; "
                  f"the incident field varies across elements there (refine the mesh near them)")

    def incident(k, pts, d0, d1):
        if incidence == "plane":
            return plane_waves(k, doa_g[d0:d1], pts, cdt, xp)
        return point_sources(k, src_g[d0:d1], pts, cdt, xp)

    def incident_dn(k, d0, d1):                       # normal derivative at the element centroids
        if incidence == "plane":
            return plane_waves_dn(k, doa_g[d0:d1], el.centroids, el.normals, cdt, xp)
        return point_sources_dn(k, src_g[d0:d1], el.centroids, el.normals, cdt, xp)
    cen_g = el.centroids
    rec_g = bk.asarray(positions, dtype=rdt)
    chief_g = bk.asarray(chief_points, dtype=rdt) if use_chief else None
    pidx_g = bk.asarray(panel_indices) if mode == "element" else None
    ridx_g = bk.asarray(r_idx) if mode == "surface" else None
    rw_g = bk.asarray(r_w, dtype=rdt) if mode == "surface" else None

    buf = np.empty((R, n_doa), dtype=np.complex64)
    fmode = "a" if (resume and os.path.exists(out_h5_path)) else "w"
    t_start = time.time()
    with h5py.File(out_h5_path, fmode) as f:
        if fmode == "w":
            f.create_dataset("p", shape=(R, n_doa, n_freq), dtype="complex64")
            f.create_dataset("freq_done", data=np.zeros(n_freq, dtype="u1"))
            f.create_dataset("doa_grid", data=doa_grid.astype("float32"))
            if incidence == "point":
                f.create_dataset("source_positions", data=sources)
            f.create_dataset("freqs", data=freqs.astype("float32"))
            f.create_dataset("positions", data=positions.astype("float32"))
            f.create_dataset("centroids", data=positions.astype("float32"))         # legacy name: where p is given
            if mode == "element":
                f.create_dataset("normals", data=np.asarray(normals)[panel_indices].astype("float32"))
                f.create_dataset("areas", data=np.asarray(areas)[panel_indices].astype("float32"))
                f.create_dataset("panel_indices_in_full_mesh", data=panel_indices)
            if use_chief:
                f.create_dataset("chief_points", data=np.asarray(chief_points, "float32"))
            f.attrs.update({
                "c_sound": c, "use_chief": use_chief, "layout": "panel_major",
                "n_panels": R, "n_full_mesh": N, "n_doa": n_doa, "n_freq": n_freq,
                "receiver_mode": mode, "integration": integration, "dyn_quad": dyn_quad,
                "precision": precision, "backend": bk.name, "convention": CONVENTION, "incidence": incidence,
                "formulation": formulation,
            })
            for key, val in (extra_metadata or {}).items():
                f.attrs[key] = val
        else:                                           # resume only the same problem, never mix two tables
            if f["p"].shape != (R, n_doa, n_freq):
                raise ValueError(f"cannot resume: existing table has shape {f['p'].shape}")
            same = [("freqs", freqs), ("doa_grid", doa_grid), ("positions", positions)]
            if incidence == "point":
                same.append(("source_positions", sources))
            for key, val in same:
                if key not in f or not np.allclose(f[key][:], val, rtol=1e-5, atol=1e-7):
                    raise ValueError(f"cannot resume: {key} differs from the existing table")
            for key, val in (("formulation", formulation), ("integration", integration), ("precision", precision),
                             ("receiver_mode", mode), ("incidence", incidence), ("n_full_mesh", N)):
                old = f.attrs.get(key)
                old = old.decode() if isinstance(old, bytes) else old
                if old is not None and old != val:
                    raise ValueError(f"cannot resume: {key} is {old!r} in the existing table, {val!r} now")

        p_dset, done = f["p"], f["freq_done"][:]
        if verbose and done.sum():
            print(f"          resume: {int(done.sum())}/{n_freq} frequencies already done")
        it = enumerate(freqs)
        if verbose:
            it = tqdm(it, total=n_freq, desc=f"[{scheme}] p_total")

        for i, freq in it:
            if done[i]:
                continue
            if freq < 1e-10 and incidence == "plane":           # DC plane wave: uniform field, p = 1
                p_dset[:, :, i] = 1.0 + 0j
                f["freq_done"][i] = 1
                continue
            k = 2.0 * np.pi * freq / c
            A = double_layer(k, cen_g, el, cdt, integration, exclude_self=True, budget_mb=budget_mb)
            A[xp.arange(N), xp.arange(N)] += 0.5
            alpha = (-1j / k) if (bm and k > 0) else None   # k = 0 (point source, DC): the CBIE is unique
            if alpha is not None:
                A += alpha * hypersingular(k, el, cdt, budget_mb=budget_mb)
            if use_chief:
                A = xp.vstack([A, double_layer(k, chief_g, el, cdt, integration, budget_mb=budget_mb)])
                fac = bk.lstsq_factor(A)
                del A
            else:
                lu = bk.lu_factor(A)
                del A
            # receiver rows of M: all at once if they fit the budget, else in chunks (field maps: many points)
            needs_m = mode in ("representation", "surface")
            r_step = max(1, int(budget_mb * 2 ** 20 // (N * np.dtype(cdt).itemsize))) if needs_m else R
            r_chunks = [(r0, min(r0 + r_step, R)) for r0 in range(0, R, r_step)]
            Mrec = (double_layer(k, rec_g, el, cdt, integration, budget_mb=budget_mb)
                    if needs_m and len(r_chunks) == 1 else None)

            for d0 in range(0, n_doa, doa_chunk):
                d1 = min(d0 + doa_chunk, n_doa)
                b = incident(k, cen_g, d0, d1)
                if alpha is not None:
                    b = b + alpha * incident_dn(k, d0, d1)
                if use_chief:
                    b = xp.vstack([b, incident(k, chief_g, d0, d1)])
                    sol = bk.lstsq_solve(fac, b)
                else:
                    sol = bk.lu_solve(lu, b)
                del b
                if not needs_m:
                    buf[:, d0:d1] = bk.asnumpy(sol[pidx_g, :]).astype(np.complex64)
                for r0, r1 in (r_chunks if needs_m else []):
                    Mc = Mrec if Mrec is not None else double_layer(k, rec_g[r0:r1], el, cdt, integration,
                                                                    budget_mb=budget_mb)
                    out = incident(k, rec_g[r0:r1], d0, d1) - Mc @ sol
                    if mode == "surface":
                        out = 2 * out
                        for kk in range(ridx_g.shape[1]):
                            out -= rw_g[r0:r1, kk, None] * sol[ridx_g[r0:r1, kk], :]
                    buf[r0:r1, d0:d1] = bk.asnumpy(out).astype(np.complex64)
                    del out
                    if Mrec is None:
                        del Mc
                del sol

            p_dset[:, :, i] = buf
            f["freq_done"][i] = 1
            f.flush()
            if use_chief:
                del fac
            else:
                del lu
            del Mrec
            bk.free()
        f.attrs["solve_seconds"] = float(f.attrs.get("solve_seconds", 0.0)) + (time.time() - t_start)

    if verbose:
        print(f"[BEM] saved table: {out_h5_path} ({os.path.getsize(out_h5_path) / 1024 ** 2:.1f} MB)")
    return out_h5_path


def compute_p_total_panel_major(freqs, doa_grid, centroids, normals, areas, panel_indices, out_h5_path,
                                chief_points=None, use_chief=True, c=C_SOUND, doa_chunk=4096, dtype=None,
                                tile=None, extra_metadata=None, resume=False, verbose=True,
                                verts=None, tris=None, integration=None, backend="auto"):
    """Backward-compatible entry point of the original package (element values at `panel_indices`).

    Without `verts`/`tris` only the original one-point centroid rule is possible (the analytic static part needs
    the triangle vertices); with them the analytic rule is used unless integration="centroid" is given."""
    if verts is None or tris is None:
        if integration not in (None, "centroid"):
            raise ValueError("integration='analytic' needs verts= and tris=")
        integration = "centroid"
        n = len(centroids)
        verts = np.repeat(np.asarray(centroids), 3, axis=0)            # placeholders, unused by the centroid rule
        tris = np.arange(3 * n).reshape(n, 3)
    precision = "double" if (dtype is not None and np.dtype(dtype) == np.complex128) else "single"
    return solve_table(verts, tris, centroids, normals, areas, freqs, doa_grid, out_h5_path,
                       panel_indices=panel_indices, chief_points=chief_points, use_chief=use_chief, c=c,
                       doa_chunk=doa_chunk, precision=precision, backend=backend,
                       integration=integration or "analytic", extra_metadata=extra_metadata,
                       resume=resume, verbose=verbose)
