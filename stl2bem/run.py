"""Command line: closed STL surface -> HDF5 table of the pressure at receivers.

    stl2bem --stl body.stl --units mm --mics mics.csv --out out/
    stl2bem --stl body.stl --units mm --n_receivers 24 --out out/ --dry_run
    stl2bem --stl body.stl --units mm --mics mics.csv --out out/ --source_distance 0.5      # near-field sources

Output: <out>/p_total_<name>_az<A>el<E>[_r<R>m]_<formulation>_<N>receivers.h5 (see README for the format).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .backend import Backend
from .bem_core import estimate_cost, solve_table, spherical_doa_grid, write_free_field
from .chief import chief_points
from .panels import auto_panel_targets, load_mic_positions, map_to_elements, surface_receivers
from .stl_io import build_mesh, valid_up_to_hz


def body_size(verts) -> float:
    """Largest dimension of the body (greatest distance between two vertices, through the convex hull)."""
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import pdist
    v = np.asarray(verts, np.float64)
    try:
        v = v[ConvexHull(v).vertices]
    except Exception:                       # flat or degenerate input: all vertices
        pass
    if len(v) > 4000:
        v = v[np.random.default_rng(0).choice(len(v), 4000, replace=False)]
    return float(pdist(v).max())


def build_parser():
    p = argparse.ArgumentParser("stl2bem", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description="Compute a BEM pressure table of a rigid body from a closed STL mesh.")
    g = p.add_argument_group("body")
    g.add_argument("--stl", required=True, help="closed STL surface of the body")
    g.add_argument("--units", required=True, choices=["m", "cm", "mm"],
                   help="units of the STL coordinates (STL files do not store them)")
    g.add_argument("--scale", type=float, default=None, help="factor to meters; overrides --units")
    g.add_argument("--recenter", default="bbox", choices=["bbox", "centroid", "none"],
                   help="origin of the phase reference: bounding-box center, volume centroid or the CAD origin")
    g.add_argument("--allow_open_mesh", action="store_true",
                   help="solve even if the surface has open or non-manifold edges (results may be wrong)")
    g.add_argument("--name", default=None, help="body name in the output file (default: STL file name)")

    g = p.add_argument_group("receivers")
    g.add_argument("--mics", default=None, help="receiver positions, one x, y, z row each (csv, txt or json)")
    g.add_argument("--mic_units", default=None, choices=["m", "cm", "mm"], help="units of --mics (default: as the STL)")
    g.add_argument("--n_receivers", "--n_panels", dest="n_receivers", type=int, default=24,
                   help="receivers spread over the surface when --mics is not given")
    g.add_argument("--receivers", default="surface", choices=["surface", "field", "snap"],
                   help="surface: on the closest surface point (second-order accurate); field: at the given points "
                        "in the air; snap: value of the nearest element, as in the original code")
    g.add_argument("--max_receiver_dist", "--max_panel_dist", dest="max_receiver_dist", type=float, default=0.05,
                   help="largest allowed distance between a given receiver and the surface [m]")
    g.add_argument("--surface_offset", type=float, default=1e-6, help="evaluation height above the surface [m]")

    g = p.add_argument_group("sources and frequencies")
    g.add_argument("--n_az", type=int, default=360, help="azimuths of the direction grid (0 to 360 deg)")
    g.add_argument("--n_el", type=int, default=181, help="elevations of the direction grid (-90 to 90 deg)")
    g.add_argument("--source_distance", type=float, default=None,
                   help="point sources at this distance [m] on the direction grid (near field); "
                        "without it: plane waves (far field)")
    g.add_argument("--fs", type=float, default=16000.0, help="sampling rate [Hz]; frequencies are the FFT bins")
    g.add_argument("--nfft", type=int, default=512, help="FFT length; nfft/2 + 1 frequencies from 0 to fs/2")
    g.add_argument("--c_sound", type=float, default=343.0, help="speed of sound [m/s]")

    g = p.add_argument_group("solver")
    g.add_argument("--formulation", default="chief", choices=["chief", "burton-miller"],
                   help="chief: CBIE + CHIEF interior points; burton-miller: CBIE + normal-derivative equation "
                        "(experimental, see docs/method.md)")
    g.add_argument("--n_chief", type=int, default=20, help="CHIEF interior points (fewer if the body is thin)")
    g.add_argument("--chief_seed", type=int, default=0, help="random seed of the CHIEF point search")
    g.add_argument("--chief_oversample", type=int, default=60, help="candidate points per CHIEF point")
    g.add_argument("--no_chief", action="store_true", help="CBIE only (wrong near the irregular frequencies)")
    g.add_argument("--integration", default="analytic", choices=["analytic", "centroid"],
                   help="analytic: exact singular part; centroid: one-point rule of the original code")
    g.add_argument("--dyn_quad", type=int, default=1, choices=[1, 3, 7],
                   help="quadrature points for the smooth part of the kernel")
    g.add_argument("--backend", default="auto", choices=["auto", "cupy", "numpy"],
                   help="auto: GPU if it can be used, otherwise CPU")
    g.add_argument("--precision", default=None, choices=["single", "double"], help="default: single")
    g.add_argument("--dtype", default="c64", choices=["c64", "c128"], help="older name of --precision")
    g.add_argument("--doa_chunk", type=int, default=4096, help="directions solved at once (memory)")
    g.add_argument("--budget_mb", type=float, default=256.0, help="memory for assembly temporaries [MB]")
    g.add_argument("--tile", type=int, default=None, help=argparse.SUPPRESS)

    g = p.add_argument_group("output")
    g.add_argument("--out", default="out", help="output folder, or a .h5 file path")
    g.add_argument("--free_field", action="store_true", help="also write the table without the body")
    g.add_argument("--sofa", action="store_true", help="also write the table(s) as SOFA (GeneralTF; needs sofar)")
    g.add_argument("--resume", action="store_true", help="continue an interrupted table with the same settings")
    g.add_argument("--dry_run", action="store_true", help="check the mesh, receivers and resources, then stop")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    t0 = time.time()
    name = args.name or Path(args.stl).stem
    use_chief = not args.no_chief and args.formulation == "chief"
    precision = args.precision or ("double" if args.dtype == "c128" else "single")
    if args.sofa:
        import importlib.util
        if importlib.util.find_spec("sofar") is None:
            raise SystemExit("--sofa needs the sofar package: pip install sofar")
    bk = Backend(args.backend)

    print("=" * 70)
    note = f"; GPU not used: {bk.fallback_reason}" if bk.fallback_reason else ""
    print(f"[stl2bem] {name}  (backend {bk.name}{note})")
    print("=" * 70)

    try:
        mesh = build_mesh(args.stl, units=args.units, scale=args.scale, recenter=args.recenter, c=args.c_sound,
                          f_max=args.fs / 2, allow_open=args.allow_open_mesh)
    except ValueError as e:
        raise SystemExit(str(e))
    if mesh.info["edge_mean_m"] <= 0:
        raise SystemExit("could not compute the element size")

    doa_grid, _, _ = spherical_doa_grid(args.n_az, args.n_el)
    freqs = np.linspace(0, args.fs / 2, args.nfft // 2 + 1)
    sources = None if args.source_distance is None else args.source_distance * doa_grid
    print(f"  doa      : {len(doa_grid)} ({args.n_az} x {args.n_el})"
          + (f", point sources at {args.source_distance:g} m" if sources is not None else ", plane waves"))
    print(f"  freq     : {len(freqs)} bins (0 ~ {args.fs / 2:.0f} Hz); mesh valid up to "
          f"~{valid_up_to_hz(mesh, args.c_sound):.0f} Hz (every element <= wavelength / 6)")
    size = body_size(mesh.verts)
    d_far = 2 * size ** 2 / (args.c_sound / (args.fs / 2))                               # Fraunhofer 2 D^2 / lambda
    print(f"  far field: plane waves are accurate for sources beyond 2D^2/lambda = {d_far:.2f} m at "
          f"{args.fs / 2:.0f} Hz")
    print(f"             (D = {size * 100:.1f} cm, largest dimension); closer sources need --source_distance")
    if sources is not None:
        from .chief import points_inside
        if points_inside(sources, mesh.verts, mesh.tris, use_gpu=bk.gpu).any():
            raise SystemExit(f"--source_distance {args.source_distance:g} m puts sources inside the body")

    pts = None
    if use_chief:
        pts = chief_points(mesh, n=args.n_chief, seed=args.chief_seed, oversample=args.chief_oversample,
                           use_gpu=bk.gpu)
        if len(pts) == 0:                    # thin-walled body: the CBIE alone (chief_points says why)
            use_chief = False

    if args.mics:
        mscale = ({"m": 1.0, "cm": 0.01, "mm": 0.001}[args.mic_units]
                  if args.mic_units else mesh.info["scale_to_m"])
        targets = load_mic_positions(args.mics, scale=mscale, shift=mesh.info["origin_shift"])
        print(f"  mics     : {len(targets)} from {args.mics} (x{mscale:g})")
    else:
        targets = auto_panel_targets(mesh, n=args.n_receivers)
        print(f"  mics     : {len(targets)} spread over the surface")
    receivers = panel_idx = rec_elem = None
    if args.receivers == "snap":
        panel_idx = map_to_elements(targets, mesh, max_dist=args.max_receiver_dist)
        positions = mesh.centroids[panel_idx]
    elif args.receivers == "surface":
        receivers, rec_elem, _ = surface_receivers(targets, mesh, max_dist=args.max_receiver_dist,
                                                   offset=args.surface_offset)
        positions = receivers
    else:
        from .chief import points_inside
        inside = points_inside(targets, mesh.verts, mesh.tris, use_gpu=bk.gpu)
        if inside.any():
            raise SystemExit(f"--receivers field: receivers {np.nonzero(inside)[0].tolist()} are inside the body")
        receivers = positions = targets
        print(f"  receivers: {len(receivers)} field points")

    n_chief = len(pts) if use_chief else 0
    est = estimate_cost(len(mesh.centroids), n_chief, len(doa_grid), len(freqs),
                        len(positions), args.doa_chunk, itemsize=8 if precision == "single" else 16, gpu=bk.gpu)
    t = est["seconds"]
    when = f"{t:.0f} s" if t < 120 else (f"{t / 60:.0f} min" if t < 7200 else f"{t / 3600:.1f} h")
    print(f"  estimate : {'GPU memory' if bk.gpu else 'RAM'} ~ {est['memory_gb']:.1f} GB, file ~ {est['file_mb']:.0f} MB, "
          f"time ~ {when} "
          f"({'RTX 4060 Ti' if bk.gpu else '20-thread CPU'} reference; fewer --n_az/--n_el or --nfft is faster)")
    free = bk.free_memory_gb()
    if free is not None and est["gpu_gb"] > free:
        print(f"  [WARN] free GPU memory {free:.1f} GB < estimate {est['gpu_gb']:.1f} GB: use fewer elements "
              f"or a smaller --doa_chunk (memory grows with the square of the element count)")

    out = Path(args.out)
    if out.suffix == ".h5":
        out.parent.mkdir(parents=True, exist_ok=True)
        out_path = out
    else:
        out.mkdir(parents=True, exist_ok=True)
        tag = "burton-miller" if args.formulation == "burton-miller" else ("chief" if use_chief else "cbie")
        dist = "" if sources is None else f"_r{args.source_distance:g}m"
        out_path = out / f"p_total_{name}_az{args.n_az}el{args.n_el}{dist}_{tag}_{len(positions)}receivers.h5"
    print(f"  output   : {out_path}")
    for i, c in enumerate(positions):
        print(f"    receiver {i:>2d}: ({c[0] * 100:>+7.2f}, {c[1] * 100:>+7.2f}, {c[2] * 100:>+7.2f}) cm")

    if args.dry_run:
        print("\n[dry_run] no solve.")
        return 0

    meta = {"body_id": name, "mesh_type": "stl", "description": f"STL: {args.stl}",
            "n_az": args.n_az, "n_el": args.n_el, "fs": args.fs, "nfft": args.nfft,
            "n_chief": n_chief, "recenter": args.recenter, "formulation": args.formulation,
            "source_distance_m": -1.0 if sources is None else args.source_distance,
            **{k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in mesh.info.items() if k != "stl_path"},
            "stl_file": str(Path(args.stl).resolve())}

    solve_table(mesh.verts, mesh.tris, mesh.centroids, mesh.normals, mesh.areas, freqs,
                doa_grid if sources is None else None, str(out_path), sources=sources,
                receivers=receivers, panel_indices=panel_idx, receiver_elements=rec_elem,
                chief_points=pts if use_chief else None, use_chief=use_chief, c=args.c_sound,
                doa_chunk=args.doa_chunk, precision=precision, backend=bk.name, integration=args.integration,
                dyn_quad=args.dyn_quad, formulation=args.formulation, budget_mb=args.budget_mb,
                extra_metadata=meta, resume=args.resume)

    written = [out_path]
    if args.free_field:
        stem = out_path.stem
        ff_stem = (stem.replace("p_total_", "p_total_free_field_", 1)
                   if stem.startswith("p_total_") else stem + "_free_field")
        ff_path = out_path.with_name(ff_stem + out_path.suffix)
        write_free_field(positions, doa_grid, freqs, str(ff_path), sources=sources,
                         c=args.c_sound, extra_metadata={**meta, "body_id": f"free_field_{name}",
                                                         "reference_body": name})
        written.append(ff_path)
    if args.sofa:
        from .sofa import write_sofa
        for h5 in written:
            print(f"  sofa     : {write_sofa(h5, h5.with_suffix('.sofa'))}")
    print(f"done ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
