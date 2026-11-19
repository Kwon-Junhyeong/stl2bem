"""CAD parts -> one closed surface for the BEM (voxel union + marching cubes).

CAD exports are usually several overlapping, open or non-manifold meshes. The BEM needs one closed, non-overlapping
surface. This module rasterizes every triangle of every part into an occupancy grid of pitch h, fills enclosed
cavities, keeps the largest connected body and extracts the boundary with marching cubes, which is closed and
consistently oriented by construction. Elements come out at about the voxel size, so h also sets the frequency
limit (c / (6 h)).

Effects to keep in mind (reported by `prep_report`):
  * features smaller than h disappear and gaps narrower than about h close;
  * plates thinner than h become one voxel thick;
  * the surface position is quantized to h / 2.

    python -m stl2bem.prep part1.stl part2.stl ... --pitch 8 --units mm --out body.stl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

from .stl_io import read_stl, resolve_scale, write_stl


def sample_triangles(tris: np.ndarray, step: float) -> np.ndarray:
    """Points on every triangle (T, 3, 3), at most `step` apart (barycentric grid per triangle)."""
    out = [tris.reshape(-1, 3)]
    e = np.stack([np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1),
                  np.linalg.norm(tris[:, 2] - tris[:, 0], axis=1),
                  np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1)], 1).max(1)
    k = np.maximum(1, np.ceil(e / step)).astype(int)
    for kk in np.unique(k):
        sel = tris[k == kk]
        u, v = np.meshgrid(np.arange(kk + 1), np.arange(kk + 1), indexing="ij")
        m = (u + v) <= kk
        u, v = u[m] / kk, v[m] / kk
        p = (sel[:, None, 0] + u[None, :, None] * (sel[:, None, 1] - sel[:, None, 0])
             + v[None, :, None] * (sel[:, None, 2] - sel[:, None, 0]))
        out.append(p.reshape(-1, 3))
    return np.concatenate(out)


def voxel_union(parts: list[np.ndarray], pitch: float, extra_points: np.ndarray | None = None,
                align_z: float | None = None, keep_largest: bool = True):
    """Closed surface (vertices (V, 3), triangles (T, 3)) enclosing the union of all parts (each (T_i, 3, 3)).

    extra_points: additional solid points (for example a slab that replaces a detailed part).
    align_z: put a voxel boundary at this height, so that a flat face there stays exactly at z = align_z.
    Returns (verts, tris, info)."""
    from skimage.measure import marching_cubes

    h = float(pitch)
    tris = np.concatenate([np.asarray(p, np.float64) for p in parts])
    pts = sample_triangles(tris, h / 3)
    if extra_points is not None:
        pts = np.concatenate([pts, np.asarray(extra_points, np.float64)])
    lo = pts.min(0) - 2 * h
    if align_z is not None:      # marching cubes puts the surface at lo + (i + 1/2) h between voxels i and i + 1
        lo[2] = align_z - 0.5 * h - np.ceil((align_z - 0.5 * h - lo[2]) / h) * h
    idx = np.floor((pts - lo) / h).astype(int)
    occ = np.zeros(idx.max(0) + 3, bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    occ = ndimage.binary_fill_holes(occ)
    lab, n = ndimage.label(occ)
    sizes = ndimage.sum(occ, lab, range(1, n + 1)) if n else np.array([0.0])
    if keep_largest and n > 1:
        occ = lab == (1 + int(np.argmax(sizes)))
    verts, faces, _, _ = marching_cubes(np.pad(occ, 1).astype(np.float32), level=0.5, spacing=(h, h, h))
    verts = verts - h + lo
    # marching cubes winds faces with normals towards the low values (outside) for this volume: check and fix
    v = verts[faces]
    vol = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6
    if vol < 0:
        faces = faces[:, ::-1]
    info = {"pitch": h, "n_input_triangles": int(len(tris)), "n_bodies": int(n),
            "largest_fraction": float(sizes.max() / max(sizes.sum(), 1)), "n_triangles": int(len(faces)),
            "volume": float(abs(vol))}
    return verts, faces.astype(np.int64), info


def prep_report(verts, tris, info, c=343.0, unit_to_m=1.0):
    v = verts[tris]
    e = np.linalg.norm(v[:, [1, 2, 0]] - v, axis=2)
    h = info["pitch"] * unit_to_m
    print(f"  input    : {info['n_input_triangles']} triangles -> voxel pitch {info['pitch']:g} "
          f"({info['n_bodies']} connected bodies; largest {100 * info['largest_fraction']:.1f}% of the voxels)")
    print(f"  output   : {len(tris)} triangles, mean edge {e.mean():.3g}, max {e.max():.3g} (input units)")
    print(f"  valid up to ~{c / (6 * e.mean() * unit_to_m):.0f} Hz (lambda/6); details below {info['pitch']:g} and "
          f"gaps narrower than about {info['pitch']:g} are lost; thin plates become >= {info['pitch']:g} thick")
    if info["n_bodies"] > 1 and info["largest_fraction"] < 0.999:
        print(f"  [WARN] {info['n_bodies']} separate bodies: only the largest was kept (use --keep_all to keep all)")


def main(argv=None):
    ap = argparse.ArgumentParser("stl2bem-prep", description="Merge overlapping STL parts into one closed surface for the BEM.")
    ap.add_argument("parts", nargs="+", help="STL files of the parts, in the same coordinate frame")
    ap.add_argument("--pitch", type=float, required=True, help="voxel size, in the STL units (sets the element size)")
    ap.add_argument("--units", default="auto", choices=["auto", "m", "cm", "mm"], help="for the frequency report")
    ap.add_argument("--align_z", type=float, default=None, help="keep a flat face exactly at this height")
    ap.add_argument("--keep_all", action="store_true", help="keep every connected body, not only the largest")
    ap.add_argument("--out", required=True, help="output STL (same units as the input)")
    args = ap.parse_args(argv)
    parts = [read_stl(p)[0] for p in args.parts]
    scale, _ = resolve_scale(np.concatenate(parts), args.units)
    verts, tris, info = voxel_union(parts, args.pitch, align_z=args.align_z, keep_largest=not args.keep_all)
    write_stl(args.out, verts[tris], header="stl2bem.prep voxel union")
    prep_report(verts, tris, info, unit_to_m=scale)
    print(f"  written  : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
