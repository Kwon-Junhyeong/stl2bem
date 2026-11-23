"""STL -> BEM elements (centroid / outward normal / area).

binary or ascii STL -> unit scaling -> vertex welding -> outward orientation -> quality report.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Mesh:
    centroids: np.ndarray   # (N, 3) element centroids [m]
    normals: np.ndarray     # (N, 3) outward unit normals
    areas: np.ndarray       # (N,)   element areas [m^2]
    verts: np.ndarray       # (V, 3) vertices [m]
    tris: np.ndarray        # (N, 3) vertex indices
    info: dict = field(default_factory=dict)


# Files

def _parse_binary(raw: bytes, n_tri: int):
    dt = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec = np.frombuffer(raw, dtype=dt, count=n_tri, offset=84)
    return rec["v"].astype(np.float64), rec["n"].astype(np.float64)


def _parse_ascii(raw: bytes):
    txt = raw.decode("utf-8", errors="ignore")
    num = r"([-+0-9.eEdD]+)"
    v = np.array(re.findall(rf"vertex\s+{num}\s+{num}\s+{num}", txt), dtype=np.float64)
    fn = np.array(re.findall(rf"facet\s+normal\s+{num}\s+{num}\s+{num}", txt), dtype=np.float64)
    if len(v) == 0 or len(v) % 3 != 0:
        raise ValueError("could not parse the ASCII STL (vertex count is not a multiple of 3)")
    v = v.reshape(-1, 3, 3)
    if len(fn) != len(v):
        fn = np.zeros((len(v), 3))
    return v, fn


def read_stl(path: str | Path):
    """Returns (triangle vertices (T, 3, 3), stored facet normals (T, 3)) in the file's units."""
    raw = Path(path).read_bytes()
    if len(raw) >= 84:
        n = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
        if n > 0 and len(raw) == 84 + 50 * n:      # exact binary size
            return _parse_binary(raw, n)
    return _parse_ascii(raw)


def write_stl(path: str | Path, tri_verts: np.ndarray, header: str = "stl2bem"):
    """Binary STL from triangle vertices (T, 3, 3); facet normals from the vertex order (right-hand rule)."""
    tri = np.asarray(tri_verts, dtype=np.float64)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    rec = np.zeros(len(tri), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as f:
        f.write(header.encode("ascii", "replace")[:80].ljust(80, b" "))
        f.write(struct.pack("<I", len(tri)))
        f.write(rec.tobytes())
    return Path(path)


# Units

def resolve_scale(tri_verts: np.ndarray, units: str = "auto") -> tuple[float, str]:
    """Factor to meters. 'auto' guesses from the largest bounding-box side (>= 100 -> mm, >= 5 -> cm, else m)."""
    table = {"m": 1.0, "cm": 0.01, "mm": 0.001}
    if units != "auto":
        return table[units], units
    L = float((tri_verts.reshape(-1, 3).max(0) - tri_verts.reshape(-1, 3).min(0)).max())
    u = "m" if L < 5.0 else ("cm" if L < 100.0 else "mm")
    return table[u], u + "(auto)"


# Welding, orientation

def _weld(tri_verts: np.ndarray, tol: float):
    flat = tri_verts.reshape(-1, 3)
    key = np.round(flat / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return flat[first], inv.reshape(-1, 3).astype(np.int64)


def _geom(verts, tris):
    v = verts[tris]
    cen = v.mean(axis=1)
    cr = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    nrm = np.linalg.norm(cr, axis=1)
    return cen, cr, nrm


def _edge_report(tris):
    """(number of edges not shared by exactly two triangles, consistent winding)."""
    de = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    _, cnt = np.unique(np.sort(de, axis=1), axis=0, return_counts=True)
    _, dcnt = np.unique(de, axis=0, return_counts=True)
    return int((cnt != 2).sum()), bool((dcnt == 1).all())


def mesh_from_arrays(verts, tris, file_normals=None, recenter: str = "none", source: str = "arrays") -> Mesh:
    """Mesh from vertices (V, 3) in meters and triangles (N, 3): welding is not repeated, orientation is checked."""
    verts = np.asarray(verts, np.float64)
    tris = np.asarray(tris, np.int64).copy()
    diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    _, _, nrm = _geom(verts, tris)
    keep = nrm > 1e-14 * diag ** 2
    n_degen = int((~keep).sum())
    tris = tris[keep]
    fn = None if file_normals is None else np.asarray(file_normals)[keep]
    open_edges, consistent = _edge_report(tris)

    cen, cr, nrm = _geom(verts, tris)
    if not consistent and fn is not None and np.linalg.norm(fn, axis=1).max() > 0:
        flip = np.einsum("ij,ij->i", cr, fn) < 0                     # per-facet fix from the stored normals
        tris[flip] = tris[flip][:, ::-1]
        cen, cr, nrm = _geom(verts, tris)
    v = verts[tris]
    vol = float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)
    if vol < 0:                                                        # whole surface inside out
        tris = tris[:, ::-1]
        cen, cr, nrm = _geom(verts, tris)
        vol = -vol
        v = verts[tris]

    normals = cr / nrm[:, None]
    areas = 0.5 * nrm
    edge = np.linalg.norm(v[:, [1, 2, 0]] - v[:, [0, 1, 2]], axis=2)
    size = edge.max(axis=1)                                            # element size: its longest edge
    info = {
        "stl_path": source, "units": "m", "scale_to_m": 1.0,
        "n_elements": len(tris), "n_verts": len(verts),
        "open_edges": open_edges, "winding_consistent": consistent,
        "n_degenerate_removed": n_degen,
        "volume_m3": vol, "area_m2": float(areas.sum()),
        "edge_mean_m": float(edge.mean()), "edge_max_m": float(edge.max()),
        "edge_p95_m": float(np.percentile(size, 95)),                 # 95 % of the elements have no longer edge
        "bbox_min": verts.min(0), "bbox_max": verts.max(0),
        "origin_shift": np.zeros(3),
    }
    if recenter != "none":
        shift = (0.5 * (verts.min(0) + verts.max(0)) if recenter == "bbox"
                 else (cen * areas[:, None]).sum(0) / areas.sum())
        verts, cen = verts - shift, cen - shift
        info["origin_shift"] = shift
        info["bbox_min"], info["bbox_max"] = verts.min(0), verts.max(0)
    return Mesh(cen, normals, areas, verts, tris, info)


def build_mesh(path: str | Path, units: str, scale: float | None = None,
               recenter: str = "bbox", c: float = 343.0, f_max: float = 8000.0,
               verbose: bool = True, allow_open: bool = False) -> Mesh:
    """Mesh (BEM input) from an STL file: meters, welded vertices, outward normals, quality report.

    units: "m", "cm" or "mm" (STL files do not store units); "auto" guesses from the size and warns.
    A surface with open or non-manifold edges raises ValueError unless allow_open=True: the boundary integral
    equations, the inside test and the normal orientation all assume a closed surface."""
    tv, fn = read_stl(path)
    s, uname = resolve_scale(tv, units)
    if units == "auto" and scale is None:
        print(f"  [WARN] units guessed from the size: {uname}. Pass units= explicitly; a small part in mm can be "
              f"taken for cm.")
    if scale is not None:
        s, uname = scale, f"custom({scale})"
    tv = tv * s
    diag = float(np.linalg.norm(tv.reshape(-1, 3).max(0) - tv.reshape(-1, 3).min(0)))
    verts, tris = _weld(tv, tol=max(1e-9, 1e-7 * diag))
    m = mesh_from_arrays(verts, tris, file_normals=fn, recenter=recenter, source=str(path))
    m.info["units"], m.info["scale_to_m"] = uname, s
    if verbose:
        print_report(m, c=c, f_max=f_max)
    if m.info["open_edges"] and not allow_open:
        raise ValueError(f"{path}: {m.info['open_edges']} open or non-manifold edges, the surface is not closed. "
                         f"Repair it (e.g. stl2bem-prep) or allow it explicitly (--allow_open_mesh).")
    return m


# Report

def element_sizes(mesh: Mesh) -> np.ndarray:
    """Longest edge of every element (m)."""
    v = mesh.verts[mesh.tris]
    return np.linalg.norm(v[:, [1, 2, 0]] - v[:, [0, 1, 2]], axis=2).max(axis=1)


def valid_up_to_hz(mesh: Mesh, c: float = 343.0, per_wavelength: float = 6.0, stat: str = "max") -> float:
    """Highest frequency at which the elements are no larger than wavelength / per_wavelength.

    stat: "max" (every element, by its longest edge; the default since 0.1.1), "p95" (95 % of the elements) or
    "mean" (the mean edge length, the check of 0.1.0, which a few large elements do not move)."""
    key = {"max": "edge_max_m", "p95": "edge_p95_m", "mean": "edge_mean_m"}[stat]
    return c / (per_wavelength * mesh.info[key])


def print_report(m: Mesh, c: float = 343.0, f_max: float = 8000.0):
    i = m.info
    print(f"  STL      : {i['stl_path']}")
    print(f"  units    : {i['units']} (x{i['scale_to_m']:g} -> m)")
    degen = f"  ({i['n_degenerate_removed']} degenerate removed)" if i["n_degenerate_removed"] else ""
    print(f"  elements : {i['n_elements']} triangles / {i['n_verts']} vertices{degen}")
    print(f"  bbox     : {np.round(i['bbox_min'], 4)} ~ {np.round(i['bbox_max'], 4)} m")
    if np.any(i["origin_shift"]):
        print(f"  recenter : shifted by {np.round(i['origin_shift'], 4)} m (phase reference = new origin)")
    print(f"  volume   : {i['volume_m3'] * 1e3:.3f} L, surface {i['area_m2'] * 1e4:.1f} cm^2")
    print(f"  edge     : mean {i['edge_mean_m'] * 1000:.2f} mm, max {i['edge_max_m'] * 1000:.2f} mm; "
          f"95 % of the elements <= {i['edge_p95_m'] * 1000:.2f} mm (longest edge)")
    lam = c / f_max
    n_over = int((element_sizes(m) > lam / 6).sum())
    if n_over == 0:
        print(f"  [OK] lambda/6 at {f_max / 1000:.1f} kHz = {lam / 6 * 1000:.2f} mm "
              f"-> mesh valid up to ~{valid_up_to_hz(m, c) / 1000:.1f} kHz (every element)")
    else:
        print(f"  [WARN] lambda/6 at {f_max / 1000:.1f} kHz = {lam / 6 * 1000:.2f} mm: {n_over} of "
              f"{i['n_elements']} elements ({100 * n_over / i['n_elements']:.1f} %) are larger")
        print(f"         -> mesh valid up to ~{valid_up_to_hz(m, c) / 1000:.1f} kHz for every element, "
              f"~{valid_up_to_hz(m, c, stat='p95') / 1000:.1f} kHz for 95 % of them; remesh with a smaller "
              f"maximum edge length or lower the band")
    if i["open_edges"]:
        print(f"  [WARN] {i['open_edges']} open or non-manifold edges: the surface is not closed "
              f"(the BEM and the CHIEF points assume a closed surface; see stl2bem-prep)")
    if not i["winding_consistent"]:
        print("  [WARN] inconsistent winding: corrected from the stored facet normals")
