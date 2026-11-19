"""Export of stl2bem tables to SOFA (AES69), convention GeneralTF 2.0.

    from stl2bem.sofa import write_sofa
    write_sofa("table.h5", "table.sofa")                    # or: python -m stl2bem.sofa table.h5 [table.sofa]

Layout: Data.Real / Data.Imag (M directions, R receivers, N frequencies); SourcePosition = (azimuth, elevation,
distance) of each source in degrees and meters, ReceiverPosition = receiver positions in the mesh
frame (meters), N = frequencies in Hz.

Meaning of the data: the complex pressure at each receiver for the source at SourcePosition, relative to the
free-field pressure that source produces at the coordinate origin. In the usual
FFT/STFT convention (kernel e^{-jwt}), the spectrum at receiver r of a source signal s(t) arriving from direction m
is S(w) * Data[m, r, w]; no conjugation is needed (see docs/method.md). For plane waves (far field) the distance
in SourcePosition is a nominal 1 m (GeneralTF requires one); for point sources (near field) it is the real distance.

Needs the optional package sofar (pip install sofar).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .analysis import Table, load_table

COMMENT = ("Transfer functions computed by the boundary element method (stl2bem). Data[m, r, n] is the complex "
           "pressure at receiver r for source m at frequency n, relative to the free-field pressure the source "
           "produces at the coordinate origin; spectra of received signals (FFT kernel exp(-jwt)) are S(w) * Data. ")
COMMENT_PLANE = "Sources are plane waves: the SourcePosition distance is nominal (1 m)."
COMMENT_POINT = "Sources are point sources (near field) at SourcePosition."


def to_sofa(table, title: str = "", author_contact: str = "", organization: str = "", license: str = ""):
    """sofar.Sofa object (GeneralTF 2.0) from an stl2bem table (path to the HDF5 file, or a Table)."""
    try:
        import sofar as sf
    except ImportError as e:  # optional dependency
        raise ImportError("SOFA export needs the package sofar: pip install sofar") from e
    from . import __version__

    t = table if isinstance(table, Table) else load_table(table)
    az, el = t.az_el
    s = sf.Sofa("GeneralTF", version="2.0")
    s.Data_Real = np.ascontiguousarray(t.p.real.transpose(1, 0, 2), dtype=np.float64)    # (M, R, N)
    s.Data_Imag = np.ascontiguousarray(t.p.imag.transpose(1, 0, 2), dtype=np.float64)
    s.N = np.asarray(t.freqs, np.float64)
    dist = np.ones_like(az) if t.sources is None else np.linalg.norm(t.sources, axis=1)   # plane waves: nominal 1 m
    s.SourcePosition = np.column_stack([az, el, dist])
    s.ReceiverPosition = np.asarray(t.positions, np.float64)
    s.GLOBAL_Title = title or str(t.attrs.get("body_id", "stl2bem table"))
    s.GLOBAL_Comment = COMMENT + (COMMENT_PLANE if t.sources is None else COMMENT_POINT)
    s.GLOBAL_Origin = f"stl2bem {__version__}"
    s.GLOBAL_History = "; ".join(f"{k}={t.attrs[k]}" for k in
                                 ("receiver_mode", "integration", "precision", "convention") if k in t.attrs)
    s.GLOBAL_AuthorContact = author_contact
    s.GLOBAL_Organization = organization
    if license:
        s.GLOBAL_License = license
    s.verify()
    return s


def write_sofa(table, path, compression: int = 4, **kwargs) -> Path:
    """Write an stl2bem table (HDF5 path or Table) as a SOFA file (GeneralTF 2.0). Returns the output path.
    Keyword arguments (title, author_contact, organization, license) go to the SOFA global attributes."""
    import sofar as sf
    path = Path(path)
    sf.write_sofa(str(path), to_sofa(table, **kwargs), compression=compression)
    return path


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python -m stl2bem.sofa table.h5 [out.sofa]")
        return 0
    src = Path(argv[0])
    dst = Path(argv[1]) if len(argv) > 1 else src.with_suffix(".sofa")
    try:
        print(f"wrote {write_sofa(src, dst)}")
    except ImportError as e:                # sofar missing: one line instead of a traceback
        raise SystemExit(str(e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
