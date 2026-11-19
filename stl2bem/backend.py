"""Array backend: CuPy on an NVIDIA GPU when available, NumPy/SciPy on the CPU otherwise.

The solver is written once against ``Backend.xp`` (the array module) and ``Backend`` linear-algebra wrappers,
so small meshes run anywhere and large ones use the GPU without code changes.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as _sp_la

import functools

try:  # optional GPU backend
    import cupy as _cp
    import cupyx.scipy.linalg as _cpx_la
    HAS_CUPY = True          # CuPy can be imported; whether a GPU can be used is checked by gpu_available()
except Exception:  # ImportError, or a CUDA runtime that cannot be loaded
    _cp = _cpx_la = None
    HAS_CUPY = False


@functools.lru_cache(maxsize=1)
def gpu_available() -> tuple[bool, str]:
    """(True, "") if CuPy can run on a GPU here, else (False, reason). Checked once, on first use: importing CuPy
    succeeds on machines without a GPU or with a mismatched driver, and fails only at the first kernel."""
    if not HAS_CUPY:
        return False, "CuPy is not installed"
    try:
        if _cp.cuda.runtime.getDeviceCount() < 1:
            return False, "no CUDA device found"
        a = _cp.arange(16, dtype=_cp.complex64).reshape(4, 4) + 4 * _cp.eye(4, dtype=_cp.complex64)
        float(_cp.abs(_cp.linalg.qr(a)[1]).sum())         # a kernel and cuSOLVER, as the solver uses them
        return True, ""
    except Exception as e:  # driver/runtime mismatch, missing libraries, ...
        return False, f"CuPy cannot use the GPU ({type(e).__name__}: {str(e).splitlines()[0][:120]})"


class Backend:
    """Thin wrapper over NumPy/SciPy or CuPy with the few operations the solver needs.

    name: "auto" (GPU if one can be used, otherwise CPU), "cupy" (GPU, error if not possible) or "numpy" (CPU)."""

    def __init__(self, name: str = "auto"):
        self.fallback_reason = ""
        if name == "auto":
            ok, why = gpu_available()
            name = "cupy" if ok else "numpy"
            self.fallback_reason = why
        elif name == "cupy":
            ok, why = gpu_available()
            if not ok:
                raise RuntimeError(f"backend 'cupy' requested but {why} "
                                   "(pip install cupy-cuda12x matching your CUDA version, or use --backend numpy)")
        if name not in ("cupy", "numpy"):
            raise ValueError(f"unknown backend {name!r} (use auto, cupy or numpy)")
        self.name = name
        self.gpu = name == "cupy"
        self.xp = _cp if self.gpu else np
        self._la = _cpx_la if self.gpu else _sp_la

    # Transfers
    def asarray(self, a, dtype=None):
        return self.xp.asarray(a, dtype=dtype)

    def asnumpy(self, a) -> np.ndarray:
        return _cp.asnumpy(a) if self.gpu else np.asarray(a)

    def free(self):
        """Return cached GPU blocks to the driver (no-op on the CPU)."""
        if self.gpu:
            _cp.get_default_memory_pool().free_all_blocks()

    def free_memory_gb(self) -> float | None:
        if not self.gpu:
            return None
        return _cp.cuda.Device().mem_info[0] / 1024 ** 3

    # Dense linear algebra
    def qr(self, a):
        return self.xp.linalg.qr(a, mode="reduced")

    def solve_triangular(self, r, b):
        return self._la.solve_triangular(r, b, lower=False)

    def lstsq_factor(self, a):
        """Factor a tall matrix (m >= n, the CHIEF-augmented system) for repeated least-squares solves.

        GPU: reduced QR (cuSOLVER). CPU: normal equations A^H A x = A^H b in double precision with a Cholesky
        factor. Dense QR is several times slower than Cholesky in common CPU LAPACK builds; forming the Gram matrix in
        complex128 keeps the rounding error (cond(A)^2 * 1e-16) far below the error already present in A
        (single-precision assembly), for the moderate condition numbers of CHIEF-augmented systems."""
        if self.gpu:
            q, r = self.xp.linalg.qr(a, mode="reduced")
            return ("qr", q, r)
        a64 = np.asarray(a, dtype=np.complex128 if np.iscomplexobj(a) else np.float64)
        gram = a64.conj().T @ a64
        c = _sp_la.cho_factor(gram, lower=False, overwrite_a=True, check_finite=False)
        return ("chol", a64, c)

    def lstsq_solve(self, fac, b):
        """Least-squares solution of the factored system for right-hand sides b (m, k)."""
        if fac[0] == "qr":
            _, q, r = fac
            return self._la.solve_triangular(r, q.conj().T @ b, lower=False)
        _, a64, c = fac
        rhs = a64.conj().T @ np.asarray(b, dtype=a64.dtype)
        return _sp_la.cho_solve(c, rhs, overwrite_b=True, check_finite=False).astype(b.dtype, copy=False)

    def lu_factor(self, a):
        return self._la.lu_factor(a, overwrite_a=True)

    def lu_solve(self, lu_piv, b):
        return self._la.lu_solve(lu_piv, b, overwrite_b=True)

    def __repr__(self):
        return f"Backend({self.name})"
