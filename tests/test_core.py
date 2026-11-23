"""Fast CPU tests (NumPy backend). Run: python -m pytest -q"""
import numpy as np
import pytest

from stl2bem import mesh_from_arrays, chief_points, solve_table, rigid_sphere, project_to_surface
from stl2bem.analysis import load_table
from stl2bem.backend import Backend
from stl2bem.kernels import Elements, double_layer
from stl2bem.panels import closest_points_on_triangles
from stl2bem.shapes import box, cylinder, icosphere

BK = Backend("numpy")


def _elements(mesh, dyn_quad=1):
    return Elements(BK, mesh.verts, mesh.tris, mesh.centroids, mesh.normals, mesh.areas,
                    dyn_quad=dyn_quad, real_dtype=np.float64)


@pytest.mark.parametrize("shape", ["sphere", "box", "cylinder"])
def test_closed_shapes(shape):
    v, t = {"sphere": lambda: icosphere(0.1, 2), "box": lambda: box(0.2, 0.1, 0.05, 6),
            "cylinder": lambda: cylinder(0.05, 0.1, 0.02)}[shape]()
    m = mesh_from_arrays(v, t)
    assert m.info["open_edges"] == 0
    assert m.info["winding_consistent"]
    assert m.info["volume_m3"] > 0


@pytest.mark.parametrize("shape", ["sphere", "box"])
def test_static_identities(shape):
    """k -> 0: (I/2 + M) rows sum to 1 on the surface; M rows sum to 1 inside and 0 outside (Gauss)."""
    v, t = icosphere(0.1, 2) if shape == "sphere" else box(0.2, 0.1, 0.06, 5)
    m = mesh_from_arrays(v, t)
    el = _elements(m)
    k = 1e-7
    A = double_layer(k, m.centroids, el, np.complex128, "analytic", exclude_self=True)
    assert np.allclose(A.sum(1).real + 0.5, 1.0, atol=1e-9)
    inside = np.array([[0.0, 0.0, 0.0], [0.02, -0.01, 0.01]])
    outside = np.array([[0.5, 0.0, 0.0], [0.0, 0.0, m.verts[:, 2].max() + 1e-6]])
    assert np.allclose(double_layer(k, inside, el, np.complex128, "analytic").sum(1).real, 1.0, atol=1e-9)
    assert np.allclose(double_layer(k, outside, el, np.complex128, "analytic").sum(1).real, 0.0, atol=1e-9)


def test_centroid_rule_matches_original_formula():
    """integration='centroid' is the original one-point kernel G (-jk - 1/r) (R.n) / r A with a zero diagonal."""
    v, t = icosphere(0.1, 1)
    m = mesh_from_arrays(v, t)
    k = 2 * np.pi * 1000 / 343
    M = double_layer(k, m.centroids, _elements(m), np.complex128, "centroid", exclude_self=True)
    R = m.centroids[:, None] - m.centroids[None]
    r = np.linalg.norm(R, axis=2)
    np.fill_diagonal(r, 1.0)
    G = np.exp(-1j * k * r) / (4 * np.pi * r)
    ref = G * (-1j * k - 1 / r) * np.einsum("ijk,jk->ij", R, m.normals) / r * m.areas[None]
    np.fill_diagonal(ref, 0)
    assert np.allclose(M, ref, rtol=1e-10, atol=1e-14)


def test_closest_point_on_triangle():
    a, b, c = np.array([[0.0, 0, 0]]), np.array([[1.0, 0, 0]]), np.array([[0.0, 1, 0]])
    for p, q in [((0.2, 0.2, 1.0), (0.2, 0.2, 0)), ((-1.0, -1.0, 0.0), (0, 0, 0)), ((2.0, 0.0, 0.5), (1, 0, 0)),
                 ((0.5, -1.0, 0.0), (0.5, 0, 0)), ((1.0, 1.0, 0.0), (0.5, 0.5, 0))]:
        assert np.allclose(closest_points_on_triangles(np.array(p), a, b, c)[0], q)


@pytest.mark.parametrize("mode,tol", [("surface", 0.008), ("representation", 0.03)])
def test_rigid_sphere_receivers(tmp_path, mode, tol):
    """Random points of the exact sphere on a 1280-element mesh against the series solution. The corrected surface
    evaluation is second-order accurate (< 0.8 %); the plain representation keeps a first-order term (< 3 %)."""
    a = 0.1
    v, t = icosphere(a, 3)
    m = mesh_from_arrays(v, t)
    rng = np.random.default_rng(0)
    mics = rng.normal(size=(6, 3))
    mics = a * mics / np.linalg.norm(mics, axis=1, keepdims=True)
    doa = np.array([[1.0, 0, 0], [0, 0, 1.0], [0.6, 0.8, 0]])
    freqs = np.array([500.0, 1500.0])
    rec, elem, _ = project_to_surface(mics, m, verbose=False)
    pts = chief_points(m, n=12, seed=0, use_gpu=False, verbose=False)
    fn = tmp_path / "t.h5"
    solve_table(m.verts, m.tris, m.centroids, m.normals, m.areas, freqs, doa, str(fn), receivers=rec,
                receiver_elements=elem if mode == "surface" else None,
                chief_points=pts, backend="numpy", precision="double", verbose=False)
    tab = load_table(fn)
    for fi, f in enumerate(freqs):
        ref = rigid_sphere(2 * np.pi * f / 343, a, mics, doa)
        err = np.abs(tab.p[:, :, fi] - ref).mean() / np.abs(ref).mean()
        assert err < tol, (f, err)


def test_receivers_on_vertices_and_edges(tmp_path):
    """Microphones exactly on mesh vertices and edge midpoints: the shared-jump correction keeps them as accurate
    as points inside elements."""
    from stl2bem import surface_receivers
    a = 0.1
    v, t = icosphere(a, 3)
    m = mesh_from_arrays(v, t)
    on_vertex = m.verts[[0, 5, 17, 40]]
    e = m.verts[m.tris[[3, 50, 200]]]
    on_edge = 0.5 * (e[:, 0] + e[:, 1])
    mics = np.concatenate([on_vertex, on_edge])
    rec, elem, _ = surface_receivers(mics, m, verbose=False)
    assert elem[0].shape[1] >= 5                                    # vertices touch 5-6 elements
    exact_pts = a * mics / np.linalg.norm(mics, axis=1, keepdims=True)
    doa = np.array([[1.0, 0, 0], [0, 0, 1.0], [0.6, 0.8, 0]])
    pts = chief_points(m, n=12, seed=0, use_gpu=False, verbose=False)
    fn = tmp_path / "v.h5"
    solve_table(m.verts, m.tris, m.centroids, m.normals, m.areas, [1500.0], doa, str(fn), receivers=rec,
                receiver_elements=elem, chief_points=pts, backend="numpy", precision="double", verbose=False)
    ref = rigid_sphere(2 * np.pi * 1500 / 343, a, exact_pts, doa)
    err = np.abs(load_table(fn).p[:, :, 0] - ref).mean() / np.abs(ref).mean()
    assert err < 0.008, err


def test_point_sources_rigid_sphere(tmp_path):
    """Near-field (point) sources at 2 and 3 radii from a rigid sphere against the exact series solution, surface
    receivers; the table keeps the source positions and its free field matches the solver's normalization."""
    from stl2bem import rigid_sphere_point, surface_receivers
    from stl2bem.analysis import free_field
    a = 0.1
    m = mesh_from_arrays(*icosphere(a, 3))
    rng = np.random.default_rng(0)
    mics = rng.normal(size=(6, 3))
    mics = a * mics / np.linalg.norm(mics, axis=1, keepdims=True)
    dirs = np.array([[1.0, 0, 0], [0, 0, 1.0], [0.6, 0.8, 0]])
    sources = np.concatenate([2 * a * dirs, 3 * a * dirs])
    freqs = np.array([200.0, 1500.0])
    rec, elem, _ = surface_receivers(mics, m, verbose=False)
    fn = tmp_path / "ps.h5"
    solve_table(m.verts, m.tris, m.centroids, m.normals, m.areas, freqs, None, str(fn), sources=sources,
                receivers=rec, receiver_elements=elem, chief_points=chief_points(m, n=12, seed=0, use_gpu=False,
                                                                                  verbose=False),
                backend="numpy", precision="double", verbose=False)
    tab = load_table(fn)
    assert tab.sources is not None and np.allclose(tab.sources, sources)
    for fi, f in enumerate(freqs):
        ref = rigid_sphere_point(2 * np.pi * f / 343, a, mics, sources)
        err = np.abs(tab.p[:, :, fi] - ref).mean() / np.abs(ref).mean()
        assert err < 0.01, (f, err)
    ff = free_field(tab)
    r = np.linalg.norm(tab.positions[:, None] - sources[None], axis=2)
    r0 = np.linalg.norm(sources, axis=1)[None]
    k = 2 * np.pi * freqs[1] / 343
    assert np.allclose(ff.p[:, :, 1], r0 / r * np.exp(-1j * k * (r - r0)), atol=1e-6)


def test_far_point_source_is_a_plane_wave(tmp_path):
    """A point source far away gives the plane-wave table (difference falls like size / distance)."""
    m = mesh_from_arrays(*box(0.2, 0.1, 0.06, 5))
    rec = np.array([[0.15, 0.0, 0.0], [0.0, 0.12, 0.05]])
    d = np.array([[0.6, 0.0, 0.8]])
    common = dict(receivers=rec, chief_points=chief_points(m, n=8, seed=0, use_gpu=False, verbose=False),
                  backend="numpy", precision="double", verbose=False)
    solve_table(m.verts, m.tris, m.centroids, m.normals, m.areas, [800.0], d, str(tmp_path / "pw.h5"), **common)
    solve_table(m.verts, m.tris, m.centroids, m.normals, m.areas, [800.0], None, str(tmp_path / "far.h5"),
                sources=200.0 * d, **common)
    pw, far = load_table(tmp_path / "pw.h5").p, load_table(tmp_path / "far.h5").p
    assert np.abs(far - pw).max() < 2e-3


def test_point_source_inside_the_body_is_rejected(tmp_path):
    m = mesh_from_arrays(*box(0.2, 0.1, 0.06, 4))
    with pytest.raises(ValueError, match="inside the body"):
        solve_table(m.verts, m.tris, m.centroids, m.normals, m.areas, [500.0], None, str(tmp_path / "x.h5"),
                    sources=[[0.01, 0.0, 0.0]], receivers=[[0.3, 0, 0]], use_chief=False, backend="numpy",
                    verbose=False)


def test_irregular_frequency_chief_and_burton_miller(tmp_path):
    """At the first irregular frequency of a rigid sphere (ka = pi) the CBIE alone fails; CBIE + CHIEF and
    Burton-Miller both stay close to the exact solution, also for point sources."""
    from stl2bem import rigid_sphere_point, surface_receivers
    a = 0.1
    m = mesh_from_arrays(*icosphere(a, 3))
    rng = np.random.default_rng(0)
    mics = rng.normal(size=(12, 3))
    mics = a * mics / np.linalg.norm(mics, axis=1, keepdims=True)
    rec, elem, _ = surface_receivers(mics, m, verbose=False)
    f = 343 / (2 * a)                                                  # ka = pi
    k = 2 * np.pi * f / 343
    doa = np.array([[1.0, 0, 0], [0, 0.6, 0.8]])
    ref = rigid_sphere(k, a, mics, doa)
    err = {}
    for name, kw in (("cbie", dict(use_chief=False)), ("chief", {}), ("bm", dict(formulation="burton-miller"))):
        fn = tmp_path / f"{name}.h5"
        solve_table(m.verts, m.tris, m.centroids, m.normals, m.areas, [f], doa, str(fn), receivers=rec,
                    receiver_elements=elem, backend="numpy", precision="double", verbose=False, **kw)
        err[name] = np.abs(load_table(fn).p[:, :, 0] - ref).mean() / np.abs(ref).mean()
    assert err["cbie"] > 0.03, err
    assert err["chief"] < 0.01 and err["bm"] < 0.01, err
    src = 0.25 * doa
    fn = tmp_path / "bm_point.h5"
    solve_table(m.verts, m.tris, m.centroids, m.normals, m.areas, [f], None, str(fn), sources=src, receivers=rec,
                receiver_elements=elem, backend="numpy", precision="double", verbose=False,
                formulation="burton-miller")
    ref = rigid_sphere_point(k, a, mics, src)
    assert np.abs(load_table(fn).p[:, :, 0] - ref).mean() / np.abs(ref).mean() < 0.01


def test_hypersingular_with_sliver_triangles_single_precision():
    """CAD exports contain slivers; the Burton-Miller operator must stay finite in single precision. One edge of a
    sphere mesh is split 1e-4 of its length from a vertex, which leaves two slivers."""
    from stl2bem.kernels import hypersingular
    v, t = icosphere(0.1, 3)
    v, t = np.asarray(v, float), np.asarray(t)
    a, b = t[0, 0], t[0, 1]
    other = [i for i in range(1, len(t)) if a in t[i] and b in t[i]][0]
    p = len(v)
    v = np.vstack([v, v[a] + 1e-4 * (v[b] - v[a])])
    tri = [list(x) for x in t]
    c = [x for x in t[0] if x not in (a, b)][0]
    d = [x for x in t[other] if x not in (a, b)][0]
    tri[0] = [a, p, c]
    tri.append([p, b, c])
    ia, ib = list(t[other]).index(a), list(t[other]).index(b)
    o = list(t[other])
    o[ib] = p                                     # (.., b, a, ..) -> (.., p, a, ..) keeps the orientation
    tri[other] = o
    o2 = list(t[other])
    o2[ia] = p                                    # and (.., b, p, ..)
    tri.append(o2)
    m = mesh_from_arrays(v, np.array(tri))
    assert m.info["open_edges"] == 0 and m.info["winding_consistent"]
    el = Elements(BK, m.verts, m.tris, m.centroids, m.normals, m.areas, real_dtype=np.float32)
    H = hypersingular(2 * np.pi * 3000 / 343, el, np.complex64)
    assert np.isfinite(H).all()


def test_chief_points_are_chosen_automatically(tmp_path):
    """Without chief_points, solve_table picks the same interior points as chief_points(mesh) (thick body) and
    falls back to the CBIE alone for a thin wall, without errors."""
    from stl2bem.shapes import plate
    m = mesh_from_arrays(*icosphere(0.05, 2))
    rec, doa = np.array([[0.1, 0.0, 0.0]]), np.array([[1.0, 0, 0]])
    kw = dict(receivers=rec, backend="numpy", precision="double", verbose=False)
    solve_table(m.verts, m.tris, m.centroids, m.normals, m.areas, [1000.0], doa, str(tmp_path / "a.h5"), **kw)
    solve_table(m.verts, m.tris, m.centroids, m.normals, m.areas, [1000.0], doa, str(tmp_path / "b.h5"),
                chief_points=chief_points(m, n=20, seed=0, use_gpu=False, verbose=False), **kw)
    assert np.allclose(load_table(tmp_path / "a.h5").p, load_table(tmp_path / "b.h5").p)
    thin = mesh_from_arrays(*plate(0.1, 0.1, 0.004, 0.006))
    solve_table(thin.verts, thin.tris, thin.centroids, thin.normals, thin.areas, [1000.0], doa,
                str(tmp_path / "c.h5"), **kw)
    import h5py
    with h5py.File(tmp_path / "c.h5") as f:
        assert not f.attrs["use_chief"]


def test_chief_points_keep_clear_of_the_surface():
    """CHIEF points closer than 0.75 element edges to the surface are rejected; a thin wall gets none (CBIE only)."""
    from stl2bem.chief import _nearest_surface_dist
    from stl2bem.shapes import plate
    thick = mesh_from_arrays(*box(0.1, 0.1, 0.1, 8))
    pts = chief_points(thick, n=10, seed=0, use_gpu=False, verbose=False)
    assert len(pts) == 10
    assert _nearest_surface_dist(pts, thick.centroids).min() >= 0.75 * thick.info["edge_mean_m"]
    thin = mesh_from_arrays(*plate(0.1, 0.1, 0.004, 0.006))
    assert len(chief_points(thin, n=10, seed=0, use_gpu=False, verbose=False)) == 0


def test_stl_round_trip_in_mm(tmp_path):
    """A mesh written as a binary STL in mm reads back in meters with the same closed, outward surface."""
    from stl2bem import build_mesh, write_stl
    v, t = cylinder(0.04, 0.1, 0.01)
    ref = mesh_from_arrays(v, t)
    fn = tmp_path / "c.stl"
    write_stl(fn, 1000 * v[t])
    m = build_mesh(fn, units="mm", recenter="none", verbose=False)
    assert len(m.tris) == len(t)
    assert m.info["open_edges"] == 0 and m.info["winding_consistent"]
    assert np.isclose(m.info["volume_m3"], ref.info["volume_m3"], rtol=1e-6)
    assert np.allclose(np.sort(m.centroids, axis=0), np.sort(ref.centroids, axis=0), atol=1e-7)


def test_frequency_limit_uses_the_longest_edge(capsys):
    from stl2bem.stl_io import print_report, valid_up_to_hz
    m = mesh_from_arrays(*box(0.2, 0.2, 0.2, (20, 20, 2)))          # 1 cm squares, sides 1 cm x 10 cm
    assert np.isclose(m.info["edge_max_m"], np.hypot(0.01, 0.1))
    assert np.isclose(valid_up_to_hz(m), 343.0 / (6 * m.info["edge_max_m"]))
    assert valid_up_to_hz(m, stat="mean") > 4 * valid_up_to_hz(m)
    f_max = 0.5 * (valid_up_to_hz(m) + valid_up_to_hz(m, stat="mean"))  # the mean passes, the long sides do not
    print_report(m, f_max=f_max)
    out = capsys.readouterr().out
    assert "[WARN]" in out and f"320 of {len(m.tris)} elements" in out
    print_report(m, f_max=0.9 * valid_up_to_hz(m))
    assert "[OK]" in capsys.readouterr().out


def test_open_mesh_is_refused_unless_allowed(tmp_path):
    from stl2bem import build_mesh, write_stl
    v, t = icosphere(0.05, 2)
    fn = tmp_path / "open.stl"
    write_stl(fn, 1000 * v[t[1:]])                                    # one triangle missing: a hole
    with pytest.raises(ValueError, match="not closed"):
        build_mesh(fn, units="mm", verbose=False)
    assert build_mesh(fn, units="mm", verbose=False, allow_open=True).info["open_edges"] > 0


def test_resume_refuses_other_settings(tmp_path):
    m = mesh_from_arrays(*icosphere(0.05, 1))
    kw = dict(receivers=np.array([[0.1, 0.0, 0.0]]), use_chief=False, backend="numpy", verbose=False)
    doa = np.array([[1.0, 0, 0]])
    fn = str(tmp_path / "r.h5")
    solve_table(m.verts, m.tris, m.centroids, m.normals, m.areas, [500.0, 600.0], doa, fn, **kw)
    solve_table(m.verts, m.tris, m.centroids, m.normals, m.areas, [500.0, 600.0], doa, fn, resume=True, **kw)
    with pytest.raises(ValueError, match="freqs"):
        solve_table(m.verts, m.tris, m.centroids, m.normals, m.areas, [500.0, 700.0], doa, fn, resume=True, **kw)
    with pytest.raises(ValueError, match="formulation"):
        solve_table(m.verts, m.tris, m.centroids, m.normals, m.areas, [500.0, 600.0], doa, fn, resume=True,
                    formulation="burton-miller", **kw)


def test_command_line_needs_units(tmp_path):
    from stl2bem.run import main
    with pytest.raises(SystemExit):
        main(["--stl", str(tmp_path / "x.stl"), "--out", str(tmp_path)])


def test_command_line(tmp_path):
    """The stl2bem command: dry run, then a small solve that writes a readable table."""
    from stl2bem import write_stl
    from stl2bem.run import main
    v, t = icosphere(0.05, 2)
    stl, mics = tmp_path / "s.stl", tmp_path / "mics.csv"
    write_stl(stl, 1000 * v[t])
    np.savetxt(mics, [[50.0, 0, 0], [0, 50.0, 0], [0, 0, -50.0]], delimiter=",")
    common = ["--stl", str(stl), "--mics", str(mics), "--units", "mm", "--out", str(tmp_path / "out"),
              "--backend", "numpy", "--n_chief", "8", "--fs", "2000", "--nfft", "4", "--n_az", "8", "--n_el", "3"]
    main(common + ["--dry_run"])
    main(common + (["--free_field", "--sofa"] if _has("sofar") else []))
    tables = sorted((tmp_path / "out").glob("*.h5"))
    assert tables
    tab = load_table([t for t in tables if "free_field" not in t.name][0])
    assert tab.p.shape[0] == 3 and np.isfinite(tab.p).all()
    if _has("sofar"):
        assert len(list((tmp_path / "out").glob("*.sofa"))) == 2
    main(common + ["--source_distance", "0.3", "--free_field"])                    # near-field sources
    near = [t for t in (tmp_path / "out").glob("*_r0.3m_*.h5") if "free_field" not in t.name]
    assert len(near) == 1
    tn = load_table(near[0])
    assert tn.sources is not None and np.allclose(np.linalg.norm(tn.sources, axis=1), 0.3)
    with pytest.raises(SystemExit):
        main(common + ["--source_distance", "0.01", "--dry_run"])                   # inside the sphere


def _has(module):
    import importlib.util
    return importlib.util.find_spec(module) is not None


def test_sofa_export(tmp_path):
    """A table written as SOFA (GeneralTF) reads back with the same values, directions and receivers."""
    sf = pytest.importorskip("sofar")
    from stl2bem.analysis import Table
    from stl2bem.sofa import write_sofa
    rng = np.random.default_rng(1)
    doa = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-0.6, 0, 0.8]])
    pos = rng.normal(size=(3, 3)) * 0.05
    freqs = np.array([0.0, 500.0, 1000.0])
    p = rng.normal(size=(3, 4, 3)) + 1j * rng.normal(size=(3, 4, 3))
    fn = write_sofa(Table(p, doa, freqs, pos, {"body_id": "test"}), tmp_path / "t.sofa")
    s = sf.read_sofa(str(fn))
    assert s.GLOBAL_SOFAConventions == "GeneralTF"
    assert np.allclose(s.Data_Real + 1j * s.Data_Imag, p.transpose(1, 0, 2))
    assert np.allclose(s.N, freqs) and np.allclose(s.ReceiverPosition, pos)
    az, el = np.radians(s.SourcePosition[:, 0]), np.radians(s.SourcePosition[:, 1])
    assert np.allclose(np.column_stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]), doa)


def test_field_receivers_far_away(tmp_path):
    """Far from a small sphere the total field tends to the incident field."""
    v, t = icosphere(0.02, 2)
    m = mesh_from_arrays(v, t)
    pts = chief_points(m, n=8, seed=0, use_gpu=False, verbose=False)
    far = np.array([[3.0, 0.0, 0.0], [0.0, -3.0, 1.0]])
    doa = np.array([[0.0, 1.0, 0.0]])
    fn = tmp_path / "f.h5"
    solve_table(m.verts, m.tris, m.centroids, m.normals, m.areas, [300.0], doa, str(fn), receivers=far,
                chief_points=pts, backend="numpy", precision="double", verbose=False)
    tab = load_table(fn)
    inc = np.exp(1j * 2 * np.pi * 300 / 343 * (far @ doa.T))
    assert np.abs(tab.p[:, :, 0] - inc).max() < 1e-3
