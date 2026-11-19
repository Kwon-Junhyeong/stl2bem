# Changelog

## 0.1.0 (first public release)

stl2bem grew from an unreleased GPU solver ("original code" below): STL -> HDF5 tables of the total pressure at
microphones, CBIE + CHIEF on the GPU (CuPy), one-point kernel integration, microphones snapped to the nearest
element centroid. That code computed the microphone responses of a robot body in two earlier projects (see the
README).

### Solver
- **Exact integration of the singular part.** The static part 1/(4 pi r) of the double-layer kernel is integrated
  exactly over every triangle (signed solid angle, Van Oosterom & Strackee 1983); the smooth remainder G - G0 uses
  a 1-, 3- or 7-point rule. The original code took the whole kernel at the element centroid, which breaks down when
  elements are close (neighbors at sharp edges, the two faces of thin walls). `--integration centroid` keeps the
  original rule.
- **Receivers at their exact positions, second-order accurate.** Microphones are projected onto the surface and
  evaluated with the corrected representation 2 (p_inc - M p)(x) - p_j, instead of taking the value of the nearest
  element centroid (2-4 mm away on typical meshes, first-order accurate). On a rigid sphere the error is 6-50 times
  smaller on the same mesh, and 320 elements are more accurate than the original code with 5,120
  (examples/validation_sphere).
  `--receivers snap` keeps the original behavior; `--receivers field` evaluates exterior points.
- **Microphones on edges and vertices.** The surface correction uses the solid-angle-weighted mean of all
  elements meeting at the point (a microphone placed on a CAD vertex otherwise kept a first-order error).
- **Minimum distance of the CHIEF points** from the element centroids (0.75 mean edges by default). Points close to the surface gave
  inaccurate CHIEF rows that distorted resonances of thin-walled bodies; such bodies are solved with the CBIE alone
  (their irregular frequencies lie far above the resolved band).
- **Point sources (near field).** `solve_table(..., sources=)`, `--source_distance`; tables relative to the free
  field at the origin; they converge to the plane-wave tables as the source distance grows. The dry run prints the Fraunhofer distance
  2D²/λ of the body.
- **Burton-Miller (experimental option).** `--formulation burton-miller`: hypersingular operator with the static
  part exact (gradient of the solid angle) and the 1/r part of the self term analytic. Compared with the default
  on the gallery against bempp-cl (examples/crosscheck_bempp/formulations.py); the default stays CBIE + CHIEF.
- **CHIEF points are chosen automatically** by `solve_table` when none are given, as on the command line.
- **CPU backend.** NumPy/SciPy when no GPU can be used; the matrix is assembled in parallel tiles and the CHIEF
  least-squares system is solved through the normal equations in double precision (Cholesky). `backend="auto"`
  checks that the GPU can run and otherwise falls back to the CPU and reports why.

### Command line
- `--units` is required (STL files do not store units); open or non-manifold surfaces stop the run unless
  `--allow_open_mesh` is given; `--resume` refuses a table computed with other settings; `--n_receivers`
  replaces `--n_panels` (kept as an alias); the dry run estimates the run time; `--sofa` checks for sofar
  before solving.

### Tools
- SOFA export (AES69, GeneralTF 2.0): `--sofa`, `python -m stl2bem.sofa`, `stl2bem.sofa.write_sofa`.
- `stl2bem.prep`: several overlapping CAD parts -> one closed surface (voxel union + marching cubes).
- `stl2bem.analysis`: table loading, free-field tables, localization error of SRP-PHAT / MUSIC caused by the body.
- `stl2bem.analytic`: rigid-sphere series solutions for plane waves and point sources, irregular frequencies of a
  sphere.
- `stl2bem.shapes`: closed test shapes (sphere, box, cylinder, plate).

### Validation, examples, documentation
- Examples: a gallery of six CAD objects (field, surface, cavity resonances), the rigid-sphere validation, a
  cross-check against bempp-cl on the gallery meshes, and a benchmark of time and memory. Figures drawn with
  pgfplots (docs/figures).
- Tests (CPU) and continuous integration; `CITATION.cff`.
- Phase convention clarified: with the package's e^{+jwt} convention, positive-frequency STFT bins (e^{-jwt}
  kernel) of a delayed signal have the same phase as the table; no conjugation is needed (the original README said
  conj() was required).

### Reproducing the original code
`--integration centroid --receivers snap` reproduces the original tables.
