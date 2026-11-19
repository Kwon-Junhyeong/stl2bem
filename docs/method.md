# Method

## Problem

A rigid (sound-hard) body S in free space is exposed to a plane wave from direction **d** (unit vector towards
the source) or to the spherical wave of a point source (see below). The total pressure p = p_inc + p_scat satisfies
the Helmholtz equation outside the body, ∂p/∂n = 0 on S, and the Sommerfeld radiation condition for p_scat.

Convention: time dependence e^{+jωt}, wavenumber k = ω/c,

    p_inc(x) = exp(+j k d·x),        G(x, y) = exp(-j k r) / (4π r),  r = |x - y|,

normals point out of the body. For a signal s(t) arriving from **d**, microphone m receives s(t - τ_m) with
τ_m = -(d·x_m)/c; its FFT/STFT bin (kernel e^{-jωt}) is S(ω) exp(+j k d·x_m), i.e. the table value times the source
spectrum. **No conjugation is needed** to combine tables with FFT-domain signals.

**Point sources (near field).** A source at a finite position **s** radiates a spherical wave. stl2bem uses

    p_inc(x) = (r0 / r) exp(-j k (r - r0)),        r = |x - s|,  r0 = |s|,

that is, the free field of a monopole normalized by its value at the origin, as for near-field HRTFs. When
r0 grows, r - r0 → -d·x with d = s/r0 and r0/r → 1, so the point source tends to the plane wave from d: the plane
wave is the far-field limit of the point source. The usual boundary between the two regimes is the Fraunhofer
distance 2D²/λ (D the size of the body or array): a spherical wavefront deviates from a plane one by less than π/8
in phase across D beyond it. Closer sources need the point-source incidence, which also brings the level
difference between receivers (1/r). The boundary integral equations below are unchanged; only p_inc differs. Unlike
a plane wave, a point source is scattered even at 0 Hz (its field is not uniform), so the DC bin is solved too.
Sources must lie outside the body; within about two element sizes of the surface the incident field varies across
elements and the mesh should be refined there.

## Boundary integral equations

With the double-layer operator

    (M p)(x) = - ∫_S p(y) ∂G(x, y)/∂n_y dS_y ,

the total pressure satisfies (rigid body, Kirchhoff-Helmholtz representation):

| where x is | equation |
|---|---|
| outside the body | p(x) = p_inc(x) - (M p)(x) |
| on the surface (smooth point) | ½ p(x) + (M p)(x) = p_inc(x)   (CBIE) |
| inside the body | 0 = p_inc(x) - (M p)(x) |

The CBIE alone is not uniquely solvable at the interior Dirichlet eigenfrequencies of the body (*irregular
frequencies*; for a sphere of radius a: zeros of j_n(ka), first ka = π). **CHIEF** (Schenck 1968) appends the
interior equation at points inside the body; the over-determined system is solved in the least-squares sense.
Candidate points are sampled uniformly in the bounding box, kept when a majority of three rays finds them inside,
filtered by their distance from the surface and spread by farthest-point sampling.

CHIEF points must lie about one element away from the surface (by default, at least 0.75 x the mean edge from the
nearest element centroid): a CHIEF row is the representation formula evaluated inside the material, and with
constant elements that formula is inaccurate close to the surface. Inaccurate CHIEF rows pull the least-squares solution away from the correct one. With a
thin-walled bottle (4 mm glass, 6 mm elements) they flattened the Helmholtz resonance and put a notch where the
peak should be, while the plain CBIE gave the expected single sharp peak. Thin-walled bodies (mugs, bells, pipes,
bottles) need no CHIEF at all in practice: the irregular frequencies belong to the material region, and for a wall
of thickness t they start near c / (2t) (about 40 kHz for 4 mm), far above the band such a mesh resolves. When no
interior point is deep enough, stl2bem solves the CBIE alone and reports it.

## Discretization

Flat triangles, constant pressure per element, collocation at the element centroids (N unknowns).

**Integration.** The kernel is split into a static part and a smooth remainder,

    ∂G/∂n_y = ∂G0/∂n_y + ∂(G - G0)/∂n_y ,     G0 = 1 / (4π r).

* Static part, exactly: ∫_T ∂G0/∂n_y dS = -Ω/(4π), where Ω is the signed solid angle of triangle T seen from x
  (Van Oosterom & Strackee 1983, tan(Ω/2) = a·(b×c) / (abc + (a·b)c + (a·c)b + (b·c)a) with a, b, c the vectors from
  x to the vertices). This is the singular and near-singular part: neighboring elements at an edge, the two faces of
  a thin plate and a receiver just above the surface are integrated exactly.
* Smooth remainder: d/dr[(e^{-jkr} - 1)/(4π r)] = ((z - 1)e^z + 1)/(4π r²), z = -jkr, which tends to -k²/(8π) as
  r → 0 (a series is used for kr < 0.1). It is integrated with a symmetric 1-, 3- or 7-point triangle rule.
* The self term (x at the centroid of its own flat element) is zero.

The original, unreleased code evaluated the whole kernel once at the element centroid (one-point rule). That rule
is exact in the far field but fails when the distance between elements is comparable to their size;
`--integration centroid` keeps it for comparison.

**Receivers.** After solving for the element pressures p_j:

* *Field points* (outside the body): p(x) = p_inc(x) - (M p)(x), with the same integration.
* *Surface points* (microphones on the body): the microphone is projected onto the closest surface point x, on
  element j. Just outside the surface, the element under the point subtends a solid angle of -2π, so the
  representation returns ½ p_j + (the rest), whereas the exact field would give ½ p(x) + (the rest). The plain
  representation therefore equals ½ (p(x) + p_j) up to second-order terms; since p(x) - p_j is first order in the
  element size (the field varies across the element), it is only first-order accurate on the surface, like taking
  the element value itself. The corrected value

      p(x) ≈ 2 [p_inc(x) - (M p)(x)] - p_j

  removes the first-order term and is second-order accurate at any point of the surface (rigid sphere: error ratio
  ≈ 4 when the element size is halved, examples/validation_sphere). It costs one extra row per microphone.
* *Original code*: the value of the element whose centroid is closest (`--receivers snap`), 2-4 mm away on typical
  meshes; first-order accurate.

**Linear algebra.** For every frequency: assemble ½I + M (N×N) and the CHIEF rows (N_c×N), then
* GPU: reduced QR (cuSOLVER) and back substitution;
* CPU: normal equations AᴴA x = Aᴴb in double precision, Cholesky. Rounding error cond(A)²·10⁻¹⁶ stays far below the
  error of the single-precision matrix for the moderate condition numbers of CHIEF-augmented systems;
* without CHIEF: LU.

All plane-wave directions are right-hand sides of the same factorization; only receiver rows are stored.

## Burton-Miller (experimental option)

`--formulation burton-miller` (`solve_table(..., formulation="burton-miller")`) replaces CHIEF by the
Burton-Miller combination of the CBIE with the normal-derivative equation N p = ∂p_inc/∂n,

    (½ I + M) p + α N p = p_inc + α ∂p_inc/∂n,        α = -j/k,        (N p)(x) = ∂/∂n_x (M p)(x),

which is uniquely solvable at every frequency without interior points (square system, LU). With constant elements
the hypersingular operator N is collocated at the centroids:

* static part, exactly: ∂/∂n_x [Ω_j(x)/4π] = n_x·∇Ω_j(x)/4π. The gradient of the solid angle of a triangle is a sum
  over its edges, (u₁×u₂)(|u₁|+|u₂|) / (|u₁||u₂|(|u₁||u₂| + u₁·u₂)) with u the vectors from x to the edge ends (the
  Biot-Savart field of the edge loop). It is continuous across the element, so the self term is the finite-part
  value directly. Near an edge the denominator is evaluated as |u₁×u₂|² / (|u₁||u₂| - u₁·u₂) to stay accurate in
  single precision on sliver triangles;
* dynamic part: ∫ [B (n_x·n_y) - A (R·n_x)(R·n_y)/r²] dS with B = f/(4πr³), A = (3f - z²e^z)/(4πr³),
  f = (z-1)e^z + 1, z = -jkr, 7-point rule. Both A and B behave like -k²/(8πr); on the element itself the 1/r part
  is integrated analytically (∫ dS/r from the centroid, a sum over the edges) and the bounded rest numerically.

Comparison with the default on the gallery, both against bempp-cl
(examples/crosscheck_bempp/formulations.py, results_formulations.json):

| | CBIE + CHIEF (default) | Burton-Miller |
|---|---|---|
| field on the slices, median difference | 0.010-0.42 % | 0.011-0.46 % |
| thick bodies (star, boat) | reference | comparable |
| thin walls (mug, pipe), 95th percentile | 2.8 %, 0.29 % | 3.6 %, 1.1 % |
| wine bottle resonance (bempp-cl: 145.35 Hz, half width 0.063 Hz, 61.4 dB) | 144.66 Hz, 0.061 Hz, 61.6 dB | 144.90 Hz, 0.166 Hz, 52.9 dB |
| elbow pipe resonance (bempp-cl: 1448.0 Hz) | 1447.0 Hz | 1442.3 Hz |
| relative solve time (GPU) | 1 | 1.1-1.5 |

Away from resonances both agree with bempp-cl to the same level. At the sharp Helmholtz resonance of the bottle the
constant-element Burton-Miller system introduces artificial damping (2.7 times the width, 8.6 dB less at the peak);
bempp-cl uses Burton-Miller with piecewise-linear elements and does not show it. CBIE + CHIEF therefore remains the
default.
Burton-Miller is kept for thick bodies at high ka, where CHIEF points may fall on nodal surfaces of an interior
mode: if CHIEF results look suspicious at some frequencies, solving with both formulations is a quick check.

## Accuracy guidelines

* Element size ≤ λ/6 (the report prints the resulting frequency limit). The error of constant elements falls with
  the square of the element size (examples/validation_sphere).
* Closed, consistently oriented surface. Assemblies exported from CAD usually are not: merge them with
  `stl2bem-prep` (voxel union).
* Thin plates: with exact static integration a plate thinner than the elements is no longer singular, but the CBIE
  of a body of vanishing thickness becomes ill-conditioned; check with two meshes.

## Memory and time

Dense matrices: memory ~ 3 N² × 8 bytes on the GPU (A, Q, R), ~ 3 N² × 16 bytes on the CPU (A, AᴴA in double).
Time per frequency grows as N³ (factorization) + N²·D (D directions). Practical sizes: up to ~15 000 elements on an
8 GB GPU, a few thousand on a laptop CPU. Large bodies at high frequencies need fast multipole methods
(e.g. NumCalc in Mesh2HRTF), which this package does not implement.

## References

* H. A. Schenck, "Improved integral formulation for acoustic radiation problems", J. Acoust. Soc. Am. 44(1), 1968.
* A. Van Oosterom, J. Strackee, "The solid angle of a plane triangle", IEEE Trans. Biomed. Eng. 30(2), 1983.
* D. A. Dunavant, "High degree efficient symmetrical Gaussian quadrature rules for the triangle",
  Int. J. Numer. Meth. Eng. 21, 1985.
* C. Ericson, *Real-Time Collision Detection*, 2005 (closest point on a triangle).
* J. H. DiBiase, "A high-accuracy, low-latency technique for talker localization in reverberant environments using
  microphone arrays", PhD thesis, Brown University, 2000 (SRP-PHAT).
