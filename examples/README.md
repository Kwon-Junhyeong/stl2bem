# Examples

Each example writes its numerical results with `run.py` (into `data/`) and turns them into rasters, animations
(`docs/img/`) and pgfplots data with `export.py`; `bash docs/figures/build.sh` then draws the figures.

| folder | what |
|---|---|
| [gallery](gallery) | six CAD objects from Fusion (star, toy boat, mug, wine bottle, bell, elbow pipe): pressure on a slice, on the surface, and inside the cavities over frequency |
| [validation_sphere](validation_sphere) | rigid sphere against the exact series solution: field map, surface pressure at ka = 1, 3, 6, convergence with the element size, comparison with the original code |
| [crosscheck_bempp](crosscheck_bempp) | the gallery meshes solved again with bempp-cl (Burton-Miller, P1 elements): field differences, surface levels, the two resonances. Needs `pip install bempp-cl` (separate environment recommended). `formulations.py` compares both stl2bem formulations with bempp-cl |
| [benchmark](benchmark) | time and peak memory per frequency against the element count, CPU and GPU |

```bash
python examples/gallery/run.py --backend cupy          # or --backend numpy (slower), --only star,mug
python examples/gallery/export.py
python examples/validation_sphere/run.py --backend cupy
python examples/validation_sphere/export.py
python examples/crosscheck_bempp/ours.py --backend cupy        # stl2bem side for the resonators
python examples/crosscheck_bempp/run.py                        # bempp-cl environment (hours on a CPU)
python examples/crosscheck_bempp/resonance.py                  # bempp-cl environment
python examples/crosscheck_bempp/verify_peak.py                # bempp-cl environment
python examples/crosscheck_bempp/export.py
python examples/crosscheck_bempp/formulations.py --backend cupy  # CHIEF vs Burton-Miller
python examples/benchmark/run.py --backend cupy && python examples/benchmark/run.py --backend numpy
python examples/benchmark/export.py
bash docs/figures/build.sh
```

## Gallery models

`gallery/models/*.stl` are the meshes used for all results, in mm, closed and consistently oriented. They were
exported from Fusion (surface deviation 0.3 mm, normal deviation 15 degrees, a maximum edge length per model).
`gallery/models/step/*.step` hold the same solids, in the same coordinates, for remeshing at another element size
in any CAD tool.

| model | triangles | mean edge | longest edge |
|---|---|---|---|
| star | 3,800 | 2.9 mm | 4.2 mm |
| boat | 9,912 | 3.8 mm | 5.8 mm |
| mug | 8,400 | 4.1 mm | 5.7 mm |
| bottle | 7,006 | 6.0 mm | 8.5 mm |
| bell | 12,012 | 3.7 mm | 8.5 mm |
| pipe | 6,040 | 4.0 mm | 5.6 mm |

On an RTX 4060 Ti one gallery object takes 11 s (star) to 9 min (bell, with its frequency sweep).
