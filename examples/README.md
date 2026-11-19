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

The STL models in `gallery/models` are in mm, closed and consistently oriented (Fusion export, maximum edge length
3-6 mm). On an RTX 4060 Ti one gallery object takes 11 s (star) to 9 min (bell, with its
frequency sweep).
