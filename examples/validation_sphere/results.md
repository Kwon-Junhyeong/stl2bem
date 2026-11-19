# Validation: rigid sphere

Field map (5120 elements, ka = 3.66, 53062 points): median |p - p_exact| 8.6e-04, 99.8 % below 1e-2; maximum 0.94 on the surface itself (118 points within 1 mm), 0.015 from 2 mm outward.

Surface ring (1280 elements), mean relative error: ka 1: stl2bem 0.17 % / original 4.18 %, ka 3: stl2bem 0.65 % / original 9.30 %, ka 6: stl2bem 1.58 % / original 17.35 %

| elements | edge (mm) | method | 500 Hz | 2000 Hz |
|---|---|---|---|---|
| 320 | 29.9 | original | 6.19 % | 17.34 % |
| 320 | 29.9 | stl2bem | 0.57 % | 2.93 % |
| 1280 | 15.1 | original | 3.44 % | 9.46 % |
| 1280 | 15.1 | stl2bem | 0.15 % | 0.74 % |
| 5120 | 7.5 | original | 1.78 % | 4.79 % |
| 5120 | 7.5 | stl2bem | 0.04 % | 0.19 % |
