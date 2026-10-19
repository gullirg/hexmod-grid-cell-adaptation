# Simulation inputs

| File | Content |
|---|---|
| `large_module_pool_70cm.npz` | the pool of 100,000 simulated grid cells of the large module (grid scale 70 vm), 18.3 MB |
| `pool_checksums.json` | sha256 of every column of the large-module pool and of the small-module pool, and of the npz file |
| `LICENSE` | Creative Commons Attribution 4.0 International (CC BY 4.0) for the files in this folder |

Distances are in virtual metres (vm) of the simulated 100 x 100 vm environment, firing rates in Hz.
The "70cm" in the file name refers to the grid scale of the module, 70 vm (70% of the side of the
environment); the code and the documentation give it in vm.

## The large-module pool

The simulated participants of the large module draw their cells from this pool: participant s uses
the 10,000 cells `numpy.random.default_rng(s).choice(100000, 10000, replace=False)`. The pool
is used for Fig 6C, the large-module half of Fig 6D, the large-module panels of Fig 7 and
Supp Fig 3E. The published results were computed with exactly this pool, so it is shipped and
loaded by default (`scripts/step21_simulation_pools.py`).

Each cell was generated as follows (all random numbers from `numpy.random.default_rng(0)`):

- a triangular lattice of field centres with spacing 70 vm and orientation 0 deg, shifted by a
  cell-specific offset (offsets form a regular 317 x 317 grid of fractions of the lattice period,
  taken in order);
- each field centre displaced in a uniformly random direction by 70 vm times a gamma variate
  (shape 2.1, scale 0.04);
- a mean rate per cell drawn from a gamma distribution (shape 3.4, scale 1.1), and a peak rate per
  field equal to that mean times a gamma variate (shape 7.2, scale 0.10);
- the rate at each position is set by the nearest field centre: a Gaussian of the distance
  (standard deviation 70/6 vm) times that field's peak rate;
- jitter and field rates are redrawn until the gridness of the rate map (3 vm bins, downsampled
  to 6 vm bins for the autocorrelogram) is at least 0.3;
- fields delineated on the final rate map give the peak positions and radii.

`python scripts/step21_simulation_pools.py --regenerate-large` rebuilds the pool from these rules
(`hexmod.sim.pool_large`). The rebuilt pool agrees with the shipped one to within rounding
(field centres within about 3e-14 vm, peak rates within about 2e-16 Hz, gridness within about
1e-11; field peaks, radii and redraw counts identical); `gridori_est`, which no analysis reads,
can differ by up to about 2.5 deg. Results computed from a rebuilt pool therefore agree with the
published values to rounding precision rather than bit for bit.

| Column | Per cell | Unit | Meaning | Read by the simulations |
|---|---|---|---|---|
| `centers` | k x 2 array | vm | field centres (lattice plus jitter) | yes |
| `peaks_x`, `peaks_y` | array | vm | peak positions of the fields delineated on the rate map | yes |
| `radius` | array | vm | radii of the delineated fields | yes |
| `peakFR_jitter` | array (one value per centre) | Hz | peak firing rate of each field | yes |
| `gridness` | number | | gridness of the rate map | Supp Fig 3E selection; mean gridness |
| `adaptEff`, `adaptTau`, `speedGamma` | number | -, s, - | adaptation strength (0.04), time constant (2 s) and speed gain (0.5) stored at generation | no (the simulations set adaptation 0.1, time constant 2 s and speed gain 0 through their parameters) |
| `cellcount` | integer | | number of draws until the gridness criterion was met | no |
| `gridscale_est`, `gridori_est` | number | vm, deg | grid scale and orientation estimated from the autocorrelogram | no |

## File layout

The npz file holds plain numpy arrays and loads without pickle support:

- `columns`: the column names in table order;
- `<name>`: a column with one number per cell (100,000 values);
- `<name>__data` and `<name>__shapes`: a column with one array per cell, stored as all arrays
  raveled and concatenated, and one row of dimensions per cell (`(100000, 2)` for `centers`,
  `(100000, 1)` for the others).

```python
from hexmod.sim.io_pool import load_pool

pool = load_pool("inputs/simulation/large_module_pool_70cm.npz")   # pandas DataFrame, 100,000 rows
```

Without the package:

```python
import numpy as np

z = np.load("inputs/simulation/large_module_pool_70cm.npz")
ends = np.cumsum(z["centers__shapes"].prod(axis=1))
centers = [a.reshape(-1, 2) for a in np.split(z["centers__data"], ends[:-1])]
gridness = z["gridness"]
```

`hexmod.sim.io_pool.save_pool` and `load_pool` convert in both directions without changing any
value, dtype, column order or row label.

## Checksums

`pool_checksums.json` lists, for each pool, the number of cells, the column order and, per column,
its kind, dtype and sha256 (`hexmod.sim.io_pool.column_checksums`: dtype, shape and little-endian
bytes of the values; for array columns the shapes, then the data). It also gives the size and sha256
of the shipped npz file.

The small-module pool (grid scale 40 vm; jitter gamma shape 2.0, scale 0.07; rates gamma 3.2, 1.4
across cells and 10.3, 0.10 within cells; 100,000 cells, base seed 0) is not shipped.
`scripts/step21_simulation_pools.py` rebuilds it with `hexmod.sim.pool.set_gridcells_pool` in
about 5 minutes on one core and checks every column against the checksums listed here.

## Licence

The files in this folder are licensed under CC BY 4.0 (see `LICENSE`). Please cite the paper when
you reuse them.
