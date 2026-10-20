# Simulation inputs

| File | Content |
|---|---|
| `large_module_pool_70cm.npz` | the pool of 100,000 simulated grid cells of the large module (grid scale 70 vm), 18.3 MB |
| `pool_checksums.json` | for both pools, the number of cells, the column order and the dtype and sha256 of every column, over values rounded to six significant digits (`hexmod.sim.io_pool.column_checksums`); for the shipped file, its size and sha256 |

Distances are in virtual metres (vm) of the simulated 100 x 100 vm environment, firing rates in Hz.
The "70cm" in the file name is the grid scale of the module, 70 vm.

## The large-module pool

The simulated participants of the large module draw their cells from this pool: participant s uses
the 10,000 cells `numpy.random.default_rng(s).choice(100000, 10000, replace=False)`. The pool is
used for Fig 6C, the large-module half of Fig 6D, the large-module panels of Fig 7 and Supp Fig 3E.
The results of the paper use exactly this pool, so it is shipped and loaded by default
(`scripts/step21_simulation_pools.py`). Each cell was generated as follows (all random numbers from
`numpy.random.default_rng(0)`):

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
(`hexmod.sim.pool_large`). The rebuilt pool agrees with the shipped one to within rounding (field
centres within about 3e-14 vm, peak rates within about 2e-16 Hz, gridness within about 1e-11), so
results computed from it can differ from those of the paper in the last digits. The small-module
pool (grid scale 40 vm, base seed 0) is not shipped: step 21 rebuilds it in about 6 minutes and
checks every column against `pool_checksums.json`. The rounding to six significant digits keeps
these checksums the same when builds of the numerical libraries differ in the last digit of a value.

The npz file holds plain numpy arrays and loads without pickle support: `columns` (the column
names), one array per column with one number per cell, and `<name>__data` and `<name>__shapes` for a
column with one array per cell. `hexmod.sim.io_pool.load_pool` reads it into a pandas DataFrame. The
simulations read `centers`, `peaks_x`, `peaks_y`, `radius` and `peakFR_jitter` (the peak rate of
each field), and `gridness` selects the cells of Supp Fig 3E.

## Licence

The files in this folder are licensed under CC BY 4.0 ([LICENSE-data](../../LICENSE-data)). Please
cite the paper when you reuse them.
