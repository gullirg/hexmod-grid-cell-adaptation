# Pipeline and results

This document lists the steps of the analysis in run order, with the command of each step and the
files it reads and writes under `results/`. Statistics are listed in [STATISTICS.md](STATISTICS.md)
and figure panels in [FIGURES.md](FIGURES.md).

## Overview

```
python scripts/data_download.py                        # D1, once
python scripts/run_pipeline.py --workers 4             # D2 to 40
```

`run_pipeline.py` runs the steps in order and stops at the first that fails; `--from 02 --to 12`
selects a range and `--list` prints the steps. D1 and D2, the download and check of the data, are
described in [DATA.md](DATA.md); D1 runs only with `--download`. Step 24, the full simulation of the
Fig 7 participants (about 6 hours with 10 workers), runs only with `--full-sim`; without it, step 25
reads the shipped participant results in `results_precomputed/`. `--quick` simulates participants 0
to 29 for Fig 7A to 7E (about 1.2 worker hours).

Steps 01 and 09 read the raw data (`HEXMOD_DATA`, default `data/`). The other steps read only the
results folder (`HEXMOD_RESULTS`, default `results/`) and the shipped inputs. Every step writes a
run record, `logs/<script name>.json`, with its arguments, software versions, runtime and the sha256
of each input and output. Steps 02, 03, 06, 07, 08, 10, 12 and 27 write their statistics to
`stats/<script name>.json`, one number per key.

| Step | Script | Reads (steps) | Writes under `results/` |
|---|---|---|---|
| D1 | `data_download.py` | | raw files in the data folder |
| D2 | `data_check.py` | D1 | `data_check.json` |
| 01 | `step01_preprocess.py` | raw data | `preprocessed/` |
| 02 | `step02_cohort.py` | 01 | `rat/cohort/` |
| 03 | `step03_adaptation_parametric.py` | 01, 02 | `rat/adaptation/param_cells.csv`, `alpha_bounds.csv` |
| 04 | `step04_adaptation_nonparametric.py` | 01, 02 | `rat/adaptation/fr_percent_change*.npy` |
| 05 | `step05_prospective_coding.py` | 01, 02, 03 | `rat/prospective/` |
| 06 | `step06_visit_sequences.py` | 01, 02 | `rat/visits/` |
| 07 | `step07_adaptation_timescales.py` | 01, 02, 05 | `rat/timescales/` |
| 08 | `step08_methods_numbers.py` | 01, 02, 04 | `rat/methods/` |
| 09 | `step09_same_cell_links.py` | 01, 02, raw data | `rat/same_cell/` |
| 10 | `step10_conjunctive_cells.py` | 02, 09 | `rat/conjunctive/` |
| 11 | `step11_example_cells.py` | 01, 02, 05 | `rat/examples/` |
| 12 | `step12_rat_statistics.py` | 02 to 05 | `rat/figure_tables/` |
| 21 | `step21_simulation_pools.py` | | `sim/pools/` |
| 22 | `step22_simulation_maps.py` | 21 | `sim/fig6/`, `sim/fig8/fig8a_ideal_grids.npz` |
| 23 | `step23_simulation_proximity.py` | 21 | `sim/fig6/fig6d_*.csv` |
| 24 | `step24_simulation_participants.py` | 21 | `sim/fig7/participants/` |
| 25 | `step25_simulation_collect.py` | 24 or `results_precomputed/` | `sim/fig7/` |
| 26 | `step26_speed_sweep.py` | | `sim/fig8/fig8d_sweep.csv` |
| 27 | `step27_simulation_statistics.py` | 22 to 26 | `stats/` only |
| 30 | `step30_figures.py` | 02 to 26 | `figures/` |
| 40 | `step40_check_statistics.py` | all `stats/*.json` | `stats/statistics_check.*` |

```
results/   data_check.json  preprocessed/ (about 1.2 GB)  figures/  stats/  logs/
  rat/     cohort/ adaptation/ prospective/ visits/ timescales/ methods/ same_cell/
           conjunctive/ examples/ figure_tables/
  sim/     pools/ fig6/ fig7/ fig8/
```

## Cell identity and order

Per-cell tables carry `row` (the position of the cell in `rat/cohort/cells.csv`, from 0),
`animal_id`, `session_id` and `cell_id` (the unit key within the session), with the labels of
`inputs/manifests/`. Rows are in canonical order: datasets as in `hexmod.datasets.CANONICAL_ORDER`,
then animals, sessions and units in manifest order. Seeded permutation tests and random draws depend
on this order.

## Units and formats

Distances are in cm (simulations: virtual metres, vm), times in s, firing rates in Hz and angles in
degrees. A column in another unit says so in its name (`_ms`, `_rad`, `_vm`, `_pct`). Grid
orientation is in [0, 60) deg and preferred directions in [0, 360) deg; `hexmod.circular.wrap60`
gives the deviation from the nearest grid axis in [-30, 30) deg. Six-fold quantities are computed on
angles multiplied by 6. CSV files hold floats at full precision and read back exactly with
`hexmod.io.read_csv`; the preprocessed recordings are read with `hexmod.io.load_cache`.

## Rounded per-cell inputs

Some statistics of the paper were computed from per-cell values rounded as after a round trip
through a CSV file. The steps apply the same rounding with two helpers of `hexmod.io`:
`csv_float_roundtrip` (pandas' default float format and parser, which returns the neighbouring
double for about one value in five) and `array_text_roundtrip` (numpy's 8-decimal text form of an
array). Step 03 passes the grid orientation after one round trip and the field rates in 8-decimal
text to the fits. Step 05 compares `alpha_original` after one round trip. Steps 09 and 10 use grid
orientation, scale, preferred direction and resultant length after two round trips, and the
per-animal tests of Fig 2E after three; the tables of steps 05 and 10 hold these rounded values. The
rounding changes only the last digits and no printed value, and the reference values include it.

## Steps

### 01. Preprocessing

`python scripts/step01_preprocess.py --workers 4` checks the size of every raw file and writes
`preprocessed/<dataset>.pkl` (one per dataset, in its own conventions, see
[PROVENANCE.md](PROVENANCE.md)), `trajectories.pkl` (all six merged, with running speed and
direction) and `sessions.csv`.

### 02. Cohort and characterisation

`python scripts/step02_cohort.py --workers 8` reads `trajectories.pkl` and keeps the units with a
gridness of at least 0.3 in the sessions with at least 30,000 position samples faster than 5 cm/s:
3,938 cells of 32 animals in 112 sessions. It writes `stats/step02_cohort.json` and, to
`rat/cohort/`, `cells.csv` (below), `dir_maps.npy` (tuning curves, 72 bins of 5 deg),
`field_locations.npz`, `cells_full.npz` (rate maps, field rates and deviations), `sessions.csv` and
`table1.csv`.

| Column of `cells.csv` | Unit | Meaning |
|---|---|---|
| `row`, `animal_id`, `session_id`, `cell_id`, `unique_cell_id`, `dataset` | | identity of the cell |
| `gridness` | | gridness score |
| `scale_cm` | cm | grid scale |
| `orientation_deg` | deg | grid orientation, in [0, 60) |
| `Rayleigh_mag` | | resultant length of the direction tuning curve, 0 to 1 (conjunctive cells: above 0.3) |
| `Rayleigh_angle` | deg | preferred movement direction, in [0, 360) |
| `meanFR_all`, `meanFR_infield`, `meanFR_outfield` | Hz | mean rate overall, in fields and outside fields |
| `n_fields` | | number of fields |

### 03. Parametric adaptation and HexMod fits

`python scripts/step03_adaptation_parametric.py --workers 8` reads `trajectories.pkl`, `cells.csv`,
`field_locations.npz` and `cells_full.npz`. It fits the parametric adaptation model (time constant
2 s, regressors T = `tprev`, `tprevXloc`; masks M = `all`, `oddMinute`, `evenMinute`, `infield`) and
the HexMod model, and writes `rat/adaptation/param_cells.csv` (below). `alpha_bounds.csv` and
`stats/step03_adaptation_parametric.json` count the cells with alpha on a fit limit (not printed;
`--no-alpha-bounds` skips them).

| Column of `param_cells.csv` | Meaning |
|---|---|
| `row`, `animal_id`, `session_id`, `cell_id` | identity of the cell |
| `speed_T_M`, `alpha_T_M`, `converge_T_M` | speed coefficient, adaptation coefficient (negative: lower rate soon after the last field visit), convergence |
| `align_T_M`, `misalign_T_M` | mean of the adaptation regressor (proximity exp(-t_prev / 2 s), times the location input for `tprevXloc`) over the samples of mask M running within 15 deg of a grid axis, and further from the axes |
| `speed_all`, `beta_all`, `converge_all` | HexMod fit: speed coefficient, amplitude beta of rate x (1 + beta cos 6(theta - phi)), convergence |

### 04. Firing rate by time since the last field visit

`python scripts/step04_adaptation_nonparametric.py --workers 8` reads `trajectories.pkl`,
`cells.csv` and `field_locations.npz`. From a non-parametric model of location, speed and t_prev it
writes `rat/adaptation/fr_percent_change.npy` (rate in the t_prev bins 0 to 1, 1 to 2, 2 to 3 and
above 3 s, in percent change from the mean) and `fr_percent_change_infield.npy` (not printed).

### 05. Prospective coding control

`python scripts/step05_prospective_coding.py --workers 8` reads `trajectories.pkl`, `cells.csv` and
`param_cells.csv`, shifts the spikes by -400 to +380 ms in steps of 20 ms and characterises and
refits each cell at its best shift (3,754 of 3,938 remain grid cells). It writes `rat/prospective/`
(`shift_ms.npy`, `si_shift.npy`, `peak_shift.csv`, `shift_cells.csv`,
`fr_percent_change_shifted.npy`, `shift_field_locations.npz`).

### 06. Field visit sequences, Fig 3F

`python scripts/step06_visit_sequences.py --workers 4` reads `trajectories.pkl` and `cells.csv`.
With the default `--rule printed` (Fig 3F), fields come from a rate map of spike counts, and t_prev
is undefined for the first visit and across recording gaps longer than 5 s and capped at 10 s;
`--rule library` uses the settings of the other rat analyses. It writes
`stats/step06_visit_sequences.json` and `rat/visits/` (`paired_cells.csv`, `summary.csv`;
`nesting.csv` and `diagnostics.csv`, not printed).

### 07. Adaptation timescales, Supp Fig 4

`python scripts/step07_adaptation_timescales.py --workers 4` reads `trajectories.pkl`, `cells.csv`,
`field_locations.npz`, the shift tables of step 05 and `inputs/rat/timescale_fit_settings.json`. It
fits one and two exponentials to the rate in nine t_prev bins (0.050 to 20.1 s), with an animal
bootstrap of 1,000 resamples, for spikes as recorded (printed) and shifted. It writes
`stats/step07_adaptation_timescales.json` and `rat/timescales/` (`finebin_percent_change.npy`,
`supp4_points.csv`, `supp4_fits.csv`, `fits.json`, `bootstrap.json`; for the shifted spikes
`finebin_percent_change_shifted.npy`; and the time constants and AIC difference of every resample
in `bootstrap_draws_unshifted.npz` and `bootstrap_draws_shifted.npz`).

### 08. Numbers quoted in the Methods

`python scripts/step08_methods_numbers.py --workers 8` reads `trajectories.pkl`, `cells.csv`,
`field_locations.npz`, `fr_percent_change.npy` and `cells_full.npz`. It writes
`rat/methods/methods_numbers.json` (visit intervals by scale third, running speed in the 100 cm
arenas) and `stats/step08_methods_numbers.json`.

### 09. Repeated recordings of the same cell

`python scripts/step09_same_cell_links.py --workers 4` reads `cells.csv`, `trajectories.pkl` and the
16 raw files of the Vollan et al. (2025) animals recorded in more than one session. It links
recordings of the same unit, keeps the earliest of each and writes `entries.csv`, `units.csv` and
`links.csv` to `rat/same_cell/`.

### 10. Conjunctive cells, Fig 2B to 2E

`python scripts/step10_conjunctive_cells.py` reads `cells.csv`, `dir_maps.npy`, `entries.csv` and
`inputs/rat/module_labels.csv`. It writes the tables of Fig 2B to 2E to `rat/conjunctive/`
(`fig2b_*`, `fig2c_*`, `fig2d_cells.csv`, `fig2e_*`) and `stats/step10_conjunctive_cells.json`.

### 11. Example cells

`python scripts/step11_example_cells.py` reads the outputs of steps 01, 02 and 05 and
`inputs/rat/example_cells.csv`, and writes the arrays of the example cells of Fig 2A, 3A, 3B, 4A, 4B
and Supp Figs 2A, 2B and 3A to `rat/examples/`.

### 12. Rat statistics and figure tables

`python scripts/step12_rat_statistics.py` reads the per-cell outputs of steps 02 to 05. It writes
the other printed rat statistics (Fig 3C, 3D, 4C, 4D, 5; Supp Figs 1, 2, 3B to 3D, 5) to
`stats/step12_rat_statistics.json`, and `cells_thirds.csv`, `thirds_edges.csv`,
`supp3bcd_arrays.npz`, `supp3bcd_fits.csv` and `supp5_widths.csv` to `rat/figure_tables/`.

### 21. Simulated grid cell pools

`python scripts/step21_simulation_pools.py` builds the small-module pool (100,000 cells, grid scale
40 vm) and copies the shipped large-module pool (70 vm; `--regenerate-large` rebuilds it), checks
both against the column checksums in `inputs/simulation/pool_checksums.json` (taken over values
rounded to six significant digits), and writes `sim/pools/small_pool_40cm.npz`,
`large_pool_70cm.npz` and `pool_checksums.json`.

### 22. Simulated maps and trajectory, Fig 6A to 6C, Supp 3E, Fig 8A

`python scripts/step22_simulation_maps.py` reads the pools. It writes
`sim/fig6/trajectory_seed0.csv` and `trajectory_seed0_objects.json` (Fig 6A), `sim/fig6/maps.npz`
(Fig 6B, 6C, Supp Fig 3E) and `sim/fig8/fig8a_ideal_grids.npz` (Fig 8A).

### 23. Proximity of simulated cells, Fig 6D

`python scripts/step23_simulation_proximity.py` reads the pools and simulates the 10,000 cells of
participant 0 on the trajectory of Fig 6A. It writes `sim/fig6/fig6d_small.csv` and
`fig6d_large.csv`, equal in content to the gzipped shipped files.

### 24. Simulated participants, Fig 7

`python scripts/step24_simulation_participants.py --module small --workers 10` (and
`--module large`) reads the pools. Participant s draws its 10,000 cells with `default_rng(s)` and
its trajectory with seed s; all 3,000 take about 62 worker hours. It writes
`sim/fig7/participants/<module>/p0000.npz` to `p1499.npz`; a rerun skips the participants already
computed.

### 25. Fig 7 values

`python scripts/step25_simulation_collect.py` reads the participant files of step 24 when all 1,500
per module are present and match the pool, otherwise the shipped participant tables.
Fig 7F uses participants 30 to 1,499 (49 experiments of 30), or 0 to 1,499 with `--experiments 50`.
It writes the participant tables, the tables of Fig 7A to 7F and `fig7_values.json` (keys `fig7a.*`
to `fig7f.*`, and `fig7f_50.*` over 50 experiments) to `sim/fig7/`.

### 26. Running-speed sweep, Fig 8D

`python scripts/step26_speed_sweep.py --workers 10` simulates ideal grids of 30, 50 and 70 vm at 37
running speeds from 5 to 149 vm/s. It writes `sim/fig8/fig8d_sweep.csv` (111 runs).

### 27. Simulation statistics

`python scripts/step27_simulation_statistics.py` reads the outputs of steps 22, 23, 25 and 26, or
the shipped files when a step has not been run. It writes the closed-form values of the Fig 1 legend
and Fig 8B and 8C, and the statistics of Figs 6 to 8, to `stats/step27_simulation_statistics.json`.

### 30. Figures

`python scripts/step30_figures.py` draws Figures 2 to 8 and Supplementary Figures 1 to 5
(`--figure Fig3` draws one) as PDF, SVG and PNG (600 dpi) in `figures/<Figure>/`, with the values of
each panel in `source_data/`. Simulation results missing from the results folder are read from
`results_precomputed/`. A figure whose inputs are missing is skipped; the exit status is 1 when a
figure fails. Figure 1 and panel 3E are schematics that the code does not draw.

### 40. Statistics check

`python scripts/step40_check_statistics.py` compares every key of `stats/*.json` with
`reference/expected_statistics.json`, checks that every printed statistic rounds to its printed
number, and writes `stats/statistics_check.csv` and `.md`. The exit status is 1 when a statistic
fails or is missing.
