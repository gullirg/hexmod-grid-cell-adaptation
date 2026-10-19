# Precomputed simulation results

The simulation results behind Figs 6 to 8 and Supp Fig 3E, in the layout that the simulation steps
write under `results/sim/` (see `docs/PIPELINE.md`, steps 22 to 26). The figure step reads
them when the corresponding step has not been run, so every simulation figure can be drawn without
running the simulations. The full Fig 7 simulation (1,500 participants per module) takes several
hours; `scripts/step25_simulation_collect.py` computes every Fig 7 value and table from the
participant summaries in `sim/fig7/` instead (`--source precomputed`; it reproduces the shipped
Fig 7 files byte for byte).

Distances are in virtual metres (vm) of the 100 x 100 vm environment, times in s, rates in Hz,
angles in degrees. CSV files are written with full float precision and should be read with
`hexmod.io.read_csv` (exact round trip). Small module: grid scale 40 vm; large module: 70 vm (the
pool in `inputs/simulation/`).

## `sim/fig6/`

| File | Content |
|---|---|
| `maps.npz` | `extent` (0, 100, 0, 100 vm); `small__example_maps`, `large__example_maps` (6 x 34 x 34, Hz): rate maps of the six example cells of Fig 6B and 6C, pool indices in `small__example_cells`, `large__example_cells` (the first six of `default_rng(1).choice(100000, 10000, replace=False)`, the same indices in both pools); `seed0__small`, `seed0__large`, `seed1__small`, `seed1__large` (34 x 34): summed rate maps of the 10,000 cells of participants 0 and 1, in percent of their mean; `supp3e__small__maps`, `supp3e__large__maps` (6 x 34 x 34, Hz): the maps of Supp Fig 3E, with their pool indices (`supp3e__<module>__cells`), gridness (`supp3e__<module>__gridness`) and peak rate (`supp3e__<module>__peak_rate_hz`, the maximum of the map), which the labels above the maps show; `supp3e__gridness_class`: low, low, medium, medium, high, high. Maps are indexed [x, y]. |
| `trajectory_seed0.csv` | the trajectory of Fig 6A and of the Fig 6D simulation: object-location task, trajectory seed 0, 23,990 samples at 10 Hz in 4 blocks of about 600 s that start at 0, 1,600, 3,200 and 4,800 s. Columns `time` (s), `x`, `y` (vm, empty during cues and inter-trial intervals), `moveSpeed` (vm/s), `moveDir` (deg), `trialId` (trial within the block), `expPhase` (-1 cue or inter-trial interval, 0 turning, 1 moving). |
| `trajectory_seed0_objects.json` | object locations of the trajectory: `objLocs` (the five objects of the first block, drawn in Fig 6A), `objLocs_by_block` (all four blocks), `block_duration_s`, `n_blocks`, `env_size_vm` |
| `fig6d_small.csv.gz`, `fig6d_large.csv.gz` | Fig 6D: the 10,000 cells of participant 0 (`default_rng(0).choice(100000, 10000, replace=False)`), simulated on the trajectory above with adaptation 0.1 and time constant 2 s. Columns `cell_index` (pool index), `aligned`, `misaligned`: mean proximity of the previous field visit, exp(-t_prev / 2 s), when running within 15 deg of a grid axis and further from the axes |

## `sim/fig7/`

| File | Content |
|---|---|
| `participants_small.csv.gz`, `participants_large.csv.gz` | one row per participant (0 to 1,499) and method (`simple`, then `location-controlled`): `module`, `participant`, `method`; `dir_0` to `dir_330`: mean firing rate of the participant's 10,000 adapting cells in 12 movement-direction bins (0 = along a grid axis), as estimated by the method, in percent change from the mean over the bins, as drawn in Fig 7A and 7D; `fold_1` to `fold_7`: cross-validated HexMod coefficient for 1- to 7-fold symmetry in percent of the mean firing rate during movement (Fig 7B, 7E, 7F); `orientation_deg_mod60`: estimated six-fold orientation, circular mean over the cross-validation runs, in [0, 60) deg (Fig 7C); `rate_dir_0_hz` to `rate_dir_330_hz`: the same 12 bins in Hz (the Fig 7A and 7D tests compare the mean of the aligned and of the misaligned bins); `mean_rate_moving_hz`: mean firing rate during movement |
| `fig7ad_direction_bins.csv` | Fig 7A and 7D: participants 0 to 29 of each module and method; `module`, `method`, `participant`, `dir_0` to `dir_330` (as in the participant tables) |
| `fig7be_beta.csv` | Fig 7B and 7E: participants 0 to 29; `module`, `method`, `participant`, `fold_1` to `fold_7` |
| `fig7c_orientation.csv` | Fig 7C: participants 0 to 29; `module`, `method`, `participant`, `orientation_deg_mod60` |
| `fig7f_detection.csv` | Fig 7F: one row per module, method and set of experiments (`source`: `rest_49_experiments`, participants 30 to 1,499, drawn in Fig 7F; `all_50_experiments`, participants 0 to 1,499); `experiments`, `participants`, `rate_fold_1` to `rate_fold_7` (share of experiments with a one-sided p < 0.05) and `count_fold_1` to `count_fold_7` |
| `fig7f_p_per_experiment.csv` | Fig 7F: one row per module, method, set of experiments and experiment of 30 participants; `p_fold_1` to `p_fold_7` (one-sided one-sample t-test of the coefficient against 0) |
| `fig7_values.json` | the statistics of Fig 7, one number per key: `fig7a.<module>.t`, `.df`, `.p`, `.n` (paired t-test, aligned against misaligned bins, participants 0 to 29, simple method) and `fig7d.*` (location-controlled); `fig7b.<module>.fold_<k>.t`, `.df`, `.p`, `.significant` (two-sided one-sample t-test of the coefficient, participants 0 to 29, simple) and `fig7e.*` (location-controlled); `fig7c.<module>.<method>.count_<deg>` (participants per 30 deg bin of six times the orientation); `fig7f.<module>.<method>.fold_<k>.count` and `.rate` (experiments of 30 participants with a one-sided p < 0.05; `fig7f` uses the 49 experiments of participants 30 to 1,499 drawn in Fig 7F, `fig7f_50` all 50 experiments of participants 0 to 1,499) |

Each participant s of a module draws its cells with `default_rng(s).choice(100000, 10000,
replace=False)` and its trajectory with seed s (`scripts/step24_simulation_participants.py`). The
coefficients are computed with the simple estimator and with the estimator that controls for
location (5 vm spatial bins).

## `sim/fig8/`

| File | Content |
|---|---|
| `fig8a_ideal_grids.npz` | the ideal grids of Fig 8A for 30, 50 and 70 vm (`small`, `medium`, `large`): `<name>__field_locations` (x, y, radius in vm, one row per field), `<name>__centers` (lattice points, vm), `<name>__grid_scale_vm` |
| `fig8d_sweep.csv` | Fig 8D: 111 simulations (3 ideal grids of 1,024 cells x 37 running speeds from 5 to 149 vm/s), each on a wall-to-wall trajectory with seed equal to the speed index, adaptation 0.1. Columns `scale` (vm), `scaleLevel`, `speedLevel`, `moveSpeed` (vm/s), `FR_align`, `FR_misalign` (Hz, mean rate of the population in the aligned and misaligned direction bins), `proximity_align`, `proximity_misalign`, `n_time` (samples), `FRby12dir_0` to `FRby12dir_330` (Hz), `FR_mis_minus_align_pct` (misaligned minus aligned rate, percent of their mean), `proximity_align_minus_mis`, `plotted` (speed drawn in Fig 8D: every second speed) |

## Licence

The files in this folder and its subfolders are licensed under CC BY 4.0 (see `LICENSE`). Please
cite the paper when you reuse them.
