# Precomputed simulation results

The simulation results behind Figs 6 to 8 and Supp Fig 3E, in the layout that steps 22 to 26 write
under `results/sim/`; only the two Fig 6D tables are shipped gzipped (step 23 writes them as plain
CSV files). Step 30 reads these files when the simulation steps have not been run, so every
simulation figure can be drawn without running the simulations. The full Fig 7 simulation takes
several hours; step 25 computes every Fig 7 value and table from the participant tables in
`sim/fig7/` instead, and then gives files equal to the shipped ones. The table below and the
docstrings of the scripts of steps 22 to 26 give the columns and keys of each file;
[docs/PIPELINE.md](../docs/PIPELINE.md) lists the steps.

Distances are in virtual metres (vm) of the 100 x 100 vm environment, times in s, rates in Hz and
angles in degrees. CSV files hold floats at full precision; read them with `hexmod.io.read_csv`. The
small module has a grid scale of 40 vm and the large module of 70 vm (the pool in
`inputs/simulation/`).

| File | Content | Step |
|---|---|---|
| `sim/fig6/maps.npz` | example and summed rate maps of Fig 6B and 6C, and the maps of Supp Fig 3E with their gridness and peak rate | 22 |
| `sim/fig6/trajectory_seed0.csv`, `trajectory_seed0_objects.json` | the trajectory of Fig 6A and 6D (seed 0, 10 Hz, four blocks of about 600 s) and its object locations | 22 |
| `sim/fig6/fig6d_small.csv.gz`, `fig6d_large.csv.gz` | mean proximity of the 10,000 cells of participant 0 for aligned and misaligned running (Fig 6D) | 23 |
| `sim/fig7/participants_small.csv.gz`, `participants_large.csv.gz` | per participant (0 to 1,499) and estimator: rate in 12 direction bins, cross-validated HexMod coefficients for 1- to 7-fold symmetry, six-fold orientation | 24, 25 |
| `sim/fig7/fig7ad_direction_bins.csv`, `fig7be_beta.csv`, `fig7c_orientation.csv` | participants 0 to 29: `module`, `method`, `participant` and the columns of the participant tables drawn in Fig 7A and 7D (`dir_*`), 7B and 7E (`fold_*`) and 7C (`orientation_deg_mod60`) | 25 |
| `sim/fig7/fig7f_detection.csv`, `fig7f_p_per_experiment.csv` | Fig 7F: per module, estimator and set of experiments of 30 participants (`source`: the 49 experiments from participant 30, drawn in the figure, or all 50), the share (`rate_fold_k`) and number (`count_fold_k`) of experiments with a one-sided p < 0.05, and the p value of each experiment (`p_fold_k`) | 25 |
| `sim/fig7/fig7_values.json` | the statistics of Fig 7, one number per key | 25 |
| `sim/fig8/fig8a_ideal_grids.npz` | the ideal grids of Fig 8A (30, 50 and 70 vm) | 22 |
| `sim/fig8/fig8d_sweep.csv` | 111 simulations of three ideal grids at 37 running speeds (Fig 8D) | 26 |

## Licence

The files in this folder and its subfolders are licensed under CC BY 4.0
([LICENSE-data](../LICENSE-data)). Please cite the paper when you reuse them.
