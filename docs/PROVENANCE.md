# Provenance

This document lists the conventions of the preprocessing and the simulations, and the seed of every
random draw.

## 1. Preprocessing and cohort conventions

Each dataset module (`hexmod.preprocessing.<dataset>`) and `hexmod.preprocessing.trajectories`
describe their conventions in their docstrings, with the size of each effect. In short:

- **Clocks.** Every dataset stays on its own clock; the time shifts applied to positions and spikes
  are those listed below.
- **Sargolini et al. (2006).** The spike times of each unit are shifted so that its first spike is at
  0, while the position times are not shifted. Relative to the positions, each unit's spikes
  therefore move earlier by its first-spike latency (0.01 to 0.74 s for the 7 cohort cells).
- **Barry et al. (2012).** The position time is shifted so that it starts at 0 (it starts at 0.02 s
  in every session) and the spike times are not, so a spike is paired with the position sample 0.02
  to 0.04 s after it. All 57 sessions of the data file are read.
- **Vollan et al. (2025).** The released time series are speed filtered (samples at locomotion
  speeds below 5 cm/s were removed by the dataset authors), so the kept samples have gaps (42% of the
  session span in the median session). Every second sample is kept (100 to 50 Hz), counting across
  the gaps; positions are not interpolated across gaps. Files `28063_4` and `28063_5` (sessions V2_4
  and V2_5) hold one recording with two spike sortings; both stay in the cohort (113 and 98 cells),
  and the analyses that need independent recordings (Fig 2D and 2E) keep one recording per cell
  (`hexmod.rat.same_cell`).
- **Hafting et al. (2005).** The Fig2c and Fig2d files of one trial hold different cells of the same
  trial with identical position records and are merged into one session. Positions are resampled
  at 50 Hz. The recordings were made in a 180 cm cylinder.
- **Brun et al. (2008).** Positions in camera pixels are converted at 2 pixels per cm; missing
  samples are filled by linear interpolation over time.
- **Ólafsdóttir et al. (2016).** Positions and spikes are kept on the stored clock; the stored sample
  rate is kept with its type. The positions keep the column-major (Fortran) memory layout of the
  matrix in the original MATLAB data.
- **Data files of Barry et al. (2012) and Ólafsdóttir et al. (2016).** The sessions of these two
  datasets are read from the HDF5 files of their Zenodo record ([DATA.md](DATA.md)), which hold the
  positions and spike times of the analysed sessions with the values of the original recordings.
  `hexmod.preprocessing.hdf5_sessions` reads them into the session structure that the dataset
  modules take, in the order of the session manifest.
- **Theta phase, head direction and EEG.** No analysis uses them, and the preprocessed recordings
  do not hold them.
- **Sample rate.** `sample_rate` keeps the value and type derived from each raw file (for example
  49.999999998908606 Hz for some Sargolini et al. files), because `1 / sample_rate` is the sample
  interval used by later steps.
- **Smoothing.** Missing positions are filled by linear interpolation over the sample index and
  smoothed with a 20-sample boxcar; the window runs over the sample index, also across the gaps of
  the Vollan et al. recordings. The merged data hold the smoothed positions under both `xy` and
  `xy_smoothed`; all analyses read `xy_smoothed`.
- **Spikes in gaps.** Each spike is assigned to the first position sample at or after its time, and
  spikes that fall in a gap of the position record longer than 1.5 sample intervals are dropped
  (`hexmod.rat.gridcell.GridCellAnalysis`). Only samples with running speed between 5 and 150 cm/s
  enter the analyses.
- **Cohort rule.** Sessions with at least 30,000 position samples faster than 5 cm/s; cells with
  gridness of at least 0.3.

## 2. Seeds

Every random draw has a fixed seed. Seeded results also depend on the order of the cells
([PIPELINE.md](PIPELINE.md), Cell identity and order).

| Analysis | Seed | Where |
|---|---|---|
| Fig 2D permutation test (20,000 permutations of directions across cells) | `numpy.random.default_rng(FIG2D_PERMUTATION_SEED)`, seed 1802094013 | `hexmod.stats.fisher_lee_permutation`, `hexmod.rat.conjunctive` |
| Fig 2D drawing order of the coloured dots | `PUBLISHED_DRAW_ORDER_SEED`, the seed of the printed panel | `hexmod.figures.fig2` |
| Fig 2E chance band (95th percentile of the resultant length of n uniform directions) | 100,000 draws from `default_rng(5600 + n)` for n up to 60 cells; `sqrt(-ln 0.05 / n)` above | `hexmod.stats.rayleigh_chance`, `rayleigh_monte_carlo` |
| Supplementary Figure 4 animal bootstrap (1,000 resamples) | `default_rng(PUBLISHED_BOOTSTRAP_SEED)`, the seed of the printed intervals; one generator shared by the two variants, unshifted first | `hexmod.rat.timescales`, `step07_adaptation_timescales.py --seed` |
| Small-module pool (100,000 cells, 40 vm) | base seed 0 | `hexmod.sim.pool.set_gridcells_pool` (step 21) |
| Large-module pool (100,000 cells, 70 vm) | base seed 0 | `hexmod.sim.pool_large` (shipped; rebuilt with `--regenerate-large`) |
| Simulated participant s (Fig 7) | cells `default_rng(s).choice(100000, 10000, replace=False)`; trajectory `trajectory_v1.make_trajectory(par, base_seed=s)` | `hexmod.sim.simulate.participant_cells`, step 24 |
| Fig 6A trajectory and Fig 6D cells | trajectory `trajectory_v2` with base seed 0; cells `default_rng(0).choice(...)` (participant 0) | steps 22 and 23 |
| Fig 6B, 6C example maps | the first six cells of `default_rng(1).choice(100000, 10000, replace=False)`; summed maps of participants 0 and 1 | step 22 |
| Supplementary Figure 3E cells | `DataFrame.sample(n=1, random_state=5 + p)` for panel p (1 to 6) within its gridness class | step 22 |
| Fig 8A ideal grids and Fig 8D sweep | perfect-grid pools with base seed 0; the trajectory of each sweep run has base seed equal to its speed index | steps 22 and 26 |

## 3. Simulation conventions

- The simulated participants of Fig 7 use trajectory generator version 1
  (`hexmod.sim.trajectory_v1`), and Figs 6A, 6D and 8D use version 2 (`hexmod.sim.trajectory_v2`).
  The two versions draw random numbers differently. For example, the balanced start-position sampler
  of version 1 stops once the least used groups of object and direction are exhausted, while
  version 2 balances only over the groups that still have rows.
- Every simulated t_prev includes the default sample interval of `extract_field_entry`
  (dt = 0.02 s, that of the recordings), although the trajectories are sampled at 10 Hz.
- The speed factor is 1 in every simulation of the paper (`speedGamma_mean = 0`), so its form does
  not affect any result.
- The simulated large-module cells use a within-cell field-rate gamma of shape 7.2 and scale 0.10
  (mean 0.72) and an across-cell mean-rate gamma of shape 3.4 and scale 1.1, the parameters with
  which the shipped pool was generated. The gamma fits to the recorded large-scale cells (step 12,
  Supplementary Figure 3C and 3D) give shape 7.10 and scale 0.141 within cells and shape 3.35 and
  scale 1.09 across cells.
- The large-module builder (`hexmod.sim.pool_large`) differs from the small-module builder in
  ways that change its random stream (layout of the rate map, no limit on redraws, two extra normal
  draws per cell); its docstring lists them. Its columns `adaptEff`, `adaptTau` and `speedGamma` hold
  the generation defaults and are not read by any simulation, which set adaptation 0.1, time constant
  2 s and speed gain 0 through their parameters.
- A pool builder keeps a cell with placeholder gridness 10 if the gridness routine fails on its map;
  no cell of either pool of the paper took this path. Wall-to-wall trajectories (Fig 8D) stop a block
  when a 100th direction draw would be needed.
- Each simulated participant estimates hexadirectional modulation for symmetries 1 to 7
  (`hexmod.sim.simulate.FOLD_SYM_LIST`) with both estimators and four left-out blocks (112 rows per
  participant); asking for more symmetries adds rows and leaves those of 1 to 7 unchanged.
