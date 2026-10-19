# Provenance

This document describes the conventions that the code keeps from the analysis behind the published
numbers, and how the library in this repository differs from the code that produced them. The
analysis functions keep the arithmetic, the order of operations and the random draws of that code,
so that the published values are reproduced (see [README](../README.md), Reproducibility notes, and
[reference/README.md](../reference/README.md)). Where a convention is a choice rather than a
necessity, the code keeps the published choice and names it; there is no silently "corrected"
variant in the default path.

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
  datasets are read from the HDF5 files of their Zenodo record (docs/DATA.md), which hold the
  positions and spike times of the analysed sessions with the values of the original recordings.
  `hexmod.preprocessing.hdf5_sessions` reads them into the session structure that the dataset
  modules take, in the order of the session manifest. The preprocessed recordings of these sessions
  are identical, array for array, to the recordings from which the published numbers were computed
  (same dtypes, shapes, memory layout and values).
- **Theta phase, head direction and EEG.** No analysis uses them, and the preprocessed recordings
  do not hold them (the data files of the two datasets above hold no EEG and no head direction).
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

## 2. Labels and order

- **Labels.** Animal, session and unit labels (for example `V10`, `V10_1`, `V10_V10_1_cell341`) and
  unit keys (`cell_id`) come from the shipped manifests `inputs/manifests/sessions.csv` and
  `units.csv` (`hexmod.preprocessing.labels`). Sessions are numbered per animal; for the Sargolini et
  al. data the unit keys number the spike files of the whole dataset in the reading order of the
  manifest column `listdir_order` of `files_sargolini2006.csv`.
- **Canonical order.** Datasets appear in the order of `hexmod.datasets.CANONICAL_ORDER`
  (`sargolini2006`, `brun2008`, `olafsdottir2016`, `vollan2025`, `hafting2005`, `barry2012`), then
  animals, sessions and units in manifest order. Every per-cell table follows this order, and no
  order is taken from a directory listing, because the seeded permutation test of Fig 2D and other
  seeded draws depend on the order of the cells.
- **Module labels.** `inputs/rat/module_labels.csv` holds the manual assignment of cells to grid
  modules. It gives the `module` column of the Fig 2D cell table and the selection subsets of the
  example cells of Fig 4A and Supplementary Figure 3A.
- **Example cells.** The cells and time windows of Fig 2A, 3A, 3B, 4A, 4B and Supplementary Figures
  2A and 3A are listed in `inputs/rat/example_cells.csv` with the rule that selects each of them;
  step 11 checks the rules of Fig 2A, 4A and Supplementary Figure 2A and records the result in its
  run record (`selection_checks`). The twelve
  cells of Supplementary Figure 3A were drawn at random from the subsets given in the file; the list
  is shipped so that the panel does not depend on that draw.
- **Table 1 text.** The environment and grid scale columns of Table 1 are text
  (`inputs/rat/table1_text.csv`); step 02 writes the measured scale ranges next to them.

## 3. Rounding of per-cell inputs

Some per-cell values entered the published statistics after a round trip through text files
(pandas' default CSV float format, or numpy's 8-decimal text form of an array). The steps reproduce
these round trips for the inputs where they change the last digits of a published value
(`hexmod.io.csv_float_roundtrip`, `array_text_roundtrip`): the grid orientation and field
rates passed to the parametric fits (step 03), the adaptation coefficient with unshifted spikes
compared in Supplementary Figure 2 (step 05), and the orientations and preferred directions of Fig 2D and 2E
(step 10). [PIPELINE.md](PIPELINE.md), section 2.7, lists every case and how the tables are written.
Printed values do not depend on these round trips.

## 4. Seeds

Every random draw has a fixed seed. Seeded results also depend on the order of the cells (section 2).

| Analysis | Seed | Where |
|---|---|---|
| Fig 2D permutation test (20,000 permutations of directions across cells) | `numpy.random.default_rng(seed_of("q2sel_kl\|one\|all\|dir_mean"))`, where `seed_of` is the crc32 of the text key modulo 2^31 | `hexmod.stats.fisher_lee_permutation`, `hexmod.rat.conjunctive` |
| Fig 2D drawing order of the coloured dots | `PUBLISHED_DRAW_ORDER_SEED`, the seed of the published panel | `hexmod.figures.fig2` |
| Fig 2E chance band (95th percentile of the resultant length of n uniform directions) | 100,000 draws from `default_rng(5600 + n)` for n up to 60 cells; `sqrt(-ln 0.05 / n)` above | `hexmod.stats.rayleigh_chance`, `rayleigh_monte_carlo` |
| Supplementary Figure 4 animal bootstrap (1,000 resamples) | `default_rng(PUBLISHED_BOOTSTRAP_SEED)`, the seed of the published intervals; one generator shared by the two variants, unshifted first (with `--variant shifted` alone the unshifted draws are skipped first, `skip_bootstrap_draws`) | `hexmod.rat.timescales`, `step07_adaptation_timescales.py --seed` |
| Small-module pool (100,000 cells, 40 vm) | base seed 0 | `hexmod.sim.pool.set_gridcells_pool` (step 21) |
| Large-module pool (100,000 cells, 70 vm) | base seed 0 | `hexmod.sim.pool_large` (shipped; rebuilt with `--regenerate-large`) |
| Simulated participant s (Fig 7) | cells `default_rng(s).choice(100000, 10000, replace=False)`; trajectory `trajectory_v1.make_trajectory(par, base_seed=s)` | `hexmod.sim.simulate.participant_cells`, step 24 |
| Fig 6A trajectory and Fig 6D cells | trajectory `trajectory_v2` with base seed 0; cells `default_rng(0).choice(...)` (participant 0) | steps 22 and 23 |
| Fig 6B, 6C example maps | the first six cells of `default_rng(1).choice(100000, 10000, replace=False)`; summed maps of participants 0 and 1 | step 22 |
| Supplementary Figure 3E cells | `DataFrame.sample(n=1, random_state=5 + p)` for panel p (1 to 6) within its gridness class | step 22 |
| Fig 8A ideal grids and Fig 8D sweep | perfect-grid pools with base seed 0; the trajectory of each sweep run has base seed equal to its speed index | steps 22 and 26 |
| Smoke run | fixed seeds in `hexmod.synthetic` | `run_pipeline.py --smoke` |

The two seeds named `PUBLISHED_...` are fixed integer seeds of the published analysis and
reproduce its draws. The text rules (`tests/test_text_rules.py`) report eight-digit numbers that
have the form of a date and allow these two values only as the named constants listed in
`SEED_CONSTANTS`.

## 5. Analysis variants and options

The defaults of the library reproduce the published analyses. Options select the variants that some
published results used; each step passes the options its printed results need.

| Option | Default | Other value, and where it is used |
|---|---|---|
| `rate_map_spike_mode` (`hexmod.rat.gridcell`, `characterise`) | `"binary"`: a position sample with several spikes counts once in the rate map | `"counts"`: spike counts, for the fields of Fig 3F |
| `tprev_rule` (`hexmod.rat.fields.extract_field_entry`) | `"standard"`: t_prev is 0 until the end of the first visit and after the last; no cap | `"gap_aware"` (Fig 3F): t_prev undefined for the first visit, for non-positive intervals and across steps of the speed-filtered time axis longer than `recording_gap_threshold` (5 s); capped at `tprev_cap` (10 s) |
| `mask_valid` (`hexmod.rat.adaptation.adaptation_nonparametric`) | `False` | `True` (Supp Fig 4): only samples within the speed bins and the t_prev bins enter the fit |
| `step06_visit_sequences.py --rule` | `printed`: the two options above, as in the published Fig 3F | `library`: binary rate map and standard t_prev, the settings of the other rat analyses; its tables go to `rat/visits/library_rule/` |
| `step25_simulation_collect.py --experiments` | `49`: Fig 7F over participants 30 to 1,499 | `50`: participants 0 to 1,499 |
| `step21_simulation_pools.py --regenerate-large` | the shipped large-module pool | rebuild it; the result agrees with the shipped pool within rounding (field centres 3e-14 vm, rates 2e-16 Hz, gridness 1e-11), not bit for bit |
| `step07_adaptation_timescales.py` | `--variant both`, `--n-boot 1000`, `--seed` the published seed | other variants, resample numbers or seeds |
| trajectory generator | version 1 (`hexmod.sim.trajectory_v1`) for the Fig 7 participants | version 2 (`hexmod.sim.trajectory_v2`) for Figs 6A, 6D and 8D; the two versions draw random numbers differently |

Supplementary Figure 4 uses the arithmetic bin centres as abscissa, the s.e.m. across the 3,936
cells with values in all nine bins, and weights 1 / s.e.m.^2 (`absolute_sigma=True`). The fit
settings are in `inputs/rat/timescale_fit_settings.json`; of the starting values listed there the
fit with the lowest residual sum of squares is kept (two starts for one exponential, one for two
exponentials). AIC is computed from the chi-square of the weighted fits and from n log(SS / n) for
the unweighted fits. The bootstrap draws animals with replacement (`rng.choice` over the sorted
animal labels) and keeps the cells with values in all nine bins. Because the slow time constant has
a flat minimum, the two-exponential refits of a resample depend on their starting values; with the
starting values of the settings file the two-exponential interval ends agree with the reference
values within 0.02 s.

The numbers of the Methods (step 08) use the standard t_prev at field entry, without the first visit
of each cell and with positive intervals only; the pooled interval statistics use intervals up to
60 s. Scale thirds are cut at the 33.3rd and 66.7th percentiles of grid scale (`numpy.digitize`).
The running speed is the mean over the samples between 5 and 150 cm/s of each cohort session in a
100 cm arena, averaged over these sessions.

## 6. Differences from the code behind the published numbers

The library differs from the code that produced the published numbers in these ways only. None of
them changes a published value.

- **Names.** Functions, arguments and modules have lower-case names that describe what they compute
  (for example `extract_field_entry`, `position_binning`, `fit_two_factor`, `fit_three_factor`,
  `measure_hexmod`, `simulate_grid_cells`, `simulate_and_measure`, `make_trajectory`,
  `subsample_rows_balanced`). The dictionary keys of the per-cell inputs and outputs are unchanged,
  so result tables keep their column names (for example `alpha_tprevXloc_all`, `beta_all`,
  `FR_inhomogeneity_regionFR`). Dataset keys are citation keys.
- **Plotting removed.** Analysis functions return values and draw nothing. The drawing helpers used
  by the figures (`draw_triangular_lattice_lines`, `polar_hist`, `plot_trajectory_segment`) are in
  `hexmod.figures.common`. Progress bars are optional (`progress=True`).
- **Unused code removed.** Functions that no pipeline step calls, unused branches and arrays, unused
  trajectory modes and an unused import of a graphical module (which failed on computers without a
  display) are not included.
- **Empty direction bins.** The movement-direction tuning curve divides the smoothed spike counts by
  the smoothed dwell time only where the dwell time is not zero. Bins without dwell are set to zero
  explicitly (`np.divide(..., out=np.zeros_like(...), where=...)` in
  `hexmod.rat.maps.compute_directional_rate_map_and_rayleigh_vector`), rather than left with
  whatever the output array held. No cohort cell has such a bin, so no published value changes;
  `tests/test_maps_synthetic.py::test_empty_direction_bins_are_zero` checks the behaviour on a
  recording that leaves half of the directions empty.
- **Errors and warnings.** Input checks that stopped the code with an assertion raise `ValueError`
  with a message (for example a time axis that is not strictly increasing); invalid option values
  raise `ValueError`. The multiplicative Poisson fits report non-convergence as a `RuntimeWarning`
  instead of printing a message. Step 01 stops with a one-line message, before any work, when a raw
  file is missing or has the wrong size; a step stops with a one-line message when its results
  folder holds results of the other kind of run (smoke run or recorded data,
  `hexmod.config.ResultsFolderError`); a figure whose inputs are missing is reported as skipped
  (`hexmod.figures.common.MissingInput`).
- **Parallel work.** Per-cell analyses run through `hexmod.parallel.map_ordered`, which returns
  results in input order; results do not depend on the number of workers.
- **File formats.** Results are written as CSV with full float precision, npy/npz and JSON; local
  caches are pickles of plain dicts of numpy arrays read with a restricted unpickler. Pools of
  simulated cells are npz files (`hexmod.sim.io_pool`) and simulated participants are one npz file
  each.
- **Statistics.** All tests are in `hexmod.stats`, written with numpy and scipy only; the
  repeated-measures ANOVA, the Holm correction and Zar's approximation of the Rayleigh p value are
  written out and tested against textbook examples (`tests/test_stats.py`).
- **Visit sequences (Fig 3F).** `hexmod.rat.visit_sequences` holds the visit extraction and the
  sequence tables of Fig 3F. Event columns, summaries and reports that no printed number uses are not
  included; cells are processed in cohort order; the event cache is a dict of numpy arrays. The
  option `rule="library"` gives the extraction with the settings of the other rat analyses.
- **Repeated recordings (Fig 2D, 2E).** `hexmod.rat.same_cell` holds the link set used in the paper
  (rules in its docstring). The labels of the Hafting et al. units are taken from the manifest
  (`raw_unit_key`). Link variants that the paper does not use are not included.

## 7. Simulation conventions

- Every simulated t_prev includes the default sample interval of `extract_field_entry`
  (dt = 0.02 s, that of the recordings), although the trajectories are sampled at 10 Hz.
- The balanced start-position sampler of trajectory version 1 stops once the least used groups of
  object and direction are exhausted; version 2 balances only over the groups that still have rows.
  This is one of the differences in the random stream of the two versions.
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

## 8. Figures

- The figure modules draw the published figures from the result files only. The panels carry no
  titles (the figure legends name them); labels that tell the plots of a panel apart are drawn. The
  panel letters of a row share one line, 1.5 mm above the highest element of the row or higher where
  a letter would otherwise come within 0.5 mm of a label (`hexmod.figures.style.settle_letters`,
  `LETTER_ABOVE`, `LETTER_CLEAR`; Figures 1 and 2 place their letters at fixed positions by the same
  rule). Count notes such as "3,938 cells" are not drawn.
- All statistics shown in a figure or written to its `source_data/` are computed from the plotted
  arrays (`hexmod.stats`, `hexmod.figures.style.holm`).
- Fig 1 is a schematic with its own type settings; the bar heights of panel D (`D_RATIO`) are
  illustrative constants, not results.
- The panel positions of Fig 2 are fixed constants in mm (`fig2.LAYOUT`).
- Fig 4B draws the arrays `grid_alignment` and `movedir_rad_pm_pi` of step 11, computed from the
  movement directions at their stored precision (single precision for the Vollan et al. data); the
  column `movement_direction_deg` of the Fig 4 source data keeps that precision.
- The effect size of the Fig 7A and 7D tests in `Fig7_statistics.csv` (`cohens_dz_ddof0`) is the mean
  paired difference over its standard deviation computed with ddof 0.
- Supplementary Figures 1 and 2E use the exact per-cell values of the result files; the same
  values after a CSV round trip (section 3) differ by less than 1e-16. The figures drawn from the
  two differ by at most one grey level in a few pixels of the PNG (102 pixels of Supplementary
  Figure 1).
- Supplementary Figure 5 finds the von Mises concentration with `hexmod.circular.kappa_from_sd`; the
  curve agrees with that of the published figure to 4e-12 percentage points.
- `hexmod.figures.style` takes Arial from `HEXMOD_FONT_DIR`, then from the system, then falls back to
  Liberation Sans or DejaVu Sans with one warning. The colour key of Figure 1B is placed for Arial;
  in a fallback font that brings it closer to the wheel labels than `fig1.KEY_GAP_MIN`, the module
  warns and draws the figure all the same. PDF and SVG files carry no creation date, SVG
  element identifiers use a fixed salt, and the identifiers of clip paths that are not rectangles
  (polar axes), which matplotlib derives from memory addresses, are renamed after their outline
  once the file is written (`hexmod.figures.style.stable_svg_clip_ids`). Repeated builds with the
  same inputs, font files and matplotlib version therefore give the same files.
