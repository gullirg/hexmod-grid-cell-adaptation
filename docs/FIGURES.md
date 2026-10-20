# Figures

Every figure is drawn by one module of `hexmod.figures` from result files only, never from raw
data. `python scripts/step30_figures.py` draws all figures; `--figure Fig3` (repeatable) draws
selected ones and `--list` lists them. The tables below give, for every panel, the step that writes
the values it shows, the result files it reads (under `results/`) and the source-data files the
figure writes. The statistics shown in the panels and legends are listed with their keys in
[STATISTICS.md](STATISTICS.md), and the steps in [PIPELINE.md](PIPELINE.md).

The code draws Figures 2 to 8 and Supplementary Figures 1 to 5. Figure 1 and panel E of Figure 3
are schematics and are not produced by the code. The keys `fig1.*` hold the closed-form values
behind the legend of Figure 1, computed by step 27.

## Output files

For each figure `F`, step 30 writes to `results/figures/F/`:

| File | Content |
|---|---|
| `F.pdf` | the figure at its printed size, with embedded TrueType fonts and editable text |
| `F.svg` | the figure with text kept as text |
| `F.png` | the figure at 600 dpi |
| `source_data/<F><panel>_<description>.csv` | the values drawn in a panel; Figures 3 to 8 and Supplementary Figures 1 and 2 also write their statistics, recomputed from these values, to `<F>_statistics.csv` |

The panels carry no titles; the figure legends of the paper name them. A figure whose input files
are missing is reported as skipped. The simulation figures (6, 7 and 8) therefore build without any
data: their inputs are taken from `results_precomputed/` when the simulation steps have not been
run. Supplementary Figure 3E also reads the shipped simulated maps when step 22 has not been run.
`tests/test_plotted_vs_stats.py` checks, after a run, that the statistics recomputed from the
source data equal the statistics files.

## Figure 2: movement-direction tuning of conjunctive grid cells relative to the grid axes

Module `hexmod.figures.fig2`.

| Panel | Content | Step | Result files | Source data |
|---|---|---|---|---|
| A | tuning curves of five conjunctive cells of session V11_5 (cells 424, 427, 431, 447, 450) | 11 (tuning curves from step 02) | `rat/examples/fig2a_cells.csv`, `fig2a_cells.npz` | `Fig2A_tuning_curves.csv`, `Fig2A_cells.csv` |
| B | deviation of the preferred direction of the 1,171 conjunctive cells from the nearest grid axis, 5 deg bins | 10 | `rat/conjunctive/fig2b_bins.csv` | `Fig2B_deviation_histogram.csv` |
| C | mean rotated tuning curve across 15 animals (band: s.e.m. across animals) | 10 | `rat/conjunctive/fig2c_per_animal_curves.csv`, `fig2c_mean.csv` | `Fig2C_per_animal_curves_and_mean.csv` |
| D | preferred direction modulo 60 deg against grid orientation for the 890 directional cells (earliest recording of each cell), coloured by animal | 10 (uses step 09) | `rat/conjunctive/fig2d_cells.csv` | `Fig2D_cells.csv` |
| E | six-fold clustering of preferred directions in each of the 12 Vollan et al. (2025) animals, with the chance band | 10 | `rat/conjunctive/fig2e_per_animal.csv`, `fig2e_band.csv` | `Fig2E_animals.csv`, `Fig2E_chance_band.csv` |

Statistics: `fig2b.*` to `fig2e.*`, written by step 10 (and three derived in step 40). In D, the
six animals with the most cells have fixed colours (`fig2.ANIMAL_COLOURS`) and the other animals are
grey. The dots are drawn in a seeded random order (`PUBLISHED_DRAW_ORDER_SEED`), so that no animal
systematically covers another.

## Figure 3: firing rate adaptation of grid cells

Module `hexmod.figures.fig3`.

| Panel | Content | Step | Result files | Source data |
|---|---|---|---|---|
| A | rate map of cell 2 of session B1_2 with 12 s of path (143 to 155 s), spikes and fields numbered by first visit | 11 | `rat/examples/fig3ab.npz`, `fig3ab_field_visits.csv` | `Fig3A_rate_map_hz.csv`, `Fig3A_path_and_spikes.csv`, `Fig3A_fields.csv` |
| B | time since the previous field visit, location input and adaptation variable along that path | 11 | as A | `Fig3B_time_courses.csv`, `Fig3B_field_visits.csv` |
| C | firing rate change by time since the last field visit (four bins), all cells | 04, statistics 12 | `rat/adaptation/fr_percent_change.npy`, `rat/cohort/cells.csv` | `Fig3C_firing_rate_change_pct.csv` |
| D | adaptation coefficient alpha of the parametric model, all cells | 03, statistics 12 | `rat/adaptation/param_cells.csv` | `Fig3D_adaptation_coefficient.csv` |
| E | schematic of three successive field visits, not produced by the code | | | |
| F | firing rate ratio of the third visit for shorter and longer second intervals (3,576 cells) | 06 | `rat/visits/paired_cells.csv` | `Fig3F_firing_rate_ratio.csv` |

Statistics: `fig3c.*`, `fig3d.*` (step 12) and `fig3f.*` (step 06); recomputed from the plotted
values in `Fig3_statistics.csv`.

## Figure 4: proximity of field visits depends on the running direction relative to the grid axes

Module `hexmod.figures.fig4`.

| Panel | Content | Step | Result files | Source data |
|---|---|---|---|---|
| A | rate maps of a small-scale (cell 341) and a large-scale (cell 770) grid cell of session V10_1, with 12 s of path (126 to 138 s) coloured by speed | 11 | `rat/examples/fig4ab_small.npz`, `fig4ab_large.npz` and their `*_field_visits.csv` | `Fig4A_rate_maps.csv`, `Fig4A_fields.csv`, `Fig4AB_paths.csv` |
| B | time since the previous field visit, movement direction (shown from -90 to 270 deg) and grid alignment cos 6(theta - phi) for both cells | 11 | as A (arrays `grid_alignment`, `movedir_rad_pm_pi`) | `Fig4AB_paths.csv`, `Fig4B_field_visits.csv` |
| C | proximity for aligned and misaligned running, all cells | 03, statistics 12 | `rat/adaptation/param_cells.csv`, `rat/cohort/cells.csv` | `Fig4CD_cells.csv`, `Fig4_box_numbers.csv` |
| D | proximity difference by thirds of grid scale | 03 and 12 | `rat/figure_tables/cells_thirds.csv` | `Fig4CD_cells.csv`, `Fig4_box_numbers.csv` |

Statistics: `fig4c.*`, `fig4d.*` (step 12); recomputed in `Fig4_statistics.csv`.

## Figure 5: hexadirectional modulation of recorded grid cells

Module `hexmod.figures.fig5`.

| Panel | Content | Step | Result files | Source data |
|---|---|---|---|---|
| A | HexMod beta of all cells | 03, statistics 12 | `rat/figure_tables/cells_thirds.csv`, `rat/cohort/cells.csv` | `Fig5_cells.csv`, `Fig5_box_numbers.csv` |
| B | top third of the proximity difference, split into thirds of the adaptation coefficient (Strong, Weak, No adaptation) | 03 and 12 | as A | as A |
| C | strong adaptation, top against bottom third of the proximity difference | 03 and 12 | as A | as A |

Statistics: `fig5.*`, `fig5a.*` to `fig5c.*` (step 12); recomputed in `Fig5_statistics.csv`.

## Figure 6: simulated grid cells with firing rate adaptation

Module `hexmod.figures.fig6`. Inputs from `results/sim/fig6/`, or from
`results_precomputed/sim/fig6/`.

| Panel | Content | Step | Result files | Source data |
|---|---|---|---|---|
| A | path of the simulated participant with seed 0 (first 600 s of its trajectory) with the five objects of the first block | 22 | `sim/fig6/trajectory_seed0.csv`, `trajectory_seed0_objects.json` | `Fig6A_path.csv`, `Fig6A_objects.csv` |
| B | small module (40 vm): six example rate maps and the summed maps of the participants with seeds 0 and 1, with two straight example runs | 22 | `sim/fig6/maps.npz` | `Fig6BC_maps.csv`, `Fig6BC_example_runs.csv`, `Fig6BC_colour_scales.csv` |
| C | the same for the large module (70 vm) | 22 | `sim/fig6/maps.npz` | as B |
| D | proximity of the 10,000 cells of the participant with seed 0 for aligned and misaligned running, both modules | 23 | `sim/fig6/fig6d_small.csv`, `fig6d_large.csv` | `Fig6D_proximity.csv`, `Fig6_box_numbers.csv` |

Statistics: `fig6c.*`, `fig6d.*` (step 27); recomputed in `Fig6_statistics.csv`.

## Figure 7: detection of hexadirectional modulation in simulated participants

Module `hexmod.figures.fig7`. Inputs from `results/sim/fig7/`, or from
`results_precomputed/sim/fig7/`. A and B show the simple estimator, D and E the location-controlled
estimator, and C and F both; small module left and large module right in every panel.

| Panel | Content | Step | Result files | Source data |
|---|---|---|---|---|
| A, D | firing rate in 12 direction bins relative to the grid, participants 0 to 29 | 25 (from 24) | `sim/fig7/fig7ad_direction_bins.csv`, `participants_<module>.csv.gz` (rates in Hz for the tests) | `Fig7AD_firing_rate_by_direction.csv` |
| B, E | cross-validated beta for 1- to 7-fold symmetry, participants 0 to 29 | 25 | `sim/fig7/fig7be_beta.csv` | `Fig7BE_beta_by_symmetry.csv`, `Fig7_box_numbers.csv` |
| C | estimated six-fold orientation of participants 0 to 29 | 25 | `sim/fig7/fig7c_orientation.csv` | `Fig7C_orientation.csv` |
| F | detection rate over experiments of 30 participants: 49 experiments (participants 30 to 1,499) by default, 50 after `step25_simulation_collect.py --experiments 50` | 25 | `sim/fig7/fig7f_p_per_experiment.csv`, `fig7f_detection.csv`, `fig7_values.json` | `Fig7F_p_per_experiment.csv` |

Statistics: `fig7a.*` to `fig7f.*` (step 25, copied into the statistics of step 27); recomputed
from the plotted arrays in `Fig7_statistics.csv`, where the effect size of the A and D tests,
`cohens_dz_ddof0`, is the mean paired difference over its standard deviation with ddof 0.

## Figure 8: prediction for the effect of running speed

Module `hexmod.figures.fig8`. Inputs from `results/sim/fig8/`, or from
`results_precomputed/sim/fig8/`.

| Panel | Content | Step | Result files | Source data |
|---|---|---|---|---|
| A | field outlines of ideal 30 and 70 vm grids with an aligned and a misaligned run | 22 | `sim/fig8/fig8a_ideal_grids.npz` | `Fig8A_fields.csv`, `Fig8A_runs.csv` |
| B | proximity at the next field against running speed (tau = 2 s; 5 to 149 vm/s in steps of 2) | closed form, `hexmod.sim.analytic` | | `Fig8BC_proximity.csv` |
| C | proximity difference against running speed, largest at 1.3327 scale / tau | closed form | | `Fig8BC_proximity.csv` |
| D | simulated firing rate, misaligned minus aligned, against running speed, for the 30 and 70 vm grids at every second speed of the sweep (the sweep also runs a 50 vm grid, which is not drawn: 111 runs = 3 grids x 37 speeds from 5 to 149 vm/s) | 26 | `sim/fig8/fig8d_sweep.csv` | `Fig8D_speed_sweep.csv` |

Statistics: `fig8.v_opt_scale_per_tau`, `fig8d.n_runs` (step 27); `Fig8_statistics.csv`.

## Supplementary Figure 1: adaptation in odd and even minutes

Module `hexmod.figures.supp1`.

| Panel | Content | Step | Result files | Source data |
|---|---|---|---|---|
| A | adaptation coefficient fitted to the odd and to the even minutes of each recording | 03, statistics 12 | `rat/adaptation/param_cells.csv` | `SuppFig1_adaptation_coefficient_odd_even_minutes.csv`, `SuppFig1A_box_numbers.csv` |
| B | odd against even minutes, one point per cell, with the least-squares line | 03 and 12 | as A | `SuppFig1_adaptation_coefficient_odd_even_minutes.csv`, `SuppFig1B_fitted_line.csv` |

Statistics: `supp1.*` (step 12); recomputed in `SuppFig1_statistics.csv`.

## Supplementary Figure 2: prospective coding control

Module `hexmod.figures.supp2`.

| Panel | Content | Step | Result files | Source data |
|---|---|---|---|---|
| A | spikes and fields of the example cell (B1_2 cell 2) with spike times shifted by minus its best shift, 0 and its best shift (120 ms) | 11 | `rat/examples/supp2a_shifted_maps.npz` | `SuppFig2A_spike_positions.csv`, `SuppFig2A_fields.csv` |
| B | spatial information of the example cell against the time shift | 11 (spatial information from 05) | `rat/examples/supp2a_shifted_maps.npz` | `SuppFig2B_spatial_information_example_cell.csv` |
| C | best time shift of every cell, with the median | 05 | `rat/prospective/peak_shift.csv` | `SuppFig2C_best_time_shift.csv`, `SuppFig2C_histogram.csv` |
| D | firing rate change by time since the last field visit with shifted spikes (3,754 cells that remain grid cells) | 05, statistics 12 | `rat/prospective/fr_percent_change_shifted.npy`, `shift_cells.csv` | `SuppFig2D_firing_rate_change_pct.csv`, `SuppFig2_box_numbers.csv` |
| E | adaptation coefficient with original and shifted spike times | 05 (and 03), statistics 12 | `rat/prospective/shift_cells.csv` | `SuppFig2E_adaptation_coefficient.csv`, `SuppFig2_box_numbers.csv` |

Statistics: `supp2.*` (step 12); recomputed in `SuppFig2_statistics.csv`.

## Supplementary Figure 3: recorded and simulated grid cells of small and large scale

Module `hexmod.figures.supp3`.

| Panel | Content | Step | Result files | Source data |
|---|---|---|---|---|
| A | rate maps of 12 recorded grid cells (small and large scale; low, medium and high gridness) with fields and fitted lattice; the label above each map gives its gridness and peak rate | 11 | `rat/examples/supp3a_cells.csv`, `supp3a_cells.npz` | `SuppFig3A_rate_maps.csv`, `SuppFig3A_fields.csv`, `SuppFig3A_cells_and_lattices.csv` |
| B | deviation of field centres from the fitted lattice, with gamma fits | 12 (from 02) | `rat/figure_tables/supp3bcd_arrays.npz`, `supp3bcd_fits.csv` | `SuppFig3BCD_values.csv`, `SuppFig3BCD_histograms.csv`, `SuppFig3BCD_gamma_fits.csv` |
| C | mean field rate across cells, with gamma fits | 12 | as B | as B |
| D | field rates within each cell, with gamma fits | 12 | as B | as B |
| E | rate maps of 12 simulated grid cells (small and large module; low, medium and high gridness); the label above each map gives its gridness and peak rate | 22 | `sim/fig6/maps.npz` | `SuppFig3E_rate_maps.csv` |

The cells of A are listed in `inputs/rat/example_cells.csv` with the subset each was drawn from.
Statistics: `supp3.*` (step 12).

## Supplementary Figure 4: timescale of adaptation

Module `hexmod.figures.supp4`.

| Panel | Content | Step | Result files | Source data |
|---|---|---|---|---|
| (single) | firing rate change in nine logarithmic bins of time since the last field visit, with 95% intervals and weighted fits of one exponential (teal, solid) and two exponentials (orange, dashed) | 07 | `rat/timescales/supp4_points.csv`, `supp4_fits.csv` | `SuppFig4_points.csv`, `SuppFig4_fits.csv`, `SuppFig4_curves.csv` |

Statistics: `supp4.*` (step 07).

## Supplementary Figure 5: six-fold component of a tuning curve against its width

Module `hexmod.figures.supp5`.

| Panel | Content | Step | Result files | Source data |
|---|---|---|---|---|
| A | six-fold component of a von Mises tuning curve against its circular SD (linear scale), with the narrow tuning of Fig 1C and cosine tuning marked | closed form, `hexmod.sim.analytic` | | `SuppFig5AB_sixfold_component_curve.csv`, `SuppFig5_marked_points.csv` |
| B | the same on a log scale, with the narrowest and the median conjunctive cell marked, and the distribution of the circular SD of the 1,171 conjunctive cells | closed form and 12 (from 02) | `rat/figure_tables/supp5_widths.csv` | `SuppFig5AB_sixfold_component_curve.csv`, `SuppFig5B_conjunctive_cell_widths.csv`, `SuppFig5B_width_histogram.csv`, `SuppFig5_marked_points.csv` |

Statistics: `supp5.*` (step 12, two derived in step 40).
