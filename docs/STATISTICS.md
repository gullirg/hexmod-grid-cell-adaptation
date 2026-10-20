# Statistics

Every statistic printed in the paper is computed by one pipeline step and written, under a key, to
`results/stats/<step>.json`. Step 40 (`scripts/step40_check_statistics.py`) collects these files,
computes the few printed numbers that follow from other statistics by a fixed formula (marked
"40 (derived)" below), and compares every value with `reference/expected_statistics.json`. Its
report is `results/stats/statistics_check.md`, with one row per statistic: the expected value, the
computed value and the tolerance. The reference values and their tolerances are described in
[reference/README.md](../reference/README.md), and the steps in [PIPELINE.md](PIPELINE.md).

## How to read the tables

- **Printed**: the statistic as printed in the paper (minus signs written as hyphens); for a
  statistic of a figure panel ("panel" under Where), the number at a fixed precision or the
  asterisks of a bracket.
- **Where**: where it is printed (text section, figure legend, figure panel or table). A note in
  parentheses adds context, for example where another part of the paper gives the value differently.
  Panel letters are those of the figures (docs/FIGURES.md). "Panel" marks a statistic of a figure
  panel; the figures show the plotted values and the asterisks of the brackets (listed as `***` or
  "no bracket"), not the numbers themselves.
- **Key**: the statistic key. A row that lists several keys gives their common printed text and value.
- **Step**: the pipeline step that writes it, `scripts/stepNN_<name>.py`:

  | Step | Script | Statistics file |
  |---|---|---|
  | 02 | `step02_cohort.py` | `stats/step02_cohort.json` |
  | 06 | `step06_visit_sequences.py` | `stats/step06_visit_sequences.json` |
  | 07 | `step07_adaptation_timescales.py` | `stats/step07_adaptation_timescales.json` |
  | 08 | `step08_methods_numbers.py` | `stats/step08_methods_numbers.json` |
  | 10 | `step10_conjunctive_cells.py` | `stats/step10_conjunctive_cells.json` |
  | 12 | `step12_rat_statistics.py` | `stats/step12_rat_statistics.json` |
  | 27 | `step27_simulation_statistics.py` | `stats/step27_simulation_statistics.json` (Fig 7 values computed by step 25) |
  | 40 (derived) | `step40_check_statistics.py` | computed in the check and listed in `stats/statistics_check.csv` |

- **Value**: the value computed by the pipeline from the raw data of the six datasets, to six
  significant digits. Counts, degrees of freedom and indicators (`significant`: 1 when p < 0.05) are
  integers; `p` is two-sided unless the key or the paper says otherwise.

The values below were computed on a Linux workstation with the package versions of
`environment.yml` and an MKL build of numpy. They agree with the reference values within the
tolerances of `reference/expected_statistics.json`, and all but seven are equal to them. The
exceptions are six bootstrap values of Supplementary Figure 4 (one-exponential interval within
1e-6 s, two-exponential intervals within 0.02 s, because their refits depend on the starting values,
and the share of resamples within 0.5 percentage points) and the last digit of `fig2d.fisher_lee.r`,
which depends on the numerical library build. An OpenBLAS build of numpy (the conda-forge default)
gives the same values within these tolerances; 16 of them differ in the eighth or a later
significant digit.

Four printed numbers differ from the computed value at the printed precision (step 40 lists them,
`KNOWN_PRINTED_DIFFERENCES`): the running speed of the Methods (19 cm/s printed, 19.5 cm/s
computed), the largest time shift (+400 ms printed, +380 ms analysed), the across-cell gamma shape
of the large-scale cells in Supplementary Figure 3C (3.3 printed, 3.35 fitted) and the upper end of
the fast time constant of the bootstrap in Supplementary Figure 4 (0.56 s printed, 0.568 s
computed). The effect sizes of Fig 7A are printed as Cohen's d_z with the SD of the paired
differences computed with ddof 0 (t / sqrt(n - 1)); every other effect size uses ddof 1.

Besides the printed statistics listed here, the reference file holds constants of the analysis
and the simulation statistics that are not printed (the Fig 7 values of all symmetries, both
estimators and 50 experiments, and a few further values of Figs 1 and 6). The steps also write
statistics of the recordings that the paper does not print and that are therefore not reference
values: the t, df and p of the post-hoc tests behind the brackets of Figs 3C, 4D and 5B and
Supplementary Figure 2D (the Holm-corrected p values that give the drawn asterisks are listed
below), the numbers of cells whose adaptation coefficient lies on a limit of the parametric fit
(step 03), the measured scale ranges of Table 1 and the number of sessions behind the running
speed. They are in `results/stats/*.json`, and step 40 lists them as `no reference`.

`tests/test_plotted_vs_stats.py` recomputes the statistics of every figure from the values the
figure draws (`results/figures/<Figure>/source_data/`) and compares them with these keys.

## Printed statistics

### Cohort and Table 1

| Printed | Where | Key | Step | Value |
|---|---|---|---|---|
| 7 | Table 1, Barry et al. (2012) | `cohort.barry2012.n_animals` | 02 | 7 |
| 118 | Table 1, Barry et al. (2012) | `cohort.barry2012.n_cells` | 02 | 118 |
| 57 | Table 1, Barry et al. (2012) | `cohort.barry2012.n_sessions` | 02 | 57 |
| 1 | Table 1, Brun et al. (2008) | `cohort.brun2008.n_animals` | 02 | 1 |
| 5 | Table 1, Brun et al. (2008) | `cohort.brun2008.n_cells` | 02 | 5 |
| 2 | Table 1, Brun et al. (2008) | `cohort.brun2008.n_sessions` | 02 | 2 |
| 21 animals | Results; Methods | `cohort.conjunctive.n_animals` | 02 | 21 |
| 1,171 | Results; Methods | `cohort.conjunctive.n_cells` | 02 | 1171 |
| 55 sessions | Methods | `cohort.conjunctive.n_sessions` | 02 | 55 |
| 2 | Table 1, Hafting et al. (2005) | `cohort.hafting2005.n_animals` | 02 | 2 |
| 23 | Table 1, Hafting et al. (2005) | `cohort.hafting2005.n_cells` | 02 | 23 |
| 3 | Table 1, Hafting et al. (2005) | `cohort.hafting2005.n_sessions` | 02 | 3 |
| 32 animals | Results; Methods; Table 1 | `cohort.n_animals` | 02 | 32 |
| 3,938 grid cells | Results; Methods; Table 1 | `cohort.n_cells` | 02 | 3938 |
| 112 sessions | Results; Methods; Table 1 | `cohort.n_sessions` | 02 | 112 |
| 6 | Table 1, Ólafsdóttir et al. (2016) | `cohort.olafsdottir2016.n_animals` | 02 | 6 |
| 61 | Table 1, Ólafsdóttir et al. (2016) | `cohort.olafsdottir2016.n_cells` | 02 | 61 |
| 24 | Table 1, Ólafsdóttir et al. (2016) | `cohort.olafsdottir2016.n_sessions` | 02 | 24 |
| at least 10 minutes | Methods; Table 1 legend (the rule counts position samples faster than 5 cm/s: 30,000 samples, 10 min at 50 Hz) | `cohort.rule.min_running_time_min` | 02 | 10 |
| 1 | Table 1, Sargolini et al. (2006) | `cohort.sargolini2006.n_animals` | 02 | 1 |
| 7 | Table 1, Sargolini et al. (2006) | `cohort.sargolini2006.n_cells` | 02 | 7 |
| 2 | Table 1, Sargolini et al. (2006) | `cohort.sargolini2006.n_sessions` | 02 | 2 |
| 15 | Table 1, Vollan et al. (2025) | `cohort.vollan2025.n_animals` | 02 | 15 |
| 3,724 | Table 1, Vollan et al. (2025) | `cohort.vollan2025.n_cells` | 02 | 3724 |
| 24 | Table 1, Vollan et al. (2025) | `cohort.vollan2025.n_sessions` | 02 | 24 |

### Figure 1

Figure 1 is a schematic and is not drawn by the code. The numbers in the legend of its panel C are
closed-form values computed by step 27.

| Printed | Where | Key | Step | Value |
|---|---|---|---|---|
| 67.5° | Fig 1C legend; Methods; Supp Fig 5A legend | `fig1.broad.circular_sd_deg` | 27 | 67.4606 |
| zero | Fig 1C legend | `fig1.broad.sixfold` | 27 | 0 |
| 15° | Fig 1C legend; Methods | `fig1.narrow.circular_sd_deg` | 27 | 15 |
| 59% | Fig 1C legend; Methods; Supp Fig 5A legend | `fig1.narrow.sixfold` | 27 | 0.592926 |

### Figure 2

| Printed | Where | Key | Step | Value |
|---|---|---|---|---|
| p = 0.22 | Results | `fig2b.binomial.p` | 10 | 0.219672 |
| 564 | Results; Fig 2B legend | `fig2b.n_aligned` | 10 | 564 |
| Of 1,171 conjunctive cells | Results | `fig2b.n_cells` | 10 | 1171 |
| 607 | Fig 2B legend | `fig2b.n_misaligned` | 10 | 607 |
| 48% | Results; Fig 2B legend | `fig2b.pct_aligned` | 10 | 48.164 |
| 52% | Fig 2B legend | `fig2b.pct_misaligned` | 40 (derived) | 51.836 |
| 73° | Results; Fig 2C legend | `fig2c.circular_sd.median_deg` | 10 | 73.0586 |
| 66.0 (IQR 66.0 to 80.0) | Fig 2C panel | `fig2c.circular_sd.q25_deg` | 10 | 65.9667 |
| 80.0 (IQR 66.0 to 80.0) | Fig 2C panel | `fig2c.circular_sd.q75_deg` | 10 | 80.0189 |
| 15 animals | Fig 2C legend | `fig2c.n_animals` | 10 | 15 |
| 1,160 cells | Fig 2C legend | `fig2c.n_cells` | 10 | 1160 |
| r = 0.001 | Results | `fig2d.fisher_lee.r` | 10 | 0.00126222 |
| 21 animals | Methods | `fig2d.n_animals` | 10 | 21 |
| n = 890 | Results; Methods; Fig 2D legend | `fig2d.n_cells` | 10 | 890 |
| 766 cells | Fig 2D legend | `fig2d.n_cells_six_largest_animals` | 40 (derived) | 766 |
| 888 | Methods | `fig2d.n_conjunctive` | 10 | 888 |
| 47 sessions | Fig 2D panel | `fig2d.n_sessions` | 10 | 47 |
| 20,000 permutations | Methods | `fig2d.permutation.n_perm` | 10 | 20000 |
| p = 0.16 | Results | `fig2d.permutation.p` | 10 | 0.163992 |
| p = 0.020 | Results | `fig2e.binomial.p` | 10 | 0.0195683 |
| 0.6 expected by chance | Results | `fig2e.expected_below_0_05` | 40 (derived) | 0.6 |
| Holm-corrected p ≥ 0.23 | Results | `fig2e.min_p_holm` | 10 | 0.231419 |
| 12 animals | Results; Methods; Fig 2E legend | `fig2e.n_animals` | 10 | 12 |
| 1,082 cells | Methods; Fig 2E legend | `fig2e.n_cells` | 10 | 1082 |
| Three | Results; Fig 2E legend | `fig2e.n_p_below_0_05` | 10 | 3 |
| 0.019 | Fig 2E panel | `fig2e.p_below_0_05.1` | 10 | 0.0192849 |
| 0.037 | Fig 2E panel | `fig2e.p_below_0_05.2` | 10 | 0.0366761 |
| 0.046 | Fig 2E panel | `fig2e.p_below_0_05.3` | 10 | 0.0457428 |

### Figure 3

| Printed | Where | Key | Step | Value |
|---|---|---|---|---|
| -6.19% | Fig 3C panel | `fig3c.mean_pct_change.0_1s` | 12 | -6.18883 |
| +0.39% | Fig 3C panel | `fig3c.mean_pct_change.1_2s` | 12 | 0.388884 |
| +2.39% | Fig 3C panel | `fig3c.mean_pct_change.2_3s` | 12 | 2.38888 |
| +2.84% | Fig 3C panel | `fig3c.mean_pct_change.above_3s` | 12 | 2.83998 |
| all pairs p < 0.001 except 2-3 s versus > 3 s | Results | `fig3c.posthoc.0_1s_vs_1_2s.p` | 12 | 1.27811e-102 |
| \*\*\* | Fig 3C panel (bracket) | `fig3c.posthoc.0_1s_vs_1_2s.p_holm` | 12 | 5.11243e-102 |
| all pairs p < 0.001 except 2-3 s versus > 3 s | Results | `fig3c.posthoc.0_1s_vs_2_3s.p` | 12 | 1.37502e-134 |
| \*\*\* | Fig 3C panel (bracket) | `fig3c.posthoc.0_1s_vs_2_3s.p_holm` | 12 | 6.87508e-134 |
| all pairs p < 0.001 except 2-3 s versus > 3 s | Results | `fig3c.posthoc.0_1s_vs_above_3s.p` | 12 | 3.7209e-187 |
| \*\*\* | Fig 3C panel (bracket) | `fig3c.posthoc.0_1s_vs_above_3s.p_holm` | 12 | 2.23254e-186 |
| all pairs p < 0.001 except 2-3 s versus > 3 s | Results | `fig3c.posthoc.1_2s_vs_2_3s.p` | 12 | 2.82206e-10 |
| all pairs p < 0.001 except 2-3 s versus > 3 s | Results | `fig3c.posthoc.1_2s_vs_above_3s.p` | 12 | 6.46346e-17 |
| t(3937) | Fig 3C panel | `fig3c.posthoc.2_3s_vs_above_3s.df` | 12 | 3937 |
| p = 0.12 | Results | `fig3c.posthoc.2_3s_vs_above_3s.p` | 12 | 0.120469 |
| t(3937) = -1.55 | Fig 3C panel | `fig3c.posthoc.2_3s_vs_above_3s.t` | 12 | -1.55315 |
| F(3,11811)=375.8 | Results | `fig3c.rm_anova.F` | 12 | 375.753 |
| F(3,11811) | Results | `fig3c.rm_anova.df1` | 12 | 3 |
| F(3,11811) | Results | `fig3c.rm_anova.df2` | 12 | 11811 |
| p<0.001 | Results; Fig 3C legend | `fig3c.rm_anova.p` | 12 | 4.12067e-233 |
| Cohen's d = -0.65 | Results | `fig3d.d` | 12 | -0.65474 |
| t(3937) | Results | `fig3d.df` | 12 | 3937 |
| 3,938 grid cells | Results | `fig3d.n_cells` | 12 | 3938 |
| p < 0.001 | Results | `fig3d.p` | 12 | 2.03399e-307 |
| 77.0% | Fig 3D panel | `fig3d.pct_negative` | 12 | 77.0442 |
| t(3937) = -41.1 | Results | `fig3d.t` | 12 | -41.0872 |
| d = -0.04 | Results | `fig3f.longer.d` | 40 (derived) | -0.0351259 |
| t(3575) | Results | `fig3f.longer.df` | 06 | 3575 |
| mean -0.006 | Results | `fig3f.longer.mean` | 06 | -0.00648752 |
| p = 0.036 | Results | `fig3f.longer.p` | 06 | 0.0357531 |
| t(3575) = -2.10 | Results | `fig3f.longer.t` | 06 | -2.10052 |
| n = 3,576 cells | Results | `fig3f.n_cells` | 06 | 3576 |
| d = 0.05 | Results | `fig3f.paired.d` | 40 (derived) | 0.0485684 |
| t(3575) | Results | `fig3f.paired.df` | 06 | 3575 |
| 0.014 | Results | `fig3f.paired.mean_diff` | 06 | 0.0141187 |
| p = 0.004 | Results | `fig3f.paired.p` | 06 | 0.00370242 |
| t(3575) = 2.90 | Results | `fig3f.paired.t` | 06 | 2.90438 |
| Cohen's d = 0.04 | Results | `fig3f.shorter.d` | 40 (derived) | 0.0422401 |
| t(3575) | Results | `fig3f.shorter.df` | 06 | 3575 |
| mean 0.008 | Results | `fig3f.shorter.mean` | 06 | 0.00763123 |
| p = 0.012 | Results | `fig3f.shorter.p` | 06 | 0.0115817 |
| t(3575) = 2.53 | Results | `fig3f.shorter.t` | 06 | 2.52594 |

### Figure 4

| Printed | Where | Key | Step | Value |
|---|---|---|---|---|
| t(3937) | Results; Fig 4C legend | `fig4c.df` | 12 | 3937 |
| 0.32285 | Fig 4C panel | `fig4c.mean_aligned` | 12 | 0.322846 |
| 0.32068 | Fig 4C panel | `fig4c.mean_misaligned` | 12 | 0.320679 |
| p<0.001 | Results; Fig 4C legend | `fig4c.p` | 12 | 4.58272e-28 |
| only 0.7% higher | Fig 4C legend (the Results say about 1% higher) | `fig4c.pct_higher` | 12 | 0.675954 |
| t(3937)=11.1 | Results; Fig 4C legend | `fig4c.t` | 12 | 11.069 |
| F(2,3935)=7.1 | Results | `fig4d.anova.F` | 12 | 7.0555 |
| F(2,3935) | Results | `fig4d.anova.df1` | 12 | 2 |
| F(2,3935) | Results | `fig4d.anova.df2` | 12 | 3935 |
| p<0.001 | Results | `fig4d.anova.p` | 12 | 0.00087361 |
| 1313 | Methods | `fig4d.n.large` | 12 | 1313 |
| 1312 | Methods | `fig4d.n.medium` | 12 | 1312 |
| 1313 | Methods | `fig4d.n.small` | 12 | 1313 |
| no bracket | Fig 4D panel (bracket) | `fig4d.posthoc.medium_vs_large.p_holm` | 12 | 0.767418 |
| \*\* | Fig 4D panel (bracket) | `fig4d.posthoc.small_vs_large.p_holm` | 12 | 0.00429885 |
| \*\* | Fig 4D panel (bracket) | `fig4d.posthoc.small_vs_medium.p_holm` | 12 | 0.00421873 |
| 27.6 cm | Fig 4D panel | `fig4d.scale_edges_cm.0` | 12 | 27.5965 |
| 62 cm | Results; Methods | `fig4d.scale_edges_cm.1` | 12 | 61.7929 |
| 86 cm | Results; Methods | `fig4d.scale_edges_cm.2` | 12 | 86.1047 |
| 154.8 cm | Fig 4D panel | `fig4d.scale_edges_cm.3` | 12 | 154.826 |

### Figure 5

| Printed | Where | Key | Step | Value |
|---|---|---|---|---|
| -0.160 | Fig 5 panel | `fig5.alpha_edges.0` | 12 | -0.159974 |
| -0.0614 | Fig 5 panel | `fig5.alpha_edges.1` | 12 | -0.0613983 |
| -0.0182 | Fig 5 panel | `fig5.alpha_edges.2` | 12 | -0.0181508 |
| 0.171 | Fig 5 panel | `fig5.alpha_edges.3` | 12 | 0.170679 |
| -0.0658 | Fig 5 panel | `fig5.proximity_edges.0` | 12 | -0.065792 |
| -0.00281 | Fig 5 panel | `fig5.proximity_edges.1` | 12 | -0.00280962 |
| 0.00604 | Fig 5 panel | `fig5.proximity_edges.2` | 12 | 0.00603544 |
| 0.0542 | Fig 5 panel | `fig5.proximity_edges.3` | 12 | 0.0541915 |
| t(3937) | Results; Fig 5A legend | `fig5a.df` | 12 | 3937 |
| p=0.8 | Results; Fig 5A legend | `fig5a.p` | 12 | 0.812034 |
| t(3937)=-0.2 | Results; Fig 5A legend | `fig5a.t` | 12 | -0.237819 |
| F(2,1310) = 4.0 | Fig 5B panel | `fig5b.anova.F` | 12 | 4.01333 |
| F(2,1310) | Fig 5B panel | `fig5b.anova.df1` | 12 | 2 |
| F(2,1310) | Fig 5B panel | `fig5b.anova.df2` | 12 | 1310 |
| p = 0.018 | Fig 5B panel | `fig5b.anova.p` | 12 | 0.0182958 |
| t(447) | Results | `fig5b.no.df` | 12 | 447 |
| n=448 | Fig 5B legend | `fig5b.no.n` | 12 | 448 |
| p=0.4 | Results | `fig5b.no.p` | 12 | 0.412959 |
| t(447) = 0.8 | Results | `fig5b.no.t` | 12 | 0.819463 |
| \* | Fig 5B panel (bracket) | `fig5b.posthoc.strong_vs_no.p_holm` | 12 | 0.0313469 |
| no bracket | Fig 5B panel (bracket) | `fig5b.posthoc.strong_vs_weak.p_holm` | 12 | 0.0564592 |
| no bracket | Fig 5B panel (bracket) | `fig5b.posthoc.weak_vs_no.p_holm` | 12 | 0.568172 |
| Cohen's d=-0.14 | Results; Fig 5B legend | `fig5b.strong.d` | 12 | -0.1364 |
| t(434) | Results; Fig 5B legend | `fig5b.strong.df` | 12 | 434 |
| n=435 | Results | `fig5b.strong.n` | 12 | 435 |
| p=0.005 | Results; Fig 5B legend | `fig5b.strong.p` | 12 | 0.00465365 |
| t(434)=-2.8 | Results; Fig 5B legend | `fig5b.strong.t` | 12 | -2.84484 |
| t(429) | Results | `fig5b.weak.df` | 12 | 429 |
| n=430 | Fig 5B legend | `fig5b.weak.n` | 12 | 430 |
| p=0.9 | Results | `fig5b.weak.p` | 12 | 0.934252 |
| t(429)=0.08 | Results | `fig5b.weak.t` | 12 | 0.0825451 |
| t(442) | Results; Fig 5C legend | `fig5c.strong_bottom.df` | 12 | 442 |
| 443 | Fig 5C panel | `fig5c.strong_bottom.n` | 12 | 443 |
| p=0.2 | Results | `fig5c.strong_bottom.p` | 12 | 0.173308 |
| t(442)=1.4 | Results; Fig 5C legend | `fig5c.strong_bottom.t` | 12 | 1.36385 |
| t(876) | Fig 5C legend | `fig5c.strong_top_vs_bottom.df` | 12 | 876 |
| p=0.003 | Fig 5C legend | `fig5c.strong_top_vs_bottom.p` | 12 | 0.00281468 |
| t(876)=-3.0 | Fig 5C legend | `fig5c.strong_top_vs_bottom.t` | 12 | -2.99577 |

### Figure 6

| Printed | Where | Key | Step | Value |
|---|---|---|---|---|
| +3.99% | Fig 6C panel (the Results say up to ±5% variation) | `fig6c.large_p0.max_pct` | 27 | 3.99114 |
| -2.63% | Fig 6C panel | `fig6c.large_p0.min_pct` | 27 | -2.62995 |
| SD 1.49 | Fig 6C panel | `fig6c.large_p0.sd_pct` | 27 | 1.48987 |
| d 0.92 | Fig 6D panel | `fig6d.large.d` | 27 | 0.917169 |
| t(9999) | Fig 6D panel | `fig6d.large.df` | 27 | 9999 |
| 0.0853 | Fig 6D panel | `fig6d.large.mean_aligned` | 27 | 0.0853383 |
| 0.0684 | Fig 6D panel | `fig6d.large.mean_misaligned` | 27 | 0.0684019 |
| t(9999) = 91.7 | Fig 6D panel | `fig6d.large.t` | 27 | 91.7169 |
| d 1.76 | Fig 6D panel | `fig6d.small.d` | 27 | 1.75774 |
| t 175.8 | Fig 6D panel | `fig6d.small.t` | 27 | 175.774 |

### Figure 7

| Printed | Where | Key | Step | Value |
|---|---|---|---|---|
| Cohen's d=-1.3 | Results (d_z with the SD of the paired differences computed with ddof 0, t / sqrt(n - 1), as in the Fig 7 statistics table; with ddof 1, as for the other effect sizes, -1.30) | `fig7a.large.d` | 40 (derived) | -1.31725 |
| t(29) | Results | `fig7a.large.df` | 27 | 29 |
| 30 virtual participants | Results; Fig 7A legend | `fig7a.large.n` | 27 | 30 |
| p<0.001 | Results | `fig7a.large.p` | 27 | 8.35287e-08 |
| t(29)=-7.1 | Results | `fig7a.large.t` | 27 | -7.09362 |
| Cohen's d=-4.4 | Results (d_z with the SD of the paired differences computed with ddof 0, t / sqrt(n - 1), as in the Fig 7 statistics table; with ddof 1, as for the other effect sizes, -4.31) | `fig7a.small.d` | 40 (derived) | -4.38718 |
| t(29) | Results | `fig7a.small.df` | 27 | 29 |
| 30 virtual participants | Results; Fig 7A legend | `fig7a.small.n` | 27 | 30 |
| p<0.001 | Results | `fig7a.small.p` | 27 | 1.7176e-20 |
| t(29)=-23.6 | Results | `fig7a.small.t` | 27 | -23.6257 |
| not significant | Results; Fig 7B legend (significant only: small modules 6-fold; large modules 2- and 3-fold) | `fig7b.large.fold_<k>.significant` for k = 1 and 4 to 7; `fig7b.small.fold_<k>.significant` for k = 1 to 5 and 7 | 27 | 0 |
| significant | Results; Fig 7B legend (large modules: 2- and 3-fold) | `fig7b.large.fold_2.significant` | 27 | 1 |
| significant | Results; Fig 7B legend (large modules: 2- and 3-fold) | `fig7b.large.fold_3.significant` | 27 | 1 |
| significant | Results; Fig 7B legend (small modules: 6-fold) | `fig7b.small.fold_6.significant` | 27 | 1 |
| -11.4143 | Fig 7D panel | `fig7d.large.t` | 27 | -11.4143 |
| -43.3605 | Fig 7D panel | `fig7d.small.t` | 27 | -43.3605 |
| not significant | Results (only 6-fold symmetry in both modules) | `fig7e.large.fold_<k>.significant` and `fig7e.small.fold_<k>.significant` for k = 1 to 5 and 7 | 27 | 0 |
| significant | Results (only 6-fold symmetry in both modules) | `fig7e.large.fold_6.significant` | 27 | 1 |
| significant | Results (only 6-fold symmetry in both modules) | `fig7e.small.fold_6.significant` | 27 | 1 |
| 19 of 49 | Fig 7F panel | `fig7f.large.simple.fold_1.count` | 27 | 19 |
| 41 of 49 | Fig 7F panel | `fig7f.large.simple.fold_2.count` | 27 | 41 |
| 17 of 49 | Fig 7F panel | `fig7f.large.simple.fold_3.count` | 27 | 17 |
| 12 of 49 | Fig 7F panel | `fig7f.large.simple.fold_4.count` | 27 | 12 |
| 6 of 49 | Fig 7F panel | `fig7f.large.simple.fold_5.count` | 27 | 6 |
| 39 of 49 | Fig 7F panel | `fig7f.large.simple.fold_6.count` | 27 | 39 |
| 1 of 49 | Fig 7F panel | `fig7f.large.simple.fold_7.count` | 27 | 1 |
| 49 | Fig 7F panel (the Results and Methods say 50 repetitions) | `fig7f.n_experiments` | 27 | 49 |

### Figure 8

| Printed | Where | Key | Step | Value |
|---|---|---|---|---|
| 1.33 × scale/τ | Results; Methods | `fig8.v_opt_scale_per_tau` | 27 | 1.33268 |
| 111 runs | Fig 8D panel | `fig8d.n_runs` | 27 | 111 |

### Supplementary Figure 1

| Printed | Where | Key | Step | Value |
|---|---|---|---|---|
| significantly negative in each half | Results; Supp Fig 1A legend | `supp1.even.p` | 12 | 1.77647e-191 |
| -31.2 | Supp Fig 1A panel | `supp1.even.t` | 12 | -31.2293 |
| significantly negative in each half | Results; Supp Fig 1A legend | `supp1.odd.p` | 12 | 1.43783e-238 |
| -35.4 | Supp Fig 1A panel | `supp1.odd.t` | 12 | -35.4004 |
| t(3937) | Supp Fig 1A panel | `supp1.odd_vs_even.df` | 12 | 3937 |
| p = 0.016 | Supp Fig 1A panel | `supp1.odd_vs_even.p` | 12 | 0.0164821 |
| t(3937) = -2.40 | Supp Fig 1A panel | `supp1.odd_vs_even.t` | 12 | -2.39909 |
| r(3936) | Supp Fig 1B legend | `supp1.pearson_df` | 12 | 3936 |
| p<0.001 | Supp Fig 1B legend | `supp1.pearson_p` | 12 | 3.22873e-88 |
| r(3936)=0.31 | Supp Fig 1B legend | `supp1.pearson_r` | 12 | 0.309571 |

### Supplementary Figure 2

| Printed | Where | Key | Step | Value |
|---|---|---|---|---|
| d = -0.667 | Supp Fig 2E panel | `supp2.alpha_original.d` | 12 | -0.666683 |
| t(3753) | Supp Fig 2E legend | `supp2.alpha_original_vs_shifted.df` | 12 | 3753 |
| p<0.001 | Supp Fig 2E legend | `supp2.alpha_original_vs_shifted.p` | 12 | 3.70435e-05 |
| t(3753)=-4.13 | Supp Fig 2E legend | `supp2.alpha_original_vs_shifted.t` | 12 | -4.13014 |
| d = -0.611 | Supp Fig 2E panel | `supp2.alpha_shifted.d` | 12 | -0.610693 |
| 120 ms | Supp Fig 2A legend | `supp2.example_best_shift_ms` | 12 | 120 |
| 40 ms | Results; Supp Fig 2C legend | `supp2.median_best_shift_ms` | 12 | 40 |
| 3,938 | Supp Fig 2 panel | `supp2.n_cells` | 12 | 3938 |
| 3,754 of 3,938 cells kept | Supp Fig 2E panel | `supp2.n_cells_kept` | 12 | 3754 |
| \*\*\* | Supp Fig 2D panel (bracket) | `supp2.posthoc.0_1s_vs_1_2s.p_holm` | 12 | 1.54995e-74 |
| \*\*\* | Supp Fig 2D panel (bracket) | `supp2.posthoc.0_1s_vs_2_3s.p_holm` | 12 | 2.62715e-106 |
| \*\*\* | Supp Fig 2D panel (bracket) | `supp2.posthoc.0_1s_vs_above_3s.p_holm` | 12 | 9.80332e-146 |
| F(3,11259)=294.7 | Supp Fig 2D legend | `supp2.rm_anova.F` | 12 | 294.674 |
| F(3,11259) | Supp Fig 2D legend | `supp2.rm_anova.df1` | 12 | 3 |
| F(3,11259) | Supp Fig 2D legend | `supp2.rm_anova.df2` | 12 | 11259 |
| p<0.001 | Supp Fig 2D legend | `supp2.rm_anova.p` | 12 | 3.62763e-184 |

### Supplementary Figure 3

| Printed | Where | Key | Step | Value |
|---|---|---|---|---|
| scale=0.04 | Supp Fig 3B legend | `supp3.large.deviation.scale` | 12 | 0.0444223 |
| shape=2.1 | Supp Fig 3B legend | `supp3.large.deviation.shape` | 12 | 2.07507 |
| 1,199 cells | Supp Fig 3B to 3D panels | `supp3.large.n_cells` | 12 | 1199 |
| 4,082 fields | Supp Fig 3B to 3D panels | `supp3.large.n_fields` | 12 | 4082 |
| scale=1.1 | Supp Fig 3C legend | `supp3.large.rate_across_cells.scale` | 12 | 1.09054 |
| shape=3.3 | Supp Fig 3C legend | `supp3.large.rate_across_cells.shape` | 12 | 3.35044 |
| scale=0.1 | Supp Fig 3D legend | `supp3.large.rate_within_cell.scale` | 12 | 0.140804 |
| shape=7.1 | Supp Fig 3D legend | `supp3.large.rate_within_cell.shape` | 12 | 7.10207 |
| scale=0.07 | Supp Fig 3B legend | `supp3.small.deviation.scale` | 12 | 0.0660245 |
| shape=2.0 | Supp Fig 3B legend | `supp3.small.deviation.shape` | 12 | 2.02251 |
| 1,178 cells | Supp Fig 3B to 3D panels | `supp3.small.n_cells` | 12 | 1178 |
| 12,667 fields | Supp Fig 3B to 3D panels | `supp3.small.n_fields` | 12 | 12667 |
| scale=1.4 | Supp Fig 3C legend | `supp3.small.rate_across_cells.scale` | 12 | 1.42212 |
| shape=3.2 | Supp Fig 3C legend | `supp3.small.rate_across_cells.shape` | 12 | 3.15232 |
| scale=0.1 | Supp Fig 3D legend | `supp3.small.rate_within_cell.scale` | 12 | 0.0969433 |
| shape=10.3 | Supp Fig 3D legend | `supp3.small.rate_within_cell.shape` | 12 | 10.3153 |

### Supplementary Figure 4

| Printed | Where | Key | Step | Value |
|---|---|---|---|---|
| from 0.05 to 20 s | Supp Fig 4 legend | `supp4.bin_edges.first_s` | 07 | 0.0497871 |
| from 0.05 to 20 s | Supp Fig 4 legend (the last edge is 20.09 s) | `supp4.bin_edges.last_s` | 07 | 20.0855 |
| 1,000 bootstrap resamples | Supp Fig 4 legend | `supp4.bootstrap.n_resamples` | 07 | 1000 |
| 99% | Supp Fig 4 legend | `supp4.bootstrap.pct_double_preferred` | 07 | 99 |
| 0.56 s | Supp Fig 4 legend (this bootstrap gives 0.568 s) | `supp4.bootstrap.tau_fast_high_s` | 07 | 0.56757 |
| 0.21 s | Supp Fig 4 legend | `supp4.bootstrap.tau_fast_low_s` | 07 | 0.212559 |
| 0.81 s | Supp Fig 4 legend | `supp4.bootstrap.tau_single_high_s` | 07 | 0.806463 |
| 0.31 s | Supp Fig 4 legend | `supp4.bootstrap.tau_single_low_s` | 07 | 0.308469 |
| more than 10 s | Supp Fig 4 legend (the upper fit bound is 50 s) | `supp4.bootstrap.tau_slow_high_s` | 07 | 50 |
| 1.3 s | Supp Fig 4 legend | `supp4.bootstrap.tau_slow_low_s` | 07 | 1.27997 |
| ΔAIC = -59 | Supp Fig 4 legend | `supp4.delta_aic` | 07 | -59.1385 |
| ΔBIC = -59 | Supp Fig 4 legend | `supp4.delta_bic` | 07 | -58.744 |
| χ² = 21 | Supp Fig 4 legend | `supp4.double.chi2` | 07 | 20.8287 |
| 4 residual degrees of freedom | Supp Fig 4 legend | `supp4.double.dof` | 07 | 4 |
| τ = 0.30 s | Supp Fig 4 legend; Discussion | `supp4.double.tau_fast_s` | 07 | 0.299966 |
| 2.6 s | Supp Fig 4 legend; Discussion | `supp4.double.tau_slow_s` | 07 | 2.64292 |
| F(2, 4) = 6.1 | Supp Fig 4 legend | `supp4.f_test.F` | 07 | 6.06264 |
| F(2, 4) | Supp Fig 4 legend | `supp4.f_test.df1` | 07 | 2 |
| F(2, 4) | Supp Fig 4 legend | `supp4.f_test.df2` | 07 | 4 |
| p = 0.06 | Supp Fig 4 legend | `supp4.f_test.p` | 07 | 0.0615326 |
| nine logarithmically spaced bins | Supp Fig 4 legend | `supp4.n_bins` | 07 | 9 |
| 3,938 cells | Supp Fig 4 legend | `supp4.n_cells` | 07 | 3938 |
| 3,936 in the last bin | Supp Fig 4 legend | `supp4.n_cells_last_bin` | 07 | 3936 |
| 84 | Supp Fig 4 legend | `supp4.single.chi2` | 07 | 83.9672 |
| with 6 (9 bins minus 3 parameters) | Supp Fig 4 legend | `supp4.single.dof` | 07 | 6 |
| τ = 0.45 s | Supp Fig 4 legend | `supp4.single.tau_s` | 07 | 0.452615 |

### Supplementary Figure 5

| Printed | Where | Key | Step | Value |
|---|---|---|---|---|
| 0.008% | Supp Fig 5A legend | `supp5.cosine_width_vonmises_sixfold_pct` | 40 (derived) | 0.00809639 |
| none can be broader than 89° | Supp Fig 5 legend | `supp5.max_conjunctive_sd_deg` | 40 (derived) | 88.909 |
| 73° | Methods; Supp Fig 5B legend | `supp5.median_sd_deg` | 12 | 73.0586 |
| 0.003% | Results; Methods; Supp Fig 5B legend | `supp5.median_sixfold_pct` | 12 | 0.00339533 |
| 1,171 conjunctive cells | Supp Fig 5B legend | `supp5.n_cells` | 12 | 1171 |
| 43° | Methods; Supp Fig 5B legend | `supp5.narrowest_sd_deg` | 12 | 43.3332 |
| 0.3% | Methods; Supp Fig 5B legend | `supp5.narrowest_sixfold_pct` | 12 | 0.320568 |

### Methods

| Printed | Where | Key | Step | Value |
|---|---|---|---|---|
| median 1.4 s | Discussion | `methods.interfield_median_s.small_third` | 08 | 1.44 |
| ~8% reduction | Methods | `methods.ratio_0_1s_to_above_3s` | 08 | 0.923155 |
| 19 cm/s | Methods | `methods.running_speed_cm_s` | 08 | 19.5059 |
| +400 ms | Results; Methods; Supp Fig 2 legend (the largest shift analysed is 380 ms) | `methods.shift_max_ms` | 08 | 380 |
| -400 ms | Results; Methods; Supp Fig 2 legend | `methods.shift_min_ms` | 08 | -400 |
| 2 s | Results; Methods | `methods.tau_s` | 08 | 2 |

### Discussion

| Printed | Where | Key | Step | Value |
|---|---|---|---|---|
| slow (2.7 m/s) | Discussion | `discussion.optimal_speed_4m_grid_m_s` | 40 (derived) | 2.66537 |
| fast (5.3 m/s) | Discussion | `discussion.optimal_speed_8m_grid_m_s` | 40 (derived) | 5.33073 |
