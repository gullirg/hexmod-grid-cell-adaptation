# Reference statistics

`expected_statistics.json` holds the statistics of the paper with the value each should have and an
absolute tolerance: every statistic printed in the paper (text, figure legends, figure panels and
Table 1), constants of the analysis, and the simulation statistics. Statistics that a step writes
but that the paper does not print (for example the measured scale ranges of Table 1 or the
post-hoc tests behind the figure brackets) are listed by step 40 as `no reference`. `scripts/step40_check_statistics.py`
(the last step of `scripts/run_pipeline.py`) compares the statistics of a run with these values and
writes `results/stats/statistics_check.md`.

## Fields

The file has the fields `schema` (`hexmod-expected-statistics/2`) and `statistics`, a list of
entries in the order of step and key. Each entry has:

| Field | Meaning |
|---|---|
| `key` | statistic key, as in `results/stats/<step>.json` (docs/PIPELINE.md, section 4) |
| `step` | script stem of the step that writes the key; `step40_check_statistics` for the derived statistics below |
| `kind` | `printed`: the statistic is printed in the paper (text, figure legend or table) or belongs to a figure panel (location "Fig X panel": the panel draws the values, and for the post-hoc tests behind a bracket the asterisks, but not the number itself); `reported`: a constant of the analysis or a simulation statistic that the paper does not print |
| `printed` | the text as printed, with minus signs written as hyphens; for a statistic of a figure panel, the number at a fixed precision, which step 40 also checks (empty for `reported` entries) |
| `location` | where it is printed: a section of the text, a figure legend, Table 1 or the figure panel it belongs to, with the panel letters of the figures (docs/FIGURES.md; Figure 1, a schematic that the code does not draw, has the letters of the paper) |
| `value` | reference value at full precision |
| `tolerance` | absolute tolerance: a statistic passes when `abs(computed - value) <= tolerance` |

## Values

The values behind the printed numbers, at full precision. Running the pipeline on the raw data
reproduces them exactly, apart from the statistics with a larger tolerance listed below. The upper
end of the two-exponential fast time constant of the animal bootstrap
(`supp4.bootstrap.tau_fast_high_s`) is the value of this pipeline, 0.568 s; the legend of
Supplementary Figure 4 prints 0.56 s.

Five values are computed from the exact per-cell values; the same statistics computed
from the per-cell values after a CSV round trip (docs/PIPELINE.md, section 2.7) differ in the 16th
significant digit: `fig6d.small.mean_aligned`, `fig6d.small.mean_misaligned`, `fig6d.small.t`,
`supp1.odd_vs_even.t` and `supp1.odd_vs_even.p`. The differences are at most 1.3e-15 relative
(2.8e-14 absolute, in the Fig 6D t of 175.77) and do not change any printed value.

## Tolerances

- **Counts, degrees of freedom and other integers:** 0.
- **Floating-point statistics:** `max(1e-9 * abs(value), 1e-15)`. On the machine that produced the
  reference values the statistics agree bit for bit; the tolerance leaves room for the last-digit
  differences that other numerical libraries give (for example an MKL against an OpenBLAS build of
  numpy, which changes the last digit of the Fisher-Lee correlation of Fig 2D, 0.0012622152849992038
  against 0.0012622152849992045), and stays far below the printed precision (at most five
  significant digits). The absolute floor of 1e-15 applies to very small p values, which the paper
  prints as p < 0.001.
- **Animal bootstrap of Supp Fig 4** (`supp4.bootstrap.*`): 1e-6 s for the single-exponential
  interval (the optimiser tolerance of the refits), 0.02 s for the two-exponential intervals (0.005 s,
  half a printed unit, for the upper end of the fast time constant, whose reference value is that of
  this pipeline) and 0.5 percentage points for the share of resamples in which the two-exponential
  model is preferred.
  The resamples are drawn with the published seed, so the same animals are drawn in each resample.
  The two-exponential refit of a resample depends on its starting values, because the slow time
  constant has a flat minimum; with the starting values in `inputs/rat/timescale_fit_settings.json` the ends of the
  two-exponential intervals differ from the reference values by up to 0.017 s and the share by 0.1
  percentage points (one resample in 1,000).

## Derived statistics

A few printed numbers follow from other statistics by a fixed formula. When no step writes them,
step 40 computes them (`DERIVED` in `scripts/step40_check_statistics.py`) and checks them like the
others:

| Key | Formula |
|---|---|
| `fig2b.pct_misaligned` | 100 x misaligned cells / conjunctive cells |
| `fig2d.n_cells_six_largest_animals` | cells of the six animals with the most cells in `rat/conjunctive/fig2d_cells.csv` |
| `fig2e.expected_below_0_05` | 0.05 x number of animals tested |
| `fig3f.shorter.d`, `fig3f.longer.d`, `fig3f.paired.d` | Cohen's d of the one-sample and paired t-tests, t / sqrt(n) |
| `fig7a.small.d`, `fig7a.large.d` | Cohen's d_z of the paired t-test of Fig 7A with the SD of the differences computed with ddof 0, t / sqrt(n - 1), as printed (-4.4 and -1.3; with ddof 1, as for the other effect sizes, -4.31 and -1.30) |
| `supp5.cosine_width_vonmises_sixfold_pct` | six-fold component of a von Mises curve as wide as cosine tuning (circular SD `fig1.broad.circular_sd_deg`) |
| `supp5.max_conjunctive_sd_deg` | circular SD of a curve whose Rayleigh vector length equals the conjunctive threshold 0.3 |
| `discussion.optimal_speed_8m_grid_m_s`, `discussion.optimal_speed_4m_grid_m_s` | `fig8.v_opt_scale_per_tau` x grid scale / 2 s |

## Printed values

Step 40 also checks that every computed value of a printed statistic rounds to the number in its
`printed` text (`printed_agreement` in the script; bounds such as `p<0.001`, words such as
`significant` and the asterisks of a bracket are checked too). Four printed numbers differ from
what the analysis gives; they are listed with the reason in `KNOWN_PRINTED_DIFFERENCES`:
`methods.running_speed_cm_s` (19 cm/s printed, 19.5 computed), `methods.shift_max_ms` (+400 ms
printed, +380 ms analysed), `supp3.large.rate_across_cells.shape` (3.3 printed, 3.35 fitted) and
`supp4.bootstrap.tau_fast_high_s` (0.56 s printed, 0.568 s computed). `tests/test_reference_statistics.py`
checks every printed reference value in the same way.

## Comparing complete results

`python scripts/step40_check_statistics.py --compare-frozen DIR` also compares every file of another
results folder `DIR` (in the layout of `results/`, for example the results of an earlier run) with the
files of this run: integers, booleans and text exactly, floating-point values with the tolerances of
`FROZEN_TOLERANCES` in the script; parts that this pipeline does not write (`NOT_COMPARED`) are left
out and named in the report. It writes `stats/frozen_comparison.csv` and `.md`.
