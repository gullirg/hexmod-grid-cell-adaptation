# Pipeline and results

This document lists every step of the analysis in run order: the command, what it reads, and
every file it writes under `results/`, with formats, columns, units and row order. It is
the reference that the step scripts, the figure code and the tests follow. File formats and
naming rules are in [CONVENTIONS.md](CONVENTIONS.md).

Contents:

1. [Overview](#1-overview) (step table, results layout)
2. [Rules shared by all steps](#2-rules-shared-by-all-steps): [results folder](#21-results-folder),
   [cell identity and order](#22-cell-identity-and-canonical-order), [formats](#23-formats),
   [units](#24-units), [statistics files](#25-statistics-files), [run records](#26-run-records),
   [rounded per-cell inputs](#27-rounded-per-cell-inputs), [step script interface](#28-step-script-interface)
3. [Steps](#3-steps): [D1](#d1-download-the-datasets), [D2](#d2-check-the-data),
   [01](#01-preprocessing), [02](#02-cohort-and-characterisation),
   [03](#03-parametric-adaptation-and-hexmod-fits),
   [04](#04-firing-rate-by-time-since-the-last-field-visit), [05](#05-prospective-coding-control),
   [06](#06-field-visit-sequences-fig-3f), [07](#07-adaptation-timescales-supp-fig-4),
   [08](#08-numbers-quoted-in-the-methods), [09](#09-repeated-recordings-of-the-same-cell),
   [10](#10-conjunctive-cells-fig-2b-to-2e), [11](#11-example-cells),
   [12](#12-rat-statistics-and-figure-tables), [21](#21-simulated-grid-cell-pools),
   [22](#22-simulated-maps-and-trajectory-fig-6a-to-6c-supp-3e-fig-8a),
   [23](#23-proximity-of-simulated-cells-fig-6d), [24](#24-simulated-participants-fig-7),
   [25](#25-fig-7-values), [26](#26-running-speed-sweep-fig-8d), [27](#27-simulation-statistics),
   [30](#30-figures), [40](#40-statistics-check)
4. [Statistic keys](#4-statistic-keys)

## 1. Overview

```
python scripts/data_download.py                        # D1, once
python scripts/run_pipeline.py --workers 4             # D2 to 40
```

`run_pipeline.py` runs the steps below in order (`--from 02 --to 12` selects a range,
`--list` prints them). It skips D1 unless `--download` is given, and skips step 24 (the full
Fig 7 simulation, about 6 hours with 10 workers) unless `--full-sim` or `--quick` is given; step 25
then reads the shipped simulation results in `results_precomputed/`. `--quick` simulates
participants 0 to 29 of both modules (about 1.2 worker hours), from which step 25 takes Fig 7A to
7E; Fig 7F then comes from the shipped results. `--smoke` skips D1 and D2, writes small synthetic
recordings made with the simulator (`hexmod.synthetic`: two sessions, 22 grid cells with adaptation)
in place of the outputs of steps 01 and 09, which read the raw files, and runs the other steps on
them with small sizes, in a temporary results folder (printed on the first line and not removed).

| Step | Script | Reads (steps) | Writes under `results/` |
|---|---|---|---|
| D1 | `data_download.py` | | `data/` (raw files) |
| D2 | `data_check.py` | D1 | `data_check.json` |
| 01 | `step01_preprocess.py` | D2 | `preprocessed/` |
| 02 | `step02_cohort.py` | 01 | `rat/cohort/` |
| 03 | `step03_adaptation_parametric.py` | 01, 02 | `rat/adaptation/param_cells.csv` |
| 04 | `step04_adaptation_nonparametric.py` | 01, 02 | `rat/adaptation/fr_percent_change*.npy` |
| 05 | `step05_prospective_coding.py` | 01, 02, 03 | `rat/prospective/` |
| 06 | `step06_visit_sequences.py` | 01, 02 | `rat/visits/` |
| 07 | `step07_adaptation_timescales.py` | 01, 02, 05 | `rat/timescales/` |
| 08 | `step08_methods_numbers.py` | 01, 02, 04 | `rat/methods/` |
| 09 | `step09_same_cell_links.py` | 01, 02, raw Vollan files | `rat/same_cell/` |
| 10 | `step10_conjunctive_cells.py` | 02, 09 | `rat/conjunctive/` |
| 11 | `step11_example_cells.py` | 01, 02, 05 | `rat/examples/` |
| 12 | `step12_rat_statistics.py` | 02 to 11 | `rat/figure_tables/`, `stats/step12_rat_statistics.json` |
| 21 | `step21_simulation_pools.py` | | `sim/pools/` |
| 22 | `step22_simulation_maps.py` | 21 | `sim/fig6/`, `sim/fig8/fig8a_ideal_grids.npz` |
| 23 | `step23_simulation_proximity.py` | 21 | `sim/fig6/fig6d_*.csv` |
| 24 | `step24_simulation_participants.py` | 21 | `sim/fig7/participants/` |
| 25 | `step25_simulation_collect.py` | 24 or `results_precomputed/` | `sim/fig7/` |
| 26 | `step26_speed_sweep.py` | | `sim/fig8/fig8d_sweep.csv` |
| 27 | `step27_simulation_statistics.py` | 22 to 26 | `stats/step27_simulation_statistics.json` |
| 30 | `step30_figures.py` | 12, 27 | `figures/` |
| 40 | `step40_check_statistics.py` | all `stats/*.json` | `stats/statistics_check.*` |

Every step also writes its run record to `results/logs/<script stem>.json` (section 2.6).

### Results layout

```
results/
  data_check.json               D2
  preprocessed/                 01  (large caches, about 1.2 GB)
  rat/cohort/ adaptation/ prospective/ visits/ timescales/ methods/ same_cell/
      conjunctive/ examples/ figure_tables/
  sim/pools/ fig6/ fig7/ fig8/
  figures/<Figure>/             30
  stats/                        one JSON per step with printed statistics; check reports (40)
  logs/                         one run record per step (D2: logs/data_check.json)
  cache/                        working caches (never read by another step)
```

## 2. Rules shared by all steps

### 2.1 Results folder

Every step reads and writes the results folder `results/` (`HEXMOD_RESULTS`, default
`<repo>/results`; `hexmod.config.results_dir`). The analysis always uses the six datasets. The steps
that read the raw recordings (01 and 09) need the data folder (`HEXMOD_DATA`); step 01 stops before
any work, with a one-line message (exit status 1), when a raw file is missing or has the wrong size.
The other steps read only results of earlier steps, so statistics, tables and figures can be rebuilt
from a results folder without the raw data.

A step run with `--smoke` marks its results folder with the file `.smoke_run`
(`hexmod.config.SMOKE_MARKER`) and refuses a folder that holds run records of steps run without
`--smoke`; a step run without `--smoke` refuses a marked folder. Synthetic results therefore never
mix with results of the recorded data.

### 2.2 Cell identity and canonical order

Per-cell tables and arrays use these identifiers:

| Column | dtype | Meaning |
|---|---|---|
| `row` | int64 | position of the cell in the cohort table `rat/cohort/cells.csv`, 0-based |
| `animal_id` | str | animal label, dataset prefix plus number, for example `V10` |
| `session_id` | str | session label, for example `V10_1` |
| `cell_id` | int64 | unit key within the session (key of `spike_times` in the preprocessed data) |
| `unique_cell_id` | str | unit label written by preprocessing, for example `V10_V10_1_cell341` |
| `dataset` | str | dataset key, for example `vollan2025` (never a nickname) |

Labels are those of the shipped manifests `inputs/manifests/sessions.csv` and `units.csv`. Their
columns, and the functions that read them and put data into canonical order
(`hexmod.preprocessing.labels`), are described in the docstring of that module.

**Canonical order.** Datasets in `hexmod.datasets.CANONICAL_ORDER` (`sargolini2006`,
`brun2008`, `olafsdottir2016`, `vollan2025`, `hafting2005`, `barry2012`), then animals, sessions
and units in the order of the manifests. All per-cell outputs are in canonical order, filtered as
stated, and carry the `row` of the cell; arrays (`.npy`) are row-aligned with the table named in
their entry. Seeded permutation tests (Fig 2D) and seeded draws depend on this order.

### 2.3 Formats

- CSV: written with `hexmod.io.write_csv` (no index, floats with `%.17g`, `\n` line ends);
  read with `hexmod.io.read_csv` (exact float round trip). Booleans are `True`/`False`; missing
  values are empty fields. `.csv.gz` files are gzip-compressed CSV.
- `.npy` and `.npz`: no object arrays. Lists of arrays of unequal shape are stored with
  `hexmod.io.save_npz(..., ragged=...)` as `<name>__data` and `<name>__shapes`.
- JSON: `hexmod.io.write_json` (sorted keys, numbers at full precision).
- Pickles only for the preprocessing caches and `cache/`, holding plain dicts of numpy arrays,
  numbers and strings; read them with `hexmod.io.load_cache`.
- Integer-valued float columns written with `%.17g` read back as int64 (for example `moveSpeed`
  of the Fig 8D sweep); code that needs floats converts explicitly.
- All writers replace a file atomically (a temporary file in the same folder, renamed on
  success) and give it the usual permissions, read and write for everyone minus the umask
  (`hexmod.io.FILE_MODE`, 0644 with umask 022).

### 2.4 Units

Positions and distances in cm (simulations: virtual metres, vm), times in s unless the column
name ends in `_ms`, firing rates in Hz (spikes/s), angles in degrees unless the name ends in
`_rad`, percentages as numbers from 0 to 100 (`pct` in the name). Column names give the unit
where it is not dimensionless.

### 2.5 Statistics files

A step that computes statistics printed in the paper writes `stats/<script stem>.json`: one flat
JSON object mapping a statistic key to one number (int or float; booleans as 0 or 1; no lists,
no NaN). Keys are dot separated, start with the figure or section in lower case and end with the
statistic in its usual notation (`F`, `t`, `df`, `p`, `d`, `n`), for example `fig3c.rm_anova.F`. Each key is written by exactly one step; section 4 lists all keys.

`reference/expected_statistics.json` holds the published values:

```json
{
  "schema": "hexmod-expected-statistics/2",
  "statistics": [
    {"key": "fig3c.rm_anova.F", "step": "step12_rat_statistics",
     "kind": "printed", "printed": "F(3,11811) = 375.8", "location": "Results, Fig 3C legend",
     "value": 375.7525820528324, "tolerance": 1e-09}
  ]
}
```

| Field | Meaning |
|---|---|
| `key` | statistic key as in `stats/<step>.json` |
| `step` | script stem of the step that writes the key |
| `kind` | `printed` (appears in the paper) or `reported` (shown in the reports only) |
| `printed` | the text as printed (empty for statistics that the paper does not print) |
| `location` | where it is printed (section, figure legend, table) |
| `value` | reference value at full precision |
| `tolerance` | absolute tolerance: the check passes when `abs(computed - value) <= tolerance` (0 for counts) |

### 2.6 Run records

`hexmod.runlog.RunRecord` writes `logs/<script stem>.json` with: `step`, `status`
(`ok` or `failed`), `error` and `traceback` (when failed), `start` and `end` (ISO 8601 with time zone),
`runtime_s`, `workers`, `argv`, `platform`, `versions` (python, hexmod, numpy, scipy, pandas,
scikit-image, matplotlib, tqdm), `inputs` and `outputs` (each `{path, bytes, sha256}`) and
`notes` (step-specific counts). Paths start with `results/` (for example
`results/rat/cohort/cells.csv`), with `data/` (raw data folder) or with `repo/`
(shipped inputs); a file reached through a symbolic link inside one of these folders (for example a
data folder of links to the raw files) is recorded under that folder. The record of D2 is `results/logs/data_check.json` and names its output
`results/data_check.json`.

### 2.7 Rounded per-cell inputs

Some statistics of the paper use per-cell values rounded as they are when written to a CSV file
and read back. Two helpers in `hexmod.io` apply this rounding:

- `csv_float_roundtrip(values, times)`: values written with pandas' default float format and read
  back with pandas' default float parser (`float_precision="high"`), which returns the
  neighbouring double for roughly one value in five; `times` counts successive round trips.
- `array_text_roundtrip(array)`: a per-field array stored in a CSV cell as numpy's text form
  (8 decimals) and parsed back.

**Inputs.** The steps apply them to these inputs, which reproduces the reference values bit for
bit (printed values are not affected):

| Step | Value | Transformation |
|---|---|---|
| 03 | `orientation_deg` passed to the fits as the grid orientation | `csv_float_roundtrip`, times 1 |
| 03 | `field_rate_hz` passed to the fits | `array_text_roundtrip` |
| 05 | fits of the re-characterised cells | none (exact values) |
| 05 | `alpha_original` (`alpha_tprevXloc_all` of step 03, copied into `shift_cells.csv`) | `csv_float_roundtrip`, times 1 |
| 10 | `orientation_deg` and preferred direction (`Rayleigh_angle`) of the Fig 2D cells | `csv_float_roundtrip`, times 2 |
| 10 | `Rayleigh_angle` in the per-animal Rayleigh tests of Fig 2E | `csv_float_roundtrip`, times 3 |

With these, per-animal R6, Zar p and Holm p of Fig 2E and the Supp Fig 2 tests of `alpha_original`
equal the reference values bit for bit; the Fig 2D permutation p is identical with or without the
transformation, and r then agrees with the reference to 5e-16 relative (the remaining difference
comes from the numerical library build).

**Tables.** A table holds the rounded values where they are the values that entered a statistic,
and the exact values where the statistics of the step use the exact values:

| Step | File | Columns | Written as |
|---|---|---|---|
| 05 | `rat/prospective/shift_cells.csv` | `alpha_original` | `csv_float_roundtrip(alpha_tprevXloc_all, 1)`, the values of the Supp Fig 2 tests |
| 10 | `rat/conjunctive/fig2b_cells.csv` | `orientation_deg`, `Rayleigh_angle`, `Rayleigh_mag` | `csv_float_roundtrip(cohort value, 2)`; `deviation_deg` and `aligned` are computed from the transformed values |
| 10 | `rat/conjunctive/fig2d_cells.csv` | `orientation_deg`, `direction_deg` | `csv_float_roundtrip(cohort value, 2)`, the values of the Fig 2D test; `deviation_deg` from them |
| 10 | `rat/conjunctive/fig2e_cells.csv` | `Rayleigh_angle` | `csv_float_roundtrip(cohort value, 2)`; the per-animal tests apply one more round trip to these values (times 3 in total) |
| 12 | `rat/figure_tables/cells_thirds.csv` | `alpha`, `beta`, `proximity_diff` | exact values of `param_cells.csv`, the values of the Fig 4D and Fig 5 statistics |
| 12 | `rat/figure_tables/thirds_edges.csv` | `edge_0` to `edge_3` | edges of the exact values (equal to the `fig4d.scale_edges_cm.*`, `fig5.alpha_edges.*` and `fig5.proximity_edges.*` statistics) |
| 12 | `rat/figure_tables/supp5_widths.csv` | `Rayleigh_mag`, `circular_sd_deg` | exact cohort values |

Statistics computed from result files read with `hexmod.io.read_csv` need no transformation. The
Supp Fig 1 paired test (`supp1.odd_vs_even.t`, `.p`) uses the exact values of `param_cells.csv`;
one `csv_float_roundtrip` of the odd- and even-minute alphas would change t by 4.4e-16 and p by
2.1e-17 (the printed values are the same), and `reference/expected_statistics.json` holds the
values from the exact inputs. CSV files are compared after exact parsing (`hexmod.io.read_csv`,
`float_precision="round_trip"`): with pandas' default parser, identical values can appear to differ
in the last digit.

### 2.8 Step script interface

Every step is `scripts/stepNN_<name>.py` with `main(argv=None) -> int` (0 on success) and
`if __name__ == "__main__": sys.exit(main())`. Common options: `--workers N` for steps with
parallel work (default `HEXMOD_WORKERS` or 4) and `--smoke` (synthetic inputs, see the smoke test).
A step checks its results folder with `hexmod.config.guard_smoke_results` (section 2.1), wraps its
work in `hexmod.runlog.RunRecord(<script stem>, workers=..., argv=argv)`, registers every input and
output file, and writes its statistics with `hexmod.io.write_json`. Functions run in worker processes live in the `hexmod` package, not in
the scripts (see CONVENTIONS.md, parallel work).

## 3. Steps

### D1. Download the datasets

`python scripts/data_download.py` downloads and verifies the files of the six datasets into the
layout below (sizes and checksums in `inputs/manifests/files_<key>.csv`): `vollan2025` from EBRAINS,
`brun2008`, `hafting2005` and `sargolini2006` from the NIRD Research Data Archive, and `barry2012` and
`olafsdottir2016` from Zenodo (doi:10.5281/zenodo.23477461; one HDF5 file per dataset with the sessions
analysed in the paper). Interrupted downloads resume; files that are already present and valid are
skipped. Writes nothing under `results/`.

| Option | Meaning |
|---|---|
| `--datasets all\|KEYS` | `all` (default) or comma-separated dataset keys |
| `--dest DIR` | data folder (default `HEXMOD_DATA` or `<repo>/data`) |
| `--list` | dry run: files, sizes and addresses |
| `--check-remote` | compare the archives' file listings with the manifests and send one HEAD request per dataset (no data are downloaded) |
| `--sargolini-full-archive` | download all three volumes of the split Sargolini archive (2,367,241,994 bytes) instead of the last volume only (270,089,994 bytes); the same 150 files are extracted |
| `--only PATTERN` | only the files whose name or path matches the shell pattern (repeatable; the completeness check is then skipped) |
| `--jobs N` | parallel downloads (default 2) |
| `--retries N` | further attempts per file after a failure (default 5) |
| `--trust-existing` | accept files already present with the right size without computing their checksum |
| `--delete-archives` | remove the archives after successful extraction |

The default download is 13,639,845,431 bytes (13.64 GB, about 23 min at 10 MB/s); with
`--sargolini-full-archive` it is 15,736,997,431 bytes.

```
data/
  vollan2025/sharing_v4/navigation/of/<rat>_<n>.mat            31 files
  vollan2025/sharing_v4/documentation/mat_files.md
  brun2008/archive/Brun2008-v3.0.0.zip                         archive (may be deleted after extraction)
  brun2008/Brun2008-v3.0.0/square/recordings/*.mat             6 files from the archive
  hafting2005/<5 files>.mat
  sargolini2006/archive/gridcell_data_sargolini.zip            last archive volume (may be deleted after extraction)
  sargolini2006/subset/<150 files>                             exactly the listed subset
  barry2012/barry2012_familiar_box_sessions.h5                 57 sessions
  olafsdottir2016/olafsdottir2016_open_field_sessions.h5       24 sessions
```

### D2. Check the data

`python scripts/data_check.py` verifies every expected file (size and sha256, and md5 where the
manifest gives one). Options: `--quick` (sizes only), `--sha256` (compare sha256 checksums, the
default), `--data DIR` (default `HEXMOD_DATA` or `<repo>/data`), `--workers N` (files checked in
parallel, default 2). The exit status is 0 when every dataset passes, 1 otherwise. With checksums
the check takes seconds when the files are in the page cache and a few minutes from a cold disk.

`results/data_check.json`:

| Key | Type | Meaning |
|---|---|---|
| `ok` | bool | every dataset passes |
| `checksums` | str | `size` or `sha256` (what was compared besides presence) |
| `datasets.<key>.status` | str | `ok`, `missing`, `incomplete`, `checksum mismatch` or `unexpected files` (every listed file is valid, but `sargolini2006/subset/`, which must hold exactly the listed files, holds others) |
| `datasets.<key>.n_expected`, `.n_ok` | int | files expected and verified |
| `datasets.<key>.bad_files` | list of str | paths of the files that failed, relative to the data folder |
| `datasets.<key>.problems` | object | `{path: reason}` for the files that failed (`missing`, `wrong size`, `checksum mismatch`) |
| `datasets.<key>.extra_files` | list of str | files in `sargolini2006/subset/` that are not in the manifest |
| `datasets.<key>.bytes` | int | total size of the verified files |

Run record: `results/logs/data_check.json`.

### 01. Preprocessing

`python scripts/step01_preprocess.py --workers 4`

Reads the raw files of the six datasets and `inputs/manifests/sessions.csv`, `units.csv`; before
any work it checks that every raw file is present with the size of its manifest. Options:
`--workers`, `--data DIR` (raw-data folder; overrides `HEXMOD_DATA`), `--smoke` (see below). With 2
workers and the raw files in the page cache the step takes about 40 s (35 s of them for vollan2025);
from a cold disk add the time to read about 12 GB. Peak memory is about 1.7 GB in the main process
plus 1.5 GB per worker. Writes:

**`preprocessed/<dataset>.pkl`** (one per dataset): the recordings of one dataset
before smoothing, as a plain nested dict

```
{animal_id: {session_id: session}}          insertion order = canonical order
session = {                                  keys listed by meaning; their insertion order differs by
                                             dataset (see the docstring of hexmod.preprocessing.<dataset>)
  "animal_id": str, "session_id": str,
  "position": {"xy": (n, 2) float64 [cm] (float32 for vollan2025),
               "time": (n,) float64 [s],
               "sample_rate": number [Hz] (see below),
               "xy2": (n, 2) float64 [cm]          sargolini2006 only (second LED)},
  "spike_times": {cell_id (int): (k,) float64 [s]},   insertion order = manifest unit order
  "unique_cell_ids": {cell_id (int): str},
  optional keys as provided by the dataset:
    "trial_id": str (sargolini2006; a numpy.str_ for olafsdottir2016),
    "mod_id": {cell_id: int or None} (vollan2025, hafting2005),
    "tetrode_ids": {cell_id: int} (hafting2005),
    "time": (n,) float64 [s], equal to position["time"] (vollan2025)
}
```

Times are on each dataset's own clock as in the published analysis (documented in
docs/PROVENANCE.md). `sample_rate` keeps the value and type derived from the raw file, because
`1 / sample_rate` is the sample interval used later: Python int 50 (vollan2025, brun2008,
hafting2005, barry2012), numpy float64 such as 49.999999998908606 (sargolini2006) or a 0-d uint8
array (olafsdottir2016). Theta phase, head direction and EEG are not kept (no analysis uses them).
The positions of olafsdottir2016 (`xy`, (n, 2)) are stored in Fortran order, the layout of the
matrix in the original MATLAB data; all other arrays are in C order.

**`preprocessed/trajectories.pkl`**: the six datasets merged and smoothed. Same nested
layout and insertion order as above (datasets in canonical order), with `position` holding:

| Key | Shape, dtype | Content |
|---|---|---|
| `xy` | (n, 2), as in the dataset | equal to `xy_smoothed` (the merged data store the smoothed positions under both keys) |
| `xy_smoothed` | (n, 2), as in the dataset | NaN interpolated linearly over the sample index, then a 20-sample boxcar (`scipy.ndimage.uniform_filter1d`, size 20, mode `nearest`) per coordinate [cm] |
| `time` | (n,) float64 | [s] |
| `sample_rate` | as above | [Hz] |
| `speed` | (n,) float64 | running speed from successive smoothed positions [cm/s] |
| `move_dir` | (n,) float64 (float32 for vollan2025) | movement direction in [0, 2 pi) [rad] |
| `xy2` | (n, 2) float64 | sargolini2006 only, unsmoothed |

Spikes, labels and optional keys are carried over unchanged from the per-dataset files. For the
sessions of the session manifest this file equals the merged trajectory file of the published
analysis (same keys, dtypes, values and dict insertion order), except that no session carries the
theta phase or the raw EEG (`theta_phases`, `eeg`, `eeg_time`), which no analysis reads. Load with
`hexmod.io.load_cache`.

**`preprocessed/sessions.csv`**: one row per preprocessed session (all sessions, also those
that later fail the cohort rule), in canonical order.

| Column | dtype | Meaning |
|---|---|---|
| `dataset`, `animal_id`, `session_id` | str | identifiers |
| `source` | str | the manifest `raw_path` of the session (relative to `<data>/<dataset>/`; hafting2005 lists its two files joined by `;`; for barry2012 and olafsdottir2016 the dataset's HDF5 file, which holds the session as the group named `session_id`) |
| `n_units` | int | units in the session |
| `n_samples` | int | position samples |
| `sample_rate_hz` | float | sampling rate |
| `duration_s` | float | last minus first position time |

Run on its own, `step01_preprocess.py --smoke` writes synthetic recordings instead, under the
labels of the three hafting2005 sessions (`preprocessed/hafting2005.pkl`, `trajectories.pkl` and
`sessions.csv`): 23 grid cells (random spacing 35 to 60 cm, orientation and phase, peak rate 12 Hz,
Bernoulli spikes per sample), 12 min of random walk at 50 Hz in a 100 cm box, no adaptation. The
smoke run of `run_pipeline.py` does not run this step; it writes the recordings of
`hexmod.synthetic` instead (section 1).

### 02. Cohort and characterisation

`python scripts/step02_cohort.py --workers 8`

Reads `preprocessed/trajectories.pkl`. Cohort rule: sessions with at least 30,000 position
samples with speed above 5 cm/s; every unit of these sessions is characterised (rate map,
autocorrelogram, gridness, scale, orientation, fields, movement-direction tuning; peak distance
2/3 of the scale, prominence 0.4) and kept when its gridness is at least 0.3: 3,938 cells, 32
animals, 112 sessions.

**`rat/cohort/cells.csv`**: one row per grid cell, canonical order.

| Column | dtype | Unit | Meaning |
|---|---|---|---|
| `row`, `animal_id`, `session_id`, `cell_id`, `unique_cell_id`, `dataset` | | | section 2.2 |
| `gridness` | float | | gridness score |
| `scale_cm` | float | cm | grid scale |
| `orientation_deg` | float | deg | grid orientation, [0, 60) |
| `Rayleigh_mag` | float | | resultant length of the movement-direction tuning curve (0 to 1) |
| `Rayleigh_angle` | float | deg | preferred movement direction, [0, 360) |
| `meanFR_all`, `meanFR_infield`, `meanFR_outfield` | float | Hz | mean rate overall, in fields, outside fields |
| `n_fields` | int | | number of delineated fields |

Conjunctive cells are those with `Rayleigh_mag > 0.3` (1,171).

**`rat/cohort/dir_maps.npy`**: (n_cells, 72) float64, row-aligned with `cells.csv`. Movement
direction tuning curve: 72 bins of 5 deg (centres 2.5 to 357.5 deg), 3-bin circular boxcar on
spike counts and dwell, in spikes per position sample.

**`rat/cohort/field_locations.npz`**: `row` (n,) int64 and ragged `field_locations`, one
(k, 3) float64 array per cell: field centre coordinates (cm, the two columns in the order used by
the field-entry code) and field radius (cm).

**`rat/cohort/cells_full.npz`**: `row` (n,) and ragged per-cell arrays:

| Name | Shape per cell | Content |
|---|---|---|
| `rate_map` | (nx, ny) float64 | rate map, first index along x [Hz] |
| `x_edges`, `y_edges` | (nx + 1,), (ny + 1,) | bin edges [cm] |
| `field_rate_hz` | (k,) | mean rate in each field's region (input of the adaptation fits) [Hz] |
| `field_deviation` | (k,) | deviation of each field centre from the fitted perfect lattice, as returned by the field-deviation measure (Supp Fig 3B) |

**`rat/cohort/sessions.csv`**: one row per cohort session, that is per session with at least one grid
cell (112, the same sessions that pass the sample rule; the run record note
`n_sessions_passing_sample_rule` counts the latter), canonical order: `dataset`,
`animal_id`, `session_id`, `n_units` (int), `n_cells` (int, grid cells kept), `n_conjunctive`
(int), `n_samples` (int), `n_samples_speed_above_5` (int), `duration_s` (float).

**`rat/cohort/table1.csv`**: one row per dataset in the order of Table 1 (`brun2008`,
`barry2012`, `olafsdottir2016`, `hafting2005`, `sargolini2006`, `vollan2025`) and a final row
`total`: `dataset`, `label`, `n_animals`, `n_sessions`, `n_cells`, `n_conjunctive` (int),
`scale_min_cm`, `scale_max_cm` (measured, float), `environment`, `grid_scale_text` (text from
`inputs/rat/table1_text.csv`).

**`stats/step02_cohort.json`**: cohort keys of section 4.

### 03. Parametric adaptation and HexMod fits

`python scripts/step03_adaptation_parametric.py --workers 8`

Reads `trajectories.pkl`, `cells.csv`, `field_locations.npz`, `cells_full.npz` (`field_rate_hz`).
Settings: 5 cm location bins; adaptation time constant 2 s; standard t_prev rule; adaptation types
`tprev` and `tprevXloc`; masks `all`, `oddMinute`, `evenMinute`, `infield`; HexMod fit with mask
`all`. Grid orientation and field rates enter the fits as described in section 2.7. Option
`--no-alpha-bounds` skips the fit-limit diagnostic below (about 4 of the 14 minutes of the step at
3 workers).

**`rat/adaptation/param_cells.csv`**: one row per cell, canonical order, 47 columns. Columns
`row`, `animal_id`, `session_id`, `cell_id`; then, for each mask M in the order `all`,
`oddMinute`, `evenMinute`, `infield` (outer loop) and each type T in the order `tprev`,
`tprevXloc` (inner loop), the five columns `speed_T_M`, `alpha_T_M`, `converge_T_M`,
`align_T_M`, `misalign_T_M` (so `speed_tprev_all` to `misalign_tprevXloc_all`, then the
`oddMinute` columns):

| Column | dtype | Meaning |
|---|---|---|
| `speed_T_M` | float | speed coefficient of the fit |
| `alpha_T_M` | float | adaptation coefficient (negative: rate lower soon after the last visit) |
| `converge_T_M` | bool | optimiser converged |
| `align_T_M`, `misalign_T_M` | float | mean of the adaptation regressor (proximity exp(-t_prev / 2 s) for `tprev`, multiplied by the location input for `tprevXloc`) over the samples of mask M running within 15 deg of a grid axis, and over those further from the axes |

and finally `speed_all`, `beta_all` (HexMod amplitude: rate x (1 + beta cos 6(theta - phi))), `converge_all`.

**`rat/adaptation/alpha_bounds.csv`** (diagnostic, not printed): one row per cell, canonical order:
`row`, `animal_id`, `session_id`, `cell_id`, then for each combination C = `<T>_<M>` (masks and
types in the order above) `alpha_lower_C`, `alpha_upper_C` (float, the limits within which the fit
keeps alpha so that the modelled rate stays positive: `[max(-1/max(a), 1/min(a)),
min(-1/min(a), 1/max(a))]` of the normalised regressor a), and finally `at_bound_C` (bool, alpha
equal to a limit) for each C.

**`stats/step03_adaptation_parametric.json`**: the counts of cells with alpha on a limit (section 4).

### 04. Firing rate by time since the last field visit

`python scripts/step04_adaptation_nonparametric.py --workers 8`

Reads `trajectories.pkl`, `cells.csv`, `field_locations.npz`. Non-parametric model of location,
speed and t_prev (5 cm location bins, 8 speed bins of equal sample count, t_prev bins 0, 1, 2, 3,
inf s, standard t_prev rule), masks `all` and `infield`. The model uses neither the grid orientation
nor the field rates.

**`rat/adaptation/fr_percent_change.npy`**: (n_cells, 4) float64, row-aligned with `cells.csv`:
firing rate in the t_prev bins 0 to 1, 1 to 2, 2 to 3 and above 3 s, as percent change from the
cell's mean rate (mask `all`).
**`rat/adaptation/fr_percent_change_infield.npy`**: the same for in-field samples (not printed).

### 05. Prospective coding control

`python scripts/step05_prospective_coding.py --workers 8`

Reads `trajectories.pkl`, `cells.csv`, `param_cells.csv`. Spikes shifted by -400 to +380 ms in
steps of 20 ms (40 shifts); best shift = shift of maximal spatial information; cells
re-characterised with spikes shifted by their best shift (3,754 of 3,938 remain grid cells) and
refitted (type `tprevXloc`, masks `all`, `infield`, `outfield`).

| File | Content |
|---|---|
| `rat/prospective/shift_ms.npy` | (40,) int64, the shifts in ms |
| `rat/prospective/si_shift.npy` | (n_cells, 40) float64, spatial information [bits/spike] per shift, row-aligned with `cells.csv` |
| `rat/prospective/peak_shift.csv` | all cells: `row`, `animal_id`, `session_id`, `cell_id`, `peak_shift_ms` (int) |
| `rat/prospective/shift_cells.csv` | cells kept after re-characterisation (3,754), canonical order: `row`, `animal_id`, `session_id`, `cell_id`, `peakShift_SI` (int, ms), `gridness`, `scale_cm`, `meanFR_all`, `meanFR_infield`, `meanFR_outfield`, then `speed_tprevXloc_M`, `alpha_tprevXloc_M`, `converge_tprevXloc_M`, `align_tprevXloc_M`, `misalign_tprevXloc_M` for M in `all`, `infield`, `outfield`, and `alpha_original` (from step 03) |
| `rat/prospective/fr_percent_change_shifted.npy` | (3,754, 4) float64, as step 04 with shifted spikes, row-aligned with `shift_cells.csv` |
| `rat/prospective/shift_field_locations.npz` | `row` (3,754,) int64 and ragged `field_locations` (k, 3) of the re-characterised cells (as in `rat/cohort/field_locations.npz`), row-aligned with `shift_cells.csv`; read by step 07 |

### 06. Field visit sequences, Fig 3F

`python scripts/step06_visit_sequences.py --workers 4`

Reads `trajectories.pkl` and, of `cells.csv`, the columns `row`, `animal_id`, `session_id`,
`cell_id`, `unique_cell_id`, `dataset`. Uses the visit extraction that produced the printed values
by default (`--rule printed`: spike-count rate maps, gap-aware t_prev capped at 10 s, recording gaps
above 5 s, re-delineated fields). `--rule library` uses the binary rate map (the cohort fields) and
the standard t_prev rule of the other rat analyses; its tables go to `rat/visits/library_rule/` (same
four files), its run record to `logs/step06_visit_sequences_library_rule.json`, and it writes no
statistics file. `--force` extracts the field visits again. A cell whose visit extraction raises
an error stops the step with the number of such cells and the first error (also in the run record,
`cells_with_errors`); `--allow-failures` continues without them and does not store the event cache.

Event cache: `cache/visit_events/visit_events_<rule>.pkl` (about 270 MB per rule), a plain dict
`{"metadata", "events", "cells"}` of numpy arrays (load with `hexmod.io.load_cache`). It is reused
while the cohort identifiers, the size and modification time of `trajectories.pkl`, the rule
settings, the hexmod version and the source code of the extraction modules (sha256 of
`hexmod.rat.visit_sequences`, `characterise`, `fields`, `gridcell` and `maps`) are unchanged.
Runtime: about 4 min at 2 workers (visit extraction 2.5 min, sequences 1.5 min); with
a valid cache about 1.5 min. The Fig 3F tests are computed with the paired cells in the order of the
text key `animal|session|cell`, which reproduces the reference values bit for bit; computed from
`paired_cells.csv` (canonical order) they agree to 3.1e-15 relative.

| File | Content |
|---|---|
| `rat/visits/paired_cells.csv` | 3,576 cells with events in both groups, canonical order: `row`, `animal_id`, `session_id`, `cell_id`, `unique_cell_id`, `dataset`, `frr_T23_lt_T12`, `frr_T23_gt_T12` (firing-rate ratio of the third visit when the 2 to 3 interval is shorter, or longer, than the 1 to 2 interval), `n_events_T23_lt_T12`, `n_events_T23_gt_T12` (int), `paired_difference`, `antisymmetry_offset` |
| `rat/visits/summary.csv` | one row: `n_paired_cells`, `mean_lt`, `t_lt`, `p_lt`, `d_lt`, `mean_gt`, `t_gt`, `p_gt`, `paired_diff`, `paired_t`, `paired_p`, `paired_d`, `pct_positive` |
| `rat/visits/nesting.csv` | `cohort` (`all`, `vollan_only`, `without_vollan`), `level` (`cell`, `session`, `animal`, `dataset`), `n`, `mean_paired_difference`, `sem_paired_difference`, `t`, `p_two_sided` |
| `rat/visits/diagnostics.csv` | one row per variant of the visit-sequence analysis (`analysis`: `triplet_0to10_observed_ge2`, `triplet_0to10_expected_ge2_fr23`, `quadruplet_0to10_expected_ge2_fr23`; the last row is the analysis of Fig 3F): `analysis`, `n_events`, `n_cells_any_group`, `n_paired_cells`, `mean_T23_lt_T12`, `mean_T23_gt_T12`, `paired_difference`, `paired_t`, `paired_p`, `antisymmetry_offset`, `antisymmetry_t`, `antisymmetry_p`, `speed3_minus2_cm_s`, `speed3_minus2_t`, `speed3_minus2_p`, `duration3_minus2_s`, `duration3_minus2_t`, `duration3_minus2_p`, `field_wall_distance3_minus2_cm`, `field_wall_distance3_minus2_t`, `field_wall_distance3_minus2_p` (differences between visits 3 and 2 with their t and p) |
| `stats/step06_visit_sequences.json` | Fig 3F keys |

### 07. Adaptation timescales, Supp Fig 4

`python scripts/step07_adaptation_timescales.py --workers 4`

Reads `trajectories.pkl`, `cells.csv`, `field_locations.npz`,
`inputs/rat/timescale_fit_settings.json`, and for the shifted variant `shift_cells.csv` and
`shift_field_locations.npz` of step 05. Nine t_prev bins with edges `exp(linspace(-3, 3, 10))` s
(0.050 to 20.1 s); valid-range mask on; fits of one and two exponentials with the settings of the
settings file, unweighted and weighted by 1 / sem^2 (of the starting values `p0` and
`extra_starts` the fit with the lowest residual sum of squares is kept); animal bootstrap with a
fixed seed, `hexmod.rat.timescales.PUBLISHED_BOOTSTRAP_SEED`, the seed of the published intervals
(seed and number of resamples in the run record notes). With it the printed one-exponential
interval (variant `unshifted`) is reproduced to 1e-6 s; the two-exponential fits of the resamples
depend on their starting values, and the two-exponential interval ends and the share of resamples
that prefer two exponentials agree with the values of the published bootstrap to 0.017 s and 0.1
point (the upper end of the fast time constant is 0.568 s, printed 0.56 s). The shifted variant
continues the random stream of the unshifted one; with `--variant shifted` alone the draws of the
unshifted resamples are first skipped (over the animals of `rat/timescales/finebin_percent_change.npy`
of an earlier unshifted run), so its bootstrap is the same as with `--variant both`. Options:
`--variant {both,unshifted,shifted}` (default both), `--n-boot N` (default 1000), `--seed N`
(default the published seed). Working files, read by no step: `cache/timescales/finebin_percent_change_shifted.npy`
((3,754, 9), row-aligned with `shift_cells.csv`) and `cache/timescales/bootstrap_draws_<variant>.npz`
(per-resample `tau_single`, `tau_fast`, `tau_slow`, `dAIC_weighted`). Two variants: `unshifted` (the 3,938 cells with their spikes as recorded; the
printed values) and `shifted` (the 3,754 cells of step 05 with every spike shifted by the cell's
best shift; reported, not printed).

| File | Content |
|---|---|
| `rat/timescales/finebin_percent_change.npy` | (n_cells, 9) float64, percent change per bin, row-aligned with `cells.csv` |
| `rat/timescales/supp4_points.csv` | 9 rows: `bin` (int, 1 to 9), `edge_low_s`, `edge_high_s`, `x_s` (abscissa used by the fits), `cells_in_bin`, `mean_pct_change`, `sem_pct_change`, `ci95_low`, `ci95_high`, `weight` (1 / sem^2) |
| `rat/timescales/supp4_fits.csv` | `model` (`one exponential`, `two exponentials`), `parameter`, `value` |
| `rat/timescales/fits.json` | `<variant>.<weighting>.<key>` with variant `unshifted`, `shifted` and weighting `unweighted`, `weighted_by_SEM` (the printed fits are `unshifted.weighted_by_SEM`). Keys: `single` (`a`, `tau`, `c`, `SS`), `double` (`a_fast`, `tau_fast`, `a_slow`, `tau_slow`, `c`, `SS`), `AIC_single`, `AIC_double`, `AICc_single`, `AICc_double`, `BIC_single`, `BIC_double`, `dAIC_double_minus_single`, `dAICc`, `dBIC`, `extra_SS_F`, `extra_SS_p`, `n_points` (int). `SS` is the (weighted) residual sum of squares, the chi2 of the weighted fits; times in s |
| `rat/timescales/bootstrap.json` | `<variant>.<key>` (variants as above) with `n_cells`, `n_animals`, `n_boot_ok` (int, resamples with converged fits), `tau_single_95`, `tau_fast_95`, `tau_slow_95`, `dAIC_weighted_95` ([low, high] percentile intervals), `tau_slow_median`, `share_slow_over_10s`, `share_boot_double_preferred_by_AIC` (fractions from 0 to 1) |
| `stats/step07_adaptation_timescales.json` | Supp 4 keys |

### 08. Numbers quoted in the Methods

`python scripts/step08_methods_numbers.py --workers 8`

Reads `trajectories.pkl`, `cells.csv`, `field_locations.npz`, `fr_percent_change.npy` and
`cells_full.npz` (only `x_edges`: the arena of a session is the median over its cohort cells of the
last x edge of the rate map; 100 cm arenas are those with 85 < size < 115 cm).

`rat/methods/methods_numbers.json`:

| Key | Content |
|---|---|
| `intervals_by_scale_third.<third>.n_cells`, `.n_events` (int) | cells and field-visit intervals of the scale third (`small`, `medium`, `large`) |
| `intervals_by_scale_third.<third>.scale_range_cm` | [min, max] grid scale of the third [cm] |
| `intervals_by_scale_third.<third>.pooled_median_s`, `.pooled_iqr_s` | median and [q25, q75] of all intervals of the third [s] |
| `intervals_by_scale_third.<third>.median_of_cell_medians_s` | median over cells of each cell's median interval [s] |
| `scale_third_edges_cm` | [lower, upper] scale edges between the thirds [cm] |
| `running_speed_100cm_arenas.n_sessions` (int), `.speed_range_cm_s` ([min, max]), `.mean_speed_cm_s` | running speed in the 100 cm arenas |

Statistics in `stats/step08_methods_numbers.json`, including the ratio of the 0 to 1 s bin to the
above-3 s bin (`methods.ratio_0_1s_to_above_3s`), which is not stored in `methods_numbers.json`.

### 09. Repeated recordings of the same cell

`python scripts/step09_same_cell_links.py --workers 4`

Reads `cells.csv`, `trajectories.pkl` (only for the tetrode datasets), `inputs/manifests/sessions.csv`
(raw file of each Vollan session) and `units.csv` (Hafting tetrode and cell labels from
`raw_unit_key`), and the raw Vollan files of the animals with cohort cells in more than one session
(16 files of 7 animals); unit features read from these files are cached in
`cache/step09_same_cell_links/` (`--no-cache` reads them again). Links recordings of the same unit
(across exports of one recording, across sessions of one day, across tetrode recordings) and keeps
the earliest recording of each unit.

| File | Content |
|---|---|
| `rat/same_cell/entries.csv` | one row per cell (all cohort cells), canonical order: `row`, `animal_id`, `session_id`, `cell_id`, `unit` (int, unit identifier under the link set used in the paper), `n_entries` (int, recordings of that unit), `kept_earliest` (bool, this row is the earliest recording of its unit) |
| `rat/same_cell/units.csv` | one row per unit: `unit`, `animal_id`, `dataset`, `kept_row`, `n_entries`, `rows` (rows of its recordings joined by `;`) |
| `rat/same_cell/links.csv` | one row per accepted link: `row_a`, `row_b` (nullable int: empty when that end is a Vollan unit that is not a cohort grid cell; such units can connect two cohort recordings), `session_a`, `cell_a`, `session_b`, `cell_b` (both ends, also outside the cohort), `rule` (`same_recording`, `waveform_and_map`, `map_only`, `tetrode_map`, `tetrode_label`), `evidence` (text), `score` (coincident-spike fraction for `same_recording`; rate-map r plus waveform r for `waveform_and_map`; rate-map r for `map_only` and `tetrode_map`; empty for `tetrode_label`). Rows: Vollan links by animal and file pair, then the tetrode links |

### 10. Conjunctive cells, Fig 2B to 2E

`python scripts/step10_conjunctive_cells.py --workers 8`

Reads `cells.csv`, `dir_maps.npy`, `entries.csv`, `inputs/rat/module_labels.csv`.

| File | Content |
|---|---|
| `rat/conjunctive/fig2b_cells.csv` | the 1,171 conjunctive cells: `row`, `animal_id`, `session_id`, `cell_id`, `dataset`, `orientation_deg`, `Rayleigh_angle`, `Rayleigh_mag`, `deviation_deg` (wrap60(Rayleigh_angle - orientation_deg)), `aligned` (bool, abs(deviation_deg) < 15), `kept_earliest` |
| `rat/conjunctive/fig2b_bins.csv` | 12 rows: `bin`, `deviation_from_deg`, `deviation_to_deg` (from included, to excluded), `half` (`aligned` or `misaligned`), `count` |
| `rat/conjunctive/fig2c_per_animal_curves.csv` | `form` (`normalised_to_cell_mean`, `not_normalised`), `animal_id`, `n_cells`, `in_mean_across_animals` (bool, at least 5 cells), `bin` (0 to 72), `direction_deg` (2.5 to 362.5, first bin repeated), `mean_curve` |
| `rat/conjunctive/fig2c_mean.csv` | `form`, `bin`, `direction_deg`, `mean_across_animals`, `sem_across_animals` (73 rows per form) |
| `rat/conjunctive/fig2d_cells.csv` | the 890 cells of Fig 2D (KL divergence of the smoothed tuning curve at least 0.15, earliest recording), canonical order: `row`, `animal_id`, `session_id`, `cell_id`, `dataset`, `module` (int), `orientation_deg`, `direction_deg` (preferred direction), `kl`, `deviation_deg`, `conjunctive` (bool) |
| `rat/conjunctive/fig2e_per_animal.csv` | 12 Vollan animals with at least 5 conjunctive cells (earliest recordings), sorted by `animal_id` as text: `animal_id`, `n`, `R6` (resultant length of 6 x preferred direction), `Z`, `p_rayleigh` (Zar), `R_crit_p05`, `phase6_deg`, `p_holm`, `R6_chance95` (95th percentile of R6 under uniform directions, band of Fig 2E) |
| `rat/conjunctive/fig2e_cells.csv` | the 1,082 cells: `row`, `animal_id`, `Rayleigh_angle` |
| `rat/conjunctive/fig2e_band.csv` | chance band over cell number: `n` (int), `R_chance_mean`, `R_chance95`, for every n drawn in the figure |
| `stats/step10_conjunctive_cells.json` | Fig 2B to 2E keys |

Fig 2D uses `hexmod.stats.fisher_lee_permutation` with 20,000 permutations and the key
`"q2sel_kl|one|all|dir_mean"`; the band uses `hexmod.stats.rayleigh_monte_carlo`.

### 11. Example cells

`python scripts/step11_example_cells.py`

Reads `trajectories.pkl`, `cells.csv`, `cells_full.npz`, `field_locations.npz`, `dir_maps.npy`,
`shift_ms.npy`, `si_shift.npy`, `peak_shift.csv`, `inputs/rat/example_cells.csv`. Writes `rat/examples/`:

| File | Content |
|---|---|
| `fig2a_cells.csv` | the five Fig 2A cells: `row`, `session_id`, `cell_id`, `Rayleigh_mag`, `Rayleigh_angle`, `orientation_deg`, `deviation_deg` |
| `fig2a_cells.npz` | `dir_maps` (5, 72) and `rows` (5,) |
| `fig3ab.npz`, `fig4ab_small.npz`, `fig4ab_large.npz` | per-sample arrays of the whole session: `x`, `y` [cm], `t` [s], `spikes` (int64 counts per sample), `speed` [cm/s], `movedir_rad`, `t_prev` [s], `in_field` (int64), `location_input`, `adaptation_variable`; Fig 4 files only: `grid_alignment` (cos((movedir_rad - orientation_deg * pi / 180) * 6), evaluated on the direction samples at their stored precision, float32 for vollan2025, then converted to float64; the curve of Fig 4B) and `movedir_rad_pm_pi` (`movedir_rad` with values above pi reduced by 2 pi; the direction axis of Fig 4B); maps: `rate_map`, `x_edges`, `y_edges`; fields: `field_locations` (k, 3), `field_rate_hz` (k,), `field_order` (k,) int64 (number of each field in the figure, by first visit in the window, 0 = not visited); scalars `time_window` (2,) [s], `gridness`, `scale_cm`, `orientation_deg`, `row` |
| `fig3ab_field_visits.csv`, `fig4ab_small_field_visits.csv`, `fig4ab_large_field_visits.csv` | one row per field visit: `t_enter`, `t_exit` [s], `field` (int), `t_prev` [s], `field_x`, `field_y`, `radius` [cm], `direction_prev_deg`, `duration_s`, `rate_hz`, `rate_prev_outfield_hz` |
| `supp2a_shifted_maps.npz` | `shift_ms` (40,), `spatial_information` (40,) of the example cell; for the best shift s of the cell (120 ms), 0 and -s (prefixes `shift_p<s>ms__`, `shift_0ms__`, `shift_m<s>ms__`; for the published cell `shift_p120ms__`, `shift_0ms__`, `shift_m120ms__`): `spike_x`, `spike_y`, `spike_count`, `field_locations`, `x_edges`, `y_edges`, `gridness`, `scale_cm`, `orientation_deg` |
| `supp3a_cells.csv` | 12 rows: `panel` (1 to 12), `row`, `session_id`, `cell_id`, `gridness`, `scale_cm`, `orientation_deg`, `rate_map_max_hz`, `env_size_cm`, `origin_x_cm`, `origin_y_cm`, `conjunctive` (bool) |
| `supp3a_cells.npz` | for each panel `pNN__rate_map`, `pNN__x_edges`, `pNN__y_edges`, `pNN__field_locations`, `pNN__origin` |

### 12. Rat statistics and figure tables

`python scripts/step12_rat_statistics.py`

Reads the outputs of steps 02 to 11. Computes every printed rat statistic not written by steps
02, 06, 07, 08 and 10 (Fig 3C, 3D, 4C, 4D, 5, Supp 1, 2, 3B to 3D, 5).

| File | Content |
|---|---|
| `rat/figure_tables/cells_thirds.csv` | all cells, canonical order: `row`, `alpha` (alpha_tprevXloc_all), `proximity_diff` (`align_tprev_infield` minus `misalign_tprev_infield`), `beta` (beta_all), `adaptation_third` (1 strong, 2 weak, 3 no adaptation), `proximity_third` (1 bottom to 3 top), `scale_third` (1 small to 3 large), `env_size_cm`, `rate_map_max_hz` |
| `rat/figure_tables/thirds_edges.csv` | `quantity` (`alpha`, `proximity_diff`, `scale_cm`), `edge_0` to `edge_3` (percentile edges, top edge raised by 0.001) |
| `rat/figure_tables/supp3bcd_arrays.npz` | `deviation_small` (12,667,), `rate_normalised_small` (12,667,), `mean_rate_small` (1,178,), `deviation_large` (4,082,), `rate_normalised_large` (4,082,), `mean_rate_large` (1,199,), `rows_small`, `rows_large` |
| `rat/figure_tables/supp5_widths.csv` | the 1,171 conjunctive cells: `row`, `Rayleigh_mag`, `circular_sd_deg` (degrees(sqrt(-2 ln Rayleigh_mag))), the widths of Supp Fig 5B |
| `rat/figure_tables/supp3bcd_fits.csv` | `module` (`small`, `large`), `distribution` (`deviation`, `rate_across_cells`, `rate_within_cell`), `shape`, `scale` (gamma fit with location 0), `n_values` |
| `stats/step12_rat_statistics.json` | keys of section 4 for Fig 3C, 3D, 4C, 4D, 5, Supp 1, 2, 3, 5 |

Small module: scale at most 0.4 of the environment size and gridness above 0.3; large module:
scale at least 0.6 of the environment size.

### 21. Simulated grid cell pools

`python scripts/step21_simulation_pools.py [--regenerate-large]`

Small module (40 vm) built with `hexmod.sim.pool` (100,000 cells, base seed 0); large module
(70 vm) copied from `inputs/simulation/large_module_pool_70cm.npz`, or rebuilt with
`--regenerate-large`. A pool that fails its checksum or tolerance check is written as
`<name>.failed.npz` (an earlier file of the standard name is removed) and the step ends with exit
status 1. Steps 22 to 24 read the pools with `hexmod.sim.io_pool.load_module_pool`, which requires
100,000 cells outside a smoke run.

| File | Content |
|---|---|
| `sim/pools/small_pool_40cm.npz`, `sim/pools/large_pool_70cm.npz` | pool in the format of `hexmod.sim.io_pool` (scalar columns as arrays of length 100,000; ragged columns `centers` (k, 2), `peaks_x`, `peaks_y`, `radius`, `peakFR_jitter` as `<name>__data`, `<name>__shapes`) |
| `sim/pools/pool_checksums.json` | one entry per pool (`small_module_pool_40cm`, `large_module_pool_70cm`) as written by `hexmod.sim.io_pool.pool_summary`, the layout of the entries of `inputs/simulation/pool_checksums.json`: `n_cells`, `column_order`, `columns.<column>` (`kind`, `dtype`, `sha256`; ragged columns also `ndim`, `n_values`), `file_bytes`, `file_sha256`. The column checksums (`hexmod.sim.io_pool.column_checksums`) equal those of `inputs/simulation/pool_checksums.json`; a large pool rebuilt with `--regenerate-large` agrees with the shipped one within tolerances, not bit for bit (absolute: field centres 1e-12 vm, peak rates 1e-14 Hz, gridness 1e-9; the other columns exactly except the orientation estimate `gridori_est`, which is only reported; measured on a Linux workstation: 2.8e-14, 1.7e-16, 9.7e-12) |

### 22. Simulated maps and trajectory, Fig 6A to 6C, Supp 3E, Fig 8A

`python scripts/step22_simulation_maps.py`

| File | Content |
|---|---|
| `sim/fig6/trajectory_seed0.csv` | 23,990 rows: `time` [s], `x`, `y` [vm], `moveSpeed` [vm/s], `moveDir` [deg, 0 to 360], `trialId` (int), `expPhase` (int: -1 cue or inter-trial interval, 0 turning, 1 moving) |
| `sim/fig6/trajectory_seed0_objects.json` | `objLocs` (object locations of block 1, 5 x [x, y] in vm), `objLocs_by_block` (4 x 5 x [x, y] in vm), `block_duration_s` (600.0), `n_blocks` (4), `env_size_vm` ([x, y]) |
| `sim/fig6/maps.npz` | `extent` (4,) [vm]; `small__example_maps`, `large__example_maps` (6, 34, 34) [Hz]; `small__example_cells`, `large__example_cells` (6,) pool indices; `seed0__small`, `seed0__large`, `seed1__small`, `seed1__large` (34, 34) summed maps of participants 0 and 1 [percent of the mean]; `supp3e__small__maps`, `supp3e__large__maps` (6, 34, 34) [Hz]; `supp3e__small__cells`, `supp3e__large__cells` (6,) pool indices; `supp3e__small__gridness`, `supp3e__large__gridness` (6,); `supp3e__small__peak_rate_hz`, `supp3e__large__peak_rate_hz` (6,); `supp3e__gridness_class` (6,) str (`low`, `low`, `medium`, `medium`, `high`, `high`) |
| `sim/fig8/fig8a_ideal_grids.npz` | for `small` (30 vm), `medium` (50 vm), `large` (70 vm): `<name>__field_locations` (k, 3), `<name>__centers` (m, 2) [vm], `<name>__grid_scale_vm` (0-d int) |

### 23. Proximity of simulated cells, Fig 6D

`python scripts/step23_simulation_proximity.py`

`sim/fig6/fig6d_small.csv`, `sim/fig6/fig6d_large.csv`: 10,000 rows (cells of participant 0):
`cell_index` (pool index), `aligned`, `misaligned` (mean proximity exp(-t_prev / 2 s) when running
within 15 deg of a grid axis, and further from the axes). Both tables equal
`results_precomputed/sim/fig6/fig6d_small.csv.gz` and `fig6d_large.csv.gz` exactly.

### 24. Simulated participants, Fig 7

`python scripts/step24_simulation_participants.py --module small --seeds 0:1500 --workers 10`
(and `--module large`; `--quick` runs seeds 0:30). Participant s draws its 10,000 cells with
`default_rng(s).choice(100000, 10000, replace=False)` and its trajectory with base seed s. One
participant takes about 85 s (small module) or 65 s (large module) on one core; all 3,000 take
about 62 worker hours (6 hours with 10 workers). A rerun skips a participant whose file is present
and records the same number of cells, pool and code version (`--force` recomputes).

`sim/fig7/participants/<module>/p0000.npz` (one file per participant, `p` plus 4 digits):
every entry of the participant result: `seed`, `module` (0-d str), `alpha`, `tau`, `n_samples`,
`n_moving`, `FR_adapt`, `FR_noadapt` (n_moving,) [Hz], `FR_adapt_mean_move`,
`FR_noadapt_mean_move`, `tprevby12dir` (12,), `tprevbyAlignMis_nanmean` (2,), `tprev_HMness`,
`ratemap_sum` (34, 34), `gridness_mean`, `foldSymList` (int array, the symmetries 1 to 7), and
the two fit tables stored column by column as `crossval__<column>` (112 rows: 2 models x 2
estimators x 4 train and test splits x 7 symmetries) and `noncross__<column>` (4 rows), and how the
file was computed: `meta_n_cells` (int), `meta_pool_fingerprint` (`hexmod.sim.io_pool.pool_fingerprint`
of the pool) and `meta_hexmod_version`. The symmetries are those of the participant results of the
paper (`hexmod.sim.simulate.FOLD_SYM_LIST`); asking for more symmetries adds rows and leaves those of
1 to 7 unchanged. `sim/fig7/participants/<module>/manifest.csv`: `participant`, `file`, `sha256`,
`meta_n_cells`, `meta_pool_fingerprint`, `meta_hexmod_version`.

### 25. Fig 7 values

`python scripts/step25_simulation_collect.py [--experiments 49|50] [--source run|precomputed] [--quick]`

Reads the participant files of step 24 when all 1,500 per module exist, otherwise
`results_precomputed/sim/fig7/participants_<module>.csv.gz`. Participant files are used only when
each records 10,000 cells, the pool that step 24 uses now (its fingerprint, from
`sim/pools/pool_checksums.json` or for the shipped large pool `inputs/simulation/pool_checksums.json`)
and this version of the code; otherwise the step stops with a message. With `--quick` (passed by
`run_pipeline.py --quick`), participants 0 to 29 come from the files of `step24 --quick` and
participants 30 to 1,499 from the shipped tables; the run record notes the source (`quick`). An experiment is a block of 30
participants; Fig 7F uses participants 30 to 1,499 (49 experiments) by default and participants
0 to 1,499 (50 experiments) with `--experiments 50`. Everything below can be rebuilt exactly from
the participant tables; row blocks are converted to C order before row-wise means. The shipped
`results_precomputed/sim/fig7/` holds the five figure tables and `fig7_values.json` as written by
`--source precomputed` (byte for byte), so that Fig 7 can be drawn without running this step.

| File | Content |
|---|---|
| `sim/fig7/participants_small.csv.gz`, `participants_large.csv.gz` | 3,000 rows, method-major (`simple` rows for participants 0 to 1,499, then `location-controlled`): `module`, `participant` (int), `method` (`simple`, `location-controlled`), `dir_0` to `dir_330` (12 movement-direction bins, firing rate as plotted in Fig 7A and 7D, percent change from the mean), `fold_1` to `fold_7` (cross-validated HexMod coefficient for 1- to 7-fold symmetry), `orientation_deg_mod60`, `rate_dir_0_hz` to `rate_dir_330_hz` (the 12 direction bins in Hz, input of the Fig 7A and 7D tests), `mean_rate_moving_hz` |
| `sim/fig7/fig7ad_direction_bins.csv` | participants 0 to 29: `module`, `method`, `participant`, `dir_0` to `dir_330` |
| `sim/fig7/fig7be_beta.csv` | participants 0 to 29: `module`, `method`, `participant`, `fold_1` to `fold_7` |
| `sim/fig7/fig7c_orientation.csv` | participants 0 to 29: `module`, `method`, `participant`, `orientation_deg_mod60` |
| `sim/fig7/fig7f_detection.csv` | `module`, `method`, `source`, `experiments`, `participants`, `rate_fold_1` to `rate_fold_7`, `count_fold_1` to `count_fold_7` |
| `sim/fig7/fig7f_p_per_experiment.csv` | `module`, `method`, `source`, `experiment`, `p_fold_1` to `p_fold_7` |
| `sim/fig7/fig7_values.json` | flat `{key: number}`, keys below; identical to `results_precomputed/sim/fig7/fig7_values.json` when computed from the shipped participant tables |

Keys of `fig7_values.json` (module `small`, `large`; method `simple`, `location_controlled`;
k = 1 to 7):

| Key | Content |
|---|---|
| `fig7a.<module>.t`, `.df`, `.p`, `.n` | paired t-test, participants 0 to 29, simple method: mean of the even against the odd direction bins of `rate_dir_*_hz` |
| `fig7d.<module>.t`, `.df`, `.p`, `.n` | the same for the location-controlled method |
| `fig7b.<module>.fold_<k>.t`, `.df`, `.p`, `.significant` | one-sample t-test of `fold_<k>` against 0 (two-sided), participants 0 to 29, simple method; `significant` is 1 when p < 0.05, else 0 |
| `fig7e.<module>.fold_<k>.t`, `.df`, `.p`, `.significant` | the same for the location-controlled method |
| `fig7c.<module>.<method>.count_<deg>` | participants 0 to 29 whose six-fold orientation (6 x `orientation_deg_mod60`) falls in the 30 deg bin centred on `<deg>` (0 to 330) |
| `fig7f.<module>.<method>.fold_<k>.count`, `.rate` | experiments (participants 30 to 1,499) in which the one-sided (greater) one-sample t-test of `fold_<k>` gives p < 0.05, and their share |
| `fig7f.n_experiments`, `fig7f.first_participant`, `fig7f.last_participant` | 49, 30, 1,499 |
| `fig7f_50.<module>.<method>.fold_<k>.count`, `.rate` | the same over 50 experiments |
| `fig7f_50.n_experiments`, `fig7f_50.first_participant`, `fig7f_50.last_participant` | 50, 0, 1,499 |
| `fig7.<module>.n_participants`, `fig7.participants_per_experiment`, `fig7.alpha_threshold` | 1,500, 30, 0.05 |

### 26. Running-speed sweep, Fig 8D

`python scripts/step26_speed_sweep.py --workers 10`

The runs are independent; each is simulated by the worker function `hexmod.sim.speed_sweep.sweep_run`.

`sim/fig8/fig8d_sweep.csv`: 111 rows (3 grid scales x 37 speeds): `scale` (grid scale in vm: 30,
50, 70), `scaleLevel`, `speedLevel` (int), `moveSpeed` [vm/s] (integer-valued, reads back as
int64), `FR_align`, `FR_misalign` [Hz], `proximity_align`, `proximity_misalign`, `n_time` (int),
`FRby12dir_0` to `FRby12dir_330`, `FR_mis_minus_align_pct`, `proximity_align_minus_mis`, `plotted`
(bool, point drawn in Fig 8D). Runtimes are not stored in the table.

### 27. Simulation statistics

`python scripts/step27_simulation_statistics.py`

Reads the outputs of steps 22 to 26 and computes the analytic values of the Fig 1 legend and of
Fig 8B/8C. Writes `stats/step27_simulation_statistics.json` (Fig 1, 6, 7, 8 keys). A missing input
stops the step unless `--allow-precomputed` is given, which takes it from `results_precomputed/sim/`;
the run record lists the source of every input (`sources`), and step 40 reports it.

### 30. Figures

`python scripts/step30_figures.py [--figure Fig3]`

Reads only `results/` and `results_precomputed/`. Figures: `Fig2` to `Fig8`, `SuppFig1` to
`SuppFig5`, each drawn by `hexmod.figures.<module>.build(results_dir, out_dir)` (modules
`fig2` to `fig8`, `supp1` to `supp5`). Figure 1 is a schematic and is not drawn by the code.

| Option | Meaning |
|---|---|
| `--figure NAME` | figure to draw (`Fig2` to `Fig8`, `SuppFig1` to `SuppFig5`, or `all`, the default); repeatable, also as a comma-separated list |
| `--results DIR` | results folder (default `HEXMOD_RESULTS` or `<repo>/results`); figures go to `DIR/figures/` |
| `--precomputed [DIR]` | take the simulation results from `results_precomputed/` (or DIR) even when `results/sim/` holds them; without it, the shipped files are used only for files missing under `results/sim/` |
| `--font-dir DIR` | folder with the Arial TrueType files (default `HEXMOD_FONT_DIR`); without Arial the figures use Liberation Sans or DejaVu Sans, with a warning |
| `--strict` | count figures that cannot be drawn (module or input files missing) as failures |
| `--smoke` | smoke run: draws whatever the synthetic results allow |
| `--list` | list the figures and their modules and exit |

How a run ends:

- Each figure is drawn into `figures/.<Figure>.building/`, which replaces `figures/<Figure>/` only
  when the build succeeds and its output passes the check (PDF, SVG and PNG written,
  `positions.csv` present, no dash characters in the SVG, CSV, JSON or text files). A failed
  build keeps the previous output.
- A figure whose module is not available or whose input files are missing
  (`hexmod.figures.common.MissingInput`) is reported as skipped; when an earlier output of that
  figure exists, the console and the run record (`stale_output`) say that it was kept and may be
  stale. The exit status is 0 unless a figure failed or, with `--strict`, a figure was skipped
  (the summary line then lists the skipped figures as failed).
- The run record `logs/step30_figures.json` lists the inputs and outputs of every drawn figure and
  notes the font, the font files, the matplotlib version, whether the shipped simulation results
  were preferred, and the status of every figure (with the reason and runtime).

For each figure F:

| File | Content |
|---|---|
| `figures/F/F.pdf`, `F.svg`, `F.png` | the figure (PDF with embedded TrueType fonts, SVG with text as text, PNG at 600 dpi) |
| `figures/F/positions.csv` | `element` (`axes` or `text`), `panel`, `name`, `text`, `x0_mm`, `y0_mm`, `x1_mm`, `y1_mm` (from the bottom left of the figure) |
| `figures/F/source_data/<F><panel>_<description>.csv` | the values drawn in each panel |

### 40. Statistics check

`python scripts/step40_check_statistics.py [--compare-frozen DIR]`

Merges all `stats/*.json` and compares each key with the entry of the same key in
`reference/expected_statistics.json`. It also checks that every computed value of a printed
statistic rounds to the printed number (`printed_agreement`; the known differences between the text
and the analysis are listed with their reasons in `KNOWN_PRINTED_DIFFERENCES`); this check is
reported in `statistics_check.md` and the console, and does not change the exit status. The report
also lists where the simulation results came from (the run records of steps 25 and 27). With
`--smoke` the step only reports (synthetic values are not expected to match) and exits with 0.

| File | Content |
|---|---|
| `stats/statistics_check.csv` | `key`, `step`, `printed`, `expected`, `computed`, `abs_diff`, `tolerance`, `status` (`pass`, `fail`, `missing`, `no reference`) |
| `stats/statistics_check.md` | the same as a readable report, with the printed values that do not round to their computed values and the sources of the simulation results |
| `stats/frozen_comparison.csv` | with `--compare-frozen DIR`: `path`, `status` (`equal`, `close`, `different`, `missing`), `max_abs_diff`, `max_rel_diff`, `rtol`, `atol`, `detail`, for every file of DIR |
| `stats/frozen_comparison.md` | the same as a readable report |

`--compare-frozen` without a folder uses `HEXMOD_FROZEN`; `--reference FILE` selects another
reference file. Parts of the compared folder that this pipeline does not write are left out of the
comparison and named in its report (`NOT_COMPARED` in the script): the `theta_phases` entries of the
preprocessed recordings, sessions of the preprocessed recordings that are not in the session manifest,
and columns of a table that the table of this run does not have. The exit status is 1 when a statistic
fails or is missing, or when a compared file is different or missing.

## 4. Statistic keys

One key per number; the step that writes it is the file name in `stats/`. Pair labels for
t_prev bins: `0_1s`, `1_2s`, `2_3s`, `above_3s`. Folds k = 1 to 7, modules `small`, `large`,
methods `simple`, `location_controlled`. Unless noted, `p` is two-sided and `d` is Cohen's d
(mean over SD with ddof = 1). Keys marked "not in the reference file" are written to the
statistics files but have no entry in `reference/expected_statistics.json` (step 40 lists them as
`no reference`): the reference file holds only printed statistics, constants of the analysis and
simulation statistics.

**`step02_cohort`**

| Key | Printed |
|---|---|
| `cohort.n_animals`, `cohort.n_sessions`, `cohort.n_cells` | 32, 112, 3,938 |
| `cohort.<dataset>.n_animals`, `.n_sessions`, `.n_cells` (each dataset key) | Table 1 counts (B 1/2/5, W 7/57/118, F 6/24/61, H 2/3/23, S 1/2/7, V 15/24/3,724) |
| `cohort.conjunctive.n_cells`, `.n_animals`, `.n_sessions` | 1,171 (21 animals, 55 sessions) |
| `table1.<dataset>.scale_min_cm`, `table1.<dataset>.scale_max_cm` | measured ranges (not in the reference file) |
| `cohort.rule.min_samples_above_speed`, `cohort.rule.speed_threshold_cm_s`, `cohort.rule.min_running_time_min` | the session rule: 30,000 samples faster than 5 cm/s, 10 min at 50 Hz (printed: sessions of at least 10 minutes) |

**`step03_adaptation_parametric`**

| Key | Printed |
|---|---|
| `diagnostic.alpha_at_bound.<T>_<M>.n_cells`, `.n_lower`, `.n_upper`, `.n_total` (8 combinations of type and mask) | not printed (not in the reference file): cells whose alpha equals its lower or upper fit limit, for example 550 and 129 of 3,938 for `tprevXloc_all` |

**`step10_conjunctive_cells`**

| Key | Printed |
|---|---|
| `fig2b.n_cells`, `fig2b.n_aligned`, `fig2b.n_misaligned`, `fig2b.pct_aligned` | 1,171; 564 (48%); 607 |
| `fig2b.binomial.p` | 0.2197 (exact binomial, two-sided) |
| `fig2c.n_animals`, `fig2c.n_cells` | 15, 1,160 |
| `fig2c.circular_sd.median_deg`, `.q25_deg`, `.q75_deg` | 73.06 (IQR 66.0 to 80.0); over the 1,171 cells |
| `fig2d.n_cells`, `fig2d.n_animals`, `fig2d.n_sessions`, `fig2d.n_conjunctive` | 890, 21, 47, 888 |
| `fig2d.fisher_lee.r`, `fig2d.permutation.p`, `fig2d.permutation.n_perm` | 0.00126, 0.164 (one-sided), 20,000 |
| `fig2e.n_animals`, `fig2e.n_cells` | 12, 1,082 |
| `fig2e.n_p_below_0_05`, `fig2e.p_below_0_05.1`, `.2`, `.3` (ascending) | 3 (0.019, 0.037, 0.046) |
| `fig2e.min_p_holm` | 0.231 |
| `fig2e.binomial.p` | 0.020 (P(X >= 3), 12 animals, 0.05) |

**`step12_rat_statistics`**

| Key | Printed |
|---|---|
| `fig3c.mean_pct_change.0_1s`, `.1_2s`, `.2_3s`, `.above_3s` | -6.19, +0.39, +2.39, +2.84 % |
| `fig3c.rm_anova.F`, `.df1`, `.df2`, `.p` | F(3,11811) = 375.8 |
| `fig3c.posthoc.<a>_vs_<b>.t`, `.df`, `.p` (6 pairs) | all p < 1e-9 except `2_3s_vs_above_3s`: t(3937) = -1.55, p = 0.12 |
| `fig3c.posthoc.<a>_vs_<b>.p_holm` (6 pairs) | Holm correction within the panel; the three pairs of the first bin give the brackets of Fig 3C (*** p < 0.001); the others are not in the reference file |
| `fig3d.n_cells`, `fig3d.t`, `fig3d.df`, `fig3d.p`, `fig3d.d`, `fig3d.pct_negative` | t(3937) = -41.09, d = -0.65, 77.0% |
| `fig4c.mean_aligned`, `fig4c.mean_misaligned`, `fig4c.pct_higher` | 0.32285, 0.32068, 0.68% |
| `fig4c.t`, `fig4c.df`, `fig4c.p` | t(3937) = 11.07 |
| `fig4d.scale_edges_cm.0` to `.3` | 27.6, 61.8, 86.1, 154.8 |
| `fig4d.n.small`, `.medium`, `.large` | 1,313, 1,312, 1,313 |
| `fig4d.anova.F`, `.df1`, `.df2`, `.p` | F(2,3935) = 7.06, p = 0.00087 |
| `fig4d.posthoc.<a>_vs_<b>.t`, `.df`, `.p`, `.p_holm` (a, b in `small`, `medium`, `large`) | Student t-tests between the thirds, Holm within the panel; `p_holm` gives the brackets of Fig 4D, the other keys are not in the reference file |
| `fig5.alpha_edges.0` to `.3`, `fig5.proximity_edges.0` to `.3` | -0.160 / -0.0614 / -0.0182 / 0.171; -0.0658 / -0.00281 / 0.00604 / 0.0542 |
| `fig5a.t`, `fig5a.df`, `fig5a.p` | t(3937) = -0.24, p = 0.81 |
| `fig5b.strong.n`, `.t`, `.df`, `.p`, `.d` | 435, t(434) = -2.845, p = 0.0047, d = -0.137 |
| `fig5b.weak.n`, `.t`, `.df`, `.p` | 430, t(429) = 0.08, p = 0.93 |
| `fig5b.no.n`, `.t`, `.df`, `.p` | 448, t(447) = 0.82, p = 0.41 |
| `fig5b.anova.F`, `.df1`, `.df2`, `.p` | F(2,1310) = 4.0, p = 0.018 |
| `fig5b.posthoc.<a>_vs_<b>.t`, `.df`, `.p`, `.p_holm` (a, b in `strong`, `weak`, `no`; top proximity third) | Student t-tests, Holm within the panel; `p_holm` gives the brackets of Fig 5B, the other keys are not in the reference file |
| `fig5c.strong_bottom.n`, `.t`, `.df`, `.p` | 443, t(442) = 1.36, p = 0.17 |
| `fig5c.strong_top_vs_bottom.t`, `.df`, `.p` | Student t(876) = -3.00, p = 0.003 |
| `supp1.odd.t`, `supp1.even.t` | -35.4, -31.2 |
| `supp1.odd.df`, `.p`, `supp1.even.df`, `.p` | significantly negative in each half (p < 1e-190); df not in the reference file |
| `supp1.odd_vs_even.t`, `.df`, `.p` | t(3937) = -2.40, p = 0.016 |
| `supp1.pearson_r`, `supp1.pearson_df`, `supp1.pearson_p` | r(3936) = 0.31, p < 0.001 |
| `supp2.n_cells`, `supp2.n_cells_kept` | 3,938; 3,754 |
| `supp2.median_best_shift_ms`, `supp2.example_best_shift_ms` | 40, 120 |
| `supp2.rm_anova.F`, `.df1`, `.df2`, `.p` | F(3,11259) = 294.7 |
| `supp2.alpha_original_vs_shifted.t`, `.df`, `.p` | t(3753) = -4.13 |
| `supp2.alpha_original.d`, `supp2.alpha_shifted.d` | -0.667, -0.611 |
| `supp2.posthoc.<a>_vs_<b>.t`, `.df`, `.p`, `.p_holm` (6 pairs of t_prev bins, shifted spikes) | paired t-tests, Holm within the panel; `p_holm` of the three pairs of the first bin gives the brackets of the panel, the other keys are not in the reference file |
| `supp3.<module>.n_cells`, `supp3.<module>.n_fields` | small 1,178 / 12,667; large 1,199 / 4,082 |
| `supp3.<module>.<distribution>.shape`, `.scale` | small 2.023/0.066, 3.152/1.422, 10.315/0.097; large 2.075/0.044, 3.350/1.091, 7.102/0.141 (distribution `deviation`, `rate_across_cells`, `rate_within_cell`) |
| `supp5.n_cells` | 1,171 |
| `supp5.narrowest_sd_deg`, `supp5.narrowest_sixfold_pct` | 43.3 deg, 0.3% |
| `supp5.median_sd_deg`, `supp5.median_sixfold_pct` | 73.06 deg, 0.003% |

**`step06_visit_sequences`**

| Key | Printed |
|---|---|
| `fig3f.n_cells` | 3,576 |
| `fig3f.shorter.mean`, `.t`, `.df`, `.p` | 0.0076, t(3575) = 2.53, p = 0.012 |
| `fig3f.longer.mean`, `.t`, `.df`, `.p` | -0.0065, t = -2.10, p = 0.036 |
| `fig3f.paired.mean_diff`, `.t`, `.df`, `.p` | 0.014, t = 2.90, p = 0.0037 |

**`step07_adaptation_timescales`**

| Key | Printed |
|---|---|
| `supp4.n_cells`, `supp4.n_cells_last_bin` | 3,938; 3,936 |
| `supp4.n_bins`, `supp4.bin_edges.first_s`, `supp4.bin_edges.last_s` | nine bins from 0.05 to 20 s (last edge 20.09 s) |
| `supp4.single.dof`, `supp4.double.dof` | 6 and 4 residual degrees of freedom |
| `supp4.single.tau_s`, `supp4.single.chi2` | 0.4526 s, 84.0 |
| `supp4.double.tau_fast_s`, `supp4.double.tau_slow_s`, `supp4.double.chi2` | 0.300 s, 2.643 s, 20.8 |
| `supp4.delta_aic`, `supp4.delta_bic` | about -59 |
| `supp4.f_test.F`, `.df1`, `.df2`, `.p` | F(2,4) = 6.06, p = 0.062 |
| `supp4.bootstrap.tau_single_low_s`, `_high_s` | 0.31 to 0.81 s |
| `supp4.bootstrap.tau_fast_low_s`, `_high_s` | 0.21 to 0.57 s (printed 0.56 s) |
| `supp4.bootstrap.tau_slow_low_s`, `_high_s` | 1.28 s to the 50 s bound (printed 1.3 to more than 10 s) |
| `supp4.bootstrap.pct_double_preferred`, `supp4.bootstrap.n_resamples` | 99.0% (printed 99%) of 1,000 resamples |

**`step08_methods_numbers`**

| Key | Printed |
|---|---|
| `methods.interfield_median_s.small_third` | 1.44 (printed 1.4 s) |
| `methods.running_speed_cm_s`, `methods.running_speed.n_sessions` | 19.5 (printed 19 cm/s), 55 (not in the reference file) |
| `methods.ratio_0_1s_to_above_3s` | 0.923 (alpha 0.1, about 8% reduction) |
| `methods.tau_s`, `methods.shift_min_ms`, `methods.shift_max_ms` | 2 s; -400 and +380 ms |

**`step27_simulation_statistics`**

| Key | Printed |
|---|---|
| `fig1.narrow.circular_sd_deg`, `fig1.narrow.kappa`, `fig1.narrow.sixfold` | 15 deg, 15.1055, 0.5929 |
| `fig1.broad.circular_sd_deg`, `fig1.broad.sixfold` | 67.46 deg (cosine tuning); six-fold component 0 (a cosine curve has no six-fold term) |
| `fig6c.large_p0.max_pct`, `.min_pct`, `.sd_pct` | +3.99, -2.63, 1.49 |
| `fig6d.<module>.mean_aligned`, `.mean_misaligned`, `.t`, `.df`, `.d` | small t 175.8, d 1.76; large 0.0853, 0.0684, t(9999) = 91.7, d 0.92 |
| every key of `sim/fig7/fig7_values.json` (step 25), copied unchanged | the printed ones below |
| `fig7a.<module>.t`, `.df`, `.p`, `.n` | small -23.6257, large -7.0936 (df 29) |
| `fig7d.<module>.t`, `.df`, `.p`, `.n` | small -43.3605, large -11.4143 |
| `fig7b.<module>.fold_<k>.t`, `.df`, `.p`, `.significant` | significant: small 6; large 2 and 3 |
| `fig7e.<module>.fold_<k>.t`, `.df`, `.p`, `.significant` | significant: 6 in both modules |
| `fig7f.n_experiments`, `fig7f.first_participant`, `fig7f.last_participant` | 49 (participants 30 to 1,499) |
| `fig7f.<module>.<method>.fold_<k>.count`, `.rate` | large, simple: 19, 41, 17, 12, 6, 39, 1 of 49 |
| `fig7f_50.<module>.<method>.fold_<k>.count`, `.rate` | over 50 experiments: large, simple, two-fold 42, six-fold 39 (reported) |
| `fig8.v_opt_scale_per_tau` | 1.3327 (v_opt = 1.3327 scale / tau) |
| `fig8d.n_runs` | 111 |

**`step40_check_statistics` (derived)**

Printed numbers that follow from other statistics by a fixed formula. Step 40 computes them when no
step writes them (`DERIVED` and `DERIVED_FROM_FILES` in the script; formulas in reference/README.md)
and lists them in `stats/statistics_check.csv`.

| Key | Printed |
|---|---|
| `fig2b.pct_misaligned` | 52% |
| `fig2d.n_cells_six_largest_animals` | 766 |
| `fig2e.expected_below_0_05` | 0.6 |
| `fig3f.shorter.d`, `fig3f.longer.d`, `fig3f.paired.d` | 0.04, -0.04, 0.05 |
| `fig7a.small.d`, `fig7a.large.d` | -4.4, -1.3: Cohen's d_z of the paired test with the SD of the differences computed with ddof 0, t / sqrt(n - 1); with ddof 1, as for the other effect sizes, -4.31 and -1.30 |
| `supp5.cosine_width_vonmises_sixfold_pct` | 0.008% |
| `supp5.max_conjunctive_sd_deg` | 89 deg |
| `discussion.optimal_speed_8m_grid_m_s`, `discussion.optimal_speed_4m_grid_m_s` | 5.3 and 2.7 m/s |
