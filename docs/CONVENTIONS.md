# Conventions

Rules for code, data files and text in this repository. `tests/test_text_rules.py` checks the
text rules on every file that would be committed.

## Repository layout

- `src/hexmod/`: the library (installed with `pip install -e .`). Subpackages:
  `preprocessing` (raw files to trajectories), `rat` (analyses of the recordings), `sim`
  (simulations), `figures` (plotting). Shared helpers: `config` (paths, results folder, workers),
  `datasets` (dataset metadata), `io` (file formats), `runlog` (run records), `parallel`
  (process pool), `stats` (statistical tests), `circular` (circular helpers).
- `scripts/`: one script per pipeline step (`stepNN_<name>.py`) plus the data scripts and
  `run_pipeline.py`. Scripts parse arguments, call library functions and write results.
- `inputs/`: small shipped inputs (manifests, settings, simulation pool). No recorded data.
- `results_precomputed/`: shipped simulation results.
- `reference/`: the reference values of every statistic (`expected_statistics.json`) and their
  tolerances (README).
- `results/` and `data/` are created locally and never committed.

## Naming

- Modules, functions and variables: `lower_case_with_underscores`; classes: `CamelCase`.
  Names describe what the code does (`fit_two_factor`, `extract_field_entry`), never who wrote it.
- Datasets are named by citation key: `vollan2025`, `brun2008`, `hafting2005`,
  `sargolini2006`, `barry2012`, `olafsdottir2016` (`hexmod.datasets`). Label prefixes in the data:
  V, B, H, S, W, F (W and F keep the labels of Barry et al. (2012) and Ólafsdóttir et al. (2016)
  distinct from B, Brun et al. (2008)).
- Animal, session and unit labels in data values (for example `V10_1`, `W5`, `F1_4`) are those of
  the shipped manifests; they are data, not names.
- Step scripts: `stepNN_<name>.py`; the run record and statistics file of a step use the script
  stem (`logs/step04_adaptation_nonparametric.json`).
- Statistic keys: dot separated, beginning with the figure in lower case and ending with the
  statistic (`fig3c.rm_anova.F`); see docs/PIPELINE.md, section 4.
- Figure files: `Fig1` to `Fig8`, `SuppFig1` to `SuppFig5`; source data
  `<Figure><panel>_<description>.csv`.

## Units and angles

Distances in cm (simulations in virtual metres, vm), times in s, firing rates in Hz, angles in
degrees, percentages from 0 to 100. A column or variable holding another unit says so in its name
(`_ms`, `_rad`, `_vm`, `_pct`). Grid orientation is in [0, 60) degrees; preferred directions in
[0, 360) degrees; `hexmod.circular.wrap60` gives the deviation from the nearest grid axis in
[-30, 30). Six-fold quantities are computed on angles multiplied by 6.

## Order of cells

Per-cell tables follow the canonical order (datasets in `hexmod.datasets.CANONICAL_ORDER`, then
the manifest order of animals, sessions and units) and carry the `row` of the cell in
`rat/cohort/cells.csv`. Never sort per-cell data by name or take the order from a directory
listing: seeded permutation tests and seeded draws depend on the order.

## File formats

| Kind | Format | Write with | Read with |
|---|---|---|---|
| tables | CSV, floats as `%.17g`, no index | `hexmod.io.write_csv` | `hexmod.io.read_csv` (exact round trip) |
| arrays | `.npy`, `.npz` without object arrays | `hexmod.io.save_npy`, `save_npz` | `load_npy`, `load_npz` |
| arrays of unequal shape | `<name>__data` + `<name>__shapes` in an `.npz` | `save_npz(..., ragged=...)` | `load_npz` (returns lists) |
| statistics, settings | JSON, sorted keys | `hexmod.io.write_json` | `read_json` |
| local caches | pickle of plain dicts of numpy arrays | `hexmod.io.save_cache` | `load_cache` (accepts numpy arrays, numpy scalars and builtin containers only) |

Pandas objects are never pickled. CSV floats must be read with `float_precision="round_trip"`
(as `hexmod.io.read_csv` does); the default pandas parser can change the last digit. The writers
replace a file atomically and give it the usual permissions (`hexmod.io.FILE_MODE`, 0644 with
umask 022).

## Reproducibility

- Seeds are fixed in the code and listed in docs/PROVENANCE.md. Draws use
  `numpy.random.default_rng(seed)`; a seed derived from a text key uses
  `hexmod.stats.seed_of(key)` (crc32 modulo 2^31). Two fixed integer seeds of the published
  analysis are kept as named constants (`PUBLISHED_BOOTSTRAP_SEED`, animal bootstrap of Supp Fig 4;
  `PUBLISHED_DRAW_ORDER_SEED`, drawing order of the dots of Fig 2D); the text rules allow their
  values only on the lines listed in `SEED_CONSTANTS` of `tests/test_text_rules.py`.
- The analysis functions keep the arithmetic and order of operations of the analysis behind the
  published numbers. Options that select other variants keep the published behaviour as their
  default.
- Analysis modules do not import matplotlib; only `hexmod.figures` does.

## Parallel work

Use `hexmod.parallel.map_ordered(func, inputs, workers)`. Results come back in input order and do
not depend on the number of workers. `func` must be a module-level function in the `hexmod`
package (so that it can be imported by worker processes started with "spawn", the macOS default),
and each input item must carry everything the function needs. Step scripts call
`hexmod.config.limit_threads()` before importing numpy, so each worker uses one thread; code that
uses the library from an interactive session should do the same.

## Configuration

Command-line flags and environment variables only: `HEXMOD_DATA` (raw data, default `data/`),
`HEXMOD_RESULTS` (default `results/`), `HEXMOD_FONT_DIR` (optional Arial fonts),
`HEXMOD_WORKERS` (default 4), and for the tests marked `frozen` `HEXMOD_FROZEN` and
`HEXMOD_FROZEN_RESULTS` (a complete results folder of an earlier run; none is distributed). Paths
in code are built from these and from the repository root; no absolute paths appear in the code. All
steps read and write one results folder (`hexmod.config.results_dir`). Step 01 stops with a one-line
message, instead of a traceback, when a raw file is missing or has the wrong size; the other steps
read only the results folder, except step 09, which reads some of the Vollan et al. (2025) files
again. Results of a smoke run are kept apart: their folder is marked and
refused by steps run without `--smoke` (`hexmod.config.guard_smoke_results`).

## Text rules

These apply to every committed file: code, comments, documentation, CSV, JSON and text written
into figures.

1. **No en or em dashes.** Use a hyphen, "to" for ranges ("0 to 1 s") or a comma. In figure labels
   the minus sign (U+2212) is used for negative numbers.
2. **No personal names in code or file names.** Author and team names and initials do not appear
   in file, module, function, class or variable names, comments, docstrings or other text, except
   in `CITATION.cff`, the citation and contact sections of the README and the full references of
   the datasets (docs/DATA.md, "Citing the data", and the `citation` entries of
   `hexmod.datasets`). Published work is otherwise cited as "First author et al. (year)", for
   example "Doeller et al. (2010)". The test holds the blocked names as salted sha256 digests.
3. **Write for a journal reader.** Describe what the code does and why. No absolute or
   machine-specific paths and no references to unpublished material.
4. **No recorded data.** Nothing derived from recorded animal data is committed (no per-cell
   tables, rate maps, spike or position arrays). The reference values of the statistics check are
   summary statistics only: those printed in the paper, constants of the analysis and simulation
   statistics. Simulation inputs and outputs are committed where a step says so.
5. **No notebooks, pickles, raw-data or font files**, and no file larger than 25 MB.

## Code style

Python 3.10, plain functions and small classes, short numpy-style docstrings (one-line summary,
`Parameters`, `Returns` where useful). Line length up to 120 characters.

## Tests

`pytest -q` runs the unit tests, the text rules and the smoke run of the pipeline on synthetic
recordings. Markers: `smoke` (synthetic end-to-end run, part of CI), `data` (needs the raw
datasets), `frozen` (compares with a complete results folder of an earlier run in `HEXMOD_FROZEN`
or `HEXMOD_FROZEN_RESULTS`; see the README, section Tests), `slow` (run with `--runslow`). Tests
that need absent data are skipped.
