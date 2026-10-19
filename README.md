# Firing rate adaptation in grid cells produces macroscopic hexadirectional modulation

Analysis code for Reggio G, Kim M, Ji Z, Bush D, Barry C and Burgess N (2026). *Firing rate adaptation in grid cells produces macroscopic hexadirectional modulation*.

The repository computes every statistic printed in the paper and draws every figure (Figures 1 to 8
and Supplementary Figures 1 to 5). The rat analyses start from the raw recordings of six published
datasets (3,938 grid cells from 32 rats), all of which are publicly available; the simulations start
from the model parameters. A final step compares each statistic with its reference value and checks
that it rounds to the printed number. [docs/STATISTICS.md](docs/STATISTICS.md) lists every printed
statistic with the step that computes it and its value, and [docs/FIGURES.md](docs/FIGURES.md) lists
every figure panel with the step and the files it is drawn from.

## Quick start

```
conda env create -f environment.yml
conda activate hexmod
pip install -e .
export HEXMOD_FONT_DIR=/path/to/arial/fonts   # optional, see below
python scripts/data_download.py
python scripts/run_pipeline.py --workers 4
```

The download fetches the six datasets (13.64 GB). The pipeline then runs every step in order,
writes its results to `results/` and ends with the statistics check
(`results/stats/statistics_check.md`). With 4 workers it needs about 12 GB of memory; more workers
run faster and need more memory (see below). Install the package in editable mode (`-e`), as
above: the library finds `inputs/`, `reference/` and `results_precomputed/` relative to the
repository.

The figures are set in Arial, which cannot be distributed with the code. macOS ships Arial; on
Linux, install it (for example with the `ttf-mscorefonts-installer` package) or set
`HEXMOD_FONT_DIR` to a folder holding the Arial TrueType files. Without Arial the figures are drawn
in Liberation Sans or DejaVu Sans, with a warning, and their text differs slightly from the
published figures (see Fonts, under Reproducibility notes).

Figures without any data:

```
python scripts/step30_figures.py
```

This draws the schematic Figure 1 and, from the simulation results shipped in
`results_precomputed/`, Figures 6, 7 and 8, in a few seconds, and writes them to
`results/figures/`; the figures of recorded data are reported as skipped until the rat steps have
been run.

A smoke run runs the pipeline in about a minute and without any data:
`python scripts/run_pipeline.py --smoke`. Small synthetic recordings made with the simulator
(`hexmod.synthetic`) take the place of the outputs of steps 01 and 09, which read the raw files,
and the other steps run on them with small sizes. Its results go to the temporary folder printed on
the first line (or to `HEXMOD_RESULTS`), which is not removed afterwards. The statistics check of a
smoke run only reports: synthetic values cannot match the published ones, so many statistics are
listed as failing.

## Requirements and runtimes

- Linux or macOS with conda. `environment.yml` (conda-forge) pins Python 3.10, numpy 1.23.5,
  scipy 1.13.0, pandas 1.5.3 and scikit-image 0.20.0, the versions with which the published numbers
  were computed, and matplotlib 3.8.0 for the figures (matplotlib 3.7 also works). h5py reads the
  two datasets that are distributed as HDF5 files.
- Disk: 13.9 GB for the data (the downloads and the files extracted from the archives; 12.2 GB
  once the archives are deleted) and 2.4 GB of results.
- Memory: with `--workers 8` the run peaked at 19.6 GB, in step 26 (the main process keeps about
  2 GB; each worker needs 1 to 2.5 GB). Plan for about 12 GB with 4 workers and 24 GB with 8.

Runtimes of a run with `--workers 8` on a desktop computer (Intel i9-12900K, 16 cores, 24 threads,
46 GB of memory, no other jobs), with the raw data in the file cache:

| Step | Script | Time |
|---|---|---|
| D2 | `data_check.py` | 4 s |
| 01 | `step01_preprocess.py` | 14 s |
| 02 | `step02_cohort.py` | 35 s |
| 03 | `step03_adaptation_parametric.py` | 5.0 min |
| 04 | `step04_adaptation_nonparametric.py` | 37 s |
| 05 | `step05_prospective_coding.py` | 4.4 min |
| 06 | `step06_visit_sequences.py` | 2.1 min |
| 07 | `step07_adaptation_timescales.py` | 80 s |
| 08 | `step08_methods_numbers.py` | 22 s |
| 09 | `step09_same_cell_links.py` | 34 s |
| 10 to 12 | conjunctive cells, example cells, rat statistics | 9 s |
| 21 | `step21_simulation_pools.py` | 5.5 min |
| 22, 23 | simulated maps and proximity | 1.8 min |
| 25 | `step25_simulation_collect.py` (from the shipped Fig 7 results) | 1 s |
| 26 | `step26_speed_sweep.py` | 4.1 min |
| 27, 30, 40 | simulation statistics, figures, statistics check | 18 s |
| all | `run_pipeline.py --workers 8` (D2 to 40) | 27 min |

From a cold disk, add the time to read about 12 GB of raw data. The full simulation of the Fig 7
participants (step 24, 1,500 simulated participants per module) is skipped by default, and step 25
then uses the shipped participant results; `python scripts/run_pipeline.py --full-sim --workers 10`
includes it. It takes about 6 hours with 10 worker processes on the computer above (about 62
worker hours, 1 GB of memory per worker, 320 MB of participant files).

## Data

The analyses use grid cell recordings from six published rat datasets. All six are publicly
available and are downloaded by `scripts/data_download.py`. No recorded data are distributed with
this repository. Details, download addresses, checksums and the folder layout are in
[docs/DATA.md](docs/DATA.md).

| Dataset | Key | Grid cells (animals, sessions) | Source | Licence | Download |
|---|---|---|---|---|---|
| Vollan et al. (2025) | `vollan2025` | 3,724 (15, 24) | EBRAINS, [doi:10.25493/R5FR-EDG](https://doi.org/10.25493/R5FR-EDG) | CC BY-SA 4.0 | 11.85 GB |
| Brun et al. (2008) | `brun2008` | 5 (1, 2) | NIRD Research Data Archive, [doi:10.11582/2018.00015](https://doi.org/10.11582/2018.00015) | CC BY 4.0 | 1.44 GB |
| Hafting et al. (2005) | `hafting2005` | 23 (2, 3) | NIRD Research Data Archive, [doi:10.11582/2017.00017](https://doi.org/10.11582/2017.00017) | CC BY 4.0 | 6.1 MB |
| Sargolini et al. (2006) | `sargolini2006` | 7 (1, 2) | NIRD Research Data Archive, [doi:10.11582/2017.00019](https://doi.org/10.11582/2017.00019) | CC BY 4.0 | 270 MB |
| Barry et al. (2012) | `barry2012` | 118 (7, 57) | Zenodo, [doi:10.5281/zenodo.23477461](https://doi.org/10.5281/zenodo.23477461) | CC BY 4.0 | 47.2 MB |
| Ólafsdóttir et al. (2016) | `olafsdottir2016` | 61 (6, 24) | Zenodo, [doi:10.5281/zenodo.23477461](https://doi.org/10.5281/zenodo.23477461) | CC BY 4.0 | 27.8 MB |

The download is 13.64 GB (13,639,845,431 bytes) and needs no login;
`python scripts/data_download.py --list` shows every file, its size and its address without
downloading anything. Every file is checked against its size and sha256 checksum. The Zenodo record
holds one HDF5 file per dataset with the sessions analysed in the paper (57 sessions of Barry et al.
(2012) and 24 sessions of Ólafsdóttir et al. (2016)): the positions and the spike times of every
unit recorded in them, with the values of the original recordings. `python scripts/data_check.py`
verifies every file again and exits with status 0 when all six datasets pass.

## Pipeline

Each step is a script `scripts/stepNN_<name>.py` that reads the outputs of earlier steps, writes its
own outputs under `results/`, writes the statistics it computes to `results/stats/` and records its
inputs, outputs, software versions and runtime in `results/logs/`.
[docs/PIPELINE.md](docs/PIPELINE.md) describes every output file, its columns and its units.

| Step | Script | What it does | Used for |
|---|---|---|---|
| D1 | `data_download.py` | downloads and verifies the six datasets | |
| D2 | `data_check.py` | checks every data file | |
| 01 | `step01_preprocess.py` | reads the raw files, merges the datasets, smooths positions, computes running speed and direction | all rat analyses |
| 02 | `step02_cohort.py` | selects the grid cells and characterises them (rate map, gridness, scale, orientation, fields, direction tuning) | Table 1, cohort |
| 03 | `step03_adaptation_parametric.py` | parametric adaptation model and HexMod fits | Figs 3D, 4C, 4D, 5; Supp Fig 1 |
| 04 | `step04_adaptation_nonparametric.py` | firing rate by time since the last field visit | Fig 3C |
| 05 | `step05_prospective_coding.py` | spikes shifted in time against positions | Supp Fig 2 |
| 06 | `step06_visit_sequences.py` | firing rate on successive field visits | Fig 3F |
| 07 | `step07_adaptation_timescales.py` | fine time bins and exponential fits | Supp Fig 4 |
| 08 | `step08_methods_numbers.py` | numbers quoted in the Methods | Methods |
| 09 | `step09_same_cell_links.py` | links repeated recordings of the same cell | Figs 2D, 2E |
| 10 | `step10_conjunctive_cells.py` | preferred directions of conjunctive cells relative to the grid axes | Figs 2B to 2E |
| 11 | `step11_example_cells.py` | arrays of the example cells | Figs 2A, 3A, 3B, 4A, 4B; Supp Figs 2A, 2B, 3A |
| 12 | `step12_rat_statistics.py` | rat statistics and figure tables | Figs 3 to 5; Supp Figs 1 to 3, 5 |
| 21 | `step21_simulation_pools.py` | pools of 100,000 simulated grid cells per module | Figs 6, 7 |
| 22 | `step22_simulation_maps.py` | simulated trajectory and rate maps | Figs 6A to 6C, 8A; Supp Fig 3E |
| 23 | `step23_simulation_proximity.py` | proximity of simulated cells | Fig 6D |
| 24 | `step24_simulation_participants.py` | simulated participants (optional, about 6 hours) | Fig 7 |
| 25 | `step25_simulation_collect.py` | Fig 7 values from the participant results | Fig 7 |
| 26 | `step26_speed_sweep.py` | running-speed sweep | Fig 8D |
| 27 | `step27_simulation_statistics.py` | simulation and closed-form statistics | Figs 1, 6 to 8 |
| 30 | `step30_figures.py` | draws the figures from the result files | all figures |
| 40 | `step40_check_statistics.py` | compares every statistic with the reference values | |

`python scripts/run_pipeline.py --list` prints the steps; `--from 02 --to 12` runs a range. Steps 01
and 09 read the raw data; step 01 stops with a one-line message, before any work, when a raw file is
missing or has the wrong size. The other steps read only the results folder, so tables, figures and
the statistics check can be rebuilt without the raw data. The figures read only result files, so a
figure can be redrawn without running any analysis again, for example
`python scripts/step30_figures.py --figure Fig3`.

## Figures and statistics

- [docs/FIGURES.md](docs/FIGURES.md): for every panel, the step and result files it is drawn from,
  the figure module and the source-data files. Each figure is written as PDF (editable text), SVG
  and PNG (600 dpi) with `positions.csv` and the values of each panel in `source_data/`. The panels
  carry no titles (the figure legends of the paper name them), and the panel letters stand above
  their panels, on one line per row.
- [docs/STATISTICS.md](docs/STATISTICS.md): every printed statistic with its key, the step that
  writes it and its value.
- [reference/README.md](reference/README.md): the reference values and the tolerances of the
  statistics check.

## Reproducibility notes

- **Reference values.** Run on the raw data with the pinned environment, the pipeline gives all
  statistics of `reference/expected_statistics.json` within their tolerances, and all but seven
  agree to the last bit: six values of the animal bootstrap of Supplementary Figure 4 (the
  two-exponential refits of a resample depend on their starting values; interval ends within
  0.02 s) and the last digit of the Fisher-Lee r of Fig 2D, which depends on the numerical library
  build. Four printed numbers differ from what the analysis gives at the printed precision; step 40
  lists them with the reason ([docs/STATISTICS.md](docs/STATISTICS.md)).
- **Seeds.** Every random draw has a fixed seed in the code: the permutation test of Fig 2D, the
  chance band of Fig 2E, the animal bootstrap of Supplementary Figure 4, the simulated cell pools,
  participants, trajectories and example cells. They are listed in
  [docs/PROVENANCE.md](docs/PROVENANCE.md).
- **Analysis variants.** Every analysis uses the defaults of the library, with these exceptions,
  which the steps select themselves:
  - Fig 3F extracts field visits from fields delineated on a spike-count rate map. The time since
    the last visit is undefined for the first visit and for intervals that span a recording gap
    longer than 5 s, and it is capped at 10 s (`step06_visit_sequences.py --rule printed`, the
    default; `--rule library` uses the settings of the other rat analyses);
  - Supplementary Figure 4 uses the valid-range mask of the non-parametric model, which leaves out
    samples outside its nine time bins;
  - the simulated participants of Fig 7 use trajectory generator version 1
    (`hexmod.sim.trajectory_v1`), and Figs 6A, 6D and 8D use version 2 (`hexmod.sim.trajectory_v2`);
    the two versions draw random numbers differently;
  - Fig 7F uses 49 experiments of 30 participants (participants 30 to 1,499);
    `step25_simulation_collect.py --experiments 50` uses all 1,500 participants.
- **Preprocessing conventions.** The timing and position conventions of the published analysis are
  kept: for example, the spikes of each Sargolini et al. (2006) unit are aligned to its first spike,
  the Barry et al. (2012) spikes are paired with positions 0.02 to 0.04 s later, and spikes that fall
  in a gap of the position record longer than 1.5 samples are dropped. Each convention and its size
  is described in the docstring of its module (`hexmod.preprocessing.<dataset>`,
  `hexmod.preprocessing.trajectories`) and summarised in [docs/PROVENANCE.md](docs/PROVENANCE.md).
- **Cell order.** Animal, session and unit labels and the order of every per-cell table come from
  the manifests `inputs/manifests/sessions.csv` and `units.csv`, never from a directory listing,
  because seeded permutation tests and random draws depend on the order of the cells. The manifests
  are described in `hexmod.preprocessing.labels`.
- **Rounded inputs.** A few per-cell values enter the published statistics rounded as when they
  are written to and read from a CSV file; the steps apply the same rounding, which changes only
  the last digits ([docs/PIPELINE.md](docs/PIPELINE.md), section 2.7).
- **Fonts.** The figures are set in Arial, which cannot be distributed here. Set `HEXMOD_FONT_DIR`
  (or `--font-dir`) to a folder holding `Arial.ttf`, `Arial Bold.ttf`, `Arial Italic.ttf` and
  `Arial Bold Italic.ttf`, or install Arial. Without Arial the figures use Liberation Sans, which
  has the same letter widths, or DejaVu Sans, with a warning. In Figure 1 the colour key of panel B
  is placed for Arial; when another font brings it closer than 3 mm to the labels of the direction
  wheel (`fig1.KEY_GAP_MIN`), step 30 warns and draws the figure all the same.
- **Configuration.** Environment variables `HEXMOD_DATA` (raw data, default `data/`),
  `HEXMOD_RESULTS` (default `results/`), `HEXMOD_WORKERS` (default 4) and `HEXMOD_FONT_DIR`, and the
  command-line options of each script. Nothing else needs to be set.

## Tests

```
pytest -q
pytest -m smoke
```

`pytest -q` runs the unit tests, the text rules of [docs/CONVENTIONS.md](docs/CONVENTIONS.md) and the
smoke run of the pipeline on synthetic recordings; none of them needs data. Tests marked `data`
read the raw data in `HEXMOD_DATA`; most of them also need `HEXMOD_FROZEN` (for example `results/`
after a run of the pipeline) or `--runslow`, and are skipped otherwise. Tests marked `slow` run
with `--runslow`. Tests marked `frozen` compare with a complete results folder of an earlier run,
which is not distributed with the repository: after a run of the pipeline,

```
HEXMOD_FROZEN_RESULTS=results pytest -m frozen tests/test_plotted_vs_stats.py
```

recomputes, for every figure, the statistics from the values the figure draws and compares them
with the statistics files of that run (the smoke tests do the same on the synthetic results).
`HEXMOD_FROZEN` points the other `frozen` tests to the results folder of an earlier run. The
continuous integration workflow (`.github/workflows/tests.yml`) installs the package, then runs the
text rules, the unit tests and the smoke tests on Ubuntu and macOS with the environment of
`environment.yml`.

## Repository layout

```
README.md, LICENSE, CITATION.cff, environment.yml, pyproject.toml
src/hexmod/            the library
  preprocessing/       raw files to trajectories, one module per dataset
  rat/                 analyses of the recordings
  sim/                 simulations and closed forms
  figures/             one module per figure, shared style
  stats.py, circular.py, io.py, config.py, datasets.py, ...
scripts/               data scripts, one script per pipeline step, run_pipeline.py
inputs/manifests/      data files with checksums, session and unit labels in canonical order
inputs/rat/            module labels, example cells, Table 1 text, fit settings
inputs/simulation/     the large-module pool of simulated grid cells
results_precomputed/   shipped simulation results (Figs 6 to 8, Supp Fig 3E)
reference/             reference values of all statistics
docs/                  DATA, PIPELINE, FIGURES, STATISTICS, PROVENANCE, CONVENTIONS
tests/                 unit tests, text rules, smoke run
data/, results/        created locally, never committed
```

## Data and code availability

This paper analyses existing, publicly available data. The recordings of Vollan et al. (2025) are
available from EBRAINS (doi:10.25493/R5FR-EDG); those of Brun et al. (2008), Hafting et al. (2005)
and Sargolini et al. (2006) from the NIRD Research Data Archive (doi:10.11582/2018.00015,
doi:10.11582/2017.00017 and doi:10.11582/2017.00019); and the sessions of Barry et al. (2012) and
Ólafsdóttir et al. (2016) analysed here from Zenodo (doi:10.5281/zenodo.23477461). All original code
is available at https://github.com/gullirg/hexmod-grid-cell-adaptation under the MIT licence. It
downloads these data, computes every statistic and draws every figure of the paper, and includes
the simulated grid cell pool and the simulation results (CC BY 4.0).

## Licences

- Code: MIT licence ([LICENSE](LICENSE)).
- Shipped simulation data, `inputs/simulation/` and `results_precomputed/`: Creative Commons
  Attribution 4.0 (CC BY 4.0), see the `LICENSE` file in each folder and their READMEs
  ([inputs/simulation/README.md](inputs/simulation/README.md),
  [results_precomputed/README.md](results_precomputed/README.md)).
- Recorded data: the datasets keep their own licences (Vollan et al. (2025): CC BY-SA 4.0; Brun et
  al. (2008), Hafting et al. (2005), Sargolini et al. (2006), Barry et al. (2012) and Ólafsdóttir et
  al. (2016): CC BY 4.0). No recorded data or per-cell results are distributed; the repository holds
  only label manifests, analysis settings and summary statistics. Please cite the original
  publications and data records of the datasets you use ([docs/DATA.md](docs/DATA.md), "Citing the
  data").

## How to cite

Reggio G, Kim M, Ji Z, Bush D, Barry C and Burgess N (2026). Firing rate adaptation in grid cells
produces macroscopic hexadirectional modulation.

Citation metadata for the paper and this code are in [CITATION.cff](CITATION.cff). Please also cite
the datasets ([docs/DATA.md](docs/DATA.md)).

## Contact

Misun Kim and Neil Burgess (UCL Institute of Cognitive Neuroscience), the corresponding authors of
the paper; their addresses are given in the published article. Questions about the code can also be
raised as issues of this repository.
