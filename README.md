# Firing rate adaptation in grid cells produces macroscopic hexadirectional modulation

Analysis code for Reggio G, Kim M, Ji Z, Bush D, Barry C and Burgess N (2026). *Firing rate adaptation in grid cells produces macroscopic hexadirectional modulation*.

The code computes every statistic printed in the paper and draws its figures. The rat analyses start
from the raw recordings of six public datasets (3,938 grid cells from 32 rats), and the simulations
start from the model parameters. A final step checks every statistic against its reference value
and the number printed in the paper.

## Quick start

```
conda env create -f environment.yml
conda activate hexmod
pip install -e .
python scripts/data_download.py
python scripts/run_pipeline.py --workers 4
```

The download fetches the six datasets (13.64 GB). The pipeline then runs every step in order, writes
its results to `results/` and ends with the statistics check (`results/stats/statistics_check.md`).
Install the package in editable mode (`-e`), as above: the code finds `inputs/`, `reference/` and
`results_precomputed/` relative to the repository.

The figures are set in Arial, which cannot be distributed with the code. macOS ships Arial. On
Linux, install it (for example with the `ttf-mscorefonts-installer` package), or set
`HEXMOD_FONT_DIR` to a folder that holds `Arial.ttf`, `Arial Bold.ttf`, `Arial Italic.ttf` and
`Arial Bold Italic.ttf`. Without Arial the figures are drawn in Liberation Sans or DejaVu Sans, with
a warning, and their text differs slightly from the printed figures.

Figures 6, 7 and 8 can be drawn in a few seconds without any data, from the simulation results
shipped in `results_precomputed/`:

```
python scripts/step30_figures.py
```

The figures of recorded data are reported as skipped until the rat steps have run.

## Requirements and runtimes

- Tested on Linux (x86-64). macOS should work, but the full pipeline has not been run there.
- The environment is built from `environment.yml` with conda 23.10 or later, mamba or micromamba
  (older conda versions are very slow to solve it). It pins Python 3.10, numpy 1.23.5, scipy 1.13.0,
  pandas 1.5.3, scikit-image 0.20.0, h5py 3.7.0, matplotlib 3.8.0 and pyparsing 3.0.9 from
  conda-forge.
- Results agree across machines to well within the printed precision. The last digits of some
  values can differ between numerical libraries (for example between the MKL and OpenBLAS builds of
  numpy), and the statistics check allows for this.
- Disk: 13.9 GB for the data (12.2 GB once the archives are deleted) and 1.4 GB of results.
- Memory: about 12 GB with 4 workers and 24 GB with 8. With 8 workers the run below peaked at
  19.4 GB, in step 26.

Runtimes of a run with `--workers 8` on a desktop computer (Intel i9-12900K, 24 threads, 46 GB of
memory), with the raw data in the file cache:

| Steps | What | Time |
|---|---|---|
| D2, 01, 02 | data check (D2), preprocessing, cohort | 53 s |
| 03 to 05 | adaptation fits, prospective coding control | 10.4 min |
| 06 to 12 | visit sequences, timescales, same-cell links, conjunctive cells, rat statistics | 4.6 min |
| 21 to 23 | simulated cell pools, maps and proximity | 7.4 min |
| 25, 26 | Fig 7 values from the shipped participant results, speed sweep | 4.3 min |
| 27, 30, 40 | simulation statistics, figures, statistics check | 12 s |
| all | `run_pipeline.py --workers 8` | 28 min |

From a cold disk, add the time to read about 12 GB of raw data. The full simulation of the Fig 7
participants (step 24, 1,500 per module) is skipped by default, and step 25 then uses the shipped
participant results. The option `--full-sim` of `run_pipeline.py` includes it; it takes about 6
hours with 10 workers (62 worker hours, 1 GB of memory per worker).

## Data

The analyses use grid cell recordings from six published rat datasets. All six are public and are
downloaded by `scripts/data_download.py`, with no login. No recorded data are distributed with this
repository. [docs/DATA.md](docs/DATA.md) lists the datasets with their sources, licences and sizes,
and describes the download, the checksums and the folder layout. The command
`python scripts/data_download.py --list` shows every file, its size and its address without
downloading anything, and `python scripts/data_check.py` checks the downloaded files.

## Pipeline

Each step is a script `scripts/stepNN_<name>.py`. It reads the outputs of earlier steps and writes
its own outputs, its statistics and a record of its inputs and outputs under `results/`.
[docs/PIPELINE.md](docs/PIPELINE.md) describes every step and the files it reads and writes.
`python scripts/run_pipeline.py --list` prints the steps, and `--from 30 --to 40` runs a range.
Steps 01 and 09 read the raw data; the other steps read only the results folder, so a figure can be
redrawn without running any analysis again, for example with
`python scripts/step30_figures.py --figure Fig3`.

The environment variables `HEXMOD_DATA` (raw data, default `data/`), `HEXMOD_RESULTS` (default
`results/`), `HEXMOD_WORKERS` (default 4) and `HEXMOD_FONT_DIR` configure a run.
[docs/PROVENANCE.md](docs/PROVENANCE.md) lists the preprocessing conventions of each dataset, the
seed of every random draw and the conventions of the simulations.

## Figures

The code draws Figures 2 to 8 and Supplementary Figures 1 to 5. Figure 1 and panel E of Figure 3 are
schematics and are not produced by the code. Each figure is written as PDF, SVG and PNG (600 dpi),
with the values of each panel in `source_data/`. [docs/FIGURES.md](docs/FIGURES.md) gives, for every
panel, the step and the result files it is drawn from.

## Statistics check

Step 40 (`scripts/step40_check_statistics.py`) compares every statistic with its reference value in
`reference/expected_statistics.json` and checks that it rounds to the number printed in the paper.
Its report is `results/stats/statistics_check.md`. [docs/STATISTICS.md](docs/STATISTICS.md) lists
every printed statistic with its key, the step that computes it and its value, and names the four
printed numbers that differ from the analysis. [reference/README.md](reference/README.md) gives the
tolerances.

## Tests

```
pytest -q
HEXMOD_RESULTS=results pytest -m results tests/test_plotted_vs_stats.py
```

The unit tests need no data. The second command checks a complete run of the pipeline: for every
figure, it recomputes the statistics from the values the figure draws and compares them with the
statistics files. The continuous integration workflow (`.github/workflows/tests.yml`) runs the unit
tests and draws Figures 6, 7 and 8 on Ubuntu.

## Repository layout

```
src/hexmod/            the library: preprocessing/, rat/, sim/, figures/ and shared modules
scripts/               one script per pipeline step, the data scripts and run_pipeline.py
inputs/                data manifests, session and unit labels, rat inputs, simulated cell pool
results_precomputed/   shipped simulation results (Figs 6 to 8, Supp Fig 3E)
reference/             reference values of all statistics
docs/                  DATA, PIPELINE, FIGURES, STATISTICS, PROVENANCE
tests/                 unit tests
```

## Data and code availability

This paper analyses existing, publicly available data. The recordings of Vollan et al. (2025) are
available from EBRAINS (doi:10.25493/R5FR-EDG); those of Brun et al. (2008), Hafting et al. (2005)
and Sargolini et al. (2006) from the NIRD Research Data Archive (doi:10.11582/2018.00015,
doi:10.11582/2017.00017 and doi:10.11582/2017.00019); and the sessions of Barry et al. (2012) and
Ólafsdóttir et al. (2016) analysed here from Zenodo (doi:10.5281/zenodo.23477461). All original code
is available at https://github.com/gullirg/hexmod-grid-cell-adaptation under the MIT licence. It
downloads these data, computes every statistic and draws every figure of the paper except the
schematics (Figure 1 and Figure 3E), and includes the simulated grid cell pool and the simulation
results (CC BY 4.0).

## Licences

- Code: MIT licence ([LICENSE](LICENSE)).
- Shipped simulation data in `inputs/simulation/` and `results_precomputed/`: Creative Commons
  Attribution 4.0 ([LICENSE-data](LICENSE-data)).
- Recorded data: the datasets keep their own licences (CC BY-SA 4.0 for Vollan et al. (2025), CC BY
  4.0 for the others). No recorded data or per-cell results are distributed here. Please cite the
  original publications and data records of the datasets you use ([docs/DATA.md](docs/DATA.md),
  "Citing the data").

## How to cite

Reggio G, Kim M, Ji Z, Bush D, Barry C and Burgess N (2026). Firing rate adaptation in grid cells
produces macroscopic hexadirectional modulation.

Citation metadata for the paper and this code are in [CITATION.cff](CITATION.cff). Please also cite
the datasets ([docs/DATA.md](docs/DATA.md)).

## Contact

Misun Kim and Neil Burgess (UCL Institute of Cognitive Neuroscience), the corresponding authors of
the paper; their addresses are given in the published article. Questions about the code can also be
raised as issues of this repository.
