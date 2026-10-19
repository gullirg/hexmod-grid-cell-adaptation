# Data

The analyses use grid cell recordings from six published rat datasets. All six are publicly
available and are downloaded by `scripts/data_download.py`: four from the data archives of the
original studies (EBRAINS and the NIRD Research Data Archive), and two from a Zenodo record that
holds the sessions of Barry et al. (2012) and Ólafsdóttir et al. (2016) analysed in the paper. No
recorded data or per-cell results are distributed with this repository; it holds only label
manifests, analysis settings and summary statistics.

## The six datasets

| Key | Label prefix | Publication | Data source | Licence | Files used | Download |
|---|---|---|---|---|---|---|
| `vollan2025` | V | Vollan et al. (2025), Nature 639, 995-1005 | EBRAINS, doi:[10.25493/R5FR-EDG](https://doi.org/10.25493/R5FR-EDG) (version 1) | CC BY-SA 4.0 | 31 open-field recordings (`sharing_v4/navigation/of/*.mat`) and the file description `mat_files.md` | 11.85 GB |
| `brun2008` | B | Brun et al. (2008), Hippocampus 18, 1200-1212 | NIRD Research Data Archive, doi:[10.11582/2018.00015](https://doi.org/10.11582/2018.00015) | CC BY 4.0 | the 6 recordings of the square-box experiment, extracted from the archive `Brun2008-v3.0.0.zip` | 1.44 GB |
| `hafting2005` | H | Hafting et al. (2005), Nature 436, 801-806 | NIRD Research Data Archive, doi:[10.11582/2017.00017](https://doi.org/10.11582/2017.00017) | CC BY 4.0 | 5 `.mat` files | 6.1 MB |
| `sargolini2006` | S | Sargolini et al. (2006), Science 312, 758-762 | NIRD Research Data Archive, doi:[10.11582/2017.00019](https://doi.org/10.11582/2017.00019) | CC BY 4.0 | 150 files from the split archive `gridcell_data_sargolini` (see below) | 270 MB |
| `barry2012` | W | Barry et al. (2012), PNAS 109, 17687-17692 | Zenodo, doi:[10.5281/zenodo.23477461](https://doi.org/10.5281/zenodo.23477461) | CC BY 4.0 | `barry2012_familiar_box_sessions.h5` (see below) | 47.2 MB |
| `olafsdottir2016` | F | Ólafsdóttir et al. (2016), Nature Neuroscience 19, 792-794 | Zenodo, doi:[10.5281/zenodo.23477461](https://doi.org/10.5281/zenodo.23477461) | CC BY 4.0 | `olafsdottir2016_open_field_sessions.h5` (see below) | 27.8 MB |

The download totals 13.64 GB (13,639,845,431 bytes). No login is needed.

Recordings read by step 01, and grid cells in the cohort (step 02):

| Key | Sessions read (animals) | Units read | Cohort sessions | Cohort animals | Grid cells | Conjunctive cells |
|---|---|---|---|---|---|---|
| `vollan2025` | 27 (16) | 5,751 | 24 | 15 | 3,724 | 1,132 |
| `brun2008` | 2 (1) | 5 | 2 | 1 | 5 | 0 |
| `hafting2005` | 3 (2) | 23 | 3 | 2 | 23 | 0 |
| `sargolini2006` | 13 (4) | 109 | 2 | 1 | 7 | 0 |
| `barry2012` | 57 (7) | 321 | 57 | 7 | 118 | 23 |
| `olafsdottir2016` | 24 (6) | 406 | 24 | 6 | 61 | 16 |
| total | 126 (36) | 6,615 | 112 | 32 | 3,938 | 1,171 |

A cohort session is a session with at least 30,000 position samples faster than 5 cm/s and at least
one grid cell (gridness of at least 0.3); conjunctive cells are grid cells whose movement-direction
tuning has a Rayleigh vector length above 0.3.

The label prefix is the first letter of every animal and session label of a dataset in the
results (for example `V10_1`); W and F keep the labels of Barry et al. (2012) and Ólafsdóttir et
al. (2016) distinct from B, Brun et al. (2008). Dataset metadata are in `src/hexmod/datasets.py`;
download locations in `src/hexmod/data_sources.py`.

## Downloading

```
python scripts/data_download.py --list                  # dry run: files, sizes and addresses
python scripts/data_download.py                         # all six datasets
python scripts/data_download.py --datasets hafting2005,brun2008
```

- Files are written to the data folder: `--dest DIR`, or the environment variable
  `HEXMOD_DATA`, or `data/` in the repository.
- Every file is checked against its size and sha256 checksum before it is accepted, and also
  against the md5 checksum where the archive publishes one. A file that fails the check is
  downloaded again.
- Interrupted downloads resume where they stopped (`<file>.part` files are continued with HTTP
  range requests). Running the command again skips everything that is already present and valid;
  `--trust-existing` skips the checksum of files that are present with the right size.
- `--jobs N` sets the number of parallel downloads (default 2), `--retries N` the number of
  further attempts per file (default 5). `--only PATTERN` downloads only the files whose name
  matches a shell pattern, for example `--datasets vollan2025 --only 28063_4.mat`.
- `--check-remote` compares the file listings of the archives (sizes and hashes) with the
  manifests and sends one HTTP HEAD request per dataset, without downloading data. It detects a
  changed release.
- Archives are kept after extraction; `--delete-archives` removes them (1.7 GB) once their
  members are extracted and verified.

At 10 MB/s the download takes about 23 minutes. The data need 13.9 GB of disk space with the
archives and 12.2 GB without them. Keep the raw files after preprocessing: step 09 reads 16 of the
Vollan et al. (2025) files again (unit waveforms and rate maps of the animals recorded in more
than one session).

### Download addresses

- `vollan2025`: each file is requested through the EBRAINS data proxy,
  `https://data-proxy.ebrains.eu/api/v1/buckets/d-4080b78d-edc5-4ae4-8144-7f6de79930ea/<key>?redirect=false`,
  which returns the address of the object in the EBRAINS object store
  (`https://rgw.cscs.ch/ebrains:d-4080b78d-edc5-4ae4-8144-7f6de79930ea/<key>`). The bucket can be
  listed with `https://data-proxy.ebrains.eu/api/v1/buckets/d-4080b78d-edc5-4ae4-8144-7f6de79930ea?prefix=sharing_v4/navigation/of/`.
- `brun2008`, `hafting2005`, `sargolini2006`: the public S3 endpoint of the NIRD Research Data
  Archive, `https://s3.nird.sigma2.no/archive-ro/<identifier>/<file>`, with the identifiers
  `54d73ada-d6ad-40d7-9dcb-85310168228e` (Brun et al.), `60452682-5815-41eb-bd0a-fb11b914623d`
  (Hafting et al.) and `dbb6250a-9793-4d9b-9265-a90f886451d4` (Sargolini et al.). The same files
  can be downloaded from the landing pages of the DOIs.
- `barry2012`, `olafsdottir2016`: Zenodo record 23477461,
  `https://zenodo.org/records/23477461/files/<file>?download=1`. The record's file listing, with
  the size and md5 checksum of every file, is `https://zenodo.org/api/records/23477461`.

### The Sargolini et al. (2006) archive

The archive is a zip file split into three volumes, `gridcell_data_sargolini.z01` (1.05 GB),
`.z02` (1.05 GB) and `.zip` (270 MB), with 1,027 files from several experiments. The analysis
uses the 150 files of 13 sessions that are stored in the last volume,
`gridcell_data_sargolini.zip`: 13 position files (`_POS`), 109 single-unit files
(`_T<tetrode>C<cell>`) and 28 EEG and EGF files (which are not read). Only that volume is therefore
downloaded, and the files are extracted from it directly (`hexmod.data_sources` reads split zip
archives; no external tool is needed). `--sargolini-full-archive` downloads all three volumes
(2.37 GB) instead and extracts the same 150 files; the download then totals 15.74 GB
(15,736,997,431 bytes).

The folder `sargolini2006/subset/` must hold exactly these 150 files. Other files of the archive
would be read as additional recordings and change the cohort, so `data_check.py` reports any
extra file there as an error.

### The Brun et al. (2008) archive

The 1.44 GB archive holds the recordings of several experiments. The analysis reads the six
recordings of the square-box experiment (`Brun2008-v3.0.0/square/recordings/*.mat`, 1.8 MB),
which are extracted into `brun2008/Brun2008-v3.0.0/square/recordings/` and checked against
their md5 checksums published in the archive and their sha256 checksums. Two of them (rat 11468)
contribute grid cells to the cohort.

### The Barry et al. (2012) and Ólafsdóttir et al. (2016) sessions

The Zenodo record (doi:10.5281/zenodo.23477461, CC BY 4.0; all versions of the record:
doi:10.5281/zenodo.23477460) holds the sessions of these two studies
that are analysed in the paper, that is the sessions with at least one cohort grid cell, as one
HDF5 file per dataset:

| File | Sessions | Animals | Units | Bytes |
|---|---|---|---|---|
| `barry2012_familiar_box_sessions.h5` | 57 trials in the familiar box (Barry et al., 2012) | 7 | 321 | 47,170,260 |
| `olafsdottir2016_open_field_sessions.h5` | 24 open-field foraging sessions (Ólafsdóttir et al., 2016) | 6 | 406 | 27,785,094 |

Each file holds, for every session, the position samples (time in s, x and y in cm, 50 Hz) and the
spike times (s) of every unit recorded in the session, not only of the grid cells; the analysis
selects the grid cells itself. The values are those of the original recordings: nothing is
smoothed, shifted, rounded or converted. The files hold no EEG, no head direction, no other sessions
of the two studies and no analysis results. The record also holds a README that documents every
group and attribute, the licence and `CHECKSUMS.sha256`.

Layout of a file:

```
/<session label>/                  one group per session, named by its label (W1_1, ..., F1_1, ...)
    position/time                  float64, s
    position/x, position/y         float64, cm
    units/unit_000/spike_times     float64, s; one group per unit, in the order of the original data
```

HDF5 lists groups alphabetically (W1_10 before W1_2), so the reader,
`hexmod.preprocessing.hdf5_sessions.read_sessions`, takes the sessions in the order of their `order`
attribute and the units of a session in the order of their `original_index` attribute, and checks
that the file has the documented layout. In `inputs/manifests/sessions.csv` the `raw_path` of these
sessions is the HDF5 file (the session is the group named by `session_id`), and in `units.csv` the
`raw_unit_key` of a unit is the name of its group (`unit_<index>`). Position and spike times share
one clock; as in the published analysis, step 01 subtracts the first position time (0.02 s) from
the position times of the Barry et al. (2012) sessions and leaves their spike times unchanged
(docs/PROVENANCE.md, section 1).

## Folder layout

```
data/
  vollan2025/sharing_v4/navigation/of/<rat>_<session>.mat      31 files
  vollan2025/sharing_v4/documentation/mat_files.md
  brun2008/archive/Brun2008-v3.0.0.zip                         may be deleted after extraction
  brun2008/Brun2008-v3.0.0/square/recordings/*.mat             6 files
  hafting2005/{Hafting_Fig2c_Trial1,Hafting_Fig2c_Trial2,Hafting_Fig2d_Trial1,Hafting_Fig2d_Trial2,rat_10925}.mat
  sargolini2006/archive/gridcell_data_sargolini.zip            may be deleted after extraction
  sargolini2006/subset/<150 files>                             exactly the listed files
  barry2012/barry2012_familiar_box_sessions.h5                 57 sessions
  olafsdottir2016/olafsdottir2016_open_field_sessions.h5       24 sessions
```

The files may also be symbolic links, for example to an existing copy of the datasets.

## Checking the data

```
python scripts/data_check.py            # sizes and sha256 checksums of all files (seconds to minutes)
python scripts/data_check.py --quick    # sizes only (seconds)
```

The check looks for every file of the manifests under the data folder (`--data DIR` or
`HEXMOD_DATA`), compares its size and sha256 checksum (and its md5 checksum where the manifest gives
one), reports each dataset and writes `results/data_check.json`:

| Key | Meaning |
|---|---|
| `ok` | every dataset passes |
| `checksums` | `sha256` or `size` (what was compared) |
| `datasets.<key>.status` | `ok`, `missing`, `incomplete`, `checksum mismatch` or `unexpected files` |
| `datasets.<key>.n_expected`, `.n_ok` | files expected and files that passed |
| `datasets.<key>.bad_files` | paths of the files that failed, relative to the data folder |
| `datasets.<key>.problems` | the reason for each failed file (`missing`, `wrong size`, `checksum mismatch`) |
| `datasets.<key>.extra_files` | files in `sargolini2006/subset/` that are not in the manifest |
| `datasets.<key>.bytes` | total size of the files that passed |

The run record is `results/logs/data_check.json`. The exit status is 0 when all six datasets pass
and 1 otherwise. Step 01 makes a quicker check of its own before any work: it stops with a one-line
message when a raw file is missing or has the wrong size.

## Manifests

`inputs/manifests/files_<key>.csv` list every file that is downloaded or read, one row per file:

| Column | Meaning |
|---|---|
| `dataset` | dataset key |
| `path` | location relative to the data folder |
| `name` | file name |
| `role` | `file` (downloaded and read as it is), `archive` (zip archive from which members are extracted), `volume` (first or second volume of a split archive, downloaded only with `--sargolini-full-archive`), `member` (file extracted from an archive and read by the analysis) |
| `bytes` | size in bytes |
| `sha256` | sha256 checksum (for the two Zenodo files, as listed in the record's `CHECKSUMS.sha256`) |
| `md5` | md5 checksum published by the archive (empty where none is published; for the two Zenodo files, the md5 checksum given in the record's file listing) |
| `source` | object key at the archive (`file`, `archive`, `volume`), or the member name inside the archive (`member`) |
| `archive` | for `member` rows, the `path` of the archive it comes from |

Extra columns:

- `files_vollan2025.csv`: `etag`, the object hash in the EBRAINS listing (an md5 checksum for
  files stored in one part, otherwise a hash of the parts followed by the number of parts), used
  by `--check-remote`; `used_in_cohort`, True for the 24 recordings that contribute grid cells to
  the cohort. Of the other seven, three give a session without cohort grid cells and four give no
  session (no entorhinal units, or no samples inside the valid time ranges of the file).
- `files_brun2008.csv`: `used_in_cohort`, True for the two recordings of rat 11468.
- `files_sargolini2006.csv`: `listdir_order`, the reading order of the 150 files, which fixes the
  unit numbers (`cell_id`) of this dataset (see `inputs/manifests/units.csv`). The member rows
  are in this order.

The session and unit manifests (`inputs/manifests/sessions.csv`, `units.csv`), which fix the labels
and the order of every recording, are described in `hexmod.preprocessing.labels`.

## Citing the data

Please cite the original publications and data records of the datasets you use:

- Vollan, A. Z., Gardner, R. J., Moser, M.-B. and Moser, E. I. (2025). Left-right-alternating theta
  sweeps in entorhinal-hippocampal maps of space. Nature 639, 995-1005.
  doi:10.1038/s41586-024-08527-1. Data: EBRAINS, doi:10.25493/R5FR-EDG.
- Brun, V. H., Solstad, T., Kjelstrup, K. B., Fyhn, M., Witter, M. P., Moser, E. I. and Moser, M.-B.
  (2008). Progressive increase in grid scale from dorsal to ventral medial entorhinal cortex.
  Hippocampus 18, 1200-1212. doi:10.1002/hipo.20504. Data: NIRD, doi:10.11582/2018.00015.
- Hafting, T., Fyhn, M., Molden, S., Moser, M.-B. and Moser, E. I. (2005). Microstructure of a
  spatial map in the entorhinal cortex. Nature 436, 801-806. doi:10.1038/nature03721. Data: NIRD,
  doi:10.11582/2017.00017.
- Sargolini, F., Fyhn, M., Hafting, T., McNaughton, B. L., Witter, M. P., Moser, M.-B. and Moser,
  E. I. (2006). Conjunctive representation of position, direction, and velocity in entorhinal
  cortex. Science 312, 758-762. doi:10.1126/science.1125572. Data: NIRD, doi:10.11582/2017.00019.
- Barry, C., Ginzberg, L. L., O'Keefe, J. and Burgess, N. (2012). Grid cell firing patterns signal
  environmental novelty by expansion. Proceedings of the National Academy of Sciences USA 109(43),
  17687-17692. doi:10.1073/pnas.1209918109. Data: Zenodo, doi:10.5281/zenodo.23477461.
- Ólafsdóttir, H. F., Carpenter, F. and Barry, C. (2016). Coordinated grid and place cell replay
  during rest. Nature Neuroscience 19, 792-794. doi:10.1038/nn.4291. Data: Zenodo,
  doi:10.5281/zenodo.23477461.

The data record of the last two datasets: Barry, C. (2026). Rat grid cell recordings: familiar-box
sessions of Barry et al. (2012) and open-field sessions of Ólafsdóttir et al. (2016). Zenodo.
doi:10.5281/zenodo.23477461.

The Vollan et al. (2025) data are licensed CC BY-SA 4.0: data derived from them and redistributed
must carry the same licence. This repository redistributes no recorded data.
