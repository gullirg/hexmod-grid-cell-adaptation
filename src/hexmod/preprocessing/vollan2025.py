"""Preprocessing of the open-field recordings of Vollan et al. (2025).

Source: the ``Dsession`` struct of each open-field file ``sharing_v4/navigation/of/<rat>_<n>.mat`` of
the EBRAINS release (doi:10.25493/R5FR-EDG; the fields are described in
``sharing_v4/documentation/mat_files.md``). The files and their session labels come from the
manifests; the four files of the folder that give no session (no MEC units, or no samples inside the
declared valid time ranges) are not read.

What the code does, per file:

- **Samples.** ``t``, ``x`` and ``y`` (m, converted to cm) are restricted to the samples inside
  ``sessions.validTimeRanges``. Time is shifted so that the first valid sample is
  at 0. Every second valid sample is kept (100 Hz to 50 Hz; ``sample_rate`` is the Python int 50).
  The decimation counts over the concatenated valid samples, so it runs across the gaps between valid
  ranges. x and y are then shifted so that their minima are 0. Positions keep the single-precision
  type of the source (``xy`` is float32; ``time`` is float64).
- **Gaps.** The time series of the release are speed filtered: samples at locomotion speeds below
  5 cm/s were removed by the dataset authors (``mat_files.md``). The kept samples therefore have gaps,
  which make up 42% of the session span in the median session (16% to 84%). Positions are not
  interpolated across these gaps; the analysis drops spikes that fall in a gap longer than 1.5 sample
  intervals (:meth:`hexmod.rat.gridcell.GridCellAnalysis.compute_spike_array`).
- **Units.** Units of ``units.mec`` with ``isGrid == 1`` are kept when they have at least 20 spikes inside
  the valid time ranges, in ascending order; spikes outside the ranges are dropped and the rest are
  shifted by the same time origin as the positions. A unit is keyed by its position in ``units.mec``,
  counted from 1. ``mod_id`` is None for every unit.
- **Not kept.** Head direction, theta phase and the speed series of the source.
- **Repeated recording.** Files ``28063_4`` and ``28063_5`` (sessions V2_4 and V2_5) hold the same
  recording session with two spike sortings: identical time stamps, positions that differ by at most
  3.1e-5 cm, and 183 and 171 kept units (64 unit keys in common). Both are kept as sessions, as in the
  published analysis; the analyses that need independent recordings (Fig 2D and 2E) handle it.

A session dict holds, in this order: ``position`` (``xy``, ``time``, ``sample_rate``),
``spike_times``, ``mod_id``, ``unique_cell_ids`` (``V<a>_V<a>_<s>_cell<i>``), ``animal_id``,
``session_id`` and ``time`` (the same array as ``position["time"]``).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .. import parallel as _parallel
from . import check_file_sizes, data_base, labels, manifest_files, refuse_unlisted_files

KEY = "vollan2025"
#: Folder of the open-field files, relative to the data folder.
FOLDER = "vollan2025/sharing_v4/navigation/of"
#: Sampling rate of the source time series (Hz).
RAW_SAMPLE_RATE = 100
#: Decimation factor.
DOWNSAMPLE = 2
#: Sampling rate after decimation (Hz, Python int).
SAMPLE_RATE = RAW_SAMPLE_RATE // DOWNSAMPLE
#: Minimum number of spikes inside the valid time ranges.
MIN_SPIKES = 20
#: Fields that ``Dsession`` must have.
REQUIRED_FIELDS = ["t", "x", "y", "dt", "units", "sessions"]


def mask_spikes(spikes: np.ndarray, valid_ranges: np.ndarray) -> np.ndarray:
    """Return the spikes inside the valid time ranges, range by range (inclusive bounds)."""
    return np.concatenate([
        spikes[(spikes >= start) & (spikes <= end)]
        for start, end in valid_ranges
    ])


def session_from_dsession(d, animal_id: str, session_id: str) -> Optional[dict]:
    """Build one session from a loaded ``Dsession`` record.

    Parameters
    ----------
    d : numpy.void
        ``loadmat(path)["Dsession"][0, 0]`` (loaded with ``simplify_cells=False``).
    animal_id, session_id : str
        Labels of the session.

    Returns
    -------
    dict or None
        The session dict, or None when the file gives no session (fields missing, unexpected sample
        interval, no valid sample, no MEC unit or no unit kept).
    """
    if not all(field in d.dtype.names for field in REQUIRED_FIELDS):
        return None

    t = d["t"].flatten()
    x = d["x"].flatten() * 100
    y = d["y"].flatten() * 100
    dt = float(d["dt"][0, 0])

    if abs(1 / dt - RAW_SAMPLE_RATE) > 1:
        return None

    valid_ranges = d["sessions"][0, 0]["validTimeRanges"]
    if valid_ranges.ndim == 1 and valid_ranges.shape[0] == 2:
        valid_ranges = np.array([valid_ranges])  # shape (1, 2)
    elif valid_ranges.shape[0] == 2 and valid_ranges.shape[1] != 2:
        valid_ranges = valid_ranges.T  # (2, N) to (N, 2)
    if valid_ranges.ndim != 2 or valid_ranges.shape[1] != 2:
        raise ValueError(f"{session_id}: invalid validTimeRanges shape {valid_ranges.shape}")

    # samples inside the valid time ranges
    mask = np.zeros_like(t, dtype=bool)
    for start, end in valid_ranges:
        mask |= (t >= start) & (t <= end)
    t_valid = t[mask]
    x_valid = x[mask]
    y_valid = y[mask]
    if len(t_valid) == 0:
        return None

    # time origin at the first valid sample
    t0 = t_valid[0]
    t_valid -= t0

    # every second valid sample
    t_ds = t_valid[::DOWNSAMPLE]
    x_ds = x_valid[::DOWNSAMPLE]
    y_ds = y_valid[::DOWNSAMPLE]

    # positions start at (0, 0)
    x_ds -= np.nanmin(x_ds)
    y_ds -= np.nanmin(y_ds)

    if not (len(t_ds) == len(x_ds) == len(y_ds)):
        return None

    position = {
        "xy": np.column_stack([x_ds, y_ds]),
        "time": t_ds,
        "sample_rate": SAMPLE_RATE,
    }

    # grid cells of the MEC units
    units = d["units"][0, 0]
    mec_units = units["mec"]
    if not isinstance(mec_units, np.ndarray):
        return None
    mec_units = mec_units.flatten()
    if len(mec_units) == 0:
        return None

    spike_times: Dict[int, np.ndarray] = {}
    mod_id: Dict[int, None] = {}
    unique_cell_ids: Dict[int, str] = {}
    for i, unit in enumerate(mec_units, start=1):
        if "isGrid" in unit.dtype.names and not unit["isGrid"].item():
            continue
        if "spikeTimes" in unit.dtype.names:
            spikes = unit["spikeTimes"].flatten()
        elif "spikeInds" in unit.dtype.names:
            spikes = unit["spikeInds"].flatten()
        else:
            warnings.warn(f"{session_id}: unit {i} has no spike field and is skipped", RuntimeWarning)
            continue

        spikes = mask_spikes(spikes, valid_ranges)
        if len(spikes) == 0:
            continue
        spikes -= t0
        if len(spikes) < MIN_SPIKES or not np.all(np.diff(spikes) >= 0):
            continue

        spike_times[i] = spikes
        mod_id[i] = None
        unique_cell_ids[i] = f"{animal_id}_{session_id}_cell{i}"

    if not spike_times:
        return None

    return {
        "position": position,
        "spike_times": spike_times,
        "mod_id": mod_id,
        "unique_cell_ids": unique_cell_ids,
        "animal_id": animal_id,
        "session_id": session_id,
        "time": t_ds,
    }


def read_session(path, animal_id: str, session_id: str) -> Optional[dict]:
    """Read one open-field file and build its session (see :func:`session_from_dsession`)."""
    from scipy.io import loadmat

    data = loadmat(str(path), simplify_cells=False)
    return session_from_dsession(data["Dsession"][0, 0], animal_id, session_id)


def _read_item(item) -> Optional[dict]:
    """Worker function: ``item = (path, animal_id, session_id)``."""
    return read_session(*item)


def preprocess(base: Optional[Path] = None, workers: int = 1) -> dict:
    """Preprocess every session of the manifest.

    Parameters
    ----------
    base : Path, optional
        Data folder (default ``HEXMOD_DATA`` or ``<repo>/data``).
    workers : int
        Files read in parallel (each worker holds one file, up to a few GB, in memory).

    Returns
    -------
    dict
        ``{animal_id: {session_id: session}}`` in canonical order.
    """
    base = data_base(base)
    rows = {r["path"]: r for r in manifest_files(KEY) if r["path"].startswith(FOLDER + "/")}
    refuse_unlisted_files(KEY, FOLDER, "*.mat", base)
    items, used = [], []
    for s in labels.sessions_of(KEY):
        rel = f"{KEY}/{s['raw_path']}"
        if rel not in rows:
            raise labels.ManifestError(f"{KEY}: session {s['session_id']} names {rel}, which is not in the file "
                                       "manifest")
        used.append(rows[rel])
        items.append((str(base / rel), s["animal_id"], s["session_id"]))
    check_file_sizes(used, base)
    results = _parallel.map_ordered(_read_item, items, workers)
    data: Dict[str, dict] = {}
    for (path, animal_id, session_id), session in zip(items, results):
        if session is None:
            raise labels.ManifestError(f"{KEY}: {Path(path).name} gives no session, but the manifest lists "
                                       f"{session_id}")
        data.setdefault(animal_id, {})[session_id] = session
    return labels.canonical_order(data, KEY)
