"""Preprocessing of the recordings of Brun et al. (2008).

Source: the square-box recordings ``Brun2008-v3.0.0/square/recordings/*.mat`` of the NIRD release
(doi:10.11582/2018.00015). Only the two recordings of rat 11468 are used (sessions B1_1 =
``11468_16030601+02.mat`` and B1_2 = ``11468_07040601+02.mat``); the four files of other rats in the
same folder are not read. The folder must hold exactly the six files of the manifest.

What the code does:

- **Positions.** ``positions`` holds time, x and y in camera pixels. x and y are shifted so that their
  minima are 0 and divided by 2, assuming 2 pixels per cm (the largest extent, 218.75 cm, agrees with
  the 220 cm box of the dataset description). Time is shifted so that its minimum is 0. Missing x and
  y samples are filled by linear interpolation over time (``numpy.interp``; samples before the first
  or after the last valid one take its value). ``sample_rate`` is set to the Python int 50.
- **Spikes.** Each row of ``units`` is one spike (time, tetrode, cell). Spike times are shifted by the
  same time origin as the positions. Units are keyed by their cell number alone; the tetrode number is
  kept only in the unit label (``B1_B1_<s>_Tet<t>_Cell<c>``, tetrode of the unit's first spike). In
  the two files used no two units share a cell number. Units appear in the order of their first spike.

A session dict holds, in this order: ``position`` (``xy``, ``time``, ``sample_rate``),
``spike_times``, ``unique_cell_ids``, ``animal_id`` and ``session_id``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import check_file_sizes, data_base, labels, manifest_files, refuse_unlisted_files

KEY = "brun2008"
#: Folder of the recordings, relative to the data folder.
FOLDER = "brun2008/Brun2008-v3.0.0/square/recordings"
#: The rat whose recordings are used.
TARGET_ANIMAL = "11468"
#: Camera pixels per cm.
SCALING_FACTOR = 2.0
#: Sampling rate assigned to the positions (Hz, Python int).
SAMPLE_RATE = 50


def interpolate_nans(values: np.ndarray, time_vector: np.ndarray) -> np.ndarray:
    """Fill NaNs by linear interpolation over time (needs at least two valid samples)."""
    if np.isnan(values).any():
        valid = ~np.isnan(values)
        if np.sum(valid) >= 2:
            return np.interp(time_vector, time_vector[valid], values[valid])
    return values


def session_from_arrays(positions: np.ndarray, units: np.ndarray, animal_id: str, session_id: str) -> dict:
    """Build one session from the ``positions`` and ``units`` arrays of a recording file.

    Parameters
    ----------
    positions : ndarray, shape (n, 3)
        Time (s), x and y (pixels).
    units : ndarray, shape (k, 3)
        One row per spike: time (s), tetrode, cell.
    animal_id, session_id : str
        Labels of the session.

    Returns
    -------
    dict
    """
    pos_x = positions[:, 1].copy()
    pos_y = positions[:, 2].copy()
    pos_time = positions[:, 0].copy()

    min_x = np.nanmin(pos_x)
    min_y = np.nanmin(pos_y)
    min_time = np.nanmin(pos_time)

    pos_x = (pos_x - min_x) / SCALING_FACTOR
    pos_y = (pos_y - min_y) / SCALING_FACTOR
    pos_time = pos_time - min_time

    pos_x = interpolate_nans(pos_x, pos_time)
    pos_y = interpolate_nans(pos_y, pos_time)

    pos_data = {
        "xy": np.vstack((pos_x, pos_y)).T,
        "time": pos_time,
        "sample_rate": SAMPLE_RATE,
    }

    spike_times: Dict[int, list] = {}
    unique_cell_ids: Dict[int, str] = {}
    for row in units:
        spike_time = row[0] - min_time
        tetrode = int(row[1])
        cell = int(row[2])
        if cell not in spike_times:
            spike_times[cell] = []
            unique_cell_ids[cell] = f"{animal_id}_{session_id}_Tet{tetrode}_Cell{cell}"
        spike_times[cell].append(spike_time)
    spikes = {cell: np.array(times) for cell, times in spike_times.items()}

    return {
        "position": pos_data,
        "spike_times": spikes,
        "unique_cell_ids": unique_cell_ids,
        "animal_id": animal_id,
        "session_id": session_id,
    }


def read_session(path, animal_id: str, session_id: str) -> Optional[dict]:
    """Read one recording file; None when it has no positions or no spikes."""
    import scipy.io as sio

    mat_data = sio.loadmat(str(path))
    positions = mat_data.get("positions", np.array([]))
    units = mat_data.get("units", np.array([]))
    if positions.size == 0 or units.size == 0:
        return None
    return session_from_arrays(positions, units, animal_id, session_id)


def preprocess(base: Optional[Path] = None, workers: int = 1) -> dict:
    """Preprocess the two sessions of rat 11468.

    Parameters
    ----------
    base : Path, optional
        Data folder (default ``HEXMOD_DATA`` or ``<repo>/data``).
    workers : int
        Not used; accepted for a uniform interface.

    Returns
    -------
    dict
        ``{animal_id: {session_id: session}}`` in canonical order.
    """
    base = data_base(base)
    rows = {r["path"]: r for r in manifest_files(KEY) if r["path"].startswith(FOLDER + "/")}
    refuse_unlisted_files(KEY, FOLDER, "*.mat", base)
    sessions = labels.sessions_of(KEY)
    target = sorted(p for p, r in rows.items() if r["name"].split("_")[0] == TARGET_ANIMAL)
    named = sorted(f"{KEY}/{s['raw_path']}" for s in sessions)
    if target != named:
        raise labels.ManifestError(f"{KEY}: the files of rat {TARGET_ANIMAL} {target} differ from the sessions {named}")
    check_file_sizes([rows[p] for p in named], base)
    data: Dict[str, dict] = {}
    for s in sessions:
        session = read_session(base / KEY / s["raw_path"], s["animal_id"], s["session_id"])
        if session is None:
            raise labels.ManifestError(f"{KEY}: {s['raw_path']} gives no session, but the manifest lists "
                                       f"{s['session_id']}")
        data.setdefault(s["animal_id"], {})[s["session_id"]] = session
    return labels.canonical_order(data, KEY)
