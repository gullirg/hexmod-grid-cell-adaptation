"""Preprocessing of the recordings of Hafting et al. (2005).

Source: the five files of the NIRD release (doi:10.11582/2017.00017), read in manifest order:
``Hafting_Fig2c_Trial1.mat``, ``Hafting_Fig2c_Trial2.mat``, ``Hafting_Fig2d_Trial1.mat``,
``Hafting_Fig2d_Trial2.mat`` (rat 11015, animal H1) and ``rat_10925.mat`` (rat 10925, animal H2).

What the code does:

- **Sessions.** The Fig2c and Fig2d files of the same trial hold different cells recorded in the same
  trial, with identical position records; they are merged into one session (Trial1 into H1_1, Trial2
  into H1_2). ``rat_10925.mat`` gives H2_1. The position of a merged session is taken from the last
  file read (Fig2d).
- **Positions.** ``pos_x`` and ``pos_y`` are shifted so that their minima are 0 and resampled at 50 Hz
  (``sample_rate`` is the Python int 50) by linear interpolation (extrapolating at the ends) over the
  valid samples, at the times ``numpy.arange(min(t), max(t), 0.02)`` of the raw time stamps
  ``pos_timeStamps``, which start at 0. Spike times stay on the raw clock, so they are aligned with
  the positions. The recordings were made in a 180 cm cylinder (the position records span 170 to
  177 cm).
- **Units.** Every variable whose name contains ``rat11015`` or ``cell`` is the spike times of one
  unit. Units are numbered per animal from 1 over the files in reading order (so H1_1 holds cells 1 to
  7 and 15 to 17, H1_2 cells 8 to 14 and 18 to 20); the manifest holds the same numbers and they are
  checked against it. ``mod_id`` is 1 for the units of rat 11015 and 2 for those of rat 10925;
  ``tetrode_ids`` is -1 (not given in the files); labels are ``H<a>_H<a>_<s>_Tet-1_Cell<i>``.

A session dict holds, in this order: ``position`` (``xy``, ``time``, ``sample_rate``),
``spike_times``, ``mod_id``, ``unique_cell_ids``, ``tetrode_ids``, ``animal_id`` and ``session_id``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import check_file_sizes, data_base, labels, manifest_files

KEY = "hafting2005"
#: Folder of the files, relative to the data folder.
FOLDER = "hafting2005"
#: Sampling rate of the resampled positions (Hz, Python int).
SAMPLE_RATE = 50
#: Name fragment of the spike variables of rat 11015 (``mod_id`` 1; other units get 2).
RAT11015_TAG = "rat11015"


def load_mat_file(path) -> dict:
    """Load a .mat file without the MATLAB header entries (keys starting with ``__``)."""
    import scipy.io as sio

    data = sio.loadmat(str(path))
    return {key: value for key, value in data.items() if not key.startswith("__")}


def resample_position(pos_x: np.ndarray, pos_y: np.ndarray, pos_time: np.ndarray) -> dict:
    """Shift positions to minima of 0 and resample them at 50 Hz (see the module docstring).

    ``pos_x`` and ``pos_y`` are modified in place.
    """
    from scipy.interpolate import interp1d

    dt = 1 / SAMPLE_RATE
    pos_x -= np.nanmin(pos_x)
    pos_y -= np.nanmin(pos_y)
    valid_idx = ~np.isnan(pos_x)
    interp_time = np.arange(np.nanmin(pos_time), np.nanmax(pos_time), dt)
    interp_x = interp1d(pos_time[valid_idx], pos_x[valid_idx], kind="linear", fill_value="extrapolate")(interp_time)
    interp_y = interp1d(pos_time[valid_idx], pos_y[valid_idx], kind="linear", fill_value="extrapolate")(interp_time)
    return {
        "xy": np.column_stack((interp_x, interp_y)),
        "time": interp_time,
        "sample_rate": SAMPLE_RATE,
    }


def spike_variables(data: dict):
    """Return the names of the spike-time variables of a loaded file, in file order."""
    return [key for key in data.keys() if RAT11015_TAG in key or "cell" in key]


def preprocess(base: Optional[Path] = None, workers: int = 1) -> dict:
    """Preprocess the three sessions.

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
    rows = manifest_files(KEY)
    check_file_sizes(rows, base)
    data: Dict[str, dict] = {}
    cell_counter: Dict[str, int] = {}
    for row in rows:
        file_name = row["name"]
        s = labels.session_for_raw_path(KEY, file_name)
        animal_id, session_id = s["animal_id"], s["session_id"]
        content = load_mat_file(base / row["path"])
        cell_counter.setdefault(animal_id, 1)

        sessions = data.setdefault(animal_id, {})
        if session_id not in sessions:
            sessions[session_id] = {
                "position": {},
                "spike_times": {},
                "mod_id": {},
                "unique_cell_ids": {},
                "tetrode_ids": {},
                "animal_id": animal_id,
                "session_id": session_id,
            }
        session = sessions[session_id]

        pos_x, pos_y, pos_time = (content["pos_x"].squeeze(), content["pos_y"].squeeze(),
                                  content["pos_timeStamps"].squeeze())
        session["position"] = resample_position(pos_x, pos_y, pos_time)

        for key in spike_variables(content):
            cell_idx = cell_counter[animal_id]
            unit = labels.unit_for_raw_key(KEY, session_id, f"{file_name}:{key}")
            if unit["cell_id"] != cell_idx:
                raise labels.ManifestError(f"{KEY}: {file_name}:{key} is unit {cell_idx} in reading order, "
                                           f"but the manifest gives cell_id {unit['cell_id']}")
            session["spike_times"][cell_idx] = content[key].squeeze()
            session["unique_cell_ids"][cell_idx] = f"{animal_id}_{session_id}_Tet-1_Cell{cell_idx}"
            session["mod_id"][cell_idx] = 1 if RAT11015_TAG in key else 2
            session["tetrode_ids"][cell_idx] = -1
            cell_counter[animal_id] += 1
    return labels.canonical_order(data, KEY)
