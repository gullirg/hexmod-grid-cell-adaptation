"""Preprocessing of the recordings of Ólafsdóttir et al. (2016).

Source: the file ``olafsdottir2016_open_field_sessions.h5`` of the Zenodo record doi:10.5281/zenodo.23477461,
in the data folder at ``olafsdottir2016/``. It holds the 24 open-field sessions analysed in the paper (6
rats, F1 to F6; one session per recording day), with the positions and the spike times of every unit as in
the original recordings (layout: :mod:`hexmod.preprocessing.hdf5_sessions`). The sessions are read in the
order of the file, matched to the session manifest by their label and checked against the manifests
(sessions, units and unit labels).

What the code does, per session:

- **Positions.** x and y are shifted so that their minima are 0 (``xy`` is stored in Fortran order, the
  layout of the matrix in the original MATLAB data). The position time is kept as it is (it starts at
  0.02 s), and so are the spike times, so positions and spikes stay aligned. ``sample_rate`` is the
  stored value, a 0-d numpy uint8 array (50). Head direction is not used.
- **Units.** The unit with ``original_index`` ``i`` (counted from 0) is unit ``i + 1``. Labels are
  ``F<a>_F<a>_<s>_TetUnknownTet_Cell<i+1>`` (the data give no tetrode number). ``trial_id`` is the
  trial code of the original data, the recording date (a numpy string).

A session dict holds, in this order: ``position`` (``xy``, ``time``, ``sample_rate``),
``spike_times``, ``unique_cell_ids``, ``animal_id``, ``session_id`` and ``trial_id``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import check_animal_map, check_file_sizes, data_base, hdf5_sessions, labels, manifest_files

KEY = "olafsdottir2016"


def spike_array(entry) -> np.ndarray:
    """Return the spike times of one unit as a flat float64 array."""
    if isinstance(entry, np.ndarray) and entry.dtype == "O":
        flattened = []
        for sub_array in entry:
            flattened.extend(np.array(sub_array).squeeze().tolist())
        return np.array(flattened, dtype=np.float64)
    return np.array(entry, dtype=np.float64).squeeze()


def session_from_release(record: dict, animal_id: str, session_id: str) -> dict:
    """Build one session from a session of the HDF5 file (:func:`hdf5_sessions.read_sessions`)."""
    trial_id_original = np.str_(record["attrs"]["original_trial_id"])

    pos_xy = np.asfortranarray(np.column_stack((record["x"], record["y"])))
    pos_time = record["time"]
    sample_rate = np.array([[record["attrs"]["position_sampling_rate_hz"]]], dtype=np.uint8).squeeze()

    pos_xy[:, 0] -= np.nanmin(pos_xy[:, 0])
    pos_xy[:, 1] -= np.nanmin(pos_xy[:, 1])

    spike_times_dict: Dict[int, np.ndarray] = {}
    unique_cell_ids: Dict[int, str] = {}
    for unit in record["units"]:
        cell_idx = unit["original_index"]
        spike_times_dict[cell_idx + 1] = spike_array(unit["spike_times"])
        unique_cell_ids[cell_idx + 1] = f"{animal_id}_{session_id}_TetUnknownTet_Cell{cell_idx + 1}"

    return {
        "position": {
            "xy": pos_xy,
            "time": pos_time,
            "sample_rate": sample_rate,
        },
        "spike_times": spike_times_dict,
        "unique_cell_ids": unique_cell_ids,
        "animal_id": animal_id,
        "session_id": session_id,
        "trial_id": trial_id_original,
    }


def preprocess(base: Optional[Path] = None, workers: int = 1) -> dict:
    """Preprocess the 24 sessions.

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
    (row,) = manifest_files(KEY)
    check_file_sizes([row], base)
    records = hdf5_sessions.read_sessions(base / row["path"])
    manifest = {s["session_id"]: s for s in labels.sessions_of(KEY)}
    data: Dict[str, dict] = {}
    rats = []
    for record in records:
        s = manifest.get(record["session_id"])
        if s is None or s["animal_id"] != record["animal_id"]:
            raise labels.ManifestError(f"{KEY}: session {record['session_id']} of animal {record['animal_id']} "
                                       "is not in the session manifest")
        rats.append((str(record["attrs"]["rat_id"]), s["animal_id"]))
        session = session_from_release(record, s["animal_id"], s["session_id"])
        data.setdefault(s["animal_id"], {})[s["session_id"]] = session
    check_animal_map(KEY, rats)
    return labels.canonical_order(data, KEY)
