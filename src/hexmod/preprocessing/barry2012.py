"""Preprocessing of the recordings of Barry et al. (2012).

Source: the file ``barry2012_familiar_box_sessions.h5`` of the Zenodo record doi:10.5281/zenodo.23477461,
in the data folder at ``barry2012/``. It holds the 57 sessions analysed in the paper (7 rats, W1 to W7; the
first and the last trial of each recording day, both in the familiar box), with the positions and the spike
times of every unit as in the original recordings (layout: :mod:`hexmod.preprocessing.hdf5_sessions`). The
sessions are read in the order of the file, matched to the session manifest by their label and checked
against the manifests (sessions, units and unit labels).

What the code does, per session:

- **Positions.** x and y are shifted so that their minima are 0. The position time is shifted so that
  its minimum is 0; it starts at 0.02 s in every session. ``sample_rate`` is the Python int 50 when
  ``1 / median(diff(time))`` is within 1 Hz of 50 (true for every session).
- **Spikes.** Spike times are not shifted, so relative to the positions every spike lies 0.02 s later
  than recorded. The analysis assigns each spike to the first position sample at or after its time
  (:meth:`hexmod.rat.gridcell.GridCellAnalysis.compute_spike_array`), so a spike is paired with the
  position recorded 0.02 to 0.04 s after it. This is kept as in the published analysis.
- **Units.** The data hold no cluster-quality, tetrode or cell-number fields, so every unit is kept; the
  unit with ``original_index`` ``i`` (counted from 0) is unit ``i``, labelled ``W<a>_W<a>_<s>_T0_C<i>``.

A session dict holds, in this order: ``position`` (``xy``, ``time``, ``sample_rate``),
``spike_times``, ``unique_cell_ids``, ``animal_id`` and ``session_id``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import check_animal_map, check_file_sizes, data_base, hdf5_sessions, labels, manifest_files

KEY = "barry2012"
#: Nominal sampling rate of the positions (Hz, Python int).
SAMPLE_RATE = 50


def session_from_release(record: dict, animal_id: str, session_id: str) -> dict:
    """Build one session from a session of the HDF5 file (:func:`hdf5_sessions.read_sessions`)."""
    # positions with minima at 0, time from 0
    x = record["x"]
    y = record["y"]
    x -= np.min(x)
    y -= np.min(y)
    t = record["time"]
    t -= np.min(t)

    computed_sample_rate = 1 / np.median(np.diff(t))
    sample_rate = SAMPLE_RATE if np.isclose(computed_sample_rate, SAMPLE_RATE, atol=1) else computed_sample_rate

    # units: without cluster-quality, tetrode and cell fields every unit is kept
    spike_times_dict: Dict[int, np.ndarray] = {}
    unique_cell_ids: Dict[int, str] = {}
    for unit in record["units"]:
        idx = unit["original_index"]
        spike_times = np.array(unit["spike_times"]).squeeze()
        if spike_times.ndim > 1:
            spike_times = spike_times.flatten()
        spike_times_dict[idx] = spike_times
        unique_cell_ids[idx] = f"{animal_id}_{session_id}_T0_C{idx}"

    return {
        "position": {
            "xy": np.column_stack((x, y)),
            "time": t,
            "sample_rate": sample_rate,
        },
        "spike_times": spike_times_dict,
        "unique_cell_ids": unique_cell_ids,
        "animal_id": animal_id,
        "session_id": session_id,
    }


def preprocess(base: Optional[Path] = None, workers: int = 1) -> dict:
    """Preprocess the 57 sessions.

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
