"""Preprocessing of the recordings of Sargolini et al. (2006).

Source: the 150 files stored in the last volume of the split archive of the NIRD release
(doi:10.11582/2017.00019), extracted to ``sargolini2006/subset/``: 13 position files
(``<rat>-<trial>_POS.mat``), 109 spike files (``<rat>-<trial>_T<t>C<c>.mat``) and 28 EEG files
(``_EEG.mat``, ``_EGF.mat``), which are not read. The folder must hold exactly these files: every
other file would be read as a spike file and change the units (the complete archive holds 106 more
position files and 519 more spike files). The files are read in the order of the manifest column
``listdir_order``.

What the code does:

- **Positions.** ``posx``, ``posy`` (first LED), ``posx2``, ``posy2`` (second LED) and ``post`` are
  each shifted so that their minimum is 0 (the first time stamp is 0 in every file, so time is
  unchanged). ``sample_rate`` is ``1 / median(diff(time))``, a numpy float64 (49.999999998908606 or
  50.000000000045475 Hz). The second LED is stored as ``position["xy2"]`` and is not smoothed later.
- **Spikes.** The spike times of each unit (``cellTS``) are shifted so that the unit's first spike is
  at time 0, while the position time is not shifted. Every unit's spikes therefore move earlier
  relative to the positions by that unit's first-spike latency (0.0035 to 8.34 s across the 26 units of
  the two sessions that enter the cohort, S2_1 and S2_5, and 0.01 to 0.74 s for the 7 of them in the
  cohort). All results of the paper use this alignment.
- **Unit keys.** ``cell_id`` numbers the spike files of the whole dataset from 1, in file order (for
  example unit ``11265-16030601+02_T1C4.mat`` of S2_1 is cell 78); the manifest holds the same
  numbers and they are checked against it. ``unique_cell_ids`` are ``S<a>_S<a>_<s>_T<t>C<c>``.
- **Labels.** Animal and session labels come from the session manifest; ``trial_id`` is the trial code
  of the file name (for example ``"16030601+02"``).

A session dict holds, in this order: ``position`` (``xy``, ``xy2``, ``time``, ``sample_rate``),
``spike_times``, ``unique_cell_ids``, ``animal_id``, ``session_id`` and ``trial_id``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import check_file_sizes, data_base, labels, manifest_files, refuse_unlisted_files

KEY = "sargolini2006"
#: Folder of the extracted files, relative to the data folder.
FOLDER = "sargolini2006/subset"


def read_position_file(path) -> dict:
    """Read a position file and return the ``position`` dict (see the module docstring)."""
    import scipy.io as sio

    pos_data = sio.loadmat(str(path))
    posx = np.array(pos_data.get("posx", [])).squeeze()
    posy = np.array(pos_data.get("posy", [])).squeeze()
    post = np.array(pos_data.get("post", [])).squeeze()
    posx2 = np.array(pos_data.get("posx2", [])).squeeze()
    posy2 = np.array(pos_data.get("posy2", [])).squeeze()

    # minima at 0
    posx -= np.nanmin(posx)
    posy -= np.nanmin(posy)
    posx2 -= np.nanmin(posx2)
    posy2 -= np.nanmin(posy2)
    post -= np.nanmin(post)

    computed_sample_rate = 1 / np.median(np.diff(post)) if len(post) > 1 else np.nan
    return {
        "xy": np.column_stack((posx, posy)),
        "xy2": np.column_stack((posx2, posy2)),
        "time": post,
        "sample_rate": computed_sample_rate,
    }


def read_spike_file(path) -> np.ndarray:
    """Read a spike file and return its spike times, shifted so that the first spike is at 0.

    An empty array is returned unchanged.
    """
    import scipy.io as sio

    spike_data = sio.loadmat(str(path))
    spike_times = np.array(spike_data.get("cellTS", [])).squeeze()
    if spike_times.size == 0:
        return spike_times
    spike_times -= np.min(spike_times)
    return spike_times


def _session_of_file(name: str) -> dict:
    """Return the session manifest row of a position or spike file (``<rat>-<trial>_...``)."""
    prefix = name.split("_")[0]
    return labels.session_for_raw_path(KEY, f"subset/{prefix}_POS.mat")


def preprocess(base: Optional[Path] = None, workers: int = 1) -> dict:
    """Preprocess the 13 sessions of the subset.

    Parameters
    ----------
    base : Path, optional
        Data folder (default ``HEXMOD_DATA`` or ``<repo>/data``).
    workers : int
        Not used (the files are small); accepted for a uniform interface.

    Returns
    -------
    dict
        ``{animal_id: {session_id: session}}`` in canonical order.
    """
    base = data_base(base)
    rows = [r for r in manifest_files(KEY) if r["path"].startswith(FOLDER + "/")]
    rows.sort(key=lambda r: int(r["listdir_order"]))
    refuse_unlisted_files(KEY, FOLDER, "*", base)
    check_file_sizes(rows, base)
    all_files = [r["name"] for r in rows]
    position_files = [f for f in all_files if "_POS.mat" in f]
    spike_files = [f for f in all_files if "_POS.mat" not in f and "_EEG.mat" not in f and "_EGF.mat" not in f]

    data: Dict[str, dict] = {}
    for pos_file in position_files:
        row = labels.session_for_raw_path(KEY, f"subset/{pos_file}")
        animal_id, session_id = row["animal_id"], row["session_id"]
        session_date = pos_file.split("_")[0].split("-")[1]
        sessions = data.setdefault(animal_id, {})
        if session_id in sessions:
            raise labels.ManifestError(f"{KEY}: two position files for session {session_id}")
        sessions[session_id] = {
            "position": read_position_file(base / FOLDER / pos_file),
            "spike_times": {},
            "unique_cell_ids": {},
            "animal_id": animal_id,
            "session_id": session_id,
            "trial_id": session_date,
        }

    cell_counter = 1  # one counter over the spike files of the whole dataset
    for spike_file in spike_files:
        row = _session_of_file(spike_file)
        animal_id, session_id = row["animal_id"], row["session_id"]
        tetrode_cell = spike_file.split("_")[1].split(".")[0]
        spike_times = read_spike_file(base / FOLDER / spike_file)
        if spike_times.size == 0:
            continue
        unit = labels.unit_for_raw_key(KEY, session_id, spike_file)
        if unit["cell_id"] != cell_counter:
            raise labels.ManifestError(f"{KEY}: {spike_file} is unit {cell_counter} in file order, "
                                       f"but the manifest gives cell_id {unit['cell_id']}")
        session = data[animal_id][session_id]
        session["spike_times"][cell_counter] = spike_times
        session["unique_cell_ids"][cell_counter] = f"{animal_id}_{session_id}_{tetrode_cell}"
        cell_counter += 1
    return labels.canonical_order(data, KEY)
