"""Reader of the HDF5 session files of Barry et al. (2012) and Ólafsdóttir et al. (2016).

Both datasets are distributed as one HDF5 file each (Zenodo, doi:10.5281/zenodo.23477461), holding the
sessions analysed in the paper with the values of the original recordings. Layout of a file::

    /<session label>/                  one group per session (for example W1_1 or F1_1)
        position/time                  float64 [s]
        position/x, position/y         float64 [cm]
        units/unit_000/spike_times     float64 [s], one group per unit
        units/unit_001/spike_times     ...

Session attributes used here: ``session_label`` (equal to the group name), ``animal_label``, ``order``
(position of the session in the file, from 0; HDF5 lists groups alphabetically, so W1_10 comes before W1_2),
``rat_id`` (animal identifier of the original data), ``original_trial_id``, ``position_sampling_rate_hz``
and ``n_units``. Unit attributes: ``original_index`` (position of the unit in the session's list of units of
the original data, from 0), ``cell_id`` and ``label``. The file's README (in the record) describes every
attribute.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np

__all__ = ["read_sessions"]


def _value(attr):
    """Return an attribute as a Python value (text attributes stored as bytes are decoded)."""
    return attr.decode("utf-8") if isinstance(attr, bytes) else attr


def read_sessions(path: Union[str, Path]) -> List[dict]:
    """Read every session of a release file, in the order of the ``order`` attribute.

    Parameters
    ----------
    path : str or Path
        The HDF5 file.

    Returns
    -------
    list of dict
        One dict per session: ``session_id``, ``animal_id`` (labels of the paper), ``attrs`` (all session
        attributes), ``time``, ``x``, ``y`` (arrays as stored) and ``units``, a list of dicts with
        ``name`` (group name), ``original_index``, ``cell_id``, ``label`` and ``spike_times``, in the order
        of ``original_index``.

    Raises
    ------
    ValueError
        When the file does not have the documented layout (group name and label differ, ``order`` not
        0 to n - 1, unit indices not 0 to k - 1, or ``n_units`` not the number of unit groups).
    """
    import h5py

    out = []
    with h5py.File(str(path), "r") as f:
        names = sorted(f.keys(), key=lambda name: int(f[name].attrs["order"]))
        if [int(f[name].attrs["order"]) for name in names] != list(range(len(names))):
            raise ValueError(f"{path}: the order attributes of the sessions are not 0 to {len(names) - 1}")
        for name in names:
            group = f[name]
            attrs = {k: _value(v) for k, v in group.attrs.items()}
            if attrs["session_label"] != name:
                raise ValueError(f"{path}: group {name} holds the session label {attrs['session_label']!r}")
            units = []
            for unit_name in group["units"]:
                unit = group["units"][unit_name]
                units.append({"name": unit_name, "original_index": int(unit.attrs["original_index"]),
                              "cell_id": int(unit.attrs["cell_id"]), "label": _value(unit.attrs["label"]),
                              "spike_times": unit["spike_times"][()]})
            units.sort(key=lambda u: u["original_index"])
            if [u["original_index"] for u in units] != list(range(len(units))) or len(units) != int(attrs["n_units"]):
                raise ValueError(f"{path}: the units of {name} are not numbered 0 to {int(attrs['n_units']) - 1}")
            out.append({
                "session_id": name,
                "animal_id": attrs["animal_label"],
                "attrs": attrs,
                "time": np.asarray(group["position/time"][()]),
                "x": np.asarray(group["position/x"][()]),
                "y": np.asarray(group["position/y"][()]),
                "units": units,
            })
    return out
