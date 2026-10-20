"""Merging of the datasets, position smoothing, running speed and movement direction.

:func:`merge_and_smooth` takes the recordings of each dataset (the output of the dataset modules) and
returns one dict ``{animal_id: {session_id: session}}`` with the datasets in
``hexmod.datasets.CANONICAL_ORDER``. Every session's ``position`` dict gains three entries; spikes,
labels and the other entries are carried over unchanged.

What the code does, per session (:func:`smooth_session`):

- **Decimation.** Positions are decimated by ``int(sample_rate / 50)``, at least 1. This is 1 for every
  dataset used (all are sampled at 50 Hz, Sargolini et al. at 49.99999999891 or 50.00000000005 Hz), so
  no sample is dropped.
- **Missing positions.** NaNs in x and y are filled by linear interpolation over the sample index;
  NaNs before the first or after the last valid sample take that sample's value.
- **Smoothing.** x and y are each smoothed with a boxcar of 20 samples (0.4 s at 50 Hz;
  ``scipy.ndimage.uniform_filter1d``, ``mode="nearest"``). The window runs over the sample index, so
  in recordings with gaps in the time series (Vollan et al., whose samples at speeds below 5 cm/s were
  removed at the source) it averages samples on both sides of a gap.
- **Speed.** ``speed`` (cm/s) is the distance between successive smoothed positions divided by their
  time difference; the last sample repeats the previous value. No sample is removed here; the
  analysis keeps samples between 5 and 150 cm/s (:mod:`hexmod.rat.gridcell`).
- **Movement direction.** ``move_dir`` is ``arctan2(dy, dx)`` of successive smoothed positions, in
  radians on [0, 2 pi); the last sample repeats the previous value. It is float32 where the positions
  are float32 (Vollan et al.).
- **Smoothed positions under both names.** The smoothing writes into the position array itself, so the
  merged data hold the smoothed positions under both ``xy`` and ``xy_smoothed`` (equal arrays). The
  unsmoothed positions remain in the per-dataset files
  (``preprocessed/<dataset>.pkl``). All analyses read ``xy_smoothed``.

The ``position`` dict of a merged session holds the entries of the dataset module (``xy``, ``xy2`` for
Sargolini et al., ``time``, ``sample_rate``) followed by ``xy_smoothed``, ``speed`` and ``move_dir``.
"""

from __future__ import annotations

import warnings
from typing import Mapping

import numpy as np

from . import labels

#: Sampling rate the positions are brought to (Hz).
SAMPLE_RATE_TARGET = 50
#: Length of the boxcar window (s).
SMOOTHING_WINDOW_S = 0.4


def nan_boxcar_smoothing(arr: np.ndarray, window_size: int) -> np.ndarray:
    """Boxcar smoothing (``uniform_filter1d``, ``mode="nearest"``) that keeps NaNs in place.

    NaNs are replaced by the mean of the valid values before filtering and restored afterwards.
    """
    from scipy.ndimage import uniform_filter1d

    nan_mask = np.isnan(arr)
    arr_filled = np.copy(arr)
    arr_filled[nan_mask] = np.nanmean(arr_filled)
    smoothed = uniform_filter1d(arr_filled, size=window_size, mode="nearest")
    smoothed[nan_mask] = np.nan
    return smoothed


def linear_interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """Fill NaNs by linear interpolation over the index, in place.

    NaNs before the first valid value take that value, NaNs after the last valid value take that
    value. An array without valid values is returned unchanged.

    Returns
    -------
    ndarray
        ``arr`` itself.
    """
    n = len(arr)
    if n == 0:
        return arr
    not_nan = ~np.isnan(arr)
    if not np.any(not_nan):
        return arr
    first_valid = np.argmax(not_nan)
    arr[:first_valid] = arr[first_valid]
    last_valid = n - 1 - np.argmax(not_nan[::-1])
    arr[last_valid + 1:] = arr[last_valid]
    indices = np.arange(n)
    nan_idx = np.isnan(arr)
    arr[nan_idx] = np.interp(indices[nan_idx], indices[not_nan], arr[not_nan])
    return arr


def smooth_session(session: dict) -> bool:
    """Add ``xy_smoothed``, ``speed`` and ``move_dir`` to a session's ``position``, in place.

    The decimated positions are written back (``position["time"]`` is replaced by the decimated time
    and the position array is smoothed in place; see the module docstring).

    Returns
    -------
    bool
        False when the session has no usable positions (fewer than two samples), which is reported
        with a warning; True otherwise.
    """
    position = session.get("position")
    if position is None or "xy" not in position or "time" not in position:
        warnings.warn(f"{session.get('session_id')}: no position data, skipped", RuntimeWarning)
        return False
    pos_xy = position["xy"]
    pos_time = position["time"]
    sample_rate = position.get("sample_rate", SAMPLE_RATE_TARGET)
    if len(pos_time) < 2 or pos_xy.shape[0] < 2:
        warnings.warn(f"{session.get('session_id')}: fewer than two position samples, skipped", RuntimeWarning)
        return False

    # decimation (a factor of 1 for every dataset used)
    downsample_factor = max(1, int(sample_rate / SAMPLE_RATE_TARGET))
    pos_xy = pos_xy[::downsample_factor]
    pos_time = pos_time[::downsample_factor]
    position["time"] = pos_time

    # missing positions, then boxcar smoothing (written into the position array)
    pos_xy[:, 0] = linear_interpolate_nans(pos_xy[:, 0])
    pos_xy[:, 1] = linear_interpolate_nans(pos_xy[:, 1])
    window_size = int(SMOOTHING_WINDOW_S * SAMPLE_RATE_TARGET)
    pos_xy[:, 0] = nan_boxcar_smoothing(pos_xy[:, 0], window_size)
    pos_xy[:, 1] = nan_boxcar_smoothing(pos_xy[:, 1], window_size)

    # running speed and movement direction from successive smoothed positions
    dt = np.diff(pos_time)
    dx = np.diff(pos_xy[:, 0])
    dy = np.diff(pos_xy[:, 1])
    speed = np.sqrt(dx**2 + dy**2) / dt
    speed = np.append(speed, speed[-1])
    move_dir = np.arctan2(dy, dx)
    move_dir = np.append(move_dir, move_dir[-1])
    move_dir[move_dir < 0] += 2 * np.pi

    position["xy_smoothed"] = pos_xy
    position["speed"] = speed
    position["move_dir"] = move_dir
    return True


def smooth_recordings(data: Mapping) -> Mapping:
    """Apply :func:`smooth_session` to every session of ``{animal_id: {session_id: session}}``, in place.

    Returns
    -------
    mapping
        ``data`` itself.
    """
    for sessions in data.values():
        for session in sessions.values():
            smooth_session(session)
    return data


def merge_and_smooth(per_dataset: Mapping[str, Mapping], strict: bool = True) -> dict:
    """Smooth the recordings of several datasets and merge them in canonical order.

    Parameters
    ----------
    per_dataset : mapping
        ``{dataset key: {animal_id: {session_id: session}}}``, the output of the dataset modules (or
        their saved files). The session dicts are modified in place.
    strict : bool
        Passed to :func:`hexmod.preprocessing.labels.merge_datasets`: require the sessions and units of
        every dataset to equal the manifests.

    Returns
    -------
    dict
        ``{animal_id: {session_id: session}}``, datasets in ``CANONICAL_ORDER``.
    """
    for data in per_dataset.values():
        smooth_recordings(data)
    return labels.merge_datasets(per_dataset, strict=strict)
