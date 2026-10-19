"""Tests of the preprocessing step (``hexmod.preprocessing``, ``scripts/step01_preprocess.py``).

The synthetic tests need no data: they build small recordings (MATLAB files written with
``scipy.io.savemat``, HDF5 files written with ``h5py``, where the reader takes a file) and check each
convention of the dataset modules and of the trajectory stage. Tests marked ``data`` read the raw datasets
(``HEXMOD_DATA``); those also marked ``frozen`` compare the result with the reference preprocessed files in
``HEXMOD_FROZEN`` (``preprocessed/<dataset>.pkl`` and ``preprocessed/trajectories.pkl``): positions, times,
spikes, labels, dict order, dtypes and the type of ``sample_rate`` must be identical. The reference may hold
entries that the preprocessing does not write (``reference_as_written``): sessions outside the session
manifest and the theta phase (``theta_phases``), which no analysis uses.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from hexmod import data_sources
from hexmod import datasets as ds
from hexmod.preprocessing import (barry2012, brun2008, check_animal_map, check_file_sizes, dataset_module,
                                  hafting2005, hdf5_sessions, labels, olafsdottir2016, refuse_unlisted_files,
                                  sargolini2006, trajectories, vollan2025)

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------- helpers
def differences(a, b, path="", out=None):
    """Return a list of differences between two nested recordings (empty when identical).

    Identical means: same dict keys in the same order, same Python types, arrays with the same dtype,
    shape and values (NaN equal to NaN), numpy scalars with the same dtype and value.
    """
    out = [] if out is None else out
    if len(out) > 20:
        return out
    if type(a) is not type(b):
        out.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
    elif isinstance(a, dict):
        if list(a) != list(b):
            out.append(f"{path}: keys or key order differ")
        else:
            for k in a:
                differences(a[k], b[k], f"{path}[{k!r}]", out)
    elif isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            out.append(f"{path}: {a.dtype}{a.shape} != {b.dtype}{b.shape}")
        elif not np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"):
            out.append(f"{path}: values differ")
    elif isinstance(a, np.generic):
        if a.dtype != b.dtype or not (a == b or (a != a and b != b)):
            out.append(f"{path}: {a!r} != {b!r}")
    elif a != b and not (isinstance(a, float) and a != a and b != b):
        out.append(f"{path}: {a!r} != {b!r}")
    return out


def frozen_recordings(frozen_dir: Path, name: str):
    path = frozen_dir / "preprocessed" / name
    if not path.is_file():
        pytest.skip(f"reference file missing: preprocessed/{name}")
    from hexmod.io import load_cache

    return reference_as_written(load_cache(path))


#: Entries of reference session dicts that the preprocessing does not write.
NOT_WRITTEN = ("theta_phases",)


def reference_as_written(data: dict) -> dict:
    """Reference recordings restricted to the sessions of the manifest, without the entries ``NOT_WRITTEN``."""
    known = set(labels.read_sessions()["session_id"])
    out = {}
    for animal, sessions in data.items():
        kept = {s: {k: v for k, v in sd.items() if k not in NOT_WRITTEN} for s, sd in sessions.items() if s in known}
        if kept:
            out[animal] = kept
    return out


def session(xy, time, sample_rate=50, sid="X1_1"):
    return {"position": {"xy": xy, "time": time, "sample_rate": sample_rate}, "spike_times": {1: np.array([0.5])},
            "unique_cell_ids": {1: "u"}, "animal_id": "X1", "session_id": sid}


def test_differences_helper():
    a = {"A": {"s": {"position": {"xy": np.zeros((3, 2)), "sample_rate": 50}, "spike_times": {1: np.ones(2)}}}}
    assert differences(a, copy.deepcopy(a)) == []
    b = copy.deepcopy(a)
    b["A"]["s"]["position"]["sample_rate"] = 50.0
    assert differences(a, b)
    b = copy.deepcopy(a)
    b["A"]["s"]["position"]["xy"] = b["A"]["s"]["position"]["xy"].astype(np.float32)
    assert differences(a, b)
    b = copy.deepcopy(a)
    b["A"]["s"]["spike_times"][1][1] += 1e-12
    assert differences(a, b)
    c = {"A": {"s": {"spike_times": {1: 1, 2: 2}}}}
    d = {"A": {"s": {"spike_times": {2: 2, 1: 1}}}}
    assert differences(c, d)


# ------------------------------------------------------------------------------ trajectory stage
def test_linear_interpolate_nans():
    arr = np.array([np.nan, 1.0, np.nan, 3.0, np.nan, np.nan])
    out = trajectories.linear_interpolate_nans(arr)
    assert out is arr
    np.testing.assert_array_equal(arr, [1, 1, 2, 3, 3, 3])
    allnan = np.full(4, np.nan)
    assert np.isnan(trajectories.linear_interpolate_nans(allnan)).all()
    assert trajectories.linear_interpolate_nans(np.array([])).size == 0


def test_nan_boxcar_smoothing():
    from scipy.ndimage import uniform_filter1d

    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    np.testing.assert_array_equal(trajectories.nan_boxcar_smoothing(x, 20),
                                  uniform_filter1d(x, size=20, mode="nearest"))
    x[[5, 50]] = np.nan
    out = trajectories.nan_boxcar_smoothing(x, 20)
    assert np.isnan(out[[5, 50]]).all() and np.isfinite(np.delete(out, [5, 50])).all()


def test_smooth_session_values_and_layout():
    from scipy.ndimage import uniform_filter1d

    rng = np.random.default_rng(2)
    n = 500
    t = np.arange(n) * 0.02
    xy = np.cumsum(rng.normal(size=(n, 2)), axis=0) + 50
    xy[[0, 10, 11, n - 1], 0] = np.nan
    raw = xy.copy()
    s = session(xy, t)
    assert trajectories.smooth_session(s)
    pos = s["position"]
    assert list(pos) == ["xy", "time", "sample_rate", "xy_smoothed", "speed", "move_dir"]
    # the smoothed positions are written into the position array itself
    np.testing.assert_array_equal(pos["xy"], pos["xy_smoothed"])
    assert np.shares_memory(pos["xy"], pos["xy_smoothed"])
    # expected values: interpolation over the index, boxcar of 20 samples
    filled = raw.copy()
    for k in range(2):
        trajectories.linear_interpolate_nans(filled[:, k])
    expect = np.column_stack([uniform_filter1d(filled[:, k], size=20, mode="nearest") for k in range(2)])
    np.testing.assert_array_equal(pos["xy_smoothed"], expect)
    d = np.diff(expect, axis=0)
    speed = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2) / np.diff(t)
    np.testing.assert_array_equal(pos["speed"], np.append(speed, speed[-1]))
    ang = np.arctan2(d[:, 1], d[:, 0])
    ang = np.append(ang, ang[-1])
    ang[ang < 0] += 2 * np.pi
    np.testing.assert_array_equal(pos["move_dir"], ang)
    assert (pos["move_dir"] >= 0).all() and (pos["move_dir"] < 2 * np.pi).all()


def test_smooth_session_dtypes_follow_positions():
    n = 100
    t = np.arange(n) * 0.02
    xy = np.column_stack([np.linspace(0, 50, n), np.linspace(10, 0, n)]).astype(np.float32)
    s = session(xy, t)
    trajectories.smooth_session(s)
    assert s["position"]["xy_smoothed"].dtype == np.float32
    assert s["position"]["move_dir"].dtype == np.float32
    assert s["position"]["speed"].dtype == np.float64


@pytest.mark.parametrize("rate, factor", [(50, 1), (np.float64(49.999999998908606), 1), (np.array(50, np.uint8), 1),
                                          (100, 2)])
def test_smooth_session_decimation(rate, factor):
    n = 200
    t = np.arange(n) / 100.0
    xy = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    s = session(xy, t, sample_rate=rate)
    trajectories.smooth_session(s)
    assert len(s["position"]["time"]) == len(range(0, n, factor))
    assert len(s["position"]["speed"]) == len(s["position"]["time"])


def test_smooth_session_skips_short_sessions():
    s = session(np.zeros((1, 2)), np.zeros(1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert not trajectories.smooth_session(s)
    assert caught and "xy_smoothed" not in s["position"]


def test_merge_and_smooth_orders_datasets(monkeypatch):
    calls = []
    monkeypatch.setattr(labels, "merge_datasets", lambda per, strict=True: calls.append(list(per)) or per)
    per = {"hafting2005": {"H1": {"H1_1": session(np.ones((30, 2)), np.arange(30) * 0.02, sid="H1_1")}}}
    trajectories.merge_and_smooth(per)
    assert calls == [["hafting2005"]]
    assert "speed" in per["hafting2005"]["H1"]["H1_1"]["position"]


# ------------------------------------------------------------------------------- dataset modules
def test_vollan_session_from_synthetic_file(tmp_path):
    from scipy.io import loadmat, savemat

    n = 2000  # 20 s at 100 Hz
    t = np.arange(n) * 0.01 + 3.0
    x = (np.linspace(0.2, 1.2, n)).astype(np.float32)
    y = (np.linspace(0.5, 0.9, n)).astype(np.float32)
    ranges = np.array([[4.0, 8.0], [10.005, 15.0]])
    spikes_in = np.sort(np.r_[np.linspace(4.1, 7.9, 30), np.linspace(10.1, 14.9, 30)])
    mec = np.zeros((4, 1), dtype=[("spikeTimes", object), ("isGrid", object)])
    mec[0, 0] = (spikes_in.reshape(-1, 1), np.array([[1]]))                   # grid cell, kept
    mec[1, 0] = (spikes_in.reshape(-1, 1), np.array([[0]]))                   # not a grid cell
    mec[2, 0] = (np.linspace(4.1, 7.9, 19).reshape(-1, 1), np.array([[1]]))  # 19 spikes in range
    mec[3, 0] = (np.r_[1.0, spikes_in, 20.0].reshape(-1, 1), np.array([[1]]))  # spikes outside dropped
    savemat(tmp_path / "f.mat", {"Dsession": {"t": t.reshape(-1, 1), "x": x.reshape(-1, 1), "y": y.reshape(-1, 1),
                                              "dt": 0.01, "units": {"mec": mec},
                                              "sessions": {"validTimeRanges": ranges}}})
    d = loadmat(tmp_path / "f.mat", simplify_cells=False)["Dsession"][0, 0]
    s = vollan2025.session_from_dsession(d, "V9", "V9_1")
    keep = ((t >= 4.0) & (t <= 8.0)) | ((t >= 10.005) & (t <= 15.0))
    tv = t[keep] - t[keep][0]
    np.testing.assert_array_equal(s["position"]["time"], tv[::2])
    xv = (x * 100)[keep][::2]
    np.testing.assert_array_equal(s["position"]["xy"][:, 0], xv - xv.min())
    assert s["position"]["xy"].dtype == np.float32
    assert s["position"]["sample_rate"] == 50 and type(s["position"]["sample_rate"]) is int
    assert list(s["spike_times"]) == [1, 4]
    np.testing.assert_array_equal(s["spike_times"][1], spikes_in - t[keep][0])
    np.testing.assert_array_equal(s["spike_times"][4], s["spike_times"][1])
    assert s["unique_cell_ids"] == {1: "V9_V9_1_cell1", 4: "V9_V9_1_cell4"}
    assert s["mod_id"] == {1: None, 4: None}
    assert s["time"] is s["position"]["time"]
    assert list(s) == ["position", "spike_times", "mod_id", "unique_cell_ids", "animal_id", "session_id", "time"]


def test_vollan_mask_spikes_inclusive():
    out = vollan2025.mask_spikes(np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0]), np.array([[1.0, 2.0], [3.0, 3.5]]))
    np.testing.assert_array_equal(out, [1.0, 1.5, 2.0, 3.0])


def test_sargolini_files(tmp_path):
    from scipy.io import savemat

    n = 300
    post = np.arange(n) * 0.02
    savemat(tmp_path / "1-2_POS.mat", {"posx": np.linspace(-40, 60, n).reshape(-1, 1),
                                       "posy": np.linspace(-10, 5, n).reshape(-1, 1),
                                       "post": post.reshape(-1, 1),
                                       "posx2": np.linspace(-38, 62, n).reshape(-1, 1),
                                       "posy2": np.linspace(-9, 6, n).reshape(-1, 1)})
    pos = sargolini2006.read_position_file(tmp_path / "1-2_POS.mat")
    assert list(pos) == ["xy", "xy2", "time", "sample_rate"]
    assert np.allclose(pos["xy"].min(axis=0), 0) and np.allclose(pos["xy2"].min(axis=0), 0)
    np.testing.assert_array_equal(pos["time"], post)
    assert type(pos["sample_rate"]) is np.float64
    assert pos["sample_rate"] == 1 / np.median(np.diff(post))
    # each unit's spikes are shifted so that its first spike is at 0 (documented convention)
    savemat(tmp_path / "1-2_T1C1.mat", {"cellTS": np.array([[2.5], [3.0], [4.25]])})
    np.testing.assert_array_equal(sargolini2006.read_spike_file(tmp_path / "1-2_T1C1.mat"), [0.0, 0.5, 1.75])


def test_brun_session_from_arrays():
    t = np.arange(6) * 0.02 + 10.0
    positions = np.column_stack([t, [100, np.nan, 104, 106, 108, np.nan], [50, 52, 54, 56, 58, 60]])
    units = np.array([[10.05, 3, 2], [10.01, 4, 1], [10.07, 3, 2], [10.09, 3, 4]])
    s = brun2008.session_from_arrays(positions, units, "B1", "B1_1")
    np.testing.assert_allclose(s["position"]["xy"][:, 0], [0, 1, 2, 3, 4, 4])
    np.testing.assert_allclose(s["position"]["xy"][:, 1], [0, 1, 2, 3, 4, 5])
    np.testing.assert_allclose(s["position"]["time"], t - 10.0)
    assert s["position"]["sample_rate"] == 50 and type(s["position"]["sample_rate"]) is int
    assert list(s["spike_times"]) == [2, 1, 4]  # order of first spike, keyed by cell number
    np.testing.assert_allclose(s["spike_times"][2], [0.05, 0.07])
    assert s["unique_cell_ids"] == {2: "B1_B1_1_Tet3_Cell2", 1: "B1_B1_1_Tet4_Cell1", 4: "B1_B1_1_Tet3_Cell4"}


def test_hafting_resample_position():
    pos_t = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    pos_x = np.array([10.0, 11.0, np.nan, 13.0, 14.0])
    pos_y = np.array([5.0, 5.0, 5.0, 5.0, 6.0])
    p = hafting2005.resample_position(pos_x, pos_y, pos_t)
    np.testing.assert_array_equal(p["time"], np.arange(0.0, 2.0, 0.02))
    np.testing.assert_allclose(p["xy"][:, 0], 2 * p["time"], atol=1e-12)
    assert p["sample_rate"] == 50 and type(p["sample_rate"]) is int
    assert hafting2005.spike_variables({"pos_x": 1, "rat11015_t5c1": 2, "cell3": 3, "other": 4}) == ["rat11015_t5c1",
                                                                                                    "cell3"]


def test_olafsdottir_spike_array():
    nested = np.empty(2, dtype=object)
    nested[0] = np.array([[1.0], [2.0]])
    nested[1] = np.array([[3.5], [4.0]])
    out = olafsdottir2016.spike_array(nested)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.5, 4.0])
    np.testing.assert_array_equal(olafsdottir2016.spike_array(np.array([[1], [2]])), [1.0, 2.0])


def _write_release_file(path, sessions, rate_dtype):
    """A small HDF5 file in the layout of the Zenodo release (``hdf5_sessions``)."""
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        for order, (label, animal, rat, trial, t, x, y, units) in enumerate(sessions):
            g = f.create_group(label)
            g.attrs.update({"session_label": label, "animal_label": animal, "order": order, "rat_id": rat,
                            "original_trial_id": trial, "n_units": len(units),
                            "position_sampling_rate_hz": np.array(50, dtype=rate_dtype)})
            for name, v in (("time", t), ("x", x), ("y", y)):
                g.create_dataset(f"position/{name}", data=v)
            for i, spikes in enumerate(units):
                u = g.create_group(f"units/unit_{i:03d}")
                u.create_dataset("spike_times", data=spikes)
                u.attrs.update({"original_index": i, "cell_id": i, "label": f"unit {i}"})


def _release_sessions(rng, labels_):
    out = []
    for label, animal, rat in labels_:
        n = 400
        t = np.arange(1, n + 1) * 0.02
        x = 20 + np.cumsum(rng.normal(0, 0.5, n))
        y = 5 + np.cumsum(rng.normal(0, 0.5, n))
        units = [np.sort(rng.uniform(0.0, t[-1], k)) for k in (30, 12, 51)]
        out.append((label, animal, rat, "20141210", t, x, y, units))
    return out


def test_hdf5_sessions_follow_the_order_attribute(tmp_path):
    rng = np.random.default_rng(3)
    spec = _release_sessions(rng, [("W1_2", "W1", "838"), ("W1_10", "W1", "838"), ("W2_1", "W2", "864")])
    _write_release_file(tmp_path / "r.h5", spec, np.int64)
    got = hdf5_sessions.read_sessions(tmp_path / "r.h5")
    assert [s["session_id"] for s in got] == ["W1_2", "W1_10", "W2_1"]   # not the alphabetical order of HDF5
    assert [u["name"] for u in got[0]["units"]] == ["unit_000", "unit_001", "unit_002"]
    np.testing.assert_array_equal(got[1]["units"][2]["spike_times"], spec[1][7][2])
    assert got[2]["animal_id"] == "W2" and got[2]["attrs"]["rat_id"] == "864"


@pytest.mark.parametrize("damage, message", [
    ("order", "order attributes"),
    ("label", "holds the session label"),
    ("n_units", "not numbered"),
    ("unit_gap", "not numbered"),
])
def test_hdf5_sessions_refuse_a_wrong_layout(tmp_path, damage, message):
    """A file whose session order, session labels or unit numbering differ from the documented layout is refused."""
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(7)
    path = tmp_path / "r.h5"
    _write_release_file(path, _release_sessions(rng, [("W1_1", "W1", "838"), ("W1_2", "W1", "838")]), np.int64)
    with h5py.File(path, "r+") as f:
        if damage == "order":
            f["W1_2"].attrs["order"] = 0                          # two sessions with order 0
        elif damage == "label":
            f["W1_2"].attrs["session_label"] = "W1_3"
        elif damage == "n_units":
            f["W1_1"].attrs["n_units"] = 4                        # the session has three unit groups
        else:
            f["W1_1/units/unit_002"].attrs["original_index"] = 3   # unit indices 0, 1 and 3
    with pytest.raises(ValueError, match=message):
        hdf5_sessions.read_sessions(path)


def test_barry2012_session_from_release(tmp_path):
    rng = np.random.default_rng(4)
    spec = _release_sessions(rng, [("W1_1", "W1", "838")])
    _write_release_file(tmp_path / "r.h5", spec, np.int64)
    (rec,) = hdf5_sessions.read_sessions(tmp_path / "r.h5")
    s = barry2012.session_from_release(rec, "W1", "W1_1")
    _, _, _, _, t, x, y, units = spec[0]
    assert list(s) == ["position", "spike_times", "unique_cell_ids", "animal_id", "session_id"]
    np.testing.assert_array_equal(s["position"]["xy"], np.column_stack((x - x.min(), y - y.min())))
    np.testing.assert_array_equal(s["position"]["time"], t - t.min())
    assert s["position"]["sample_rate"] == 50 and type(s["position"]["sample_rate"]) is int
    assert list(s["spike_times"]) == [0, 1, 2]
    np.testing.assert_array_equal(s["spike_times"][1], units[1])     # spike times are not shifted
    assert s["unique_cell_ids"] == {i: f"W1_W1_1_T0_C{i}" for i in range(3)}


def test_olafsdottir2016_session_from_release(tmp_path):
    rng = np.random.default_rng(5)
    spec = _release_sessions(rng, [("F1_1", "F1", "R2117")])
    _write_release_file(tmp_path / "r.h5", spec, np.uint8)
    (rec,) = hdf5_sessions.read_sessions(tmp_path / "r.h5")
    s = olafsdottir2016.session_from_release(rec, "F1", "F1_1")
    _, _, _, _, t, x, y, units = spec[0]
    assert list(s) == ["position", "spike_times", "unique_cell_ids", "animal_id", "session_id", "trial_id"]
    xy = s["position"]["xy"]
    assert xy.flags["F_CONTIGUOUS"] and xy.dtype == np.float64
    np.testing.assert_array_equal(xy, np.column_stack((x - x.min(), y - y.min())))
    np.testing.assert_array_equal(s["position"]["time"], t)          # time kept as recorded
    rate = s["position"]["sample_rate"]
    assert isinstance(rate, np.ndarray) and rate.shape == () and rate.dtype == np.uint8 and rate == 50
    assert list(s["spike_times"]) == [1, 2, 3]
    np.testing.assert_array_equal(s["spike_times"][3], units[2])
    assert s["unique_cell_ids"][1] == "F1_F1_1_TetUnknownTet_Cell1"
    assert type(s["trial_id"]) is np.str_ and s["trial_id"] == "20141210"


def test_release_preprocessing_checks_the_manifest(tmp_path, monkeypatch):
    """A session of the file that the manifest does not list stops the preprocessing."""
    rng = np.random.default_rng(6)
    path = tmp_path / "barry2012" / "barry2012_familiar_box_sessions.h5"
    path.parent.mkdir()
    _write_release_file(path, _release_sessions(rng, [("W9_1", "W9", "1")]), np.int64)
    row = dict(data_sources.read_manifest("barry2012")[0], bytes=path.stat().st_size)
    monkeypatch.setattr(barry2012, "manifest_files", lambda key: [row])
    with pytest.raises(labels.ManifestError, match="W9_1"):
        barry2012.preprocess(tmp_path)


# ------------------------------------------------------------------------------ manifest checks
def test_refuse_unlisted_files(tmp_path):
    rows = [r for r in data_sources.read_manifest("sargolini2006") if r["role"] == "member"]
    folder = tmp_path / "sargolini2006" / "subset"
    folder.mkdir(parents=True)
    for r in rows:
        (tmp_path / r["path"]).touch()
    (folder / "download.mat.part").touch()
    (folder / ".DS_Store").touch()
    refuse_unlisted_files("sargolini2006", "sargolini2006/subset", "*", tmp_path)
    (folder / "11265-99999999_T1C1.mat").touch()
    with pytest.raises(labels.ManifestError, match="not in the manifest"):
        refuse_unlisted_files("sargolini2006", "sargolini2006/subset", "*", tmp_path)


def test_check_file_sizes(tmp_path):
    rows = data_sources.read_manifest("hafting2005")
    with pytest.raises(FileNotFoundError, match="missing"):
        check_file_sizes(rows, tmp_path)


def test_check_animal_map():
    check_animal_map("barry2012", [("838", "W1"), ("838", "W1"), ("864", "W2")])
    with pytest.raises(labels.ManifestError):
        check_animal_map("barry2012", [("838", "W1"), ("864", "W1")])
    with pytest.raises(labels.ManifestError):
        check_animal_map("barry2012", [("838", "W1"), ("838", "W2")])


def test_dataset_modules():
    for key in ds.CANONICAL_ORDER:
        module = dataset_module(key)
        assert module.KEY == key and callable(module.preprocess)
    with pytest.raises(ValueError):
        dataset_module("unknown")


def test_preprocessing_does_not_import_matplotlib():
    code = ("import sys; import hexmod.preprocessing.trajectories, hexmod.preprocessing.vollan2025, "
            "hexmod.preprocessing.barry2012, hexmod.preprocessing.sargolini2006, hexmod.preprocessing.brun2008, "
            "hexmod.preprocessing.hafting2005, hexmod.preprocessing.olafsdottir2016, "
            "hexmod.preprocessing.hdf5_sessions; assert 'matplotlib' not in sys.modules")
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.fixture
def step01(tmp_path, monkeypatch):
    """The step script as a module, with results and data folders in ``tmp_path``."""
    import importlib.util

    monkeypatch.setenv("HEXMOD_RESULTS", str(tmp_path))
    monkeypatch.setenv("HEXMOD_DATA", str(tmp_path / "no_data"))
    spec = importlib.util.spec_from_file_location("step01_preprocess", REPO / "scripts" / "step01_preprocess.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_step01_reports_missing_data(step01, tmp_path, capsys):
    """Without data the step stops before any work, naming every dataset that is missing."""
    assert step01.main(["--workers", "1"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and all(f"{key}:" in err for key in ds.CANONICAL_ORDER)
    from hexmod.io import read_json

    assert read_json(tmp_path / "logs" / "step01_preprocess.json")["status"] == "failed"
    assert not (tmp_path / "preprocessed" / "sargolini2006.pkl").exists()


@pytest.mark.smoke
def test_step01_smoke(step01, tmp_path):
    assert step01.main(["--smoke", "--workers", "1"]) == 0
    from hexmod.io import load_cache, read_csv

    out = tmp_path / "preprocessed"
    merged = load_cache(out / "trajectories.pkl")
    assert list(merged) == ["H1", "H2"]
    assert sum(len(s["spike_times"]) for v in merged.values() for s in v.values()) == 23
    pos = merged["H1"]["H1_1"]["position"]
    assert list(pos) == ["xy", "time", "sample_rate", "xy_smoothed", "speed", "move_dir"]
    assert len(read_csv(out / "sessions.csv")) == 3
    assert (tmp_path / "logs" / "step01_preprocess.json").is_file()


# ------------------------------------------------------------------------------------ data tests
def _require_files(key: str, data_dir: Path) -> None:
    if not all((data_dir / p).exists() for p in ds.DATASETS[key]["raw_subpaths"]):
        pytest.skip(f"files of {key} not found in the data folder")


@pytest.mark.data
@pytest.mark.frozen
@pytest.mark.parametrize("key", ["sargolini2006", "brun2008", "hafting2005", "olafsdottir2016", "barry2012"])
def test_small_datasets_equal_reference(key, frozen_dir, data_dir):
    _require_files(key, data_dir)
    got = dataset_module(key).preprocess()
    assert differences(got, frozen_recordings(frozen_dir, f"{key}.pkl")) == []


@pytest.mark.data
@pytest.mark.frozen
@pytest.mark.parametrize("key", ["olafsdottir2016", "barry2012"])
def test_release_datasets_keep_the_memory_layout_of_the_reference(key, frozen_dir, data_dir):
    _require_files(key, data_dir)
    got = dataset_module(key).preprocess()
    ref = frozen_recordings(frozen_dir, f"{key}.pkl")
    for animal, sessions in ref.items():
        for sid, s in sessions.items():
            a, b = got[animal][sid]["position"]["xy"], s["position"]["xy"]
            assert (a.flags["C_CONTIGUOUS"], a.flags["F_CONTIGUOUS"]) == (b.flags["C_CONTIGUOUS"],
                                                                          b.flags["F_CONTIGUOUS"]), sid


@pytest.mark.data
@pytest.mark.frozen
@pytest.mark.slow
def test_vollan_equals_reference(frozen_dir):
    got = vollan2025.preprocess(workers=2)
    assert differences(got, frozen_recordings(frozen_dir, "vollan2025.pkl")) == []


@pytest.mark.data
@pytest.mark.slow
def test_vollan_files_without_session_give_none(data_dir):
    used = {f"vollan2025/{s['raw_path']}" for s in labels.sessions_of("vollan2025")}
    unused = [r for r in data_sources.read_manifest("vollan2025")
              if r["path"].startswith(vollan2025.FOLDER) and r["path"] not in used]
    assert len(unused) == 4
    for r in unused:
        assert vollan2025.read_session(data_dir / r["path"], "V0", "V0_1") is None


@pytest.mark.frozen
@pytest.mark.slow
def test_trajectory_stage_reproduces_reference(frozen_dir):
    """Smoothing the reference per-dataset files gives the reference merged file."""
    ref = frozen_recordings(frozen_dir, "trajectories.pkl")
    present = [k for k in ds.CANONICAL_ORDER if (frozen_dir / "preprocessed" / f"{k}.pkl").is_file()]
    keys = sorted({ds.key_from_label(a) for a in ref}, key=ds.canonical_rank)
    if keys != present:
        pytest.skip("per-dataset reference files do not cover the merged reference")
    per = {k: frozen_recordings(frozen_dir, f"{k}.pkl") for k in keys}
    assert differences(trajectories.merge_and_smooth(per), ref) == []
