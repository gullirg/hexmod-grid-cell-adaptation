"""Tests of the shared helpers: file formats, ordered process pool, run records and configuration."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hexmod import config, datasets, io, parallel, runlog


def _square(x):
    return x * x


def _row_sum(item):
    return float(np.sum(item["a"]) * item["k"])


# ------------------------------------------------------------------------------------------- io
def test_csv_round_trip_is_exact(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"row": np.arange(500), "x": rng.standard_normal(500) * 1e3, "flag": rng.random(500) > 0.5,
                       "name": [f"V{i}_1" for i in range(500)], "tiny": rng.random(500) * 1e-300})
    df.loc[3, "x"] = np.nan
    for name in ("t.csv", "t.csv.gz"):
        io.write_csv(df, tmp_path / name)
        back = io.read_csv(tmp_path / name)
        assert back["row"].dtype.kind == "i" and back["flag"].dtype == bool
        assert np.array_equal(back["x"].to_numpy(), df["x"].to_numpy(), equal_nan=True)
        assert np.array_equal(back["tiny"].to_numpy(), df["tiny"].to_numpy())
    a = (tmp_path / "t.csv.gz").read_bytes()
    io.write_csv(df, tmp_path / "t.csv.gz")
    assert (tmp_path / "t.csv.gz").read_bytes() == a          # gzip output does not depend on the time
    with pytest.raises(TypeError):
        io.write_csv(pd.DataFrame({"a": [np.zeros(2), np.zeros(3)]}), tmp_path / "bad.csv")


def test_json_round_trip(tmp_path):
    obj = {"b": np.float64(0.1), "a": np.int64(3), "c": np.arange(3), "d": {"x": np.bool_(True)}}
    io.write_json(obj, tmp_path / "s.json")
    assert io.read_json(tmp_path / "s.json") == {"a": 3, "b": 0.1, "c": [0, 1, 2], "d": {"x": True}}


def test_ragged_npz_round_trip(tmp_path):
    items = [np.arange(6.0).reshape(2, 3), np.zeros((0, 3)), np.ones((4, 3))]
    data, shapes = io.pack_ragged(items)
    assert shapes.tolist() == [[2, 3], [0, 3], [4, 3]] and data.shape == (18,)
    back = io.unpack_ragged(data, shapes)
    assert all(np.array_equal(a, b) for a, b in zip(items, back))
    io.save_npz(tmp_path / "r.npz", arrays={"row": np.arange(3)}, ragged={"fields": items})
    z = io.load_npz(tmp_path / "r.npz")
    assert np.array_equal(z["row"], np.arange(3)) and all(np.array_equal(a, b) for a, b in zip(items, z["fields"]))
    raw = io.load_npz(tmp_path / "r.npz", unpack=False)
    assert set(raw) == {"row", "fields__data", "fields__shapes"}
    with pytest.raises(TypeError):
        io.save_npz(tmp_path / "o.npz", arrays={"o": np.array([np.zeros(2), np.zeros(3)], dtype=object)})


def test_cache_accepts_plain_data_only(tmp_path):
    obj = {"V1": {"V1_1": {"position": {"xy": np.zeros((3, 2), np.float32), "sample_rate": np.float64(50.0)},
                           "spike_times": {1: np.array([0.1, 0.2])}, "trial_id": np.str_("x")}}}
    io.save_cache(obj, tmp_path / "c.pkl")
    back = io.load_cache(tmp_path / "c.pkl")
    assert back["V1"]["V1_1"]["position"]["xy"].dtype == np.float32
    with pytest.raises(TypeError):
        io.save_cache({"df": pd.DataFrame({"a": [1]})}, tmp_path / "d.pkl")
    with open(tmp_path / "evil.pkl", "wb") as f:
        pickle.dump(pd.DataFrame({"a": [1]}), f)
    with pytest.raises(pickle.UnpicklingError):
        io.load_cache(tmp_path / "evil.pkl")


def test_written_files_have_the_usual_permissions(tmp_path):
    mask = os.umask(0)
    os.umask(mask)
    io.write_csv(pd.DataFrame({"a": [1]}), tmp_path / "t.csv")
    io.write_json({"a": 1}, tmp_path / "t.json")
    io.save_npz(tmp_path / "t.npz", arrays={"a": np.arange(2)})
    for name in ("t.csv", "t.json", "t.npz"):
        assert (tmp_path / name).stat().st_mode & 0o777 == 0o666 & ~mask == io.FILE_MODE
    assert not list(tmp_path.glob(".tmp_*"))


def test_text_round_trips():
    v = np.array([59.50042804250833, 0.1, 1e-5, np.nan, 92.3529438010587])
    once = io.csv_float_roundtrip(v)
    assert once.shape == v.shape and np.isnan(once[3])
    ok = ~np.isnan(v)
    assert np.all(np.abs(once[ok] - v[ok]) <= np.spacing(np.abs(v[ok])))
    assert io.array_text_roundtrip(np.array([7.730083677, 1.0])).tolist() == [7.73008368, 1.0]


# ------------------------------------------------------------------------------------- parallel
@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_map_ordered_keeps_order(method):
    if method == "fork" and sys.platform == "win32":
        pytest.skip("fork not available")
    items = [{"a": np.arange(i + 1), "k": i} for i in range(23)]
    expected = [_row_sum(x) for x in items]
    assert parallel.map_ordered(_row_sum, items, workers=1) == expected
    assert parallel.map_ordered(_row_sum, items, workers=3, start_method=method) == expected
    assert parallel.map_ordered(_square, list(range(10)), workers=2, start_method=method) == [i * i for i in range(10)]


# --------------------------------------------------------------------------------------- runlog
def test_run_record(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXMOD_RESULTS", str(tmp_path))
    out = tmp_path / "rat" / "x.csv"
    with runlog.RunRecord("step99_example", workers=2) as rec:
        out.parent.mkdir(parents=True)
        out.write_text("a\n1\n")
        rec.add_output(out)
        rec.note("n_cells", 1)
    r = json.loads((tmp_path / "logs" / "step99_example.json").read_text())
    assert r["status"] == "ok" and r["workers"] == 2 and r["notes"] == {"n_cells": 1} and "mode" not in r
    assert r["outputs"][0]["path"] == "results/rat/x.csv" and len(r["outputs"][0]["sha256"]) == 64
    assert r["versions"]["numpy"] == np.__version__
    assert runlog.read_record("step99_example") == r
    with runlog.RunRecord("data_check") as rec:   # a record of a file at the results root
        (tmp_path / "data_check.json").write_text("{}")
        rec.add_output(tmp_path / "data_check.json")
    r = json.loads((tmp_path / "logs" / "data_check.json").read_text())
    assert r["outputs"][0]["path"] == "results/data_check.json"
    with pytest.raises(ValueError):
        with runlog.RunRecord("step98_fail"):
            raise ValueError("boom")
    assert json.loads((tmp_path / "logs" / "step98_fail.json").read_text())["status"] == "failed"


def test_run_record_keeps_linked_inputs_under_their_base(tmp_path, monkeypatch):
    """Inputs reached through symbolic links inside the data or results folder keep their place in that folder."""
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "brun2008").mkdir(parents=True)
    (elsewhere / "brun2008" / "r1.mat").write_bytes(b"raw")
    (elsewhere / "cells.csv").write_text("row\n0\n")
    data = tmp_path / "data"
    data.mkdir()
    os.symlink(elsewhere / "brun2008", data / "brun2008")
    results = tmp_path / "results"
    (results / "rat" / "cohort").mkdir(parents=True)
    os.symlink(elsewhere / "cells.csv", results / "rat" / "cohort" / "cells.csv")
    monkeypatch.setenv("HEXMOD_DATA", str(data))
    monkeypatch.setenv("HEXMOD_RESULTS", str(results))
    with runlog.RunRecord("step97_links") as rec:
        rec.add_input(data / "brun2008" / "r1.mat")
        rec.add_input(results / "rat" / "cohort" / "cells.csv")
        rec.add_input(elsewhere / "cells.csv")
    r = json.loads((results / "logs" / "step97_links.json").read_text())
    assert [e["path"] for e in r["inputs"]] == ["data/brun2008/r1.mat", "results/rat/cohort/cells.csv",
                                               "cells.csv"]


# --------------------------------------------------------------------------------------- config
def test_paths_and_workers(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXMOD_DATA", str(tmp_path / "d"))
    monkeypatch.setenv("HEXMOD_RESULTS", str(tmp_path / "r"))
    monkeypatch.setenv("HEXMOD_WORKERS", "3")
    assert config.data_dir() == tmp_path / "d" and config.results_dir() == tmp_path / "r"
    assert config.workers() == 3 and config.workers(5) == 5
    with pytest.raises(ValueError):
        config.workers(0)


def test_smoke_results_are_kept_apart(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("HEXMOD_RESULTS", str(tmp_path / "smoke"))
    res = config.results_dir()
    config.guard_smoke_results(smoke=True)          # an empty folder: marked as a smoke folder
    assert (res / config.SMOKE_MARKER).is_file()
    config.guard_smoke_results(smoke=True)
    with pytest.raises(config.ResultsFolderError) as info:     # a step without --smoke refuses it
        config.guard_smoke_results(smoke=False)
    assert "smoke run" in str(info.value.code)
    monkeypatch.setenv("HEXMOD_RESULTS", str(tmp_path / "recorded"))
    other = config.results_dir()
    (other / "logs").mkdir(parents=True)
    (other / "logs" / "step02_cohort.json").write_text(json.dumps({"argv": ["step02_cohort.py"]}))
    config.guard_smoke_results(smoke=False)
    with pytest.raises(config.ResultsFolderError):             # a smoke step refuses results of the recorded data
        config.guard_smoke_results(smoke=True)
    assert not (other / config.SMOKE_MARKER).exists()


def test_dataset_metadata():
    assert datasets.CANONICAL_ORDER == ["sargolini2006", "brun2008", "olafsdottir2016", "vollan2025", "hafting2005",
                                        "barry2012"]
    assert datasets.key_from_label("W5_3") == "barry2012" and datasets.key_from_label("F1_4") == "olafsdottir2016"
    assert sorted(d["prefix"] for d in datasets.DATASETS.values()) == sorted("BFHSVW")
    for key, meta in datasets.DATASETS.items():
        assert meta["key"] == key and meta["licence"].startswith("CC BY") and meta["data_doi"] and meta["repository"]
        assert meta["manifest"] == f"inputs/manifests/files_{key}.csv"
        assert all(p == key or p.startswith(key + "/") for p in meta["raw_subpaths"])
    assert datasets.DATASETS["barry2012"]["data_doi"] == datasets.DATASETS["olafsdottir2016"]["data_doi"] == \
        "10.5281/zenodo.23477461"


def test_marked_results_folder_stops_a_step_with_one_line(tmp_path):
    (tmp_path / config.SMOKE_MARKER).write_text("x")
    env = dict(os.environ, HEXMOD_RESULTS=str(tmp_path), PYTHONPATH=str(Path(config.__file__).parents[1]))
    code = "from hexmod import config; config.guard_smoke_results(smoke=False)"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 1
    lines = out.stderr.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("error: ") and "smoke run" in lines[0]
