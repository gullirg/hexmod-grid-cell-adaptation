"""Simulation steps 21 to 27: participant files, Fig 7 values, maps, speed sweep and statistics.

The quick tests use small pools and the shipped results in ``results_precomputed/``; the smoke
test runs steps 21 to 27 with their ``--smoke`` sizes in a temporary results folder.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hexmod import io as hio

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
PRE = REPO / "results_precomputed" / "sim"
FOLDS = list(range(1, 8))
DIRS = [30 * j for j in range(12)]


def load_script(name: str):
    """Import a step script as a module."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def results(tmp_path, monkeypatch):
    """Empty results folder for a step run."""
    monkeypatch.setenv("HEXMOD_RESULTS", str(tmp_path))
    return tmp_path


@pytest.fixture(scope="module")
def records():
    """Two simulated participants (seeds 0 and 1) of 100 cells from a 300-cell small pool."""
    from hexmod.sim import pool, simulate

    table = pool.set_gridcells_pool(pool.module_param("small"), 300, 0)
    par = pool.module_param("small", adaptEff_mean=0.1, speedGamma_mean=0, nGridCells=100)
    return [simulate.simulate_participant((par, simulate.participant_cells(table, par, s), s, None,
                                           simulate.FOLD_SYM_LIST))
            for s in (0, 1)]


def _same_table(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for col in a.columns:
        x, y = a[col].to_numpy(), b[col].to_numpy()
        if x.dtype == object or y.dtype == object:
            for u, v in zip(x, y):
                if isinstance(u, np.ndarray) or isinstance(v, np.ndarray):
                    both = isinstance(u, np.ndarray) and isinstance(v, np.ndarray)
                    if not (both and np.array_equal(u, v, equal_nan=True)):
                        return False
                elif not (u == v or (u is None and (v is None or v != v)) or (v is None and u != u)):
                    return False
        elif x.dtype != y.dtype or not np.array_equal(x, y, equal_nan=True):
            return False
    return True


def test_participant_file_round_trip(records, tmp_path):
    step24 = load_script("step24_simulation_participants")
    step25 = load_script("step25_simulation_collect")
    rec = records[0]
    path = step24.save_participant(rec, "small", tmp_path / step24.participant_file(rec["seed"]))
    assert path.name == "p0000.npz"
    with np.load(path, allow_pickle=False) as z:
        assert all(z[k].dtype != object for k in z.files)
        assert len(z["crossval__beta"]) == 2 * 2 * 4 * 7 and len(z["noncross__model"]) == 4
        assert z["noncross__FRby12dir"].shape == (4, 12)
    back = step25.load_participant(path)
    assert back["module"] == "small" and back["seed"] == 0
    for key in ("crossvalResult", "noncrossResult"):
        assert _same_table(back[key], rec[key]), key
    for key, value in rec.items():
        if key not in ("crossvalResult", "noncrossResult"):
            assert np.array_equal(np.asarray(back[key]), np.asarray(value), equal_nan=True), key
    assert step24.is_complete(path, 0, "small") and not step24.is_complete(path, 1, "small")


def test_participant_resume_and_collect_check_how_files_were_computed(records, tmp_path):
    step24 = load_script("step24_simulation_participants")
    step25 = load_script("step25_simulation_collect")
    rec = records[0]
    meta = step24.participant_metadata(10000, "f" * 64)
    path = step24.save_participant(rec, "small", tmp_path / "p0000.npz", meta)
    assert step24.read_metadata(path) == meta
    assert step24.is_complete(path, 0, "small", meta)
    assert not step24.is_complete(path, 0, "small", step24.participant_metadata(200, "f" * 64))   # other size
    assert not step24.is_complete(path, 0, "small", step24.participant_metadata(10000, "e" * 64))  # other pool
    bare = step24.save_participant(rec, "small", tmp_path / "bare" / "p0000.npz")
    assert step24.is_complete(bare, 0, "small") and not step24.is_complete(bare, 0, "small", meta)
    assert step25.check_metadata(path, 0, "small", "f" * 64) is None
    assert "pool" in step25.check_metadata(path, 0, "small", "e" * 64)
    assert "does not record" in step25.check_metadata(bare, 0, "small", "f" * 64)
    small = step24.save_participant(rec, "small", tmp_path / "s" / "p0000.npz",
                                    step24.participant_metadata(200, "f" * 64))
    assert "200 cells" in step25.check_metadata(small, 0, "small", "f" * 64)
    manifest = hio.read_csv(step24.update_manifest(tmp_path, [0]))
    assert list(manifest.columns) == step24.MANIFEST_COLUMNS and int(manifest["meta_n_cells"].iloc[0]) == 10000


def test_pool_fingerprint_from_summary():
    from hexmod.sim import io_pool

    table = io_pool.load_pool(REPO / "inputs" / "simulation" / "large_module_pool_70cm.npz").iloc[:20]
    table = table.reset_index(drop=True)
    assert io_pool.fingerprint_from_summary(io_pool.pool_summary(table)) == io_pool.pool_fingerprint(table)
    shipped = hio.read_json(REPO / "inputs" / "simulation" / "pool_checksums.json")["large_module_pool_70cm"]
    assert io_pool.fingerprint_from_summary(shipped) is not None
    assert io_pool.fingerprint_from_summary({"n_cells": 3}) is None


def test_module_pool_size_is_checked(tmp_path):
    from hexmod.sim import io_pool

    table = io_pool.load_pool(REPO / "inputs" / "simulation" / "large_module_pool_70cm.npz").iloc[:30]
    io_pool.save_pool(table.reset_index(drop=True), tmp_path / "sim" / "pools" / io_pool.POOL_FILES["small"])
    with pytest.raises(SystemExit):
        io_pool.load_module_pool(tmp_path, "small")
    small, path = io_pool.load_module_pool(tmp_path, "small", smoke=True)
    assert len(small) == 30 and path.name == "small_pool_40cm.npz"
    with pytest.raises(SystemExit):
        io_pool.load_module_pool(tmp_path / "empty", "small")


def test_participant_rows(records):
    step25 = load_script("step25_simulation_collect")
    rows = step25.participant_rows(list(reversed(records)), "small")
    shipped = hio.read_csv(PRE / "fig7" / "participants_small.csv.gz", nrows=2)
    assert list(rows.columns) == list(shipped.columns)
    assert rows["method"].tolist() == ["simple", "simple", "location-controlled", "location-controlled"]
    assert rows["participant"].tolist() == [0, 1, 0, 1]
    for i, rec in enumerate(records):
        for est, method in (("simple", "simple"), ("multiplicative", "location-controlled")):
            row = rows[(rows.participant == i) & (rows.method == method)].iloc[0]
            nc = rec["noncrossResult"]
            rate = nc[(nc.model == "adapt") & (nc.estType == est)]["FRby12dir"].iloc[0]
            assert np.array_equal(row[[f"rate_dir_{d}_hz" for d in DIRS]].to_numpy(float), rate)
            assert np.allclose(row[[f"dir_{d}" for d in DIRS]].to_numpy(float), rate / rate.mean() * 100 - 100,
                               atol=1e-12)
            cv = rec["crossvalResult"]
            beta = cv[(cv.model == "adapt") & (cv.estType == est)].groupby("fold")["beta"].mean()
            expected = beta.loc[FOLDS].to_numpy() * 100 / abs(rec["FR_adapt_mean_move"])
            assert np.allclose(row[[f"fold_{k}" for k in FOLDS]].to_numpy(float), expected, rtol=1e-12, atol=0)
            assert 0 <= row["orientation_deg_mod60"] < 60
            assert row["mean_rate_moving_hz"] == rec["FR_adapt_mean_move"]


def test_collect_from_precomputed(results):
    step25 = load_script("step25_simulation_collect")
    assert step25.main(["--source", "precomputed"]) == 0
    out = results / "sim" / "fig7"
    values = hio.read_json(out / "fig7_values.json")
    assert values == hio.read_json(PRE / "fig7" / "fig7_values.json")
    for name in ("participants_small.csv.gz", "participants_large.csv.gz", "fig7ad_direction_bins.csv",
                 "fig7be_beta.csv", "fig7c_orientation.csv", "fig7f_detection.csv", "fig7f_p_per_experiment.csv",
                 "fig7_values.json"):
        assert (out / name).read_bytes() == (PRE / "fig7" / name).read_bytes(), name   # the shipped files
    printed = {"fig7a.small.t": -23.6257, "fig7a.large.t": -7.0936,
               "fig7d.small.t": -43.3605, "fig7d.large.t": -11.4143}
    assert {k: round(values[k], 4) for k in printed} == printed
    assert [k for k in FOLDS if values[f"fig7b.small.fold_{k}.significant"]] == [6]
    assert [k for k in FOLDS if values[f"fig7b.large.fold_{k}.significant"]] == [2, 3]
    assert [k for k in FOLDS if values[f"fig7e.large.fold_{k}.significant"]] == [6]
    assert [values[f"fig7f.large.simple.fold_{k}.count"] for k in FOLDS] == [19, 41, 17, 12, 6, 39, 1]
    assert (values["fig7f.n_experiments"], values["fig7f.first_participant"]) == (49, 30)
    detection = hio.read_csv(out / "fig7f_detection.csv")
    assert len(detection) == 8 and set(detection["source"]) == {"rest_49_experiments", "all_50_experiments"}
    assert len(hio.read_csv(out / "fig7ad_direction_bins.csv")) == 120


def test_quick_tables_take_participants_0_to_29_from_the_run(monkeypatch):
    """--quick: participants 0 to 29 from step 24, the others from the shipped tables, in table order."""
    step25 = load_script("step25_simulation_collect")
    shipped = {m: hio.read_csv(PRE / "fig7" / f"participants_{m}.csv.gz") for m in ("small", "large")}
    run = {m: t[t["participant"] < 30].copy() for m, t in shipped.items()}
    for t in run.values():
        t["fold_6"] = t["fold_6"] + 1.0          # mark the rows that come from the run
    monkeypatch.setattr(step25, "tables_from_run", lambda participants: run)
    out = step25.tables_quick(shipped)
    for m in ("small", "large"):
        assert list(out[m].columns) == list(shipped[m].columns) and len(out[m]) == len(shipped[m])
        assert out[m][["method", "participant"]].equals(shipped[m][["method", "participant"]])
        first = (out[m]["participant"] < 30).to_numpy()
        assert np.array_equal(out[m]["fold_6"].to_numpy()[first], shipped[m]["fold_6"].to_numpy()[first] + 1.0)
        assert np.array_equal(out[m]["fold_6"].to_numpy()[~first], shipped[m]["fold_6"].to_numpy()[~first])


def test_collect_fifty_experiments():
    step25 = load_script("step25_simulation_collect")
    tables = {m: hio.read_csv(PRE / "fig7" / f"participants_{m}.csv.gz") for m in ("small", "large")}
    values, frames = step25.fig7_values(tables, experiments=50)
    meta = (values["fig7f.n_experiments"], values["fig7f.first_participant"], values["fig7f.last_participant"])
    assert meta == (50, 0, 1499)
    keys = [k[len("fig7f_50."):] for k in values if k.startswith("fig7f_50.")]
    assert keys and all(values["fig7f." + k] == values["fig7f_50." + k] for k in keys)
    assert (values["fig7f.large.simple.fold_2.count"], values["fig7f.large.simple.fold_6.count"]) == (42, 39)
    assert len(frames["fig7f_p_per_experiment"]) == 2 * 2 * (49 + 50)


def test_supp3e_title():
    step22 = load_script("step22_simulation_maps")
    assert step22.supp3e_title(0.7270, 0.5440) == "0.7, 0.5 Hz"
    assert step22.supp3e_title(0.6012, 3.2213) == "0.6, 3 Hz"
    assert step22.supp3e_title(1.2170, 2.0170) == "1.2, 2 Hz"


def test_shipped_supp3e_cells():
    """The shipped Supp Fig 3E cells follow the selection rule of step 22 on the shipped large pool."""
    from hexmod.sim import io_pool, pool

    step22 = load_script("step22_simulation_maps")
    table = io_pool.load_pool(REPO / "inputs" / "simulation" / "large_module_pool_70cm.npz")
    cells, gridness, peaks, maps = step22.supp3e_cells(table, pool.module_param("large"))
    shipped = hio.load_npz(PRE / "fig6" / "maps.npz")
    assert cells.tolist() == [95331, 83646, 88, 72374, 36659, 2061]
    assert np.array_equal(cells, shipped["supp3e__large__cells"])
    assert np.array_equal(gridness, shipped["supp3e__large__gridness"])
    assert np.array_equal(peaks, shipped["supp3e__large__peak_rate_hz"])
    assert np.array_equal(maps, shipped["supp3e__large__maps"])
    assert [step22.supp3e_title(g, p) for g, p in zip(gridness, peaks)] == [
        "0.6, 3 Hz", "0.5, 2 Hz", "0.7, 0.5 Hz", "0.8, 6 Hz", "1.2, 2 Hz", "1.2, 4 Hz"]


def test_large_pool_tolerance_check():
    from hexmod.sim import io_pool

    step21 = load_script("step21_simulation_pools")
    path = REPO / "inputs" / "simulation" / "large_module_pool_70cm.npz"
    table = io_pool.load_pool(path).iloc[:50].reset_index(drop=True)
    assert all(d["ok"] and d["max_abs_diff"] == 0 for d in step21.compare_with_tolerance(table, table).values())
    assert step21.LARGE_TOLERANCES == {"centers": 1e-12, "peakFR_jitter": 1e-14, "gridness": 1e-9}
    shifted = table.copy()
    shifted["centers"] = [c + 5e-13 for c in table["centers"]]
    shifted["gridness"] = table["gridness"] + 5e-10
    diffs = step21.compare_with_tolerance(shifted, table)
    assert diffs["centers"]["ok"] and diffs["gridness"]["ok"]
    shifted["centers"] = [c + 1e-11 for c in table["centers"]]
    shifted["gridness"] = table["gridness"] + 1e-8
    diffs = step21.compare_with_tolerance(shifted, table)
    assert not diffs["centers"]["ok"] and not diffs["gridness"]["ok"]


def test_statistics_from_precomputed(results):
    step27 = load_script("step27_simulation_statistics")
    with pytest.raises(SystemExit):                        # no simulation results and no --allow-precomputed
        step27.main([])
    assert step27.main(["--allow-precomputed"]) == 0
    values = hio.read_json(results / "stats" / "step27_simulation_statistics.json")
    assert all(isinstance(v, (int, float)) and np.isfinite(v) for v in values.values())
    rounded = {"fig6d.small.t": (175.8, 1), "fig6d.small.d": (1.76, 2), "fig6d.large.t": (91.7, 1),
               "fig6d.large.d": (0.92, 2), "fig6d.large.mean_aligned": (0.0853, 4),
               "fig6d.large.mean_misaligned": (0.0684, 4), "fig6c.large_p0.max_pct": (3.99, 2),
               "fig6c.large_p0.min_pct": (-2.63, 2), "fig6c.large_p0.sd_pct": (1.49, 2),
               "fig1.narrow.sixfold": (0.5929, 4), "fig1.narrow.kappa": (15.1055, 4),
               "fig1.broad.circular_sd_deg": (67.46, 2), "fig8.v_opt_scale_per_tau": (1.3327, 4),
               "fig7a.small.t": (-23.6257, 4)}
    assert {k: round(values[k], n) for k, (_, n) in rounded.items()} == {k: v for k, (v, _) in rounded.items()}
    assert values["fig6d.small.df"] == values["fig6d.large.df"] == 9999 and values["fig8d.n_runs"] == 111
    fig7 = hio.read_json(PRE / "fig7" / "fig7_values.json")
    assert all(values[k] == v for k, v in fig7.items())


def test_parse_seeds():
    step24 = load_script("step24_simulation_participants")
    assert step24.parse_seeds("1470:1500") == range(1470, 1500)
    with pytest.raises(SystemExit):
        step24.parse_seeds("30")


@pytest.mark.smoke
def test_smoke_chain(results):
    """Steps 21 to 27 with their smoke sizes write every file of the results contract."""
    argv = ["--smoke"]
    for name in ("step21_simulation_pools", "step22_simulation_maps", "step23_simulation_proximity"):
        assert load_script(name).main(argv) == 0, name
    assert load_script("step24_simulation_participants").main(argv + ["--workers", "1"]) == 0
    assert load_script("step25_simulation_collect").main(argv) == 0
    assert load_script("step26_speed_sweep").main(argv + ["--workers", "1"]) == 0
    assert load_script("step27_simulation_statistics").main(argv) == 0
    sim = results / "sim"
    for rel in ("pools/small_pool_40cm.npz", "pools/large_pool_70cm.npz", "pools/pool_checksums.json",
                "fig6/trajectory_seed0.csv", "fig6/trajectory_seed0_objects.json", "fig6/maps.npz",
                "fig6/fig6d_small.csv", "fig6/fig6d_large.csv", "fig8/fig8a_ideal_grids.npz", "fig8/fig8d_sweep.csv",
                "fig7/participants/small/p0001.npz", "fig7/participants/large/manifest.csv", "fig7/fig7_values.json",
                "fig7/participants_small.csv.gz"):
        assert (sim / rel).is_file(), rel
    traj = hio.read_csv(sim / "fig6" / "trajectory_seed0.csv")
    shipped = hio.read_csv(PRE / "fig6" / "trajectory_seed0.csv")
    assert list(traj.columns) == list(shipped.columns) and len(traj) == len(shipped) == 23990
    assert all(np.array_equal(traj[c].to_numpy(), shipped[c].to_numpy(), equal_nan=True) for c in traj.columns)
    maps = hio.load_npz(sim / "fig6" / "maps.npz")
    assert set(maps) == set(hio.load_npz(PRE / "fig6" / "maps.npz"))
    fig8a = hio.load_npz(sim / "fig8" / "fig8a_ideal_grids.npz")
    shipped8a = hio.load_npz(PRE / "fig8" / "fig8a_ideal_grids.npz")
    assert set(fig8a) == set(shipped8a) and all(np.array_equal(fig8a[k], shipped8a[k]) for k in fig8a)
    sweep = hio.read_csv(sim / "fig8" / "fig8d_sweep.csv")
    assert list(sweep.columns) == list(hio.read_csv(PRE / "fig8" / "fig8d_sweep.csv", nrows=1).columns)
    assert len(sweep) == 9 and sweep["scale"].tolist() == [30] * 3 + [50] * 3 + [70] * 3
    manifest = hio.read_csv(sim / "fig7" / "participants" / "small" / "manifest.csv")
    assert manifest["participant"].tolist() == [0, 1]
    assert len(hio.read_csv(sim / "fig6" / "fig6d_large.csv")) == 200


@pytest.mark.frozen
def test_collect_matches_reference(results, frozen_path):
    """Fig 7 tables from the shipped participant tables equal the reference tables value for value."""
    step25 = load_script("step25_simulation_collect")
    assert step25.main(["--source", "precomputed"]) == 0
    out = results / "sim" / "fig7"
    for name in ("fig7ad_direction_bins", "fig7be_beta", "fig7c_orientation", "fig7f_detection",
                 "fig7f_p_per_experiment"):
        mine, ref = hio.read_csv(out / f"{name}.csv"), hio.read_csv(frozen_path(f"sim/fig7/{name}.csv"))
        assert list(mine.columns) == list(ref.columns) and len(mine) == len(ref), name
        for col in ref.columns:
            a, b = mine[col].to_numpy(), ref[col].to_numpy()
            if a.dtype.kind in "fi":
                assert a.astype(float).tobytes() == b.astype(float).tobytes(), (name, col)
            else:
                assert (a.astype(str) == b.astype(str)).all(), (name, col)
