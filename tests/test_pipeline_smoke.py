"""End-to-end smoke run of the pipeline on synthetic recordings (no data needed).

``scripts/run_pipeline.py --smoke`` simulates two sessions of grid cells with known adaptation
(``hexmod.synthetic``), runs the rat analyses (steps 02 to 08 and 10 to 12; step 09 needs raw files and is
replaced by one unit per cell), the simulation steps with small sizes (21 to 27), draws every figure
(step 30) and writes the statistics report (step 40, which has no reference values for synthetic results).
The tests check that the run succeeds within the time budget (``HEXMOD_SMOKE_BUDGET_S``, default 300 s), that
every file of the results contract is written, and that the analyses recover the simulated grid scale,
orientation, direction tuning and adaptation.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[1]
BUDGET_S = float(os.environ.get("HEXMOD_SMOKE_BUDGET_S", "300"))


def _helper():
    name = "hexmod_tests_smoke_run"
    if name not in sys.modules:
        path = Path(__file__).resolve().parent / "fixtures" / "smoke_run.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.fixture(scope="session")
def smoke(tmp_path_factory):
    run = _helper().smoke_run(tmp_path_factory)
    if run["returncode"] != 0:
        pytest.fail(f"the smoke run failed (exit status {run['returncode']}):\n{_helper().log_tail(run)}")
    return run


@pytest.fixture(scope="session")
def res(smoke) -> Path:
    return smoke["results"]


#: Steps run by the smoke pipeline, with the stem of their run record.
RUN_STEPS = {
    "01": "smoke_recordings", "02": "step02_cohort", "03": "step03_adaptation_parametric",
    "04": "step04_adaptation_nonparametric", "05": "step05_prospective_coding", "06": "step06_visit_sequences",
    "07": "step07_adaptation_timescales", "08": "step08_methods_numbers", "09": "smoke_separate_units",
    "10": "step10_conjunctive_cells", "11": "step11_example_cells", "12": "step12_rat_statistics",
    "21": "step21_simulation_pools", "22": "step22_simulation_maps", "23": "step23_simulation_proximity",
    "24": "step24_simulation_participants", "25": "step25_simulation_collect", "26": "step26_speed_sweep",
    "27": "step27_simulation_statistics", "30": "step30_figures", "40": "step40_check_statistics",
}

#: Files of the results contract (docs/PIPELINE.md) that the smoke run must write, by folder.
CONTRACT_FILES = {
    "preprocessed": ["vollan2025.pkl", "hafting2005.pkl", "trajectories.pkl", "sessions.csv"],
    "rat/cohort": ["cells.csv", "dir_maps.npy", "field_locations.npz", "cells_full.npz", "sessions.csv", "table1.csv"],
    "rat/adaptation": ["param_cells.csv", "alpha_bounds.csv", "fr_percent_change.npy",
                       "fr_percent_change_infield.npy"],
    "rat/prospective": ["shift_ms.npy", "si_shift.npy", "peak_shift.csv", "shift_cells.csv",
                        "fr_percent_change_shifted.npy", "shift_field_locations.npz"],
    "rat/visits": ["paired_cells.csv", "summary.csv", "nesting.csv", "diagnostics.csv"],
    "rat/timescales": ["finebin_percent_change.npy", "supp4_points.csv", "supp4_fits.csv", "fits.json",
                       "bootstrap.json"],
    "rat/methods": ["methods_numbers.json"],
    "rat/same_cell": ["entries.csv", "units.csv"],
    "rat/conjunctive": ["fig2b_cells.csv", "fig2b_bins.csv", "fig2c_per_animal_curves.csv", "fig2c_mean.csv",
                        "fig2d_cells.csv", "fig2e_per_animal.csv", "fig2e_cells.csv", "fig2e_band.csv"],
    "rat/examples": ["fig2a_cells.csv", "fig2a_cells.npz", "fig3ab.npz", "fig3ab_field_visits.csv",
                     "fig4ab_small.npz", "fig4ab_small_field_visits.csv", "fig4ab_large.npz",
                     "fig4ab_large_field_visits.csv", "supp2a_shifted_maps.npz", "supp3a_cells.csv",
                     "supp3a_cells.npz"],
    "rat/figure_tables": ["cells_thirds.csv", "thirds_edges.csv", "supp3bcd_arrays.npz", "supp5_widths.csv",
                          "supp3bcd_fits.csv"],
    "sim/pools": ["small_pool_40cm.npz", "large_pool_70cm.npz", "pool_checksums.json"],
    "sim/fig6": ["trajectory_seed0.csv", "trajectory_seed0_objects.json", "maps.npz", "fig6d_small.csv",
                 "fig6d_large.csv"],
    "sim/fig7": ["participants_small.csv.gz", "participants_large.csv.gz", "fig7ad_direction_bins.csv",
                 "fig7be_beta.csv", "fig7c_orientation.csv", "fig7f_detection.csv", "fig7f_p_per_experiment.csv",
                 "fig7_values.json"],
    "sim/fig7/participants/small": ["p0000.npz", "p0001.npz", "manifest.csv"],
    "sim/fig7/participants/large": ["p0000.npz", "p0001.npz", "manifest.csv"],
    "sim/fig8": ["fig8a_ideal_grids.npz", "fig8d_sweep.csv"],
    "stats": ["step02_cohort.json", "step03_adaptation_parametric.json", "step06_visit_sequences.json",
              "step07_adaptation_timescales.json", "step08_methods_numbers.json", "step10_conjunctive_cells.json",
              "step12_rat_statistics.json", "step27_simulation_statistics.json", "statistics_check.csv",
              "statistics_check.md"],
}


def _stats(res: Path) -> dict:
    from hexmod import io

    values = {}
    for path in sorted((res / "stats").glob("*.json")):
        values.update(io.read_json(path))
    return values


def _truth(res: Path):
    """True parameters of the synthetic units, from the run record of the synthetic recordings."""
    import pandas as pd

    from hexmod import io

    rec = io.read_json(res / "logs" / "smoke_recordings.json")
    return pd.DataFrame(rec["notes"]["units"]), rec["notes"]["model"]


def _cohort_with_truth(res: Path):
    from hexmod import io

    cells = io.read_csv(res / "rat" / "cohort" / "cells.csv")
    truth, _ = _truth(res)
    merged = cells.merge(truth, on=["session_id", "cell_id"], how="left", suffixes=("", "_true"), validate="1:1")
    assert merged["grid_scale_cm"].notna().all(), "cohort cells that are not synthetic units"
    return merged


# ------------------------------------------------------------------------------------------- the run
def test_smoke_run_succeeds_within_budget(smoke):
    if smoke["reused"]:
        pytest.skip("results of an earlier smoke run (HEXMOD_SMOKE_RESULTS)")
    log = smoke["log"]
    for sid in RUN_STEPS:
        assert re.search(rf"^\[{sid}\] done in ", log, re.M), f"step {sid} did not finish:\n{_helper().log_tail(smoke)}"
    assert "[09] step09_same_cell_links.py: skipped in the smoke run" in log
    assert "failed: none" in log
    assert smoke["runtime_s"] < BUDGET_S, f"smoke run took {smoke['runtime_s']:.0f} s (budget {BUDGET_S:.0f} s)"


def test_every_step_writes_a_run_record(res):
    from hexmod import io

    for sid, stem in RUN_STEPS.items():
        path = res / "logs" / f"{stem}.json"
        assert path.is_file(), f"run record of step {sid} missing: {path.name}"
        rec = io.read_json(path)
        assert rec["status"] == "ok", f"{stem}: {rec.get('error')}"
        if sid not in ("01", "09"):
            assert "--smoke" in rec["argv"], f"{stem} was not run with --smoke"


def test_contract_files_written(res):
    missing = [f"{folder}/{name}" for folder, names in CONTRACT_FILES.items() for name in names
               if not (res / folder / name).is_file()]
    assert not missing, f"files of the results contract missing: {missing}"


def test_statistics_files_are_flat_and_finite(res):
    from hexmod import io

    for path in sorted((res / "stats").glob("*.json")):
        for key, value in io.read_json(path).items():
            assert isinstance(value, (int, float)) and not isinstance(value, bool), f"{path.name}: {key}"
            assert np.isfinite(value), f"{path.name}: {key} = {value}"


def test_statistics_report_lists_every_statistic(res):
    from hexmod import io

    report = io.read_csv(res / "stats" / "statistics_check.csv")
    assert list(report.columns) == ["key", "step", "printed", "expected", "computed", "abs_diff", "tolerance",
                                    "status"]
    assert set(report["status"]) <= {"pass", "fail", "missing", "no reference"}
    reported = set(report.loc[report["status"] != "missing", "key"])
    unreported = sorted(set(_stats(res)) - reported)
    assert not unreported, f"statistics missing from the report of step 40: {unreported}"


# --------------------------------------------------------------------------------- synthetic inputs
def test_synthetic_recordings_follow_the_preprocessed_layout(res):
    from hexmod import io
    from hexmod.datasets import CANONICAL_ORDER
    from hexmod.preprocessing import labels

    sessions = io.read_csv(res / "preprocessed" / "sessions.csv")
    assert list(sessions.columns) == ["dataset", "animal_id", "session_id", "source", "n_units", "n_samples",
                                      "sample_rate_hz", "duration_s"]
    assert list(sessions["dataset"]) == sorted(sessions["dataset"], key=CANONICAL_ORDER.index)
    dataset_of = dict(zip(sessions["session_id"], sessions["dataset"]))
    data = io.load_cache(res / "preprocessed" / "trajectories.pkl")
    assert [s for sessions_ in data.values() for s in sessions_] == list(sessions["session_id"])
    for animal_id, by_session in data.items():
        for session_id, s in by_session.items():
            pos = s["position"]
            n = len(pos["time"])
            assert pos["xy"].shape == pos["xy_smoothed"].shape == (n, 2)
            assert pos["speed"].shape == pos["move_dir"].shape == (n,)
            assert np.all((pos["move_dir"] >= 0) & (pos["move_dir"] < 2 * np.pi))
            assert np.sum(pos["speed"] > 5) >= 30000, "a synthetic session fails the cohort sample rule"
            units = labels.units_of(dataset_of[session_id])[session_id]
            manifest = {u["cell_id"]: u["unique_cell_id"] for u in units}
            assert list(s["spike_times"]) == [c for c in manifest if c in s["spike_times"]]
            for c, spikes in s["spike_times"].items():
                assert s["unique_cell_ids"][c] == manifest[c]
                assert np.all(np.diff(spikes) >= 0) and spikes.min() >= pos["time"][0]
    vollan = io.load_cache(res / "preprocessed" / "vollan2025.pkl")
    first = next(iter(next(iter(vollan.values())).values()))
    assert first["position"]["xy"].dtype == np.float32  # as in the preprocessed recordings of this dataset


def test_synthetic_recordings_are_reproducible():
    from hexmod import synthetic

    spec = dict(synthetic.SMOKE_SESSIONS[1], n_cells=3, n_conjunctive=1)
    a, truth_a = synthetic.synthetic_session(spec)
    b, truth_b = synthetic.synthetic_session(spec)
    assert truth_a == truth_b
    assert np.array_equal(a["position"]["xy"], b["position"]["xy"])
    for c in a["spike_times"]:
        assert np.array_equal(a["spike_times"][c], b["spike_times"][c])


def test_same_cell_substitute_has_one_unit_per_cell(res):
    from hexmod import io

    cells = io.read_csv(res / "rat" / "cohort" / "cells.csv")
    entries = io.read_csv(res / "rat" / "same_cell" / "entries.csv")
    assert list(entries.columns) == ["row", "animal_id", "session_id", "cell_id", "unit", "n_entries",
                                     "kept_earliest"]
    assert np.array_equal(entries["row"], cells["row"])
    assert entries["unit"].is_unique and entries["kept_earliest"].all() and (entries["n_entries"] == 1).all()


# ----------------------------------------------------------------------------- recovery of the truth
def test_cohort_keeps_most_synthetic_cells(res):
    cells = _cohort_with_truth(res)
    truth, _ = _truth(res)
    assert len(cells) >= 0.75 * len(truth), f"{len(cells)} of {len(truth)} synthetic cells kept"
    assert set(cells["animal_id"]) == set(truth["animal_id"])


def test_grid_scale_and_orientation_recovered(res):
    cells = _cohort_with_truth(res)
    rel = np.abs(cells["scale_cm"] / cells["grid_scale_cm"] - 1)
    assert np.median(rel) < 0.15, f"median relative scale error {np.median(rel):.3f}"
    assert np.mean(rel < 0.25) >= 0.8
    err = np.abs((cells["orientation_deg"] - cells["orientation_deg_true"] + 30) % 60 - 30)
    assert np.median(err) < 6, f"median orientation error {np.median(err):.1f} deg"


def test_direction_tuning_recovered(res):
    cells = _cohort_with_truth(res)
    conj = cells["conjunctive"].astype(bool).to_numpy()
    assert conj.sum() >= 5 and (~conj).sum() >= 5
    assert np.array_equal(cells["Rayleigh_mag"].to_numpy() > 0.3, conj), "conjunctive cells not recovered"
    pref = cells.loc[conj, "preferred_direction_deg"].to_numpy(float)
    err = np.abs((cells.loc[conj, "Rayleigh_angle"].to_numpy(float) - pref + 180) % 360 - 180)
    assert np.max(err) < 30, f"preferred direction errors {np.round(err, 1)} deg"


def test_adaptation_recovered(res):
    from hexmod import io, stats

    s = _stats(res)
    _, model = _truth(res)
    assert model["adaptation"] > 0
    # rate lower soon after a field visit than long after it (Fig 3C)
    assert s["fig3c.mean_pct_change.0_1s"] < 0 < s["fig3c.mean_pct_change.above_3s"]
    assert s["fig3c.rm_anova.p"] < 0.05
    # negative adaptation coefficient of the proximity model (t_prev only)
    param = io.read_csv(res / "rat" / "adaptation" / "param_cells.csv")
    r = stats.one_sample_t(param["alpha_tprev_all"])
    assert r["t"] < -2 and np.mean(param["alpha_tprev_all"] < 0) >= 0.8, r
    assert s["methods.ratio_0_1s_to_above_3s"] < 0.95


# ----------------------------------------------------------------------------- protection of results
@pytest.mark.parametrize("existing", ["preprocessed/trajectories.pkl", "logs/step02_cohort.json"])
def test_smoke_run_refuses_results_of_the_recorded_data(tmp_path, existing):
    path = tmp_path / existing
    path.parent.mkdir(parents=True)
    # recordings that are not synthetic, or the run record of a step run without --smoke
    path.write_text('{"argv": ["step02_cohort.py"]}' if path.suffix == ".json" else "")
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    env = dict(os.environ, HEXMOD_RESULTS=str(tmp_path), PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "run_pipeline.py"), "--smoke"], cwd=str(tmp_path),
                          env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.returncode == 2
    assert "would overwrite them" in proc.stderr and existing in proc.stderr
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
