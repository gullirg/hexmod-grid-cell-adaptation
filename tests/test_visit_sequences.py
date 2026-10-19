"""Visit-sequence test of Fig 3F: sequence selection, cell values and statistics.

The unit tests use hand-made visit tables and a simulated grid cell (no data needed). The
``frozen`` tests recompute the summary statistics from the reference cell table, and the slow
test runs step 06 on the reference inputs and compares every output with the reference results.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from hexmod.rat import visit_sequences as vs

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------ hand-made data
def visit_table(t_prev, fields, fr, dur=None, spikes=None, valid=None, key="V1|V1_1|7", row=3):
    """Visit table of one cell in the layout of ``events_frame``."""
    n = len(t_prev)
    dur = np.full(n, 0.5) if dur is None else np.asarray(dur, float)
    spikes = np.full(n, 5.0) if spikes is None else np.asarray(spikes, float)
    t_prev = np.asarray(t_prev, float)
    valid = np.isfinite(t_prev) if valid is None else np.asarray(valid, bool)
    animal, session, cell = key.split("|")
    return pd.DataFrame({
        "row": row, "key": key, "animal_id": animal, "session_id": session, "cell_id": int(cell),
        "unique_cell_id": f"{session}_cell{cell}", "dataset": "vollan2025",
        "event_idx": np.arange(n), "t_prev": t_prev, "t_prev_valid": valid, "fieldId": np.asarray(fields, int),
        "FR": np.asarray(fr, float), "dur_eff": dur, "spike_count": spikes, "mean_speed": np.arange(n, dtype=float),
        "prop_speed_gt5": np.ones(n), "wall_distance": np.full(n, 10.0),
    })


TRIPLET = vs.SequenceSpec("triplet", 0.0, 10.0, False, "spike_ge2")
QUADRUPLET = vs.SequenceSpec("quadruplet", 0.0, 10.0, True, "expected_ge2_fr23")

# six visits to fields 0, 1, 2, 0, 1, 1; the first visit has no t_prev
BASE = dict(t_prev=[np.nan, 2.0, 1.0, 3.0, 4.0, 0.5], fields=[0, 1, 2, 0, 1, 1], fr=[10, 12, 8, 10, 6, 9])


def test_extraction_rules():
    assert vs.extraction_settings("printed") == {"rate_map_spike_mode": "counts", "tprev_rule": "gap_aware",
                                                 "tprev_cap": 10.0, "recording_gap_threshold": 5.0, "dt": 0.02}
    assert vs.extraction_settings("library")["rate_map_spike_mode"] == "binary"
    assert vs.extraction_settings("library")["tprev_rule"] == "standard"
    with pytest.raises(ValueError):
        vs.extraction_settings("other")
    assert vs.FIG3F_SPECS[-1].name == vs.FIG3F_ANALYSIS and vs.FIG3F_SPECS[-1].require_fourth


def test_triplets_hand_made():
    out = vs.triplet_rows_for_cell(visit_table(**BASE), TRIPLET)
    # visit 3 = visits 2, 3, 4; visits 3, 4, 5 go twice to field 1 and are dropped
    assert list(out["t12"]) == [2.0, 1.0, 3.0] and list(out["t23"]) == [1.0, 3.0, 4.0]
    assert list(out["group"]) == ["T23<T12", "T23>T12", "T23>T12"]
    np.testing.assert_allclose(out["frr_23"], [(12 - 8) / 20, (8 - 10) / 18, (10 - 6) / 16])
    assert list(out["speed3_minus2"]) == [1.0, 1.0, 1.0]
    assert (out["key"] == "V1|V1_1|7").all() and (out["row"] == 3).all() and (out["analysis"] == "triplet").all()


def test_quadruplet_needs_a_fourth_visit_to_another_field():
    out = vs.triplet_rows_for_cell(visit_table(**BASE), QUADRUPLET)
    # the third sequence is followed by a visit to the same field as its visit 3
    assert list(out["t34"]) == [3.0, 4.0]
    assert list(out["group"]) == ["T23<T12", "T23>T12"]
    # too few visits for a quadruplet
    assert vs.triplet_rows_for_cell(visit_table([np.nan, 1.0, 2.0], [0, 1, 2], [1, 2, 3]), QUADRUPLET).empty


def test_gap_window_duration_and_validity():
    t_prev = [np.nan, 2.0, 1.0, 12.0, 4.0, 0.5]  # T = 12 s is outside the window
    out = vs.triplet_rows_for_cell(visit_table(t_prev, BASE["fields"], BASE["fr"]), TRIPLET)
    assert list(out["t23"]) == [1.0]
    dur = [0.5, 0.1, 0.5, 0.5, 0.5, 0.5]  # visits must last longer than 0.1 s
    out = vs.triplet_rows_for_cell(visit_table(BASE["t_prev"], BASE["fields"], BASE["fr"], dur=dur), TRIPLET)
    assert list(out["t23"]) == [4.0]
    valid = [False, True, False, True, True, True]  # an interval spanning a recording gap
    out = vs.triplet_rows_for_cell(visit_table(**BASE, valid=valid), TRIPLET)
    assert list(out["t23"]) == [4.0]
    tied = [np.nan, 2.0, 2.0, 3.0, 4.0, 0.5]  # ties are dropped
    out = vs.triplet_rows_for_cell(visit_table(tied, BASE["fields"], BASE["fr"]), TRIPLET)
    assert list(out["t23"]) == [3.0, 4.0]


def test_spike_rules():
    spikes = [1, 5, 5, 5, 5, 5]  # one spike on the first visit
    ev = visit_table(**BASE, spikes=spikes)
    observed = vs.triplet_rows_for_cell(ev, TRIPLET)
    assert list(observed["t23"]) == [3.0, 4.0]  # the sequence starting with that visit is dropped
    # expected counts use the cell's mean in-field rate (26 spikes in 3 s) times the duration
    response = vs.triplet_rows_for_cell(ev, vs.SequenceSpec("t", 0.0, 10.0, False, "expected_ge2_fr23"))
    assert list(response["t23"]) == [1.0, 3.0, 4.0]
    short = visit_table(**BASE, dur=[0.5, 0.15, 0.5, 0.5, 0.5, 0.5])  # expected spikes below 2 on the second visit
    response = vs.triplet_rows_for_cell(short, vs.SequenceSpec("t", 0.0, 10.0, False, "expected_ge2"))
    assert list(response["t23"]) == [4.0]
    with pytest.raises(ValueError):
        vs.triplet_rows_for_cell(ev, vs.SequenceSpec("t", 0.0, 10.0, False, "unknown"))


def sequences(cells):
    """Kept-sequence table from ``{key: (dataset, [(group, frr), ...])}``."""
    rows = []
    for i, (key, (dataset, seqs)) in enumerate(cells.items()):
        animal, session, cell = key.split("|")
        for group, frr in seqs:
            rows.append(dict(key=key, row=i, animal_id=animal, session_id=session, cell_id=int(cell),
                             unique_cell_id=key, dataset=dataset, group=group, frr_23=frr, speed3_minus2=1.0,
                             dur3_minus2=-0.1, wall3_minus2=2.0, analysis="q"))
    return pd.DataFrame(rows)


CELLS = {
    "V1|V1_1|2": ("vollan2025", [("T23<T12", 0.2), ("T23<T12", 0.4), ("T23>T12", -0.1)]),
    "V1|V1_1|5": ("vollan2025", [("T23<T12", 0.1)]),  # one group only: not paired
    "V2|V2_1|1": ("vollan2025", [("T23<T12", 0.0), ("T23>T12", 0.1), ("T23>T12", 0.3)]),
    "B1|B1_1|1": ("brun2008", [("T23<T12", 0.5), ("T23>T12", 0.1)]),
}


def test_paired_cell_values():
    paired = vs.paired_cell_values(sequences(CELLS))
    assert list(paired["key"]) == ["B1|B1_1|1", "V1|V1_1|2", "V2|V2_1|1"]  # sorted by key
    assert list(paired["n_events_T23_lt_T12"]) == [1, 2, 1] and list(paired["n_events_T23_gt_T12"]) == [1, 1, 2]
    np.testing.assert_allclose(paired["frr_T23_lt_T12"], [0.5, 0.3, 0.0])
    np.testing.assert_allclose(paired["frr_T23_gt_T12"], [0.1, -0.1, 0.2])
    np.testing.assert_allclose(paired["paired_difference"], [0.4, 0.4, -0.2])
    np.testing.assert_allclose(paired["antisymmetry_offset"], [0.6, 0.2, 0.2])
    assert list(paired["row"]) == [3, 0, 2] and list(paired["dataset"]) == ["brun2008", "vollan2025", "vollan2025"]


def test_summary_statistics_and_keys():
    paired = vs.paired_cell_values(sequences(CELLS))
    s = vs.visit_sequence_summary(paired)
    diff = paired["paired_difference"].to_numpy()
    r = stats.ttest_1samp(diff, 0.0)
    assert s["n_paired_cells"] == 3
    assert s["paired_t"] == r.statistic and s["paired_p"] == r.pvalue
    assert s["paired_d"] == pytest.approx(diff.mean() / diff.std(ddof=1))
    assert s["pct_positive"] == pytest.approx(200 / 3)
    assert s["t_lt"] == stats.ttest_1samp(paired["frr_T23_lt_T12"], 0.0).statistic
    keys = vs.fig3f_statistics(s)
    expected = {"fig3f.n_cells"} | {f"fig3f.{g}.{k}" for g in ("shorter", "longer") for k in ("mean", "t", "df", "p")} \
        | {f"fig3f.paired.{k}" for k in ("mean_diff", "t", "df", "p")}
    assert set(keys) == expected
    assert keys["fig3f.paired.df"] == 2 and keys["fig3f.shorter.t"] == s["t_lt"]


def test_nesting_levels():
    paired = vs.paired_cell_values(sequences(CELLS))
    nest = vs.nesting_sensitivity(paired).set_index(["cohort", "level"])
    assert nest.loc[("all", "cell"), "n"] == 3 and nest.loc[("all", "session"), "n"] == 3
    assert nest.loc[("all", "animal"), "n"] == 3 and nest.loc[("all", "dataset"), "n"] == 2
    assert nest.loc[("vollan_only", "cell"), "n"] == 2 and nest.loc[("without_vollan", "cell"), "n"] == 1
    assert np.isnan(nest.loc[("without_vollan", "cell"), "t"])  # one value: no test
    assert nest.loc[("all", "cell"), "t"] == pytest.approx(vs.visit_sequence_summary(paired)["paired_t"])


def test_sampling_diagnostic():
    d = vs.sampling_diagnostic(sequences(CELLS), "q")
    assert (d["n_events"], d["n_cells_any_group"], d["n_paired_cells"]) == (9, 4, 3)
    assert d["speed3_minus2_cm_s"] == 1.0 and d["duration3_minus2_s"] == pytest.approx(-0.1)
    summary = vs.visit_sequence_summary(vs.paired_cell_values(sequences(CELLS)))
    assert d["paired_t"] == pytest.approx(summary["paired_t"])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_no_sequences():
    """Too few visits: empty tables with the usual columns and no statistics."""
    empty = vs.build_triplet_table(visit_table([np.nan, 1.0], [0, 1], [1, 2]), vs.FIG3F_SPECS)
    assert empty.empty and {"analysis", "group", "frr_23"} <= set(empty.columns)
    paired = vs.paired_cell_values(empty)
    assert paired.empty and {"paired_difference", "row", "dataset"} <= set(paired.columns)
    assert (vs.nesting_sensitivity(paired)["n"] == 0).all()
    assert vs.sampling_diagnostic(empty, "q")["n_paired_cells"] == 0
    assert vs.visit_sequence_summary(paired)["n_paired_cells"] == 0


def test_visit_cache_follows_the_extraction_code(tmp_path, monkeypatch):
    """The visit cache records the source of the extraction modules; editing one of them invalidates it."""
    sys.path.insert(0, str(REPO / "scripts"))
    import step06_visit_sequences as step

    folder = Path(vs.__file__).resolve().parent
    copy = tmp_path / "rat"
    copy.mkdir()
    for name in step.EXTRACTION_MODULES:
        (copy / f"{name}.py").write_bytes((folder / f"{name}.py").read_bytes())
    original = step.extraction_source_sha256()
    monkeypatch.setattr(step.vs, "__file__", str(copy / "visit_sequences.py"))
    assert step.extraction_source_sha256() == original
    with open(copy / "fields.py", "a", encoding="utf-8") as f:
        f.write("\n# edited\n")
    edited = step.extraction_source_sha256()
    assert edited != original
    trajectories = tmp_path / "trajectories.pkl"
    trajectories.write_bytes(b"0")
    cells = pd.DataFrame({c: ["a"] for c in step.ID_COLUMNS})
    assert step.cache_metadata("printed", {"k": 1}, cells, trajectories)["source_sha256"] == edited


# ---------------------------------------------------------------------------- simulated grid cell
def _simulated_cell(seed=0, duration_s=900.0, box=100.0, fs=50.0, scale=40.0, grid=True):
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    xy = np.empty((n, 2))
    heading = np.empty(n)
    speed = np.empty(n)
    p, h, v = np.array([box / 2, box / 2]), 0.0, 20.0
    for k in range(n):
        h += rng.normal(0, 0.15)
        v = float(np.clip(v + rng.normal(0, 1.0), 8.0, 45.0))
        p = p + v / fs * np.array([np.cos(h), np.sin(h)])
        for d in range(2):  # reflect at the walls
            if p[d] < 0 or p[d] > box:
                p[d] = np.clip(p[d], 0, box)
                h = np.pi - h if d == 0 else -h
        xy[k], heading[k], speed[k] = p, h % (2 * np.pi), v
    t = np.arange(n) / fs
    if grid:
        th = np.deg2rad(10.0)
        a1 = scale * np.array([np.cos(th), np.sin(th)])
        a2 = scale * np.array([np.cos(th + np.pi / 3), np.sin(th + np.pi / 3)])
        ij = np.arange(-6, 7)
        centres = np.array([5.0, 9.0]) + ij[:, None, None] * a1 + ij[None, :, None] * a2
        centres = centres.reshape(-1, 2)
        d2 = ((xy[:, None, :] - centres[None]) ** 2).sum(-1)
        rate = 0.2 + 15.0 * np.exp(-d2 / (2 * (0.15 * scale) ** 2)).max(axis=1)
    else:
        rate = np.zeros(n)  # a silent unit: no rate map, no fields
    spikes = np.repeat(t, rng.poisson(rate / fs))
    position = {"xy": xy, "xy_smoothed": xy, "time": t, "sample_rate": 50, "speed": speed, "move_dir": heading}
    return {"position": position, "spike_times": spikes, "unique_cell_id": "sim_cell1", "animal_id": "V1",
            "session_id": "V1_1", "cell_id": 1}


@pytest.fixture(scope="module")
def simulated_cell():
    return _simulated_cell()


def test_visit_events_of_a_simulated_grid_cell(simulated_cell):
    cells = pd.DataFrame({"row": [0], "animal_id": ["V1"], "session_id": ["V1_1"], "cell_id": [1],
                          "unique_cell_id": ["sim_cell1"], "dataset": ["vollan2025"]})
    found = {}
    for rule in ("printed", "library"):
        res = vs.visit_events_for_cell({"row": 0, "cell_data": simulated_cell, **vs.extraction_settings(rule)})
        assert res["reason"] is None and res["n_fields"] >= 4
        assert abs(res["scale_cm"] - 40.0) < 5.0
        ev = res["events"]
        assert set(ev) == set(vs.EVENT_COLUMNS)
        n = len(ev["event_idx"])
        assert n > 50 and all(len(v) == n for v in ev.values())
        assert np.all(np.diff(ev["t_enter"]) >= 0) and np.all(ev["dur_eff"] >= 0)
        # field centres are rate-map bin centres (3 cm bins), at most half a bin outside the trajectory
        assert np.all(ev["spike_count"] >= 0) and np.all(ev["wall_distance"] >= -1.5)
        found[rule] = ev
        table = vs.events_frame({"row": np.zeros(n, np.int64), **ev}, cells)
        seqs = vs.build_triplet_table(table, vs.FIG3F_SPECS)
        assert set(seqs["analysis"]) <= {s.name for s in vs.FIG3F_SPECS} and len(seqs) > 0
        assert set(seqs["group"]) <= {"T23<T12", "T23>T12"}
    printed, library = found["printed"], found["library"]
    # gap-aware rule: no t_prev before the first visit; standard rule: 0 and always defined
    assert np.isnan(printed["t_prev"][0]) and not printed["t_prev_valid"][0]
    assert printed["t_prev_valid"][1:].mean() > 0.9
    assert library["t_prev"][0] == 0.0 and library["t_prev_valid"].all() and not library["spans_recording_gap"].any()


def test_cell_without_fields_is_reported():
    cell = _simulated_cell(seed=1, duration_s=120.0, grid=False)
    res = vs.visit_events_for_cell({"row": 5, "cell_data": cell, **vs.extraction_settings("printed")})
    assert res["events"] is None and res["reason"] and res["row"] == 5 and res["n_fields"] == 0


# ------------------------------------------------------------------------------- reference results
def _by_key(paired):
    paired = paired.copy()
    paired["key"] = [vs.cell_key(a, s, c) for a, s, c in zip(paired["animal_id"], paired["session_id"],
                                                             paired["cell_id"])]
    return paired.sort_values("key").reset_index(drop=True)


@pytest.mark.frozen
def test_frozen_summary_and_statistics(frozen_path):
    from hexmod.io import read_csv, read_json

    paired = read_csv(frozen_path("rat/visits/paired_cells.csv"))
    assert len(paired) == 3576 and paired["row"].is_monotonic_increasing
    summary = vs.visit_sequence_summary(_by_key(paired))  # in the order of the cell keys, as computed
    ref = read_csv(frozen_path("rat/visits/summary.csv")).iloc[0].to_dict()
    assert set(summary) == set(ref)
    for key, value in ref.items():
        assert summary[key] == value, key
    ref_stats = read_json(frozen_path("stats/step06_visit_sequences.json"))
    assert vs.fig3f_statistics(summary) == ref_stats
    # printed values
    assert (round(summary["t_lt"], 2), round(summary["t_gt"], 2), round(summary["paired_t"], 2)) == (2.53, -2.10, 2.90)
    p_values = (round(summary["p_lt"], 3), round(summary["p_gt"], 3), round(summary["paired_p"], 4))
    assert p_values == (0.012, 0.036, 0.0037)
    # the order of the rows changes the summary only in the last digits
    canonical = vs.visit_sequence_summary(paired)
    assert canonical == pytest.approx(ref, rel=1e-12)


@pytest.mark.frozen
def test_frozen_nesting(frozen_path):
    from hexmod.io import read_csv

    paired = _by_key(read_csv(frozen_path("rat/visits/paired_cells.csv")))
    nest = vs.nesting_sensitivity(paired)
    ref = read_csv(frozen_path("rat/visits/nesting.csv"))
    assert list(nest["cohort"]) == list(ref["cohort"]) and list(nest["level"]) == list(ref["level"])
    assert list(nest["n"]) == list(ref["n"])
    for col in ("mean_paired_difference", "sem_paired_difference", "t", "p_two_sided"):
        # dataset means are averaged in the order of the dataset keys, which differs from the
        # order of the reference computation; the other levels are identical
        np.testing.assert_allclose(nest[col], ref[col], rtol=1e-12, atol=0, equal_nan=True)
    cell_rows = (nest["level"] != "dataset").to_numpy()
    assert np.array_equal(nest.loc[cell_rows, "t"].to_numpy(), ref.loc[cell_rows, "t"].to_numpy(), equal_nan=True)
    session, animal = (nest[(nest["cohort"] == "all") & (nest["level"] == lv)].iloc[0] for lv in ("session", "animal"))
    assert (session["n"], round(session["t"], 2)) == (101, 0.93)
    assert (animal["n"], round(animal["t"], 2)) == (31, 1.29)


@pytest.mark.slow
@pytest.mark.frozen
def test_step06_reproduces_reference(frozen_dir, tmp_path, monkeypatch):
    """Step 06 on the reference inputs: every output equals the reference results."""
    from hexmod.io import read_csv, read_json

    traj = frozen_dir / "preprocessed" / "trajectories.pkl"
    cells = frozen_dir / "rat" / "cohort" / "cells.csv"
    if not (traj.is_file() and cells.is_file()):
        pytest.skip("reference inputs of step 06 missing")
    monkeypatch.setenv("HEXMOD_RESULTS", str(tmp_path))
    res = tmp_path
    (res / "preprocessed").mkdir(parents=True)
    (res / "rat" / "cohort").mkdir(parents=True)
    os.symlink(traj, res / "preprocessed" / "trajectories.pkl")
    os.symlink(cells, res / "rat" / "cohort" / "cells.csv")
    sys.path.insert(0, str(REPO / "scripts"))
    import step06_visit_sequences as step

    assert step.main([]) == 0
    for name in ("paired_cells.csv", "summary.csv", "diagnostics.csv"):
        got = read_csv(res / "rat" / "visits" / name)
        ref = read_csv(frozen_dir / "rat" / "visits" / name)
        pd.testing.assert_frame_equal(got, ref, check_exact=True)
    assert read_json(res / "stats" / "step06_visit_sequences.json") == \
        read_json(frozen_dir / "stats" / "step06_visit_sequences.json")


def test_visit_extraction_reports_errors_separately():
    """An exception in one cell is returned as an error, not as a cell without fields."""
    out = vs.visit_events_for_cell({"row": 5, "cell_data": {}, **vs.extraction_settings("printed")})
    assert out["row"] == 5 and out["events"] is None
    assert out["error"] is not None and out["reason"].startswith("error:")
