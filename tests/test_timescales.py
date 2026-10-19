"""Tests of the adaptation timescales (Supp Fig 4) and the Methods numbers (hexmod.rat.timescales).

Synthetic tests run everywhere. Tests marked ``frozen`` recompute the published values from the
reference results in ``HEXMOD_FROZEN`` (per-cell profiles, cohort table) and compare them with the
reference fits, points, bootstrap intervals and statistics.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hexmod import io as hio
from hexmod.rat import timescales as ts


@pytest.fixture(scope="module")
def settings():
    return ts.load_fit_settings()


def _profile(x, a1=-26.0, t1=0.3, a2=-5.0, t2=2.6, c=3.8):
    return ts.exp_double(x, a1, t1, a2, t2, c)


# ------------------------------------------------------------------------------------- synthetic
def test_fine_bins():
    e = ts.FINE_TPREV_EDGES
    assert len(e) == 10 and e[0] == pytest.approx(0.0498, abs=1e-4) and e[-1] == pytest.approx(20.0855, abs=1e-4)
    x = ts.bin_centres()
    assert len(x) == 9 and np.all((x > e[:-1]) & (x < e[1:]))


def test_settings_file(settings):
    for model, k in (("single", 3), ("double", 5)):
        s = settings[model]
        assert len(s["p0"]) == k and all(len(p) == k for p in s["extra_starts"])
        assert len(s["bounds"][0]) == len(s["bounds"][1]) == k and s["method"] == "trf"


def test_percent_change_and_ratio():
    rates = [np.array([9.0, 10.0, 11.0]), np.array([1.0, 2.0, 3.0])]
    pct = ts.percent_change(rates, [10.0, 2.0])
    assert np.allclose(pct, [[-10.0, 0.0, 10.0], [-50.0, 0.0, 50.0]])
    # ratio of first to last bin rate, averaged over cells: (0.9/1.1 + 0.5/1.5) / 2
    assert ts.first_to_last_bin_ratio(pct) == pytest.approx((9 / 11 + 1 / 3) / 2, rel=1e-12)


def test_bin_summary():
    rng = np.random.default_rng(1)
    pct = rng.normal(size=(50, 9))
    pct[3, 8] = np.nan
    s = ts.bin_summary(pct)
    assert s["n_cells"] == 50 and s["n_cells_complete"] == 49
    assert list(s["cells_in_bin"]) == [50] * 8 + [49]
    assert np.allclose(s["sem_pct_change"][0], np.std(pct[:, 0], ddof=1) / np.sqrt(49))
    assert np.allclose(s["weight"], 1 / s["sem_pct_change"] ** 2)
    assert np.allclose(s["ci95_high"] - s["ci95_low"], 2 * 1.96 * s["sem_pct_change"])


@pytest.mark.parametrize("weighted", [True, False])
def test_fit_recovers_parameters(settings, weighted):
    x = ts.bin_centres()
    y = _profile(x)
    sem = np.full(9, 0.3) if weighted else None
    d = ts.fit_model("double", x, y, sem, settings["double"])
    assert d["tau_fast"] <= d["tau_slow"]
    assert d["tau_fast"] == pytest.approx(0.3, rel=1e-4) and d["tau_slow"] == pytest.approx(2.6, rel=1e-4)
    assert d["SS"] < 1e-10
    s = ts.fit_model("single", x, ts.exp_single(x, -20.0, 0.5, 3.0), sem, settings["single"])
    assert s["tau"] == pytest.approx(0.5, rel=1e-6) and s["a"] == pytest.approx(-20.0, rel=1e-6)


def test_model_comparison_formulas(settings):
    x = ts.bin_centres()
    rng = np.random.default_rng(2)
    sem = np.full(9, 0.4)
    y = _profile(x) + rng.normal(scale=0.4, size=9)
    w = ts.model_comparison(x, y, sem, settings, weighted=True)
    s1, s2 = w["single"]["SS"], w["double"]["SS"]
    assert w["AIC_single"] == pytest.approx(s1 + 6) and w["AIC_double"] == pytest.approx(s2 + 10)
    assert w["BIC_double"] == pytest.approx(s2 + 5 * math.log(9))
    assert w["AICc_double"] == pytest.approx(w["AIC_double"] + 20)
    assert w["dAIC_double_minus_single"] == pytest.approx(w["AIC_double"] - w["AIC_single"])
    F = ((s1 - s2) / 2) / (s2 / 4)
    assert w["extra_SS_F"] == pytest.approx(F) and 0 <= w["extra_SS_p"] <= 1
    u = ts.model_comparison(x, y, sem, settings, weighted=False)
    assert u["AIC_single"] == pytest.approx(9 * math.log(u["single"]["SS"] / 9) + 6)
    assert u["n_points"] == 9


def test_keep_best_start(settings):
    # a start that fails (here outside the bounds) is skipped and the next one is used
    x = ts.bin_centres()
    y = ts.exp_single(x, -20.0, 0.5, 3.0)
    s = dict(settings["single"], p0=[0.0, 100.0, 0.0], extra_starts=[[-20.0, 0.5, 3.0]])
    f = ts.fit_model("single", x, y, None, s)
    assert f["tau"] == pytest.approx(0.5, rel=1e-6)


def test_animal_bootstrap_synthetic(settings):
    rng = np.random.default_rng(3)
    x = ts.bin_centres()
    animals = np.repeat([f"A{i}" for i in range(12)], 40)
    offsets = np.repeat(rng.normal(scale=1.0, size=12), 40)
    pct = _profile(x)[None, :] + offsets[:, None] + rng.normal(scale=8.0, size=(len(animals), 9))
    pct[0, 8] = np.nan
    b1 = ts.animal_bootstrap(pct, animals, settings, n_boot=40, rng=np.random.default_rng(7))
    b2 = ts.animal_bootstrap(pct, animals, settings, n_boot=40, rng=np.random.default_rng(7))
    assert b1["tau_single_95"] == b2["tau_single_95"] and b1["tau_slow_95"] == b2["tau_slow_95"]
    assert b1["n_cells"] == len(animals) - 1 and b1["n_animals"] == 12 and b1["n_boot_ok"] == 40
    lo, hi = b1["tau_fast_95"]
    assert lo <= hi and 0.01 <= lo
    assert 0 <= b1["share_boot_double_preferred_by_AIC"] <= 1
    assert len(b1["draws"]["tau_single"]) == 40


def test_interval_statistics():
    scale = np.arange(30.0)
    intervals = [np.array([1.0, 2.0, 100.0]) * (1 + i // 10) for i in range(30)]
    s = ts.interval_statistics(intervals, scale)
    small = s["intervals_by_scale_third"]["small"]
    assert small["n_cells"] == 10 and small["n_events"] == 30   # the 100 s intervals count as events
    assert small["pooled_median_s"] == 1.5                       # but not in the pooled statistics
    assert small["median_of_cell_medians_s"] == 2.0
    assert s["intervals_by_scale_third"]["large"]["pooled_median_s"] == 4.5


def test_running_speed_summary():
    speed = {"s1": {"mean_speed_moving_cm_s": 10.0}, "s2": {"mean_speed_moving_cm_s": 30.0},
             "s3": {"mean_speed_moving_cm_s": 99.0}}
    arena = {"s1": 100.0, "s2": 114.9, "s3": 150.0}
    r = ts.running_speed_summary(["s1", "s2", "s3"], speed, arena)
    assert r["n_sessions"] == 2 and r["mean_speed_cm_s"] == 20.0 and r["speed_range_cm_s"] == [5, 150]
    moving = ts.session_speed({"speed": np.array([1.0, 5.0, 10.0, 150.0, 151.0, np.nan])})
    assert moving["mean_speed_moving_cm_s"] == 55.0 and moving["n_moving"] == 3
    assert ts.arena_size_by_session(["a", "a", "b"], [np.arange(0, 100.0, 3), np.arange(0, 103.0, 3),
                                                      np.arange(10, 160.0, 3)]) == {"a": 100.5, "b": 157.0}


# ---------------------------------------------------------------------------------------- frozen
def _frozen_profile(frozen_path):
    pct = hio.load_npy(frozen_path("rat/timescales/finebin_percent_change.npy"))
    cells = hio.read_csv(frozen_path("rat/cohort/cells.csv"))
    assert len(pct) == len(cells)
    return pct, cells


@pytest.mark.frozen
def test_frozen_points(frozen_path):
    pct, _ = _frozen_profile(frozen_path)
    ref = hio.read_csv(frozen_path("rat/timescales/supp4_points.csv"))
    s = ts.bin_summary(pct)
    assert list(ref["bin"]) == list(range(1, 10))
    for col in ("edge_low_s", "edge_high_s", "x_s", "cells_in_bin", "mean_pct_change", "sem_pct_change",
                "ci95_low", "ci95_high", "weight"):
        assert np.array_equal(np.asarray(s[col], dtype=float), ref[col].to_numpy(float)), col


@pytest.mark.frozen
def test_frozen_weighted_fits(frozen_path, settings):
    pct, _ = _frozen_profile(frozen_path)
    s = ts.bin_summary(pct)
    ref = hio.read_json(frozen_path("rat/timescales/fits.json"))["unshifted"]
    for weighting, weighted in (("weighted_by_SEM", True), ("unweighted", False)):
        got = ts.model_comparison(s["x_s"], s["mean_pct_change"], s["sem_pct_change"], settings, weighted)
        exp = ref[weighting]
        assert got["single"] == exp["single"], weighting
        if weighted:   # the printed fits: every value bit for bit
            assert got == exp
        else:          # the unweighted two-exponential fit is not printed; it agrees to 1e-5
            for k, v in exp["double"].items():
                assert got["double"][k] == pytest.approx(v, rel=1e-5), k
    table = hio.read_csv(frozen_path("rat/timescales/supp4_fits.csv"))
    w = ref["weighted_by_SEM"]
    values = [w["single"]["a"], w["single"]["tau"], w["single"]["c"], w["single"]["SS"], w["double"]["a_fast"],
              w["double"]["tau_fast"], w["double"]["a_slow"], w["double"]["tau_slow"], w["double"]["c"],
              w["double"]["SS"]]
    assert table["value"].tolist() == values


@pytest.mark.frozen
def test_frozen_bootstrap_reproduces_reference(frozen_path, settings):
    """Same cells, animals, number of resamples and seed as the reference bootstrap. The intervals of the
    one-exponential time constant agree to 1e-6 s (the optimiser tolerance). The two-exponential fits of the
    resamples are poorly constrained and sensitive to the starting values, so their intervals agree to 0.02 s
    and the shares to one resample in 500 (from seed to seed the interval ends vary by about 0.03 s)."""
    pct, cells = _frozen_profile(frozen_path)
    ref = hio.read_json(frozen_path("rat/timescales/bootstrap.json"))["unshifted"]
    b = ts.animal_bootstrap(pct, cells["animal_id"].to_numpy(), settings, n_boot=ts.N_BOOTSTRAP,
                            rng=np.random.default_rng(ts.PUBLISHED_BOOTSTRAP_SEED))
    assert b["n_cells"] == ref["n_cells"] and b["n_animals"] == ref["n_animals"] and b["n_boot_ok"] == ref["n_boot_ok"]
    assert np.allclose(b["tau_single_95"], ref["tau_single_95"], rtol=0, atol=1e-6)
    for key in ("tau_fast_95", "tau_slow_95"):
        assert np.allclose(b[key], ref[key], rtol=0, atol=0.02), key
    for key in ("share_boot_double_preferred_by_AIC", "share_slow_over_10s"):
        assert abs(b[key] - ref[key]) <= 0.002, key


@pytest.mark.frozen
def test_frozen_methods_ratio(frozen_path):
    pct = hio.load_npy(frozen_path("rat/adaptation/fr_percent_change.npy"))
    # printed with eight decimals by the published analysis: 0.92315494
    assert abs(ts.first_to_last_bin_ratio(pct) - 0.92315494) < 5e-9


@pytest.mark.frozen
def test_frozen_profile_first_cells(frozen_path):
    """The nine-bin profile of the first cohort cells equals the reference rows exactly."""
    from hexmod.rat.gridcell import make_cell_data

    pct_ref, cells = _frozen_profile(frozen_path)
    traj = hio.load_cache(frozen_path("preprocessed/trajectories.pkl"))
    fl = hio.load_npz(frozen_path("rat/cohort/field_locations.npz"))["field_locations"]
    rows = list(range(4))
    out = []
    for i in rows:
        r = cells.iloc[i]
        cd = make_cell_data(traj[r.animal_id][r.session_id], int(r.cell_id), r.animal_id, r.session_id)
        out.append(ts.fine_bin_cell(ts.fine_bin_inputs(cd, fl[i])))
    pct = ts.percent_change([o["rate_hz"] for o in out], [o["mean_rate_hz"] for o in out])
    assert np.array_equal(pct, pct_ref[rows], equal_nan=True)


@pytest.mark.frozen
def test_frozen_statistics_keys(frozen_path, settings):
    ref = hio.read_json(frozen_path("stats/step07_adaptation_timescales.json"))
    pct, _ = _frozen_profile(frozen_path)
    s = ts.bin_summary(pct)
    w = ts.model_comparison(s["x_s"], s["mean_pct_change"], s["sem_pct_change"], settings, True)
    assert ref["supp4.n_cells"] == len(pct) and ref["supp4.n_cells_last_bin"] == int(s["cells_in_bin"][-1])
    assert ref["supp4.single.tau_s"] == w["single"]["tau"] and ref["supp4.double.tau_slow_s"] == w["double"]["tau_slow"]
    assert ref["supp4.f_test.p"] == w["extra_SS_p"] and ref["supp4.delta_bic"] == w["dBIC"]
