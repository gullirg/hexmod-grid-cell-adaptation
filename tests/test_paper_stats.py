"""Tests of hexmod.rat.paper_stats: grouping rules, statistic keys and the alpha fit limits.

The synthetic tests need no data. Tests marked ``frozen`` recompute the printed statistics from
the reference per-cell results in ``HEXMOD_FROZEN`` and compare them with the reference
statistics files.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from hexmod import stats
from hexmod.rat import paper_stats as ps
from hexmod.rat.adaptation import adaptation_parametric
from hexmod.rat.fields import extract_field_entry, field_visit_input, position_binning
from hexmod.rat.gridcell import GridCellAnalysis


# ------------------------------------------------------------------------------ grouping rules
def test_thirds_rule():
    x = np.arange(1.0, 10.0)
    edges, cat = ps.thirds(x)
    assert edges[:3] == pytest.approx(np.percentile(x, [0, 100 / 3, 200 / 3]))
    assert edges[3] == pytest.approx(9.001)
    assert cat.tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 3]  # the largest value falls in the top third
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    e, c = ps.thirds(y)
    assert set(c) == {1, 2, 3} and np.bincount(c)[1:].tolist() == [100, 100, 100]
    assert (y[c == 1].max() < e[1]) and (y[c == 3].min() >= e[2])


def test_conjunctive_mask_is_strictly_above_0_3():
    assert ps.conjunctive_mask([0.2, 0.3, 0.31]).tolist() == [False, False, True]


def _toy_cells():
    return pd.DataFrame({
        "row": np.arange(6),
        "animal_id": ["V1", "V1", "V2", "B1", "B1", "B1"],
        "session_id": ["V1_1", "V1_2", "V2_1", "B1_1", "B1_1", "B1_2"],
        "cell_id": [1, 2, 3, 4, 5, 6],
        "dataset": ["vollan2025"] * 3 + ["brun2008"] * 3,
        "scale_cm": [40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        "Rayleigh_mag": [0.1, 0.5, 0.4, 0.3, 0.05, 0.6],
    })


def test_cohort_statistics_and_table1():
    cells = _toy_cells()
    s = ps.cohort_statistics(cells)
    assert (s["cohort.n_animals"], s["cohort.n_sessions"], s["cohort.n_cells"]) == (3, 5, 6)
    v = [s[f"cohort.vollan2025.{k}"] for k in ("n_animals", "n_sessions", "n_cells")]
    assert v == [2, 3, 3]
    assert (s["cohort.brun2008.n_sessions"], s["cohort.brun2008.n_cells"]) == (2, 3)
    c = [s[f"cohort.conjunctive.{k}"] for k in ("n_cells", "n_animals", "n_sessions")]
    assert c == [3, 3, 3]
    assert (s["table1.brun2008.scale_min_cm"], s["table1.brun2008.scale_max_cm"]) == (70.0, 90.0)
    # a dataset without cells gets zero counts and no scale range
    s2 = ps.cohort_statistics(cells, ["brun2008", "vollan2025", "hafting2005"])
    assert s2["cohort.hafting2005.n_cells"] == 0 and "table1.hafting2005.scale_min_cm" not in s2
    text = pd.DataFrame({"dataset": ["brun2008", "vollan2025"], "environment": ["square", "square"],
                         "grid_scale_text": ["60 to 150 cm", "50 to 150 cm"]})
    t = ps.table1(cells, text)
    assert t["dataset"].tolist() == ["brun2008", "vollan2025", "total"]  # order of Table 1
    assert t["n_conjunctive"].tolist() == [1, 2, 3]
    assert t.loc[0, "label"] == "Brun et al. (2008)" and t.loc[0, "grid_scale_text"] == "60 to 150 cm"
    assert (t.loc[2, "scale_min_cm"], t.loc[2, "scale_max_cm"]) == (40.0, 90.0)


# ---------------------------------------------------------------------------- statistic keys
def test_fig3c_keys_and_values():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4)) + np.array([-6, 0, 2, 3])
    X[3, 2] = np.nan  # rows with a missing bin are left out
    s = ps.fig3c_statistics(X)
    r = stats.rm_anova(X)
    assert s["fig3c.rm_anova.F"] == r["F"] and (s["fig3c.rm_anova.df1"], s["fig3c.rm_anova.df2"]) == (3, 3 * 48)
    assert s["fig3c.mean_pct_change.0_1s"] == pytest.approx(np.delete(X, 3, axis=0)[:, 0].mean())
    t = sps.ttest_rel(np.delete(X[:, 2], 3), np.delete(X[:, 3], 3))
    assert s["fig3c.posthoc.2_3s_vs_above_3s.t"] == pytest.approx(t.statistic)
    assert len([k for k in s if k.startswith("fig3c.posthoc.")]) == 18


def test_fig4_fig5_supp1_supp2_keys():
    rng = np.random.default_rng(2)
    n = 300
    alpha = rng.normal(-0.05, 0.05, n)
    pa = rng.uniform(0.3, 0.4, n)
    pm = pa - rng.normal(0.002, 0.01, n)
    beta = rng.normal(0, 0.02, n)
    scale = rng.uniform(30, 150, n)
    s4c = ps.fig4c_statistics(pa, pm)
    assert s4c["fig4c.pct_higher"] == pytest.approx((pa - pm).mean() / pm.mean() * 100)
    s4d = ps.fig4d_statistics(pa - pm, scale)
    assert s4d["fig4d.n.small"] + s4d["fig4d.n.medium"] + s4d["fig4d.n.large"] == n
    assert s4d["fig4d.scale_edges_cm.3"] == pytest.approx(scale.max() + 0.001)
    s5 = ps.fig5_statistics(alpha, pa - pm, beta)
    _, ac = ps.thirds(alpha)
    _, pc = ps.thirds(pa - pm)
    strong_top = beta[(ac == 1) & (pc == 3)]
    assert s5["fig5b.strong.n"] == len(strong_top)
    assert s5["fig5b.strong.t"] == pytest.approx(sps.ttest_1samp(strong_top, 0).statistic)
    assert s5["fig5c.strong_top_vs_bottom.df"] == len(strong_top) + s5["fig5c.strong_bottom.n"] - 2
    n_top = sum(s5[f"fig5b.{g}.n"] for g in ("strong", "weak", "no"))
    assert s5["fig5b.anova.df1"] == 2 and s5["fig5b.anova.df2"] == n_top - 3
    odd, even = alpha + rng.normal(0, 0.02, n), alpha + rng.normal(0, 0.02, n)
    odd[0] = np.nan
    s1 = ps.supp1_statistics(odd, even)
    assert s1["supp1.odd_vs_even.df"] == n - 2 and s1["supp1.pearson_df"] == n - 3
    peak = np.array([0, 40, 160, 120, 140, -120])
    s2 = ps.supp2_statistics(peak, alpha[:100], alpha[:100] + rng.normal(0.01, 0.01, 100), rng.normal(size=(100, 4)))
    assert s2["supp2.example_best_shift_ms"] == 120 and s2["supp2.median_best_shift_ms"] == 80.0
    assert (s2["supp2.n_cells"], s2["supp2.n_cells_kept"], s2["supp2.rm_anova.df2"]) == (6, 100, 297)
    assert ps.example_shift_index([0, 20, 160]) is None


# ------------------------------------------------------------------------------- alpha limits
def _walk(duration_s, seed, box=100.0, fs=50.0):
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    xy, heading, speed = np.empty((n, 2)), np.empty(n), np.empty(n)
    p, h, v = np.array([box / 2, box / 2]), 0.0, 20.0
    for k in range(n):
        h += rng.normal(0, 0.15)
        v = np.clip(v + rng.normal(0, 1.0), 8.0, 45.0)
        if k:
            p = p + v / fs * np.array([np.cos(h), np.sin(h)])
        for d in range(2):
            if p[d] < 0 or p[d] > box:
                p[d] = np.clip(p[d], 0, box)
                h = np.pi - h if d == 0 else -h
        xy[k], heading[k], speed[k] = p, h % (2 * np.pi), v
    return xy, np.arange(n) / fs, speed, heading


def _simulated_cell(alpha_true, seed, duration_s=900.0, scale=50.0, ori=10.0):
    """Grid cell on a random walk whose rate is multiplied by 1 + alpha_true * (z-scored regressor)."""
    xy, t, speed, heading = _walk(duration_s, seed)
    th = np.deg2rad(ori)
    a1 = scale * np.array([np.cos(th), np.sin(th)])
    a2 = scale * np.array([np.cos(th + np.pi / 3), np.sin(th + np.pi / 3)])
    c = np.array([np.array([20.0, 25.0]) + i * a1 + j * a2 for i in range(-4, 5) for j in range(-4, 5)])
    c = c[(np.abs(c[:, 0] - 50) < 100) & (np.abs(c[:, 1] - 50) < 100)]
    peaks = np.random.default_rng(seed + 1).uniform(8.0, 20.0, len(c))
    sigma = 0.2 * scale
    _, _, _, ix, iy = position_binning(xy[:, 0], xy[:, 1], 5)
    bx, by = xy[:, 0].min() + 5 * (ix + 0.5), xy[:, 1].min() + 5 * (iy + 0.5)
    place = 0.3 + (peaks * np.exp(-((bx[:, None] - c[:, 0]) ** 2 + (by[:, None] - c[:, 1]) ** 2)
                                  / (2 * sigma ** 2))).max(axis=1)
    inside = (c[:, 0] > 0) & (c[:, 0] < 100) & (c[:, 1] > 0) & (c[:, 1] < 100)
    fieldLoc = np.c_[c[inside], np.full(inside.sum(), 1.2 * sigma)]
    peakFR = peaks[inside]
    position = {"xy": xy, "xy_smoothed": xy, "time": t, "sample_rate": 50.0, "speed": speed, "move_dir": heading}
    labels = {"unique_cell_id": "sim", "animal_id": "sim", "session_id": "sim_1", "cell_id": 0}
    g = GridCellAnalysis({"position": position, "spike_times": np.array([]), **labels})
    ev = extract_field_entry(g.x, g.y, g.time, fieldLoc[:, 0], fieldLoc[:, 1], fieldLoc[:, 2])
    w = field_visit_input(g.x, g.y, g.time, ev["ev_df"], peakFR, g.speed)["adaptWeight_loc"]
    a = np.exp(-ev["t_prev_vec"] / 2.0) * w / np.max(w)
    a = (a - a.mean()) / a.std()
    rate = np.clip(place * (1 + alpha_true * a), 0, None)
    counts = np.random.default_rng(seed + 2).poisson(rate * 0.02)
    return {"cell_data": {"position": position, "spike_times": np.repeat(t, counts), **labels},
            "fieldLoc": fieldLoc, "peakFR": peakFR, "gridOri": ori, "tau_adapt": 2.0, "spatialResolution": 5,
            "adapt_type": ["tprev", "tprevXloc"], "mask_type": ["all", "infield"]}


def test_alpha_bounds_equal_the_limits_of_the_fit():
    inputs = _simulated_cell(-0.5, seed=3)  # adaptation beyond the limit: the fit stops at the lower limit
    fit = adaptation_parametric(inputs)
    b = ps.alpha_bounds(inputs)
    assert (b["session_id"], b["cell_id"]) == ("sim_1", 0)
    for c in ("tprev_all", "tprevXloc_all", "tprev_infield", "tprevXloc_infield"):
        assert b[f"alpha_lower_{c}"] < 0 < b[f"alpha_upper_{c}"]
        assert b[f"alpha_lower_{c}"] <= fit[f"alpha_{c}"] <= b[f"alpha_upper_{c}"]
    assert fit["alpha_tprevXloc_all"] == b["alpha_lower_tprevXloc_all"]  # exactly the clipped value
    inside = _simulated_cell(-0.02, seed=4)
    fit2 = adaptation_parametric(inside)
    b2 = ps.alpha_bounds(inside)
    assert b2["alpha_lower_tprevXloc_all"] < fit2["alpha_tprevXloc_all"] < b2["alpha_upper_tprevXloc_all"]
    param = pd.DataFrame({"alpha_tprevXloc_all": [fit["alpha_tprevXloc_all"], fit2["alpha_tprevXloc_all"]]})
    bounds = pd.DataFrame({k: [b[k], b2[k]] for k in ("alpha_lower_tprevXloc_all", "alpha_upper_tprevXloc_all")})
    counts = ps.alpha_bound_counts(param, bounds)
    assert counts == {"diagnostic.alpha_at_bound.tprevXloc_all.n_cells": 2,
                      "diagnostic.alpha_at_bound.tprevXloc_all.n_lower": 1,
                      "diagnostic.alpha_at_bound.tprevXloc_all.n_upper": 0,
                      "diagnostic.alpha_at_bound.tprevXloc_all.n_total": 1}


# ------------------------------------------------------------------- frozen reference results
CORE_PREFIXES = ("fig3c.", "fig3d.", "fig4c.", "fig4d.", "fig5.", "fig5a.", "fig5b.", "fig5c.", "supp1.", "supp2.")


@pytest.mark.frozen
def test_frozen_cohort_statistics(frozen_path):
    from hexmod.io import read_csv, read_json

    cells = read_csv(frozen_path("rat/cohort/cells.csv"))
    ref = read_json(frozen_path("stats/step02_cohort.json"))
    s = ps.cohort_statistics(cells)
    assert s == ref  # counts and scale ranges, exactly


@pytest.mark.frozen
def test_frozen_core_statistics(frozen_dir, frozen_path):
    from hexmod.io import read_json

    for rel in ("rat/cohort/cells.csv", "rat/adaptation/param_cells.csv", "rat/prospective/shift_cells.csv"):
        frozen_path(rel)  # skips when a reference file is missing
    s = ps.core_statistics_from_results(frozen_dir)
    ref = {k: v for k, v in read_json(frozen_path("stats/step12_rat_statistics.json")).items()
           if k.startswith(CORE_PREFIXES)}
    assert set(ref) <= set(s), sorted(set(ref) - set(s))     # s may hold keys added since the reference
    for k, v in ref.items():
        assert abs(s[k] - v) <= 1e-9 and abs(s[k] - v) <= 1e-9 * abs(v) + 1e-300, (k, s[k], v)
    # the printed values
    assert round(s["fig3c.rm_anova.F"], 1) == 375.8 and round(s["fig3d.t"], 2) == -41.09
    assert round(s["fig4c.t"], 2) == 11.07 and round(s["fig4d.anova.F"], 2) == 7.06
    assert (s["fig5b.strong.n"], s["fig5b.weak.n"], s["fig5b.no.n"]) == (435, 430, 448)
    assert round(s["fig5b.strong.t"], 3) == -2.845 and round(s["supp2.rm_anova.F"], 1) == 294.7
