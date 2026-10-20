"""Tests of the example-cell arrays (hexmod.rat.examples), the rat figure tables and the statistics
they add (hexmod.rat.figure_tables).

The tests use synthetic data and need no recordings.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from hexmod import circular, stats
from hexmod.rat import examples as ex
from hexmod.rat import figure_tables as ft
from hexmod.rat.characterise import characterise_gridcell


# ------------------------------------------------------------------------------- synthetic data
def _lattice(scale, orientation_deg, extent, origin=(11.0, 6.0)):
    th = np.deg2rad(orientation_deg)
    a1 = scale * np.array([np.cos(th), np.sin(th)])
    a2 = scale * np.array([np.cos(th + np.pi / 3), np.sin(th + np.pi / 3)])
    n = int(np.ceil(3 * extent / scale)) + 2
    i, j = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), indexing="ij")
    pts = np.asarray(origin) + i.reshape(-1, 1) * a1 + j.reshape(-1, 1) * a2
    keep = (pts[:, 0] > -extent) & (pts[:, 0] < 2 * extent) & (pts[:, 1] > -extent) & (pts[:, 1] < 2 * extent)
    return pts[keep]


def _grid_rate(x, y, scale, orientation_deg, extent=100.0, peak=15.0, base=0.2):
    centres = _lattice(scale, orientation_deg, extent)
    d2 = (x[..., None] - centres[:, 0]) ** 2 + (y[..., None] - centres[:, 1]) ** 2
    return base + peak * np.exp(-d2 / (2 * (0.15 * scale) ** 2)).max(axis=-1)


def _random_walk(duration_s=1200.0, box=100.0, fs=50.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    dt = 1 / fs
    heading, speed, xy = np.empty(n), np.empty(n), np.empty((n, 2))
    xy[0] = (box / 2, box / 2)
    h, v = 0.0, 20.0
    for k in range(n):
        h = h + rng.normal(0, 0.15)
        v = np.clip(v + rng.normal(0, 1.0), 8.0, 45.0)
        p = xy[k - 1] + v * dt * np.array([np.cos(h), np.sin(h)]) if k else xy[0]
        for d in range(2):
            if p[d] < 0 or p[d] > box:
                p[d] = np.clip(p[d], 0, box)
                h = np.pi - h if d == 0 else -h
        xy[k] = p
        heading[k] = h % (2 * np.pi)
        speed[k] = v
    return xy, np.arange(n) * dt, speed, heading


GRIDS = ((42.0, 12.0), (38.0, 31.0), (47.0, 50.0))


@pytest.fixture(scope="module")
def synthetic_session():
    """One synthetic session with three grid cells (unit keys 1 to 3) and their characterisation."""
    xy, t, speed, heading = _random_walk(seed=1)
    spikes, labels_, chars = {}, {}, {}
    for k, (scale, ori) in enumerate(GRIDS, start=1):
        counts = np.random.default_rng(10 + k).poisson(_grid_rate(xy[:, 0], xy[:, 1], scale, ori) * 0.02)
        spikes[k] = np.repeat(t, counts)
        labels_[k] = f"V1_V1_1_cell{k}"
    session = {"animal_id": "V1", "session_id": "V1_1", "spike_times": spikes, "unique_cell_ids": labels_,
               "position": {"xy": xy, "xy_smoothed": xy, "time": t, "sample_rate": 50, "speed": speed,
                            "move_dir": heading}}
    from hexmod.rat.gridcell import make_cell_data

    for k in spikes:
        chars[k] = characterise_gridcell({"cell_data": make_cell_data(session, k, "V1", "V1_1")})
        assert chars[k] is not None and chars[k]["fieldLoc"].shape[0] >= 3
    return session, chars


# ---------------------------------------------------------------------------- figure tables
def _toy_cohort(n=30, seed=0):
    rng = np.random.default_rng(seed)
    cells = pd.DataFrame({"row": np.arange(n), "animal_id": "V1", "session_id": "V1_2",
                          "cell_id": np.arange(100, 100 + n),
                          "gridness": rng.uniform(0.31, 1.4, n), "scale_cm": rng.uniform(30, 110, n),
                          "orientation_deg": rng.uniform(0, 60, n), "Rayleigh_mag": rng.uniform(0.0, 0.7, n),
                          "Rayleigh_angle": rng.uniform(0, 360, n)})
    param = pd.DataFrame({"row": np.arange(n), "alpha_tprevXloc_all": rng.normal(-0.03, 0.05, n),
                          "align_tprev_infield": rng.uniform(0.3, 0.35, n),
                          "misalign_tprev_infield": rng.uniform(0.3, 0.35, n),
                          "beta_all": rng.normal(0, 0.05, n), "alpha_tprevXloc_oddMinute": rng.normal(-0.03, 0.05, n),
                          "alpha_tprevXloc_evenMinute": rng.normal(-0.03, 0.05, n)})
    return cells, param


def test_cells_thirds_rule_and_columns():
    cells, param = _toy_cohort()
    x_edges = [np.arange(0, 151, 3.0) for _ in range(len(cells))]
    maps = [np.full((50, 50), float(i)) for i in range(len(cells))]
    maps[3][0, 0] = np.nan
    table, edges = ft.cells_thirds(cells, param, x_edges, maps)
    assert list(table.columns) == ["row", "alpha", "proximity_diff", "beta", "adaptation_third", "proximity_third",
                                   "scale_third", "env_size_cm", "rate_map_max_hz"]
    alpha = param["alpha_tprevXloc_all"].to_numpy()
    b = np.percentile(alpha, np.linspace(0, 100, 4))
    b[-1] += 0.001
    assert edges.loc[0, ["edge_0", "edge_1", "edge_2", "edge_3"]].tolist() == b.tolist()
    assert table["adaptation_third"].tolist() == np.digitize(alpha, b).tolist()
    assert np.bincount(table["scale_third"])[1:].tolist() == [10, 10, 10]
    assert np.array_equal(table["proximity_diff"], param["align_tprev_infield"] - param["misalign_tprev_infield"])
    assert table["env_size_cm"].eq(150.0).all() and table.loc[3, "rate_map_max_hz"] == 3.0
    assert list(edges["quantity"]) == ["alpha", "proximity_diff", "scale_cm"]
    with pytest.raises(ValueError):
        ft.cells_thirds(cells, param.iloc[::-1].reset_index(drop=True), x_edges, maps)


def test_supp3_groups_arrays_and_gamma_fits():
    cells = pd.DataFrame({"row": np.arange(5), "scale_cm": [30.0, 40.0, 50.0, 90.0, 95.0],
                          "gridness": [0.5, 0.2, 0.9, 0.31, 0.3]})
    x_edges = [np.array([0.0, 100.0])] * 5
    dev = [np.array([0.1, 0.2]), np.array([0.3]), np.array([0.05, 0.06, 0.07]), np.array([0.4]), np.array([0.5])]
    rates = [np.array([1.0, 3.0]), np.array([2.0]), np.array([4.0, 4.0, 1.0]), np.array([5.0, 7.0]), np.array([9.0])]
    arr = ft.supp3bcd_arrays(cells, x_edges, dev, rates)
    assert arr["rows_small"].tolist() == [0] and arr["rows_large"].tolist() == [3]  # gridness must exceed 0.3
    assert arr["deviation_small"].tolist() == [0.1, 0.2] and arr["mean_rate_small"].tolist() == [2.0]
    assert arr["rate_normalised_small"].tolist() == [0.5, 1.5]
    assert arr["rate_normalised_large"].tolist() == [5 / 6, 7 / 6]
    fits = ft.gamma_fits(arr)
    dists = ("deviation", "rate_across_cells", "rate_within_cell")
    assert fits[["module", "distribution"]].values.tolist() == [[m, d] for m in ("small", "large") for d in dists]
    assert np.isnan(fits.loc[1, "shape"]) and fits.loc[1, "n_values"] == 1   # one value: no fit
    rng = np.random.default_rng(3)
    big = {k: rng.gamma(4.0, 0.5, 20000) for k in ("deviation_small", "mean_rate_small", "rate_normalised_small",
                                                     "deviation_large", "mean_rate_large", "rate_normalised_large")}
    big.update(rows_small=np.arange(3), rows_large=np.arange(4))
    f = ft.gamma_fits(big)
    assert f["shape"].between(3.8, 4.2).all() and f["scale"].between(0.47, 0.53).all()
    s = ft.supp3_statistics(big, f)
    assert s["supp3.small.n_cells"] == 3 and s["supp3.large.n_fields"] == 20000
    assert s["supp3.large.rate_within_cell.shape"] == f.loc[5, "shape"]
    expected = sps.gamma.fit(big["deviation_small"], floc=0)
    assert (s["supp3.small.deviation.shape"], s["supp3.small.deviation.scale"]) == (expected[0], expected[2])


def test_supp5_widths_and_statistics():
    cells = pd.DataFrame({"row": [0, 1, 2, 3], "Rayleigh_mag": [0.2, np.exp(-0.5), 0.3, 0.9]})
    w = ft.supp5_widths(cells)
    assert w["row"].tolist() == [1, 3]                       # strictly above 0.3
    assert w.loc[0, "circular_sd_deg"] == pytest.approx(np.degrees(1.0))
    s = ft.supp5_statistics(w)
    assert s["supp5.n_cells"] == 2 and s["supp5.narrowest_sd_deg"] == w["circular_sd_deg"].min()
    assert s["supp5.narrowest_sixfold_pct"] == 100 * circular.sixfold_from_sd(w["circular_sd_deg"].min())
    assert ft.supp5_statistics(w.iloc[:0]) == {"supp5.n_cells": 0}


def test_posthoc_statistics_holm_within_panel():
    rng = np.random.default_rng(11)
    n = 60
    fr = rng.normal([-6, 0, 2, 3], 4, (n, 4))
    pa, pm = rng.uniform(0.3, 0.35, n), rng.uniform(0.3, 0.35, n)
    s = ft.posthoc_statistics(fr, pa - pm, rng.uniform(30, 110, n), rng.normal(size=n), rng.normal(size=n), fr[:50])
    pairs = ["0_1s_vs_1_2s", "0_1s_vs_2_3s", "0_1s_vs_above_3s", "1_2s_vs_2_3s", "1_2s_vs_above_3s", "2_3s_vs_above_3s"]
    p = [sps.ttest_rel(fr[:, i], fr[:, j]).pvalue for i in range(4) for j in range(i + 1, 4)]
    assert [s[f"fig3c.posthoc.{k}.p_holm"] for k in pairs] == stats.holm(p).tolist()
    assert "fig3c.posthoc.0_1s_vs_1_2s.t" not in s           # written with the core statistics
    for prefix, names in (("fig4d", ("small", "medium", "large")), ("fig5b", ("strong", "weak", "no"))):
        for a, b in (names[:2], names[::2], names[1:]):
            base = f"{prefix}.posthoc.{a}_vs_{b}"
            assert s[base + ".p_holm"] >= s[base + ".p"] and isinstance(s[base + ".df"], int)
    assert s["supp2.posthoc.2_3s_vs_above_3s.df"] == 49


# ------------------------------------------------------------------------------- example cells
def test_first_visit_order_and_window():
    ev = pd.DataFrame({"t_enter": [1.0, 3.0, 5.0, 7.0, 9.0], "t_exit": [2.0, 4.0, 6.0, 8.0, 10.0],
                       "fieldId": [4, 1, 4, 2, 0]})
    # a visit overlaps the window when it ends after its start and starts before its end
    assert ex.visits_in_window(ev, [3.5, 9.0]) == [1, 4, 2]
    assert ex.first_visit_order(ev, [3.5, 9.0], 6).tolist() == [0, 1, 3, 0, 2, 0]
    assert [ex.shift_prefix(s) for s in (120, 0, -120)] == ["shift_p120ms", "shift_0ms", "shift_m120ms"]


def test_lattice_origin_of_a_perfect_lattice():
    scale, ori = 50.0, 17.0
    pts = _lattice(scale, ori, 100.0, origin=(9.0, 4.0))
    pts = pts[(pts[:, 0] > 0) & (pts[:, 0] < 150) & (pts[:, 1] > 0) & (pts[:, 1] < 150)]
    pts = pts + np.random.default_rng(2).normal(0, 0.5, pts.shape)  # field centres are never exact
    origin = ex.lattice_origin(np.c_[pts, np.full(len(pts), 10.0)], scale, ori)
    th = np.deg2rad(ori)
    basis = scale * np.array([[np.cos(th), np.cos(th + np.pi / 3)], [np.sin(th), np.sin(th + np.pi / 3)]])
    coef = np.linalg.solve(basis, (pts - origin).T)
    assert np.max(np.abs(coef - np.round(coef))) < 0.05      # the fitted lattice passes through the field centres


def test_trajectory_and_shifted_outputs(synthetic_session):
    session, chars = synthetic_session
    cells = pd.DataFrame({"row": [0], "animal_id": ["V1"], "session_id": ["V1_1"], "cell_id": [1],
                          "gridness": [chars[1]["gridness"]], "scale_cm": [chars[1]["scale_cm"]],
                          "orientation_deg": [chars[1]["orientation_deg"]], "Rayleigh_mag": [chars[1]["Rayleigh_mag"]]})
    full = {"rate_map": [chars[1]["rate_map"]], "x_edges": [chars[1]["x_edges"]], "y_edges": [chars[1]["y_edges"]],
            "field_rate_hz": [chars[1]["FR_inhomogeneity_regionFR"]]}
    arrays, visits, seq = ex.trajectory_outputs(session, cells, 0, [chars[1]["fieldLoc"]], full, [100.0, 112.0], True)
    n = len(arrays["t"])
    for k in ("x", "y", "spikes", "speed", "movedir_rad", "t_prev", "in_field", "location_input", "adaptation_variable",
              "grid_alignment", "movedir_rad_pm_pi"):
        assert arrays[k].shape == (n,), k
    assert arrays["spikes"].dtype == np.int64 and set(np.unique(arrays["in_field"])) <= {0, 1, 2}
    assert np.nanmax(arrays["adaptation_variable"]) <= 1.0 and np.all(arrays["speed"] >= 5)
    ori_rad = np.deg2rad(cells.loc[0, "orientation_deg"])
    assert np.allclose(arrays["grid_alignment"], np.cos(6 * (arrays["movedir_rad"] - ori_rad)))
    d = arrays["movedir_rad_pm_pi"]
    assert np.all((d > -np.pi - 1e-12) & (d <= np.pi)) and np.allclose(np.mod(d, 2 * np.pi), arrays["movedir_rad"])
    order = arrays["field_order"]
    assert sorted(order[order > 0].tolist()) == list(range(1, len(set(seq)) + 1))
    assert list(visits.columns) == list(ex.VISIT_COLUMNS.values())
    si = np.zeros(40)
    si[26] = 1.0                                             # best shift 120 ms
    a2, best = ex.supp2a_outputs(session, cells, 0, np.arange(-400, 400, 20), si)
    assert best == 120
    for prefix in ("shift_p120ms", "shift_0ms", "shift_m120ms"):
        assert a2[f"{prefix}__spike_x"].shape == a2[f"{prefix}__spike_count"].shape
        assert a2[f"{prefix}__field_locations"].shape[1] == 3 and float(a2[f"{prefix}__gridness"]) > 0.3
    assert a2["shift_0ms__spike_count"].sum() == arrays["spikes"].sum()
