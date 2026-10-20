"""Tests of the conjunctive-cell analyses (Fig 2B to 2E) and of the same-cell links.

The tests use synthetic inputs with known answers.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hexmod import stats
from hexmod.rat import conjunctive as cj
from hexmod.rat import same_cell as sc


# ============================================================================== conjunctive cells
def test_deviation_from_axes_wraps_to_plus_minus_30():
    d = cj.deviation_from_axes([10.0, 75.0, 359.0, 30.0, 0.0], [5.0, 0.0, 0.0, 0.0, 30.0])
    np.testing.assert_allclose(d, [5.0, 15.0, -1.0, -30.0, -30.0])
    assert np.all((d >= -30) & (d < 30))


def test_fig2b_bins_counts_and_halves():
    dev = np.array([-30.0, -25.0, -14.9, -14.0, 0.0, 4.999, 14.99, 15.0, 29.9])
    b = cj.fig2b_bins(dev)
    assert list(b["bin"]) == list(range(1, 13))
    assert list(b["deviation_from_deg"]) == list(range(-30, 30, 5))
    assert b["count"].sum() == len(dev)
    assert list(b["half"][b["half"] == "aligned"].index) == [3, 4, 5, 6, 7, 8]
    # lower edge included: 15 falls in the bin from 15 (misaligned)
    assert int(b.loc[b.deviation_from_deg == -15, "count"].iloc[0]) == 2
    assert int(b.loc[b.deviation_from_deg == 15, "count"].iloc[0]) == 1
    assert int(b.loc[b.half == "aligned", "count"].sum()) == int((np.abs(dev) < 15).sum())


def test_tuning_width_inverts_resultant_length():
    sd = np.array([20.0, 45.0, 73.0])
    R = np.exp(-np.radians(sd) ** 2 / 2)
    np.testing.assert_allclose(cj.tuning_width_deg(R), sd, rtol=1e-12)


def _von_mises_curves(pref_deg, kappa=2.0):
    th = np.radians(cj.BIN_CENTRES_DEG)
    return np.exp(kappa * np.cos(th[None, :] - np.radians(np.asarray(pref_deg))[:, None]))


def test_aligned_tuning_curves_put_the_preferred_bin_first():
    pref = np.array([2.5, 100.0, 357.4, 182.6])
    A, ang = cj.aligned_tuning_curves(_von_mises_curves(pref), pref)
    assert A.shape == (4, 73) and ang.shape == (73,)
    assert ang[0] == 2.5 and ang[-1] == 2.5 and ang[71] == 357.5
    assert np.all(np.argmax(A[:, :72], axis=1) == 0)
    np.testing.assert_array_equal(A[:, 72], A[:, 0])


def test_per_animal_curves_mean_and_sem_across_animals():
    rng = np.random.default_rng(3)
    ids = np.array(["V2"] * 6 + ["V10"] * 5 + ["W5"] * 2)
    A = rng.random((len(ids), 73)) + 0.5
    curves, mean, summary = cj.per_animal_curves(A, ids, np.arange(73.0))
    assert summary == {"n_animals": 2, "n_cells": 11}
    nn = mean[mean.form == "not_normalised"]
    m2, m10 = A[ids == "V2"].mean(0), A[ids == "V10"].mean(0)
    np.testing.assert_allclose(nn["mean_across_animals"], (m2 + m10) / 2, rtol=1e-14)
    np.testing.assert_allclose(nn["sem_across_animals"], np.std([m2, m10], axis=0, ddof=1) / np.sqrt(2), rtol=1e-12)
    # animals by decreasing cell count, included animals first; both forms; normalised form first in the mean table
    order = list(dict.fromkeys(curves.loc[curves.form == "not_normalised", "animal_id"]))
    assert order == ["V2", "V10", "W5"]
    assert not curves.loc[curves.animal_id == "W5", "in_mean_across_animals"].any()
    assert list(dict.fromkeys(mean.form)) == ["normalised_to_cell_mean", "not_normalised"]
    norm = curves[(curves.form == "normalised_to_cell_mean") & (curves.animal_id == "V2")]["mean_curve"].to_numpy()
    np.testing.assert_allclose(norm, (A[ids == "V2"] / A[ids == "V2", :72].mean(1, keepdims=True)).mean(0), rtol=1e-14)


def test_kl_divergence_of_flat_and_peaked_curves():
    flat = np.ones((1, 72))
    peak = np.full((1, 72), 1e-12)
    peak[0, 10] = 1.0
    assert cj.kl_from_uniform(flat)[0] == pytest.approx(0.0, abs=1e-14)
    assert cj.kl_from_uniform(peak)[0] == pytest.approx(math.log(72), rel=1e-9)
    sm = cj.gaussian_smooth_circular(np.vstack([flat, peak]), cj.KL_SMOOTH_SD_DEG)
    np.testing.assert_allclose(sm.sum(1), [72.0, peak.sum()], rtol=1e-12)  # the kernel sums to 1 (circular)
    np.testing.assert_allclose(sm[0], 1.0, rtol=1e-12)
    assert cj.KL_SMOOTH_SD_DEG == pytest.approx(9.0139, abs=1e-4)
    assert 0 < cj.tuning_kl(peak)[0] < math.log(72)


def test_direction_orientation_correlation_is_seeded_and_detects_coupling():
    rng = np.random.default_rng(7)
    ori = rng.uniform(0, 60, 80)
    coupled = cj.direction_orientation_correlation(ori, ori + 120 + rng.normal(0, 2, 80), n_perm=500)
    assert coupled["r"] > 0.9 and coupled["p_one_sided_positive"] == pytest.approx(1 / 501)
    indep = rng.uniform(0, 360, 80)
    a = cj.direction_orientation_correlation(ori, indep, n_perm=500)
    b = cj.direction_orientation_correlation(ori, indep, n_perm=500)
    assert a == b  # same seed, same permutations
    a6, b6 = np.radians(6 * ori), np.radians(6 * (indep % 60))
    assert a["r"] == stats.fisher_lee(a6, b6)
    expected = stats.fisher_lee_permutation(a6, b6, 500, seed=cj.FIG2D_PERMUTATION_SEED)["p_one_sided_positive"]
    assert a["p_one_sided_positive"] == expected


def test_per_animal_rayleigh_columns_order_and_rules():
    rng = np.random.default_rng(11)
    ids = np.array(["V2"] * 40 + ["V10"] * 70 + ["V9"] * 4)
    d = np.r_[rng.vonmises(0, 4, 40) * 180 / np.pi / 6 + 10, rng.uniform(0, 360, 70), rng.uniform(0, 360, 4)]
    pa = cj.per_animal_rayleigh(ids, d)
    assert list(pa.columns) == [
        "animal_id",
        "n",
        "R6",
        "Z",
        "p_rayleigh",
        "R_crit_p05",
        "phase6_deg",
        "p_holm",
        "R6_chance95",
    ]
    assert list(pa.animal_id) == ["V10", "V2"]  # sorted as text; V9 has fewer than 5 cells
    r = pa.set_index("animal_id")
    assert r.loc["V2", "p_rayleigh"] < 1e-6 and r.loc["V2", "phase6_deg"] == pytest.approx(10, abs=3)
    np.testing.assert_allclose(pa.p_holm, stats.holm(pa.p_rayleigh))
    for n, rc in zip(pa.n, pa.R_crit_p05):
        assert stats.zar_p(int(n), rc) == pytest.approx(0.05, rel=1e-8)
    assert r.loc["V10", "R6_chance95"] == math.sqrt(-math.log(0.05) / 70)  # closed form above 60 cells
    assert r.loc["V2", "R6_chance95"] == stats.rayleigh_monte_carlo(0.0, 40)[2]  # draws with seed 5600 + n


def test_chance_band_rule():
    band = cj.chance_band([5, 60, 61, 300])
    assert list(band.n) == [5, 60, 61, 300]
    assert band.R_chance95.iloc[2] == math.sqrt(-math.log(0.05) / 61)
    assert band.R_chance_mean.iloc[3] == math.sqrt(math.pi / (4 * 300))
    nul = stats.rayleigh_chance(5)
    assert band.R_chance95.iloc[0] == float(np.quantile(nul, 0.95))
    full = cj.chance_band()
    assert list(full.n) == list(range(5, 301)) and full.R_chance95.is_monotonic_decreasing


# ================================================================================= same-cell links
def test_spike_coincidence_and_same_recording():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.5, 1.0004, 2.0009, 3.01, 4.0, 5.0])
    f, n = sc.spike_coincidence(a, b)
    assert n == 4 and f == pytest.approx(0.75)
    assert sc.spike_coincidence(b, a) == (f, n)
    ra = {"t0": 10.0, "t1": 100.0, "nt": 5}
    assert sc.same_recording(ra, {"t0": 10.3, "t1": 99.6, "nt": 5})
    assert not sc.same_recording(ra, {"t0": 10.3, "t1": 99.6, "nt": 6})


def test_session_number_and_hafting_label():
    assert sc.session_number("V10_2") == 2 and sc.session_number("W5_12") == 12
    assert sc.hafting_label("Hafting_Fig2c_Trial1.mat:rat11015_t5c1_timeStamps") == "t5c1"
    assert sc.hafting_label("Hafting_Fig2d_Trial2.mat:rat11015_t2c2") == "t2c2"
    assert sc.hafting_label("rat_10925.mat:cell3_timeStamps") == "cell3"


def test_unit_components_join_through_units_outside_the_cohort():
    entries = [("V2_1", 3), ("V2_4", 5), ("V2_5", 5), ("V2_5", 8), ("B1_1", 2), ("B1_2", 4), ("B1_2", 6)]
    files = {"V2_1": "f1", "V2_4": "f4", "V2_5": "f5"}
    vl = pd.DataFrame(
        {"file_a": ["f4", "f1", "f1"], "idx_a": [5, 3, 9], "file_b": ["f5", "f4", "f5"], "idx_b": [5, 99, 8]}
    )
    # f1:3 - f4:99 (not a cohort cell) does not join V2_4:5; f1:9 (not in the cohort) - f5:8 joins nothing in the cohort
    units = sc.unit_components(entries, vl, files, [(("B1_1", 2), ("B1_2", 6))])
    assert list(units) == [0, 1, 1, 2, 3, 4, 3]
    vl2 = pd.concat(
        [vl, pd.DataFrame({"file_a": ["f4"], "idx_a": [99], "file_b": ["f5"], "idx_b": [8]})], ignore_index=True
    )
    units2 = sc.unit_components(entries, vl2, files, [])
    assert units2[0] == units2[3]  # V2_1:3 - f4:99 - V2_5:8
    kept = sc.kept_earliest(units, [sc.session_number(s) for s, _ in entries])
    assert list(kept) == [True, True, False, True, True, True, False]


def test_kept_earliest_prefers_lowest_session_then_order():
    units = np.array([0, 0, 0, 1, 1])
    assert list(sc.kept_earliest(units, [3, 1, 1, 2, 5])) == [False, True, False, True, False]


def test_map_similarity_finds_one_bin_shift():
    rng = np.random.default_rng(5)
    a = rng.random((20, 20))
    b = np.full_like(a, np.nan)
    b[1:, :] = a[:-1, :]
    assert sc.map_similarity(a, b) == pytest.approx(1.0, abs=1e-12)
    assert sc.map_similarity(a, b, max_shift=0) < 0.5
    assert np.isnan(sc.map_similarity(np.full((5, 5), np.nan), a[:5, :5]))


def test_tetrode_rate_map_dwell_threshold():
    spikes = np.zeros((10, 10), np.float32)
    dwell = np.zeros((10, 10), np.float32)
    dwell[5, 5] = 10.0
    spikes[5, 5] = 20.0
    r = sc.tetrode_rate_map(spikes, dwell)
    assert np.isfinite(r).sum() > 1 and np.nanmax(r) == pytest.approx(2.0)
    assert np.isnan(sc.tetrode_rate_map(spikes, np.zeros_like(dwell))).all()


def _grid_map(n, phase, scale=8.0, ori=0.3):
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    g = sum(
        np.cos(
            2
            * np.pi
            / scale
            * (np.cos(ori + k * np.pi / 3) * (x - phase[0]) + np.sin(ori + k * np.pi / 3) * (y - phase[1]))
        )
        for k in range(3)
    )
    return np.exp(0.5 * g)


def test_link_tetrode_links_the_same_cell_one_to_one():
    rng = np.random.default_rng(2)
    phases = [(0, 0), (3, 1), (5, 4)]
    rows, maps = [], []
    for s, sid in enumerate(["H1_1", "H1_2"]):
        for c, ph in enumerate(phases):
            maps.append(_grid_map(50, ph) + rng.normal(0, 0.05, (50, 50)))
            rows.append(
                {
                    "row": len(rows),
                    "animal_id": "H1",
                    "session_id": sid,
                    "cell_id": 10 * s + c,
                    "dataset": "hafting2005",
                    "orientation_deg": 17.0,
                    "scale_cm": 40.0 + s,
                    "Ex_cm": 100.0,
                    "Ey_cm": 100.0,
                    "rate_hz": 1.0,
                    "label": ["t1c1", "t1c2", "x"][c] if s == 0 else ["t1c1", "zz", "t2c1"][c],
                }
            )
    pairs, thr = sc.link_tetrode(pd.DataFrame(rows), np.stack(maps))
    assert thr["hafting2005"]["n_null"] == 6 and thr["hafting2005"]["rm_threshold"] >= 0.5
    linked = pairs[pairs.map_link]
    assert len(pairs) == 9 and sorted(zip(linked.cell_a, linked.cell_b)) == [(0, 10), (1, 11), (2, 12)]
    assert list(pairs[pairs.label_link][["cell_a", "cell_b"]].itertuples(index=False, name=None)) == [(0, 10)]
    assert pairs.link.sum() == 3


def _vollan_record(t0, maps, waves, idx, positions):
    nsite = waves[0].shape[1]
    pcm = [
        {
            "xcoords": np.arange(nsite, dtype=float),
            "ycoords": np.arange(nsite, dtype=float) * 20,
            "shankInd": np.ones(nsite),
            "connected": np.ones(nsite),
        }
    ]
    units = [
        {
            "idx": i,
            "probeId": 1,
            "shank": 1,
            "shankPos": p,
            "spikeTimes": np.sort(t0 + np.arange(50) * 1.7 + i),
            "wv": w.astype(np.float32),
            "rm": m.astype(np.float32),
            "rm_valid": np.ones(m.shape, bool),
        }
        for i, m, w, p in zip(idx, maps, waves, positions)
    ]
    return {"t0": t0, "t1": t0 + 100.0, "nt": 5000, "gv_pos": np.arange(96.0), "pcm": pcm, "units": units}


def test_link_vollan_rate_map_and_waveform_rule():
    rng = np.random.default_rng(9)
    maps = [rng.random((96, 96)) for _ in range(4)]
    waves = [rng.normal(0, 1, (61, 12)) for _ in range(4)]
    a = _vollan_record(0.0, maps, waves, [1, 2, 3, 4], [100, 110, 120, 130])
    order = [2, 0, 3, 1]  # the same four units in another order, slightly changed
    b = _vollan_record(
        5000.0,
        [maps[k] + rng.normal(0, 0.05, (96, 96)) for k in order],
        [waves[k] + rng.normal(0, 0.05, (61, 12)) for k in order],
        [7, 8, 9, 10],
        [101, 112, 118, 133],
    )
    links, det = sc.link_vollan({"fa": a, "fb": b}, [["fa", "fb"]])
    assert len(det["null"]) == 6 and len(det["candidates"]) == 16
    assert set(links.rule) == {"waveform_and_map"}
    assert sorted(zip(links.idx_a, links.idx_b)) == [(1, 8), (2, 10), (3, 7), (4, 9)]
    # exports of one recording: coincident spikes
    c = _vollan_record(0.0, maps, waves, [1, 2, 3, 4], [100, 110, 120, 130])
    links2, _ = sc.link_vollan({"fa": a, "fc": c}, [["fa", "fc"]])
    assert set(links2.rule) == {"same_recording"} and list(zip(links2.idx_a, links2.idx_b)) == [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
    ]
