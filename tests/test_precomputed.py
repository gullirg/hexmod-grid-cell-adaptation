"""Shipped simulation inputs and precomputed simulation results are complete and consistent.

Checks the large-module pool against its checksums, the npz round trip of ``hexmod.sim.io_pool``,
and recomputes the printed simulation statistics (Fig 6D, Fig 7) from the shipped summaries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from hexmod import io as hio
from hexmod.sim import io_pool

REPO = Path(__file__).resolve().parents[1]
SIM_IN = REPO / "inputs" / "simulation"
PRE = REPO / "results_precomputed" / "sim"
DIRS = [30 * j for j in range(12)]
FOLDS = list(range(1, 8))


@pytest.fixture(scope="module")
def large_pool():
    return io_pool.load_pool(SIM_IN / "large_module_pool_70cm.npz")


def test_large_pool_matches_checksums(large_pool):
    sums = hio.read_json(SIM_IN / "pool_checksums.json")["large_module_pool_70cm"]
    assert hio.sha256_file(SIM_IN / "large_module_pool_70cm.npz") == sums["file_sha256"]
    assert large_pool.shape == (100000, 12)
    assert list(large_pool.columns) == sums["column_order"]
    got = io_pool.column_checksums(large_pool, sums["significant_digits"])
    assert {c: sums["columns"][c]["sha256"] for c in got} == got


def test_pool_npz_round_trip(tmp_path, large_pool):
    sub = large_pool.loc[np.random.default_rng(0).choice(len(large_pool), 300, replace=False)]
    io_pool.save_pool(sub, tmp_path / "sub.npz")
    back = io_pool.load_pool(tmp_path / "sub.npz")
    assert io_pool.pool_differences(sub, back) == []
    with np.load(tmp_path / "sub.npz", allow_pickle=False) as z:
        assert all(z[k].dtype != object for k in z.files)


def test_fig6d_statistics():
    expected = {"small": (175.8, 1.76), "large": (91.7, 0.92)}
    for module, (t_printed, d_printed) in expected.items():
        table = hio.read_csv(PRE / "fig6" / f"fig6d_{module}.csv.gz")
        assert len(table) == 10000
        assert table["cell_index"].tolist() == np.random.default_rng(0).choice(100000, 10000, replace=False).tolist()
        t, _ = stats.ttest_rel(table["aligned"], table["misaligned"])
        diff = table["aligned"] - table["misaligned"]
        assert round(t, 1) == t_printed
        assert round(diff.mean() / diff.std(ddof=1), 2) == d_printed
    large = hio.read_csv(PRE / "fig6" / "fig6d_large.csv.gz")
    assert (round(large["aligned"].mean(), 4), round(large["misaligned"].mean(), 4)) == (0.0853, 0.0684)


def _participants(module):
    table = hio.read_csv(PRE / "fig7" / f"participants_{module}.csv.gz")
    assert len(table) == 3000
    return table


def test_fig7_statistics_from_participant_summaries():
    printed_t = {("simple", "small"): -23.6257, ("simple", "large"): -7.0936,
                 ("location-controlled", "small"): -43.3605, ("location-controlled", "large"): -11.4143}
    significant = {("simple", "small"): [6], ("simple", "large"): [2, 3],
                   ("location-controlled", "small"): [6], ("location-controlled", "large"): [6]}
    values = hio.read_json(PRE / "fig7" / "fig7_values.json")
    for module in ("small", "large"):
        table = _participants(module)
        for method in ("simple", "location-controlled"):
            first = table[(table["method"] == method) & (table["participant"] < 30)]
            # C order, as the per-participant profiles were stacked: the order of a reduction can
            # change the last bit of a mean
            rates = np.ascontiguousarray(first[[f"rate_dir_{d}_hz" for d in DIRS]].to_numpy())
            t = stats.ttest_rel(rates[:, 0::2].mean(axis=1), rates[:, 1::2].mean(axis=1)).statistic
            assert round(t, 4) == printed_t[(method, module)]
            panel = "fig7a" if method == "simple" else "fig7d"
            # recomputed values: the last bits can differ between CPU architectures and numerical libraries
            assert values[f"{panel}.{module}.t"] == pytest.approx(t, rel=1e-9, abs=0)
            pct = rates / rates.mean(axis=1, keepdims=True) * 100 - 100
            # the same for the percent changes, which can be zero
            assert np.allclose(pct, first[[f"dir_{d}" for d in DIRS]].to_numpy(), rtol=1e-9, atol=1e-12)
            beta = first[[f"fold_{k}" for k in FOLDS]].to_numpy()
            p = stats.ttest_1samp(beta, 0, axis=0).pvalue
            assert [k for k in FOLDS if p[k - 1] < 0.05] == significant[(method, module)]


def test_fig7f_detection_counts():
    table = _participants("large")
    beta = table[table["method"] == "simple"][[f"fold_{k}" for k in FOLDS]].to_numpy()
    counts = {}
    for label, rows in (("49", beta[30:]), ("50", beta)):
        p = np.array([stats.ttest_1samp(rows[e * 30:(e + 1) * 30], 0, axis=0, alternative="greater").pvalue
                      for e in range(len(rows) // 30)])
        counts[label] = (p < 0.05).sum(axis=0).tolist()
    assert counts["49"] == [19, 41, 17, 12, 6, 39, 1]
    assert (counts["50"][1], counts["50"][5]) == (42, 39)


def test_maps_and_sweep_layout():
    maps = hio.load_npz(PRE / "fig6" / "maps.npz")
    for module in ("small", "large"):
        assert maps[f"{module}__example_maps"].shape == (6, 34, 34)
        assert maps[f"supp3e__{module}__maps"].shape == (6, 34, 34)
        assert np.array_equal(maps[f"supp3e__{module}__maps"].reshape(6, -1).max(axis=1),
                              maps[f"supp3e__{module}__peak_rate_hz"])
    assert maps["supp3e__large__cells"].tolist() == [95331, 83646, 88, 72374, 36659, 2061]
    trajectory = hio.read_csv(PRE / "fig6" / "trajectory_seed0.csv")
    assert len(trajectory) == 23990
    sweep = hio.read_csv(PRE / "fig8" / "fig8d_sweep.csv")
    assert len(sweep) == 111 and sorted(sweep["scale"].unique().tolist()) == [30, 50, 70]
