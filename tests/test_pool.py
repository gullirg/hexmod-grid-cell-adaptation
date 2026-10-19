"""Tests of the simulated grid cell pools (hexmod.sim.pool, hexmod.sim.pool_large) and of a
small simulated participant (hexmod.sim.simulate).

The reference values below come from 500-cell pools with base seed 0. Checksums are taken over
values rounded to 6 decimals, so that last-digit differences between numerical library builds do
not matter, and leave out ``gridori_est``, whose ring-peak estimate can move by a few degrees
with such differences (it is not used by any analysis); the sums are compared to 1e-12 relative.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from hexmod.sim import pool as simpool
from hexmod.sim import pool_large as builder
from hexmod.sim import simulate, trajectory_v1

SMALL_CHECKSUM6 = "54cddca554f67a93f7cee8ada3f59f5a5a3c772221a9a1ff579a45a2cf3301d4"
LARGE_CHECKSUM6 = "7046b179cf3e0838fa1cecea5b68d5907823d6dfc34184a002a6ca504be925d2"
SMALL_CHECKED = [c for c in simpool.POOL_COLUMNS if c != "gridori_est"]
LARGE_CHECKED = [c for c in builder.LARGE_POOL_COLUMNS if c != "gridori_est"]
SMALL_SUMS = {
    "peaks_x": (5272, 270074.7400000016),
    "peaks_y": (5272, 269783.6350000013),
    "radius": (5272, 50404.599999999984),
    "gridness": (500, 574.199809137285),
    "gridscale_est": (500, 20395.39300408183),
}
LARGE_SUMS = {
    "radius": (2138, 35538.79999999998),
    "gridness": (500, 542.132578057847),
    "gridscale_est": (500, 35430.5134274837),
    "peakFR_jitter": (3431, 8840.407205722435),
}


def rounded_checksum(table, columns, decimals=6):
    """sha256 over every value rounded to ``decimals`` decimals (with the shape of each array)."""
    h = hashlib.sha256()
    for c in columns:
        for v in table[c].to_numpy():
            a = np.round(np.asarray(v, dtype=float), decimals) + 0.0
            h.update(np.ascontiguousarray(a).tobytes())
            h.update(str(a.shape).encode())
    return h.hexdigest()


def column_sum(table, col):
    vals = table[col].to_numpy()
    if vals.dtype == object:
        return int(sum(np.asarray(v).size for v in vals)), float(sum(np.asarray(v, float).sum() for v in vals))
    return len(vals), float(np.sum(vals.astype(float)))


@pytest.fixture(scope="module")
def small_pool():
    return simpool.set_gridcells_pool(simpool.module_param("small"), 500, 0)


@pytest.fixture(scope="module")
def large_pool():
    return builder.set_gridcells_pool_large(builder.large_pool_param(), 500, 0)


def test_module_parameters():
    """Field jitter and rate parameters of the two modules (gamma shape and scale, Supp Fig 3)."""
    small = simpool.module_param("small")
    large = simpool.module_param("large")
    assert (small["gridScale"], large["gridScale"]) == (40, 70)
    assert (small["peakCoords_shape"], small["peakCoords_scale"]) == (2.0, 0.07)
    assert (small["FR_acrossCell_shape"], small["FR_acrossCell_scale"]) == (3.2, 1.4)
    assert (small["FR_withinCell_shape"], small["FR_withinCell_scale"]) == (10.3, 0.10)
    assert (large["peakCoords_shape"], large["peakCoords_scale"]) == (2.1, 0.04)
    assert (large["FR_acrossCell_shape"], large["FR_acrossCell_scale"]) == (3.4, 1.1)
    assert (large["FR_withinCell_shape"], large["FR_withinCell_scale"]) == (7.2, 0.10)
    assert small["minGridness"] == 0.3 and small["fieldSigma"] == pytest.approx(1 / 6)
    assert len(small["xBins_center"]) == 34 and small["xBins_center"][0] == 1.5
    with pytest.raises(ValueError):
        simpool.set_param({"gridscale": 40})
    lp = builder.large_pool_param()
    for k, v in simpool.MODULE_PARAMS["large"].items():
        assert lp[k] == v


def test_small_pool_reference(small_pool):
    assert list(small_pool.columns) == list(simpool.POOL_COLUMNS)
    assert len(small_pool) == 500
    assert rounded_checksum(small_pool, SMALL_CHECKED) == SMALL_CHECKSUM6
    for col, (n, total) in SMALL_SUMS.items():
        n_got, total_got = column_sum(small_pool, col)
        assert n_got == n
        assert total_got == pytest.approx(total, rel=1e-12)
    assert small_pool["centers"][0][0] == pytest.approx([0.4662492105059246, 11.70076774233753], rel=1e-12)
    assert small_pool["peakFR_jitter"][0][0] == pytest.approx(2.2517391420793946, rel=1e-12)


def test_small_pool_properties(small_pool):
    g = small_pool["gridness"].to_numpy()
    assert (g >= 0.3).all() and (g < 2).all()
    for c, r, px, py in zip(small_pool["centers"], small_pool["peakFR_jitter"], small_pool["peaks_x"], small_pool["peaks_y"]):
        assert c.ndim == 2 and c.shape[1] == 2 and len(r) == len(c)
        assert (r > 0).all()
        assert len(px) == len(py) > 0
        # delineated peaks lie on rate-map bin centres (1.5 to 100.505 vm)
        assert (px >= 1.5).all() and (px <= 100.505).all() and (py >= 1.5).all() and (py <= 100.505).all()
    # estimated scale close to the lattice spacing of 40 vm
    assert np.median(small_pool["gridscale_est"]) == pytest.approx(40, abs=4)


def test_pool_reproducible():
    par = simpool.module_param("small")
    a = simpool.set_gridcells_pool(par, 60, 0)
    b = simpool.set_gridcells_pool(par, 60, 0)
    c = simpool.set_gridcells_pool(par, 60, 1)
    assert rounded_checksum(a, simpool.POOL_COLUMNS, 15) == rounded_checksum(b, simpool.POOL_COLUMNS, 15)
    assert rounded_checksum(a, simpool.POOL_COLUMNS) != rounded_checksum(c, simpool.POOL_COLUMNS)


def test_perfect_grid():
    par = simpool.set_param(dict(gridScale=30, gridtype="perfectGrid"))
    p = simpool.set_gridcells_pool(par, 4, 0)
    c = p["centers"][0]
    assert p["peakFR_jitter"][0] is None and (p["gridness"] == 10).all()
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    d[d == 0] = np.inf
    assert d.min(axis=1) == pytest.approx(np.full(len(c), 30.0))
    rm = simpool.ratemap_from_centers(c, None, par)
    assert rm.shape == (34, 34) and rm.max() <= 1.0


def test_large_pool_builder(large_pool):
    assert list(large_pool.columns) == list(builder.LARGE_POOL_COLUMNS)
    assert rounded_checksum(large_pool, LARGE_CHECKED) == LARGE_CHECKSUM6
    for col, (n, total) in LARGE_SUMS.items():
        n_got, total_got = column_sum(large_pool, col)
        assert n_got == n
        assert total_got == pytest.approx(total, rel=1e-12)
    assert (large_pool["adaptEff"] == 0.04).all() and (large_pool["speedGamma"] == 0.5).all()
    assert np.bincount(large_pool["cellcount"].to_numpy()).tolist() == [0, 463, 32, 4, 1]
    assert (large_pool["gridness"] >= 0.3).all()
    # the two builders share cell 0 but then follow different random streams
    other = simpool.set_gridcells_pool(simpool.module_param("large"), 500, 0)
    assert np.array_equal(other["centers"][0], large_pool["centers"][0])
    assert not any(np.array_equal(a, b) for a, b in zip(other["centers"][1:], large_pool["centers"][1:]))


def test_pool_npz_round_trip(small_pool, tmp_path):
    io_pool = pytest.importorskip("hexmod.sim.io_pool")
    path = io_pool.save_pool(small_pool, tmp_path / "pool.npz")
    back = io_pool.load_pool(path)
    assert io_pool.pool_differences(small_pool, back) == []
    assert io_pool.column_checksums(back) == io_pool.column_checksums(small_pool)


def test_small_participant(small_pool):
    """A 20-cell participant: adaptation lowers the rate; chunked and single runs agree."""
    par = simpool.module_param("small", nGridCells=20)
    traj = trajectory_v1.make_trajectory(par, base_seed=0)
    cells = simulate.participant_cells(small_pool, par, 0)
    assert list(cells.index) == list(np.random.default_rng(0).choice(500, 20, replace=False))
    rec = simulate.simulate_and_measure(par, base_seed=0, traj=traj, gc_table=cells)
    assert rec["n_samples"] == 23966 and rec["n_moving"] == 9379 and len(rec["FR_adapt"]) == 9379
    assert (rec["FR_adapt"] <= rec["FR_noadapt"]).all()
    assert rec["FR_adapt_mean_move"] == pytest.approx(0.8875267303911653, rel=1e-10)
    assert rec["FR_noadapt_mean_move"] == pytest.approx(0.9093399690421894, rel=1e-10)
    assert rec["tprev_HMness"] == pytest.approx(0.0392006306330621, rel=1e-9)
    cr = rec["crossvalResult"]
    assert len(cr) == 2 * 2 * 4 * 7 and sorted(cr["fold"].unique()) == list(range(1, 8))
    b6 = cr[(cr.fold == 6) & (cr.model == "adapt") & (cr.estType == "simple")]["beta"].to_numpy()
    assert b6 == pytest.approx([0.0037739257671214436, 0.02919298540740774, -0.0044403550032883305,
                                0.023513758079796512], rel=1e-8, abs=1e-12)
    chunked = simulate.simulate_and_measure(par, base_seed=0, traj=traj, gc_table=cells, chunk=7)
    assert np.max(np.abs(chunked["FR_adapt"] - rec["FR_adapt"])) < 1e-12
    pd.testing.assert_frame_equal(chunked["crossvalResult"].drop(columns="FRby12dir_test"),
                                  rec["crossvalResult"].drop(columns="FRby12dir_test"), check_exact=False, rtol=1e-9)
    # without adaptation the adapted rate equals zero and no field visits are needed
    out = simulate.simulate_grid_cells({"gcTable": cells.iloc[:3], "traj": traj, "par": par, "isAdapt": False})
    assert not out["FR_adapt"].any() and out["FR_noadapt"].any()


def test_simulate_many_order(small_pool):
    """simulate_many returns one result per seed, in the order given, equal to single runs."""
    par = simpool.module_param("small", nGridCells=12)
    res = simulate.simulate_many(par, [2, 1], small_pool, workers=1, chunk=None)
    assert [r["seed"] for r in res] == [2, 1]
    one = simulate.simulate_participant((par, simulate.participant_cells(small_pool, par, 1), 1, None,
                                         simulate.FOLD_SYM_LIST))
    assert np.array_equal(res[1]["FR_adapt"], one["FR_adapt"])
