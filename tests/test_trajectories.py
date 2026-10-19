"""Tests of the two trajectory generators (hexmod.sim.trajectory_v1 and trajectory_v2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hexmod.sim import trajectory_v1, trajectory_v2
from hexmod.sim.pool import module_param, set_param

COLUMNS = ["time", "x_unsmooth", "y_unsmooth", "x", "y", "trialId", "expPhase", "moveSpeed", "moveDir", "blockIdx"]


@pytest.fixture(scope="module")
def traj_v1():
    return trajectory_v1.make_trajectory(set_param(), base_seed=0)


@pytest.fixture(scope="module")
def traj_v2():
    return trajectory_v2.make_trajectory(set_param(), base_seed=0)


def test_sample_counts_seed0(traj_v1, traj_v2):
    """Seed 0 gives 23,966 samples with version 1 (Fig 7, participant 0) and 23,990 with version 2 (Fig 6A)."""
    assert len(traj_v1) == 23966
    assert len(traj_v2) == 23990
    # samples analysed (faster than 5 vm/s) for participant 0 of Fig 7
    with np.errstate(invalid="ignore"):
        assert int((traj_v1["moveSpeed"] > 5).sum()) == 9379


def test_layout(traj_v1, traj_v2):
    for t in (traj_v1, traj_v2):
        assert list(t.columns) == COLUMNS
        assert sorted(t["blockIdx"].unique()) == [1, 2, 3, 4]
        for b in range(1, 5):
            tb = t["time"].to_numpy()[t["blockIdx"].to_numpy() == b]
            assert tb[0] == pytest.approx((b - 1) * 1600.0)  # 600 s blocks, 1000 s apart
            assert np.allclose(np.diff(tb), 0.1)
            assert tb[-1] - tb[0] <= 600.0 + 1e-9
        hidden = t["expPhase"].to_numpy() == -1
        assert np.isnan(t["x"].to_numpy()[hidden]).all() and np.isnan(t["moveSpeed"].to_numpy()[hidden]).all()
        x, y = t["x"].to_numpy(), t["y"].to_numpy()
        ok = ~np.isnan(x)
        assert (x[ok] >= 0).all() and (x[ok] <= 100).all() and (y[ok] >= 0).all() and (y[ok] <= 100).all()
        run = t["expPhase"].to_numpy() == 1
        # during runs the speed is the set speed (100/6 vm/s) away from the smoothed run ends
        speed = t["moveSpeed"].to_numpy()[run]
        assert np.nanmedian(speed) == pytest.approx(100.0 / 6.0, rel=1e-9)


def test_directions_balanced(traj_v1):
    """Running directions cover the 12 bins of 30 deg about evenly."""
    t = traj_v1
    with np.errstate(invalid="ignore"):
        moving = (t["moveSpeed"] > 5).to_numpy()
    d = t["moveDir"].to_numpy()[moving]
    counts = np.bincount(((d + 15) % 360 // 30).astype(int), minlength=12)
    assert counts.min() > 0.6 * counts.mean()


def test_deterministic_and_seed_dependent():
    par = module_param("small")
    for gen in (trajectory_v1, trajectory_v2):
        a = gen.make_trajectory(par, base_seed=3)
        b = gen.make_trajectory(par, base_seed=3)
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        c = gen.make_trajectory(par, base_seed=4)
        assert len(c) != len(a) or not np.array_equal(c["x"].to_numpy(), a["x"].to_numpy(), equal_nan=True)


def test_trajectory_does_not_depend_on_module():
    """Grid parameters do not enter the trajectory (the small and large modules share it)."""
    a = trajectory_v1.make_trajectory(module_param("small"), base_seed=7)
    b = trajectory_v1.make_trajectory(module_param("large"), base_seed=7)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_wall2wall():
    """Fig 8D runs: straight runs from wall to wall, no pauses, until the block is full."""
    par = set_param(dict(minLen=70, maxLen=70, moveSpeed=45.0, dur_ITI=0, dur_cue=0, moveConstraint="wall2wall"))
    t = trajectory_v2.make_trajectory(par, base_seed=10)
    for b in range(1, 5):
        tb = t[t["blockIdx"] == b]
        assert tb["time"].iloc[-1] - tb["time"].iloc[0] >= 600.0 - 0.1
        assert tb["trialId"].max() > 10
    run = t["expPhase"].to_numpy() == 1
    assert run.mean() > 0.95
    assert np.nanmedian(t["moveSpeed"].to_numpy()[run]) == pytest.approx(45.0, rel=1e-9)
    with pytest.raises(ValueError):
        trajectory_v1.make_trajectory(par, base_seed=0)


def test_subsample_rows_balanced():
    df = pd.DataFrame({"g": ["a"] * 1 + ["b"] * 5 + ["c"] * 3, "v": np.arange(9)})
    # version 2 balances only across groups that still have rows: draws every row
    out = trajectory_v2.subsample_rows_balanced(df, "g", 9, random_state=0)
    assert sorted(out["v"]) == list(range(9))
    out = trajectory_v2.subsample_rows_balanced(df, "g", 3, random_state=0)
    assert sorted(out["g"]) == ["a", "b", "c"]
    # version 1 stops once every least-used group is exhausted ("a" after one row each)
    out = trajectory_v1.subsample_rows_balanced(df, "g", 9, random_state=0)
    assert out["g"].value_counts().to_dict() == {"b": 2, "c": 2, "a": 1}
    for gen in (trajectory_v1, trajectory_v2):
        a = gen.subsample_rows_balanced(df, "g", 5, random_state=1)
        b = gen.subsample_rows_balanced(df, "g", 5, random_state=1)
        pd.testing.assert_frame_equal(a, b)
