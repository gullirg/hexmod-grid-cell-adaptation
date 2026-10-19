"""Recovery of a known adaptation coefficient from a simulated grid cell (no data needed).

A grid cell runs a smooth random walk in a 100 cm box. Its rate is the place rate (constant within
the 5 cm location bins of the model) times ``1 + alpha_true * a``, where ``a`` is the z-scored
adaptation regressor of the paper (proximity ``exp(-t_prev / 2 s)`` times the location input of the
previous field visit), computed from the same fields the fit uses. The parametric fit must return a
negative alpha close to alpha_true, and a cell without adaptation an alpha close to zero.

The regressor is strongly right-skewed (long visits through field centres give large inputs), so the
bounds that keep ``1 + alpha a`` positive limit alpha to about -1 / max(a); alpha_true = -0.05 lies
well inside them, and is typical of the recorded cells.
"""

import numpy as np
import pytest

from hexmod.rat.adaptation import adaptation_nonparametric, adaptation_parametric
from hexmod.rat.fields import extract_field_entry, field_visit_input, position_binning
from hexmod.rat.gridcell import GridCellAnalysis

SCALE, ORIENTATION, BOX = 50.0, 10.0, 100.0


def random_walk(duration_s, seed, fs=50.0):
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    dt = 1 / fs
    xy = np.empty((n, 2))
    heading = np.empty(n)
    speed = np.empty(n)
    p = np.array([BOX / 2, BOX / 2])
    h, v = 0.0, 20.0
    for k in range(n):
        h += rng.normal(0, 0.15)
        v = np.clip(v + rng.normal(0, 1.0), 8.0, 45.0)
        if k:
            p = p + v * dt * np.array([np.cos(h), np.sin(h)])
        for d in range(2):  # reflect at the walls
            if p[d] < 0 or p[d] > BOX:
                p[d] = np.clip(p[d], 0, BOX)
                h = np.pi - h if d == 0 else -h
        xy[k], heading[k], speed[k] = p, h % (2 * np.pi), v
    return xy, np.arange(n) * dt, speed, heading


def lattice(origin=(20.0, 25.0)):
    th = np.deg2rad(ORIENTATION)
    a1 = SCALE * np.array([np.cos(th), np.sin(th)])
    a2 = SCALE * np.array([np.cos(th + np.pi / 3), np.sin(th + np.pi / 3)])
    pts = np.array([np.asarray(origin) + i * a1 + j * a2 for i in range(-4, 5) for j in range(-4, 5)])
    return pts[(pts[:, 0] > -SCALE) & (pts[:, 0] < BOX + SCALE) & (pts[:, 1] > -SCALE) & (pts[:, 1] < BOX + SCALE)]


def simulate(alpha_true, seed, duration_s=1500.0):
    """Simulated session; returns the inputs of the adaptation fits."""
    xy, t, speed, heading = random_walk(duration_s, seed)
    centres = lattice()
    rng = np.random.default_rng(seed + 100)
    peaks = rng.uniform(8.0, 20.0, len(centres))
    sigma = 0.2 * SCALE
    # place rate constant within the 5 cm location bins of the model (evaluated at the bin centres)
    loc_idx, nbx, nby, ix, iy = position_binning(xy[:, 0], xy[:, 1], 5)
    bx = xy[:, 0].min() + 5 * (ix + 0.5)
    by = xy[:, 1].min() + 5 * (iy + 0.5)
    d2 = (bx[:, None] - centres[:, 0]) ** 2 + (by[:, None] - centres[:, 1]) ** 2
    place_rate = 0.3 + (peaks * np.exp(-d2 / (2 * sigma ** 2))).max(axis=1)

    inside = (centres[:, 0] > 0) & (centres[:, 0] < BOX) & (centres[:, 1] > 0) & (centres[:, 1] < BOX)
    fieldLoc = np.c_[centres[inside], np.full(inside.sum(), 1.2 * sigma)]
    peakFR = peaks[inside]

    position = {"xy": xy, "xy_smoothed": xy, "time": t, "sample_rate": 50.0, "speed": speed, "move_dir": heading}
    labels = {"unique_cell_id": "sim", "animal_id": "sim", "session_id": "sim_1", "cell_id": 0}
    # the regressor depends only on the trajectory and the fields
    g = GridCellAnalysis({"position": position, "spike_times": np.array([]), **labels})
    assert len(g.time) == len(t)  # every sample passes the speed filter
    ev = extract_field_entry(g.x, g.y, g.time, fieldLoc[:, 0], fieldLoc[:, 1], fieldLoc[:, 2])
    w = field_visit_input(g.x, g.y, g.time, ev["ev_df"], peakFR, g.speed)["adaptWeight_loc"]
    a = np.exp(-ev["t_prev_vec"] / 2.0) * w / np.max(w)
    a = (a - a.mean()) / a.std()

    rate = np.clip(place_rate * (1 + alpha_true * a), 0, None)
    counts = np.random.default_rng(seed + 200).poisson(rate * 0.02)
    cell_data = {"position": position, "spike_times": np.repeat(t, counts), **labels}
    return {"cell_data": cell_data, "fieldLoc": fieldLoc, "peakFR": peakFR, "gridOri": ORIENTATION,
            "tprev_bins": np.array([0, 1, 2, 3, np.inf]), "tau_adapt": 2.0, "spatialResolution": 5}


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_negative_alpha_is_recovered(seed):
    alpha_true = -0.05
    inputs = simulate(alpha_true, seed)
    out = adaptation_parametric({**inputs, "adapt_type": ["tprevXloc"], "mask_type": ["all"]})
    alpha = out["alpha_tprevXloc_all"]
    assert out["converge_tprevXloc_all"]
    assert alpha < 0
    assert abs(alpha - alpha_true) < 0.025
    assert abs(out["speed_tprevXloc_all"]) < 0.04  # no speed modulation was simulated


@pytest.mark.parametrize("seed", [1, 5])
def test_no_adaptation_gives_alpha_near_zero(seed):
    out = adaptation_parametric({**simulate(0.0, seed), "adapt_type": ["tprevXloc"], "mask_type": ["all"]})
    assert abs(out["alpha_tprevXloc_all"]) < 0.025


def test_nonparametric_rate_rises_with_time_since_last_visit():
    inputs = simulate(-0.05, 1)
    out = adaptation_nonparametric({**inputs, "mask_type": ["all"]})
    rate = out["tprev_nonparam_all"]
    t = inputs["cell_data"]["position"]["time"]
    assert rate.shape == (4,) and out["tprev_bincounts_all"].sum() == (t > 0).sum()  # mask "all" is t > 0
    assert rate[0] < rate[-1]  # lower rate soon after a field visit
