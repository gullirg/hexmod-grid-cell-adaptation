"""Grid measures on synthetic grids, and direction tuning with empty direction bins (no data needed)."""

import numpy as np
import pytest

from hexmod.rat import maps
from hexmod.rat.characterise import characterise_gridcell


def lattice_centres(scale, orientation_deg, extent, origin=(13.0, 7.0)):
    """Points of a triangular lattice covering [-extent, 2 extent] in x and y."""
    th = np.deg2rad(orientation_deg)
    a1 = scale * np.array([np.cos(th), np.sin(th)])
    a2 = scale * np.array([np.cos(th + np.pi / 3), np.sin(th + np.pi / 3)])
    n = int(np.ceil(3 * extent / scale)) + 2
    i, j = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), indexing="ij")
    pts = np.asarray(origin) + i.reshape(-1, 1) * a1 + j.reshape(-1, 1) * a2
    keep = (pts[:, 0] > -extent) & (pts[:, 0] < 2 * extent) & (pts[:, 1] > -extent) & (pts[:, 1] < 2 * extent)
    return pts[keep]


def grid_rate(x, y, scale, orientation_deg, extent, peak=15.0, base=0.2):
    """Rate (Hz) of a perfect grid: Gaussian fields (SD 0.15 x scale) at the lattice points."""
    centres = lattice_centres(scale, orientation_deg, extent)
    sigma = 0.15 * scale
    d2 = (x[..., None] - centres[:, 0]) ** 2 + (y[..., None] - centres[:, 1]) ** 2
    return base + peak * np.exp(-d2 / (2 * sigma ** 2)).max(axis=-1)


def angle_diff60(a, b):
    """Absolute difference of two orientations modulo 60 degrees."""
    d = (a - b) % 60
    return min(d, 60 - d)


@pytest.mark.parametrize("scale, orientation", [(45.0, 20.0), (36.0, 7.0), (60.0, 41.0)])
def test_perfect_grid_rate_map(scale, orientation):
    """Autocorrelogram of a perfect grid: gridness above 1, scale and orientation recovered."""
    bin_cm, extent = 3.0, 150.0
    centres = np.arange(0, extent, bin_cm) + bin_cm / 2  # bin centres; first index along x
    X, Y = np.meshgrid(centres, centres, indexing="ij")
    rate_map = grid_rate(X, Y, scale, orientation, extent)
    acorr = maps.spatial_autocorrelogram_fft2d(rate_map, 1, 1)
    gridness, scale_est, ori_est = maps.compute_gridness_scale_ori(acorr, rate_map_bin_size=bin_cm)
    assert gridness > 1.0
    assert abs(scale_est - scale) < bin_cm
    assert angle_diff60(ori_est, orientation) < 3.0


def test_irregular_map_is_not_a_grid():
    """A smoothed random map does not pass the cohort threshold (gridness above 0.3)."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(5)
    rate_map = gaussian_filter(rng.gamma(2.0, 1.0, (40, 40)), 3)
    acorr = maps.spatial_autocorrelogram_fft2d(rate_map, 1, 1)
    gridness, scale_est, ori_est = maps.compute_gridness_scale_ori(acorr, rate_map_bin_size=3)
    assert not gridness > 0.3
    assert np.isnan(scale_est) and np.isnan(ori_est)


def random_walk(duration_s=1200.0, box=100.0, fs=50.0, seed=0):
    """Smooth random walk in a square box: positions (cm), time (s), speed (cm/s), direction (rad)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    dt = 1 / fs
    heading = np.empty(n)
    speed = np.empty(n)
    xy = np.empty((n, 2))
    xy[0] = (box / 2, box / 2)
    h, v = 0.0, 20.0
    for k in range(n):
        h = h + rng.normal(0, 0.15)
        v = np.clip(v + rng.normal(0, 1.0), 8.0, 45.0)
        step = v * dt * np.array([np.cos(h), np.sin(h)])
        p = xy[k - 1] + step if k else xy[0]
        for d in range(2):  # reflect at the walls
            if p[d] < 0 or p[d] > box:
                p[d] = np.clip(p[d], 0, box)
                h = np.pi - h if d == 0 else -h
        xy[k] = p
        heading[k] = h % (2 * np.pi)
        speed[k] = v
    t = np.arange(n) * dt
    return xy, t, speed, heading


def poisson_spike_times(rate_hz, t, seed):
    """Spike times (s) from per-sample Poisson counts; each spike is placed on its sample time."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(rate_hz * (t[1] - t[0]))
    return np.repeat(t, counts)


def test_characterise_simulated_grid_cell():
    """End to end: a simulated grid cell on a random walk is recognised with its scale and orientation."""
    scale, orientation = 42.0, 12.0
    xy, t, speed, heading = random_walk(seed=1)
    rate = grid_rate(xy[:, 0], xy[:, 1], scale, orientation, 100.0)
    spikes = poisson_spike_times(rate, t, seed=2)
    position = {"xy": xy, "xy_smoothed": xy, "time": t, "sample_rate": 50.0, "speed": speed, "move_dir": heading}
    cell_data = {"position": position, "spike_times": spikes, "unique_cell_id": "sim", "animal_id": "sim",
                 "session_id": "sim_1", "cell_id": 0}
    out = characterise_gridcell({"cell_data": cell_data})
    assert out is not None
    assert out["gridness"] > 0.8
    assert abs(out["scale_cm"] - scale) < 4.0
    assert angle_diff60(out["orientation_deg"], orientation) < 5.0
    assert out["fieldLoc"].shape[1] == 3 and out["fieldLoc"].shape[0] >= 4
    assert out["dir_map"].shape == (72,)
    assert 0 <= out["Rayleigh_mag"] < 0.3  # no direction tuning was simulated


class _NumpyWithStaleDivide:
    """numpy, except that ``divide(..., where=...)`` without ``out`` leaves a large value in the
    elements it skips, as uninitialised memory may."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def divide(a, b, *args, **kwargs):
        if "where" in kwargs and "out" not in kwargs:
            kwargs["out"] = np.full(np.broadcast(a, b).shape, 1e300)
        return np.divide(a, b, *args, **kwargs)


def test_empty_direction_bins_are_zero(monkeypatch):
    """Direction bins without dwell (after smoothing) get a rate of exactly zero, never stale memory."""
    rng = np.random.default_rng(3)
    direction = rng.uniform(0, 170, 20000)  # no samples between 170 and 360 degrees
    spikes = rng.poisson(0.1, direction.size)
    monkeypatch.setattr(maps, "np", _NumpyWithStaleDivide())
    mag, ang, rate, dwell = maps.compute_directional_rate_map_and_rayleigh_vector(direction, spikes, bin_size_deg=5,
                                                                                  smooth_nBins=3)
    monkeypatch.undo()
    empty = dwell == 0
    assert empty.sum() >= 30
    assert np.all(rate[empty] == 0.0)
    assert np.all(np.isfinite(rate)) and np.all(rate >= 0)
    assert 0 <= ang < 180 and 0 < mag <= 1


def test_direction_tuning_recovers_preferred_direction():
    """A von Mises tuned cell has its Rayleigh angle near the preferred direction."""
    rng = np.random.default_rng(4)
    direction = rng.uniform(0, 360, 200000)
    rate = 0.05 * np.exp(2.0 * np.cos(np.deg2rad(direction - 250.0)))
    spikes = rng.poisson(rate)
    mag, ang, curve, dwell = maps.compute_directional_rate_map_and_rayleigh_vector(direction, spikes)
    assert abs(((ang - 250.0) + 180) % 360 - 180) < 3.0
    assert mag > 0.5
    assert curve.shape == (72,)
