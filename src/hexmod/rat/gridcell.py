"""Per-cell trajectory and spike samples, and the firing-rate map of a recorded cell.

``GridCellAnalysis`` turns the preprocessed data of one cell (positions sampled at 50 Hz, spike
times) into the samples used by every rat analysis:

- positions are the smoothed positions (``xy_smoothed``, falling back to ``xy``);
- each spike is assigned to the position sample at or after its time; spikes that fall inside a
  gap in the position record longer than 1.5 sample intervals are dropped;
- only samples with running speed between 5 and 150 cm/s are kept (all arrays are filtered
  with the same mask, so the time axis has gaps where the animal was slow).

The rate map uses 3 cm bins and a Gaussian filter with a standard deviation of one bin. By
default a position sample that holds several spikes counts once (``rate_map_spike_mode=
"binary"``), which is the map used for all published analyses except the field-visit sequences
of Fig 3F, which use spike counts (``"counts"``).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

__all__ = ["GridCellAnalysis", "make_cell_data", "SPIKE_MODES"]

SPIKE_MODES = ("binary", "counts")


def make_cell_data(session_data, cell_id, animal_id, session_id, spike_shift_s=None):
    """Build the per-cell input dictionary used by the rat analyses.

    Parameters
    ----------
    session_data : dict
        One session of the preprocessed trajectories: ``position`` (dict with ``xy``,
        ``xy_smoothed``, ``time``, ``sample_rate``, ``speed``, ``move_dir``), ``spike_times``
        (dict unit key to spike times in s) and ``unique_cell_ids`` (dict unit key to label).
    cell_id : int
        Unit key within the session.
    animal_id, session_id : str
        Labels of the animal and the session.
    spike_shift_s : float, optional
        Time added to every spike time (prospective coding control). ``None`` leaves the spike
        times untouched.

    Returns
    -------
    dict
        Keys ``position``, ``spike_times``, ``unique_cell_id``, ``animal_id``, ``session_id``,
        ``cell_id``.
    """
    spikes = session_data["spike_times"][cell_id]
    if spike_shift_s is not None:
        spikes = spikes + spike_shift_s
    return {"position": session_data["position"], "spike_times": spikes,
            "unique_cell_id": session_data["unique_cell_ids"][cell_id],
            "animal_id": animal_id, "session_id": session_id, "cell_id": cell_id}


class GridCellAnalysis:
    """Speed-filtered samples of one cell and its firing-rate map.

    Parameters
    ----------
    cell_data : dict
        As returned by :func:`make_cell_data`. An optional key ``rate_map_spike_mode``
        (``"binary"`` or ``"counts"``) sets the default spike mode of the rate map.
    rate_map_spike_mode : str, optional
        Overrides ``cell_data["rate_map_spike_mode"]``; the default is ``"binary"``.

    Attributes
    ----------
    x, y, time, speed, move_dir : ndarray
        Position (cm), time (s), running speed (cm/s) and movement direction (rad) of the samples
        with speed in [5, 150] cm/s.
    spike_array : ndarray of int
        Number of spikes assigned to each kept sample.
    dt : float
        Sample interval (s), ``1 / sample_rate``.
    rate_map_bin_size : float
        Bin size of the rate map (3 cm).
    """

    def __init__(self, cell_data, rate_map_spike_mode=None):
        self.animal_id = cell_data.get("animal_id", None)
        self.session_id = cell_data.get("session_id", None)
        self.cell_id = cell_data.get("cell_id", None)
        if rate_map_spike_mode is None:
            rate_map_spike_mode = cell_data.get("rate_map_spike_mode", "binary")
        if rate_map_spike_mode not in SPIKE_MODES:
            raise ValueError("rate_map_spike_mode must be 'binary' or 'counts'")
        self.rate_map_spike_mode = rate_map_spike_mode

        position = cell_data["position"]
        self.xy = position.get("xy_smoothed", position["xy"])
        self.x = self.xy[:, 0]
        self.y = self.xy[:, 1]

        self.time = position["time"]
        if not np.all(np.diff(self.time) > 0):
            raise ValueError("Time values must be strictly increasing.")

        self.speed = position.get("speed", np.full_like(self.time, np.nan))
        self.move_dir = position.get("move_dir", np.full_like(self.time, np.nan))
        self.sample_rate = position["sample_rate"]
        self.dt = 1 / self.sample_rate

        self.spike_times = cell_data["spike_times"]
        self.unique_cell_id = cell_data["unique_cell_id"]
        self.rate_map_bin_size = 3  # cm

        self.spike_array = self.compute_spike_array()

        # keep samples with speed in [min_speed, max_speed]; speed itself is filtered last
        min_speed = 5  # cm/s
        max_speed = 150  # cm/s
        self.x = GridCellAnalysis.apply_speed_thresholding(self.x, self.speed, min_speed, max_speed)
        self.y = GridCellAnalysis.apply_speed_thresholding(self.y, self.speed, min_speed, max_speed)
        self.time = GridCellAnalysis.apply_speed_thresholding(self.time, self.speed, min_speed, max_speed)
        self.spike_array = GridCellAnalysis.apply_speed_thresholding(self.spike_array, self.speed, min_speed, max_speed)
        self.move_dir = GridCellAnalysis.apply_speed_thresholding(self.move_dir, self.speed, min_speed, max_speed)
        self.speed = GridCellAnalysis.apply_speed_thresholding(self.speed, self.speed, min_speed, max_speed)

    @staticmethod
    def apply_speed_thresholding(input_array, speed_array, min_speed, max_speed=np.inf):
        """Return the samples of ``input_array`` with ``min_speed <= speed <= max_speed``."""
        mask = (speed_array >= min_speed) & (speed_array <= max_speed)
        return input_array[mask]

    def compute_spike_array(self):
        """Count spikes per position sample, ignoring spikes inside gaps in the position record.

        A gap is an interval between consecutive samples longer than 1.5 sample intervals. A
        spike is kept when it lies in ``[start, end)`` of a gap-free stretch and is assigned to
        the first sample at or after its time.

        Returns
        -------
        ndarray of int
            Spike count of every position sample (before speed filtering).
        """
        spike_array = np.zeros_like(self.time, dtype=int)

        time_diff = np.diff(self.time)
        gap_indices = np.where(time_diff > 1.5 * self.dt)[0]

        interval_starts = np.concatenate(([self.time[0]], self.time[gap_indices + 1]))
        interval_ends = np.concatenate((self.time[gap_indices], [self.time[-1]]))

        spike_times = np.asarray(self.spike_times)
        mask = np.zeros_like(spike_times, dtype=bool)
        for start, end in zip(interval_starts, interval_ends):
            mask |= (spike_times >= start) & (spike_times < end)
        valid_spike_times = spike_times[mask]

        spike_indices = np.searchsorted(self.time, valid_spike_times)
        spike_indices = spike_indices[spike_indices < len(self.time)]
        np.add.at(spike_array, spike_indices, 1)

        return spike_array

    def compute_firing_rate_map(self, spike_mode=None):
        """Firing-rate map on 3 cm bins, smoothed with a Gaussian of one bin.

        Parameters
        ----------
        spike_mode : {"binary", "counts"}, optional
            ``"binary"`` counts each sample with spikes once; ``"counts"`` uses the number of
            spikes. Default: the mode given at construction.

        Returns
        -------
        rate_map : ndarray
            Smoothed rate map (Hz), first index along x.
        occupancy_map : ndarray
            Time spent in each bin (s).
        spike_map : ndarray
            Spikes in each bin.
        x_edges, y_edges : ndarray
            Bin edges (cm).
        """
        bin_size = self.rate_map_bin_size
        x_edges = np.arange(np.min(self.x), np.max(self.x) + bin_size, bin_size)
        y_edges = np.arange(np.min(self.y), np.max(self.y) + bin_size, bin_size)

        # rows = x, columns = y
        occupancy_map, _, _ = np.histogram2d(self.x, self.y, bins=[x_edges, y_edges])
        occupancy_map *= self.dt

        if spike_mode is None:
            spike_mode = self.rate_map_spike_mode
        if spike_mode not in SPIKE_MODES:
            raise ValueError("spike_mode must be 'binary' or 'counts'")
        spike_indices = np.flatnonzero(self.spike_array > 0)
        spike_x = self.x[spike_indices]
        spike_y = self.y[spike_indices]
        weights = self.spike_array[spike_indices] if spike_mode == "counts" else None
        spike_map, _, _ = np.histogram2d(spike_x, spike_y, bins=[x_edges, y_edges], weights=weights)

        rate_map = spike_map / (occupancy_map + 1e-10)  # avoids division by zero
        rate_map = gaussian_filter(rate_map, sigma=1)

        return rate_map, occupancy_map, spike_map, x_edges, y_edges
