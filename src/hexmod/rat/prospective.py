"""Prospective coding control: spatial information of the rate map with time-shifted spikes.

``check_prospective_coding`` shifts the spike train of a cell against its positions by each
time in ``shift_time_list`` (default -400 to +380 ms in steps of 20 ms, one 50 Hz sample per
20 ms) and computes, for each shift, the spatial information (bits per spike) and the mean
squared rate of the smoothed rate map (3 cm bins, Gaussian filter of one bin). Samples at the
ends of the recording that could receive no spike under the largest shifts are set to zero for
every shift, so all shifts use the same samples. The shift with the highest spatial information
is the cell's best shift (Supp Fig 2). The comparison of spatial information across shifts
follows Chaudhuri-Vayalambrone et al. (2023).
"""

from __future__ import annotations

import numpy as np

from .gridcell import GridCellAnalysis
from .maps import compute_directional_rate_map_and_rayleigh_vector, compute_ratemap

__all__ = ["check_prospective_coding", "DEFAULT_SHIFTS_MS"]

DEFAULT_SHIFTS_MS = np.arange(-400, 400, 20)


def check_prospective_coding(inputs):
    """Spatial information of the rate map for time-shifted spikes.

    Parameters
    ----------
    inputs : dict
        ``cell_data`` and optional ``shift_time_list`` (ms; default ``np.arange(-400, 400, 20)``),
        ``isSplitHalfTest`` (False, or ``"firstSecond"``, ``"evenOddMin"``, ``"fastSlow"``:
        compute the measures separately in two halves of the samples) and
        ``isDirectionalTest`` (False; True replaces the rate map by the movement-direction tuning
        curve and adds its Rayleigh vector length).

    Returns
    -------
    dict
        ``SI`` (spatial information, bits per spike) and ``MSQ`` (mean squared rate) per shift,
        shape (n_shifts,) or (2, n_shifts) for split halves; ``rayleigh_mag`` with
        ``isDirectionalTest``.
    """
    cell_data = inputs["cell_data"]
    shift_time_list = inputs.get("shift_time_list", np.arange(-400, 400, 20))
    isSplitHalfTest = inputs.get("isSplitHalfTest", False)
    isDirectionalTest = inputs.get("isDirectionalTest", False)

    gc_obj = GridCellAnalysis(cell_data)
    x, y, t, spikes, speed, movedirInRad = gc_obj.x, gc_obj.y, gc_obj.time, gc_obj.spike_array, gc_obj.speed, gc_obj.move_dir
    bin_size = gc_obj.rate_map_bin_size
    x_edges = np.arange(np.min(x), np.max(x) + bin_size, bin_size)
    y_edges = np.arange(np.min(y), np.max(y) + bin_size, bin_size)

    match (isSplitHalfTest):
        case "firstSecond":
            tmparray = np.arange(len(t))
            mask_1st = (tmparray < tmparray.max()/2)
            mask_2nd = (tmparray >= tmparray.max()/2)
        case "evenOddMin":
            mask_1st = (np.ceil(t/60) % 2 == 0)
            mask_2nd = (np.ceil(t/60) % 2 == 1)
        case "fastSlow":
            speed_median = np.median(speed)
            mask_1st = (speed < speed_median)
            mask_2nd = (speed >= speed_median)
        case False:
            pass
        case _:
            raise ValueError('invalid isSplitHalfTest value, should be "fastSlow","firstSecond" or "evenOddMin"')

    if (isSplitHalfTest is not False):
        SI = np.zeros((2, len(shift_time_list)))
        MSQ = np.zeros((2, len(shift_time_list)))
        rayleigh_mag = np.zeros((2, len(shift_time_list)))
    else:
        SI = np.zeros(len(shift_time_list))
        MSQ = np.zeros(len(shift_time_list))
        rayleigh_mag = np.zeros(len(shift_time_list))

    for shift_id, shift_time in enumerate(shift_time_list):
        shift = int(shift_time/20)  # samples of 20 ms
        spikes_shifted = np.roll(spikes, shift)
        # samples that received wrapped-around spikes get no spike
        if shift > 0:
            spikes_shifted[:shift] = 0
        elif shift < 0:
            spikes_shifted[shift:] = 0
        # the same samples are used for every shift
        spikes_shifted[int(shift_time_list.min()/20):] = 0
        spikes_shifted[:int(shift_time_list.max()/20)] = 0

        if (isSplitHalfTest is not False):
            for mask_id, mask in enumerate([mask_1st, mask_2nd]):
                if isDirectionalTest:
                    rayleigh_mag[mask_id, shift_id], rayleighh_ang, rate_map, occupancy_map = compute_directional_rate_map_and_rayleigh_vector(
                        np.rad2deg(movedirInRad[mask]) % 360, spikes_shifted[mask], bin_size_deg=5, smooth_nBins=3)
                else:
                    rate_map, occupancy_map = compute_ratemap(x[mask], y[mask], spikes_shifted[mask], x_edges, y_edges, smooth=1)
                # mean squared rate (zero-lag autocorrelation of the map)
                MSQ[mask_id, shift_id] = np.mean(rate_map**2)
                # spatial information per bin (bits per spike), summed over bins
                tmpSI = occupancy_map/occupancy_map.sum()*rate_map/(spikes_shifted[mask].mean())*np.log2(rate_map/spikes_shifted[mask].mean()+1e-10)
                SI[mask_id, shift_id] = np.nansum(tmpSI)
        else:
            if isDirectionalTest:
                rayleigh_mag[shift_id], rayleighh_ang, rate_map, occupancy_map = compute_directional_rate_map_and_rayleigh_vector(
                    np.rad2deg(movedirInRad) % 360, spikes_shifted, bin_size_deg=5, smooth_nBins=3)
            else:
                rate_map, occupancy_map = compute_ratemap(x, y, spikes_shifted, x_edges, y_edges, smooth=1)

            # mean squared rate (zero-lag autocorrelation of the map)
            MSQ[shift_id] = np.mean(rate_map**2)
            # spatial information per bin (bits per spike), summed over bins
            tmpSI = occupancy_map/occupancy_map.sum()*rate_map/(spikes_shifted.mean())*np.log2(rate_map/spikes_shifted.mean()+1e-10)
            SI[shift_id] = np.nansum(tmpSI)

    if isDirectionalTest:
        result = {'MSQ': MSQ, 'SI': SI, 'rayleigh_mag': rayleigh_mag}
    else:
        result = {'MSQ': MSQ, 'SI': SI}

    return result
