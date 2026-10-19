"""Characterisation of a recorded cell: grid measures, fields and movement-direction tuning.

``characterise_gridcell`` computes the rate map (3 cm bins), its autocorrelogram (Gaussian
pre- and post-smoothing of one bin), gridness, scale and orientation. Cells with gridness of
at least ``MIN_GRIDNESS_SCORE`` (0.3) are kept; for them it delineates the fields (minimum peak
distance 2/3 of the scale, radius between 1/8 of the scale and half the peak distance, rate
threshold at the 40th percentile of the map, smoothing of 1/8 of the scale before peak
detection), measures the mean rate inside each field and the deviation of the field centres from
a perfect lattice, the mean firing rate in and out of fields, and the movement-direction tuning
curve (5 degree bins, 3-bin boxcar) with its Rayleigh vector.

With ``rate_map_spike_mode="counts"`` the rate map, and therefore the grid measures and the
fields, are computed from spike counts instead of samples with spikes (used for the field-visit
sequences of Fig 3F).
"""

from __future__ import annotations

import numpy as np

from .gridcell import GridCellAnalysis
from .fields import delineate_fields
from .maps import (compute_directional_rate_map_and_rayleigh_vector, compute_gridness_scale_ori,
                   quantify_field_deviation, quantify_fr_inhomogeneity, spatial_autocorrelogram_fft2d)

__all__ = ["characterise_gridcell"]


def characterise_gridcell(inputs):
    """Grid measures, fields and direction tuning of one cell.

    Parameters
    ----------
    inputs : dict
        ``cell_data`` (see :func:`hexmod.rat.gridcell.make_cell_data`) and optional settings:
        ``MIN_GRIDNESS_SCORE`` (0.3), ``min_dist_btw_peaks_rel`` (2/3, minimum distance between
        field peaks relative to the grid scale), ``prominence`` (0.4, rate quantile of the field
        threshold), ``bin_size_deg`` (5) and ``smooth_nBins`` (3) of the direction tuning curve,
        and ``rate_map_spike_mode`` (default: ``cell_data.get("rate_map_spike_mode", "binary")``).

    Returns
    -------
    dict or None
        None when the gridness is below ``MIN_GRIDNESS_SCORE``. When no field is found: the cell
        labels, ``gridness``, ``scale_cm``, ``orientation_deg`` and ``rate_map``. Otherwise also
        ``x_edges``, ``y_edges``, ``optimal_origin`` (lattice offset), ``fieldLoc`` ((K, 3): field
        centre x, y and radius, cm), ``distFromPerfectGrid`` (K,), ``FR_inhomogeneity_regionFR``
        (mean rate in each field, Hz), ``meanFR_infield``, ``meanFR_outfield``, ``meanFR_all``
        (Hz), ``Rayleigh_mag``, ``Rayleigh_angle`` (degrees) and ``dir_map`` (72-bin direction
        tuning curve, spikes per sample).
    """
    MIN_GRIDNESS_SCORE = inputs.get("MIN_GRIDNESS_SCORE", 0.3)
    min_dist_btw_peaks_rel = inputs.get("min_dist_btw_peaks_rel", 2/3)  # relative to the grid scale
    prominence = inputs.get("prominence", 0.4)  # rate quantile for field delineation
    bin_size_deg = inputs.get("bin_size_deg", 5)  # direction tuning curve
    smooth_nBins = inputs.get("smooth_nBins", 3)  # direction tuning curve
    cell_data = inputs["cell_data"]
    gc_obj = GridCellAnalysis(cell_data, rate_map_spike_mode=inputs.get("rate_map_spike_mode"))
    rate_map, occupancy_map, spike_map, x_edges, y_edges = gc_obj.compute_firing_rate_map()

    presmoothing = 1
    aftersmoothing = 1
    acorr = spatial_autocorrelogram_fft2d(rate_map, presmoothing, aftersmoothing)
    gridness, scale_cm, orientation_deg = compute_gridness_scale_ori(acorr, rate_map_bin_size=gc_obj.rate_map_bin_size)
    gridscaleInBins = scale_cm/gc_obj.rate_map_bin_size

    x, y, spikes, movedirInRad = gc_obj.x, gc_obj.y, gc_obj.spike_array, gc_obj.move_dir
    row_coords = x_edges[:-1]+np.diff(x_edges)/2
    col_coords = y_edges[:-1]+np.diff(y_edges)/2

    if not (gridness >= MIN_GRIDNESS_SCORE):
        return None

    min_dist_between_peak = gridscaleInBins*min_dist_btw_peaks_rel
    max_radius = min_dist_between_peak/2
    min_radius = gridscaleInBins/8
    peaks_row, peaks_col, radius = delineate_fields(rate_map, min_dist_between_peak, max_radius, min_radius, row_coords,
                                                    col_coords, smooth_kernel=min_radius, thres_prominence=prominence)
    fieldLoc = np.c_[peaks_row, peaks_col, radius]
    if fieldLoc.shape[0] == 0:  # no field found
        return {
            "animal_id": cell_data['animal_id'],
            "session_id": cell_data['session_id'],
            "unique_cell_id": cell_data['unique_cell_id'],
            "cell_id": cell_data['cell_id'],
            "gridness": gridness,
            "scale_cm": scale_cm,
            "orientation_deg": orientation_deg,
            "rate_map": rate_map,
        }
    # deviation of the rate map from a perfect grid
    regionFR, FR_shape, FR_scale = quantify_fr_inhomogeneity(rate_map, row_coords, col_coords, fieldLoc)
    optimal_origin, assigned_lattice_coords, distFromPerfectGrid, loc_shape, loc_scale = quantify_field_deviation(
        fieldLoc[:, :-1], lattice_spacing=scale_cm, lattice_orientation=np.deg2rad(orientation_deg))

    # mean rate in and out of fields along the trajectory
    mask_infield = np.zeros_like(x, dtype=bool)
    for i in range(fieldLoc.shape[0]):
        dist_from_field_center = np.sqrt((x-fieldLoc[i, 0])**2+(y-fieldLoc[i, 1])**2)
        mask_infield = mask_infield | (dist_from_field_center < fieldLoc[i, 2])
    fs = 50  # sampling rate (Hz)
    meanFR = np.array([spikes[mask_infield].mean(), spikes[~mask_infield].mean(), spikes.mean()])*fs
    # movement-direction tuning
    Rayleigh_mag, Rayleigh_angle, dirTuning, _ = compute_directional_rate_map_and_rayleigh_vector(
        np.rad2deg(movedirInRad) % 360, spikes, bin_size_deg=bin_size_deg, smooth_nBins=smooth_nBins)

    return {
        "animal_id": cell_data['animal_id'],
        "session_id": cell_data['session_id'],
        "unique_cell_id": cell_data['unique_cell_id'],
        "cell_id": cell_data['cell_id'],
        "gridness": gridness,
        "scale_cm": scale_cm,
        "orientation_deg": orientation_deg,
        "rate_map": rate_map,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "optimal_origin": optimal_origin,
        "fieldLoc": fieldLoc,
        "distFromPerfectGrid": distFromPerfectGrid,
        "FR_inhomogeneity_regionFR": regionFR,
        "meanFR_infield": meanFR[0],
        "meanFR_outfield": meanFR[1],
        "meanFR_all": meanFR[2],
        "Rayleigh_mag": Rayleigh_mag,
        "Rayleigh_angle": Rayleigh_angle,
        "dir_map": dirTuning,
    }
