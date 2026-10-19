"""Hexadirectional modulation of a recorded cell, controlling for location and speed.

``measure_hexmod`` fits ``rate = f(location) (1 + gamma s) (1 + beta cos 6(theta - phi))`` with
5 cm location bins, ``s`` the z-scored log running speed, ``theta`` the movement direction and
``phi`` the grid orientation. ``beta`` is the hexadirectional modulation (Fig 5).
"""

from __future__ import annotations

import numpy as np

from .gridcell import GridCellAnalysis
from .fields import extract_field_entry, position_binning
from .poisson import fit_location_parametric_poisson

__all__ = ["measure_hexmod"]


def measure_hexmod(inputs):
    """Fit the hexadirectional modulation ``beta`` of one cell.

    Parameters
    ----------
    inputs : dict
        ``cell_data``, ``gridOri`` (grid orientation, degrees) and optional
        ``spatialResolution`` (5 cm), ``mask_type`` (``["all"]``; also ``evenMinute``,
        ``oddMinute`` and, with ``fieldLoc``, ``infield``, ``outfield``, ``1stHalf``,
        ``2ndHalf``) and ``normaliseType`` (``"zscore"`` or ``"meanSubtract"``).

    Returns
    -------
    dict
        For each mask M: ``speed_M`` (gamma), ``beta_M`` and ``converge_M``; plus
        ``session_id`` and ``cell_id``.
    """
    cell_data = inputs["cell_data"]
    gridOri = inputs["gridOri"]
    spatialResolution = inputs.get('spatialResolution', 5)  # cm
    mask_type = inputs.get("mask_type", ["all"])
    normaliseType = inputs.get("normaliseType", "zscore")

    gc_obj = GridCellAnalysis(cell_data)
    x, y, t, spikes, speed, movedirInRad = gc_obj.x, gc_obj.y, gc_obj.time, gc_obj.spike_array, gc_obj.speed, gc_obj.move_dir

    mask_time = {}
    mask_time['all'] = (t > 0)
    mask_time['evenMinute'] = (np.ceil(t/60) % 2 == 0)
    mask_time['oddMinute'] = (np.ceil(t/60) % 2 == 1)

    if any(w in mask_type for w in ['infield', 'outfield', '1stHalf', '2ndHalf']):
        fieldLoc = inputs["fieldLoc"]
        peaks_row, peaks_col, radius = fieldLoc[:, 0], fieldLoc[:, 1], fieldLoc[:, 2]
        tmp = extract_field_entry(x, y, t, peaks_row, peaks_col, radius, spikes)
        in_field = tmp['in_field']
        mask_time['infield'] = (in_field > 0)
        mask_time['outfield'] = (in_field == 0)
        mask_time['1stHalf'] = (in_field == 1)
        mask_time['2ndHalf'] = (in_field == 2)

    loc_idx, nBin_x, nBin_y, _, _ = position_binning(x, y, spatialResolution)
    speedFactor_raw = np.log(speed)
    dirFactor_raw = np.cos(6*(movedirInRad-np.deg2rad(gridOri)))

    result_cell = dict()
    for mask_name in mask_type:
        mask = mask_time[mask_name]
        spikes_masked = spikes[mask]
        locFactor_masked = loc_idx[mask]
        dirFactor_masked = dirFactor_raw[mask]
        if normaliseType == 'meanSubtract':
            speedFactor_masked = speedFactor_raw[mask]-speedFactor_raw[mask].mean()
        elif normaliseType == 'zscore':
            speedFactor_masked = (speedFactor_raw[mask]-speedFactor_raw[mask].mean())/speedFactor_raw[mask].std()
        else:
            raise ValueError('Invalid normaliseType')
        # bounds keep (1 + gamma s) and (1 + beta d) positive for every sample
        speedBound = [np.max([-1/speedFactor_masked.max(), 1/speedFactor_masked.min()]),
                      np.min([-1/speedFactor_masked.min(), 1/speedFactor_masked.max()])]
        dirBound = [np.max([-1/dirFactor_masked.max(), 1/dirFactor_masked.min()]),
                    np.min([-1/dirFactor_masked.min(), 1/dirFactor_masked.max()])]

        loc_PxSxD, speed_PxSxD, HM_poisson, converge_PxSxD = fit_location_parametric_poisson(
            spikes_masked, locFactor_masked, nBin_x*nBin_y, speedFactor_masked, dirFactor_masked,
            gamma_bounds=speedBound, beta_bounds=dirBound, max_iter=100)
        result_cell[f'speed_{mask_name}'] = speed_PxSxD
        result_cell[f'beta_{mask_name}'] = HM_poisson
        result_cell[f'converge_{mask_name}'] = converge_PxSxD
    result_cell['session_id'] = cell_data['session_id']
    result_cell['cell_id'] = cell_data['cell_id']

    return result_cell
