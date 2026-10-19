"""Firing-rate adaptation: rate as a function of the time since the last field visit (t_prev).

``adaptation_nonparametric`` fits a multiplicative Poisson model
``rate = f(location) g(speed) h(t_prev)`` with 5 cm location bins, speed in octiles (8 bins of
equal sample count) and t_prev in the bins given (default 0, 1, 2, 3, inf s), and returns
``g`` and ``h`` in Hz (Fig 3C, Supp Fig 4).

``adaptation_parametric`` fits
``rate = f(location) (1 + gamma s) (1 + alpha a)`` where ``s`` is the z-scored log speed and
``a`` the z-scored adaptation regressor: the proximity ``exp(-t_prev / tau)`` (``tprev``),
optionally multiplied by the location input of the previous field visit normalised to its
maximum (``tprevXloc``, the regressor of the paper), by its speed-weighted version or by the
spike count of the previous visit. A negative ``alpha`` means a lower rate soon after a field
visit (Fig 3D, 4C, 4D, 5, Supp Fig 1). It also returns the mean regressor over samples running
within 15 degrees of a grid axis (``align``) and further away (``misalign``).

Masks select the samples used in a fit: ``all``, ``infield``, ``outfield``, ``1stHalf`` and
``2ndHalf`` (of field visits), ``oddMinute`` and ``evenMinute``. Both functions use the standard
t_prev rule of :func:`hexmod.rat.fields.extract_field_entry`.

With ``mask_valid=True`` the non-parametric fit uses only samples whose speed lies within the
speed bins and whose t_prev lies within ``[tprev_bins[0], tprev_bins[-1])``; this is needed when
the t_prev bins do not cover all values (the nine bins from 0.05 to 20.1 s of Supp Fig 4) and
removes no sample with the default bins.
"""

from __future__ import annotations

import numpy as np

from .gridcell import GridCellAnalysis
from .fields import extract_field_entry, field_visit_input, position_binning
from .poisson import fit_location_parametric_poisson, fit_model

__all__ = ["adaptation_nonparametric", "adaptation_parametric", "time_masks", "ADAPTATION_TAU_S"]

#: Time constant [s] of the adaptation regressor exp(-t_prev / tau) of the parametric model.
ADAPTATION_TAU_S = 2.0


def time_masks(t, in_field):
    """Sample masks used by the adaptation fits.

    Parameters
    ----------
    t : ndarray
        Sample times (s).
    in_field : ndarray
        0 outside fields, 1 and 2 in the first and second half of a field visit.

    Returns
    -------
    dict
        Boolean masks ``all``, ``infield``, ``outfield``, ``1stHalf``, ``2ndHalf``,
        ``evenMinute``, ``oddMinute``.
    """
    mask_time = {}
    mask_time['all'] = (t > 0)
    mask_time['infield'] = (in_field > 0)
    mask_time['outfield'] = (in_field == 0)
    mask_time['1stHalf'] = (in_field == 1)
    mask_time['2ndHalf'] = (in_field == 2)
    mask_time['evenMinute'] = (np.ceil(t/60) % 2 == 0)
    mask_time['oddMinute'] = (np.ceil(t/60) % 2 == 1)
    return mask_time


def adaptation_nonparametric(inputs):
    """Non-parametric fit of firing rate by location, speed and t_prev.

    Parameters
    ----------
    inputs : dict
        ``cell_data``, ``fieldLoc`` ((K, 3) field centres and radii) and optional
        ``spatialResolution`` (5 cm), ``tprev_bins`` (``[0, 1, 2, 3, inf]`` s), ``n_bins_speed``
        (8), ``mask_type`` (``["infield", "outfield", "all"]``) and ``mask_valid`` (False).

    Returns
    -------
    dict
        For each mask M: ``speed_nonparam_M`` (rate per speed bin, Hz), ``tprev_nonparam_M``
        (rate per t_prev bin, Hz), ``tprev_bincounts_M`` (samples per t_prev bin) and
        ``meanFR_M`` (Hz); plus ``session_id`` and ``cell_id``.
    """
    cell_data = inputs["cell_data"]
    fieldLoc = inputs["fieldLoc"]
    spatialResolution = inputs.get('spatialResolution', 5)  # cm
    tprev_bins = inputs.get("tprev_bins", np.array([0, 1, 2, 3, np.inf]))  # s
    n_bins_speed = inputs.get("n_bins_speed", 8)
    mask_type = inputs.get("mask_type", ["infield", "outfield", "all"])
    use_mask_valid = inputs.get("mask_valid", False)

    gc_obj = GridCellAnalysis(cell_data)
    x, y, t, spikes, speed = gc_obj.x, gc_obj.y, gc_obj.time, gc_obj.spike_array, gc_obj.speed

    peaks_row, peaks_col, radius = fieldLoc[:, 0], fieldLoc[:, 1], fieldLoc[:, 2]
    tmp = extract_field_entry(x, y, t, peaks_row, peaks_col, radius, spikes)
    t_prev_vec, in_field = tmp['t_prev_vec'], tmp['in_field']

    loc_idx, nBin_x, nBin_y, _, _ = position_binning(x, y, spatialResolution)

    # speed bins of equal sample count; the last edge is raised so that the maximum is included
    speed_bins = np.percentile(speed, np.linspace(0, 100, n_bins_speed+1))
    speed_bins[-1] = speed_bins[-1]+0.001
    speed_idx = np.digitize(speed, speed_bins)-1
    tprev_idx = np.digitize(t_prev_vec, tprev_bins)-1

    mask_time = time_masks(t, in_field)

    if use_mask_valid:
        # samples with speed and t_prev inside the bin ranges
        mask_valid = ((speed >= speed_bins[0]) & (speed < speed_bins[-1]) & (t_prev_vec >= tprev_bins[0])
                      & (t_prev_vec < tprev_bins[-1]))

    Fs = 50  # sampling rate (Hz)
    result_cell = dict()
    for mask_name in mask_type:
        mask = mask_time[mask_name]
        if use_mask_valid:
            mask = mask & mask_valid
        in_data = {"spikes": spikes[mask], "factors": [loc_idx[mask], speed_idx[mask], tprev_idx[mask]],
                   "modelsToCompare": [[0, 1, 2]], "maxBins": [nBin_x*nBin_y, n_bins_speed, len(tprev_bins)-1]}
        out = fit_model(in_data)
        speed_est_nonparam = out['model'][0]['B_ml']*Fs
        tprev_est_nonparam = out['model'][0]['C_ml']*Fs
        tprev_bincounts, _ = np.histogram(tprev_idx[mask], np.arange(len(tprev_bins)))

        result_cell[f'speed_nonparam_{mask_name}'] = speed_est_nonparam
        result_cell[f'tprev_nonparam_{mask_name}'] = tprev_est_nonparam
        result_cell[f'tprev_bincounts_{mask_name}'] = tprev_bincounts
        result_cell[f'meanFR_{mask_name}'] = spikes[mask].mean()*Fs

    result_cell['session_id'] = cell_data['session_id']
    result_cell['cell_id'] = cell_data['cell_id']

    return result_cell


def adaptation_parametric(inputs):
    """Parametric fit of the adaptation coefficient alpha.

    Parameters
    ----------
    inputs : dict
        ``cell_data``, ``fieldLoc`` ((K, 3)), ``peakFR`` (mean rate in each field, Hz),
        ``gridOri`` (grid orientation, degrees) and optional ``spatialResolution`` (5 cm),
        ``tau_adapt`` (2 s), ``mask_type`` (``["infield", "outfield", "all"]``), ``adapt_type``
        (``["tprev", "tprevXloc", "tprevXspike"]``; also ``tprevXlocXspeed``) and
        ``normaliseType`` (``"zscore"`` or ``"meanSubtract"``).

    Returns
    -------
    dict
        For each adaptation type T and mask M: ``speed_T_M`` (gamma), ``alpha_T_M``,
        ``converge_T_M``, ``align_T_M`` and ``misalign_T_M`` (mean raw regressor where
        ``cos 6(direction - orientation)`` is positive or negative); plus ``session_id`` and
        ``cell_id``.
    """
    cell_data = inputs["cell_data"]
    fieldLoc = inputs["fieldLoc"]
    peakFR = inputs["peakFR"]
    gridOri = inputs["gridOri"]
    spatialResolution = inputs.get('spatialResolution', 5)  # cm
    tau_adapt = inputs.get("tau_adapt", ADAPTATION_TAU_S)  # s
    mask_type = inputs.get("mask_type", ["infield", "outfield", "all"])
    adapt_type = inputs.get("adapt_type", ["tprev", "tprevXloc", "tprevXspike"])
    normaliseType = inputs.get("normaliseType", "zscore")
    gc_obj = GridCellAnalysis(cell_data)
    x, y, t, spikes, speed, movedirInRad = gc_obj.x, gc_obj.y, gc_obj.time, gc_obj.spike_array, gc_obj.speed, gc_obj.move_dir

    peaks_row, peaks_col, radius = fieldLoc[:, 0], fieldLoc[:, 1], fieldLoc[:, 2]
    tmp = extract_field_entry(x, y, t, peaks_row, peaks_col, radius, spikes)
    ev_df, t_prev_vec, in_field = tmp['ev_df'], tmp['t_prev_vec'], tmp['in_field']
    tprevFactor_raw = np.exp(-t_prev_vec/tau_adapt)

    # input received during the previous field visit
    tmp2 = field_visit_input(x, y, t, ev_df, peakFR, speed)
    adaptWeight_loc = tmp2['adaptWeight_loc']
    adaptWeight_locXspeed = tmp2['adaptWeight_locXspeed']
    adaptWeight_spikes = tmp2['adaptWeight_spikes']

    mask_time = time_masks(t, in_field)

    loc_idx, nBin_x, nBin_y, _, _ = position_binning(x, y, spatialResolution)
    speedFactor_raw = np.log(speed)

    # adaptation regressors, the weighted ones normalised to their maximum
    adaptFactors = {}
    adaptFactors['tprev'] = tprevFactor_raw
    adaptFactors['tprevXloc'] = tprevFactor_raw*adaptWeight_loc/np.max(adaptWeight_loc)
    adaptFactors['tprevXlocXspeed'] = tprevFactor_raw*adaptWeight_locXspeed/np.max(adaptWeight_locXspeed)
    adaptFactors['tprevXspike'] = tprevFactor_raw*adaptWeight_spikes/np.max(adaptWeight_spikes)

    gridAlign_raw = np.cos(6*(movedirInRad-np.deg2rad(gridOri)))

    result_cell = dict()
    for mask_name in mask_type:
        mask = mask_time[mask_name]
        spikes_masked = spikes[mask]
        locFactor_masked = loc_idx[mask]
        for adaptName in adapt_type:
            adaptFactor_raw = adaptFactors[adaptName]

            if normaliseType == 'meanSubtract':
                speedFactor_masked = speedFactor_raw[mask]-speedFactor_raw[mask].mean()
                adaptFactor_masked = adaptFactor_raw[mask]-adaptFactor_raw[mask].mean()
            elif normaliseType == 'zscore':
                speedFactor_masked = (speedFactor_raw[mask]-speedFactor_raw[mask].mean())/speedFactor_raw[mask].std()
                adaptFactor_masked = (adaptFactor_raw[mask]-adaptFactor_raw[mask].mean())/adaptFactor_raw[mask].std()
            else:
                raise ValueError('Invalid normaliseType')

            # bounds keep (1 + gamma s) and (1 + alpha a) between 0 and 2 for every sample (|alpha a| <= 1)
            speedBound = [np.max([-1/speedFactor_masked.max(), 1/speedFactor_masked.min()]),
                          np.min([-1/speedFactor_masked.min(), 1/speedFactor_masked.max()])]
            adaptBound = [np.max([-1/adaptFactor_masked.max(), 1/adaptFactor_masked.min()]),
                          np.min([-1/adaptFactor_masked.min(), 1/adaptFactor_masked.max()])]

            loc_PxSxT, speed_PxSxT, tprev_PxSxT, converge_PxSxT = fit_location_parametric_poisson(
                spikes_masked, locFactor_masked, nBin_x*nBin_y, speedFactor_masked, adaptFactor_masked,
                gamma_bounds=speedBound, beta_bounds=adaptBound, max_iter=100)

            result_cell[f'speed_{adaptName}_{mask_name}'] = speed_PxSxT
            result_cell[f'alpha_{adaptName}_{mask_name}'] = tprev_PxSxT
            result_cell[f'converge_{adaptName}_{mask_name}'] = converge_PxSxT

            # mean regressor in aligned and misaligned running directions
            result_cell[f'align_{adaptName}_{mask_name}'] = adaptFactor_raw[mask & (gridAlign_raw > 0)].mean()
            result_cell[f'misalign_{adaptName}_{mask_name}'] = adaptFactor_raw[mask & (gridAlign_raw < 0)].mean()

    result_cell['session_id'] = cell_data['session_id']
    result_cell['cell_id'] = cell_data['cell_id']

    return result_cell
