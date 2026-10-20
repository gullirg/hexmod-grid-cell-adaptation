"""Grid fields, field visits and the time since the last field visit (t_prev).

- ``delineate_fields``: local maxima of a rate map above a rate percentile, separated by a
  minimum distance, each with a radius within which at least 90% of the bins exceed that rate.
- ``extract_field_entry``: entries into and exits from the fields along the trajectory, and
  t_prev at every sample.
- ``field_visit_input``: the location input a cell receives during each field visit (field rate
  times a Gaussian of the distance from the field centre, summed over the visit), held until the
  next visit; it weights the adaptation regressor.
- ``position_binning``: linear index of 5 cm position bins.

t_prev rules (``extract_field_entry``, argument ``tprev_rule``):

``"standard"`` (default; used for every analysis of the paper except Fig 3F)
    During a visit t_prev is the interval from the exit of the previous visit to the entry of
    this one (plus one sample); between visits it grows from one sample after the exit. It is 0
    until the end of the first visit and after the end of the last visit. The first visit is not
    marked as in-field.
``"gap_aware"`` (Fig 3F)
    As the standard rule, but t_prev is undefined (NaN) until the end of the first visit, after the end of
    the last visit, and for intervals (and the visits that end them) that are not positive
    (overlapping fields) or that span a step in the speed-filtered time axis longer than
    ``recording_gap_threshold`` (5 s); it is capped at ``tprev_cap`` (10 s). Every visit,
    including the first, is marked as in-field.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist, squareform
from skimage.feature import peak_local_max

__all__ = ["delineate_fields", "field_visit_input", "extract_field_entry", "position_binning", "TPREV_RULES"]

TPREV_RULES = ("standard", "gap_aware")


def delineate_fields(data, min_dist_between_peak, max_radius, min_radius, row_coord=None, col_coord=None,
                     smooth_kernel=0.0, thres_prominence=0.5):
    """Find grid fields in a rate map.

    Parameters
    ----------
    data : (m, n) ndarray
        Rate map (may contain NaN).
    min_dist_between_peak : float
        Minimum distance between field peaks (bins); of two closer peaks the lower is removed.
    max_radius : float
        Largest field radius considered (bins).
    min_radius : float
        Fields with a radius not above this (bins) are removed (when positive).
    row_coord, col_coord : ndarray, optional
        Coordinates of the bin centres along rows and columns (default: bin indices).
    smooth_kernel : float
        Standard deviation (bins) of a Gaussian filter applied before peak detection.
    thres_prominence : float
        Peaks must exceed the rate at this quantile (0 to 1) of the map's bins.

    Returns
    -------
    peaks_row, peaks_col : ndarray
        Peak coordinates (units of ``row_coord`` and ``col_coord``).
    radius : ndarray
        Field radii (units of the coordinates): the largest radius, in steps of 0.2 bins down
        from ``max_radius``, within which more than 90% of the bins exceed the threshold rate.
    """
    if row_coord is None or col_coord is None:
        row_coord = np.arange(data.shape[0])
        col_coord = np.arange(data.shape[1])
    binSize = row_coord[1]-row_coord[0]
    nan_mask = np.isnan(data)
    data_nonan = np.nan_to_num(data, nan=0.0)

    if smooth_kernel > 0:
        data_smooth = gaussian_filter(data_nonan, sigma=smooth_kernel)
    else:
        data_smooth = data_nonan.copy()

    # candidate peaks
    candidate_coords = peak_local_max(data_smooth, exclude_border=False)
    row, col = candidate_coords[:, 0], candidate_coords[:, 1]
    peak_values = data_smooth[row, col]

    # rate threshold
    fr_sorted = np.sort(data[~nan_mask].ravel())
    min_fr = fr_sorted[int(round(len(fr_sorted) * thres_prominence))]
    prominence = peak_values

    keep = prominence > min_fr
    filtered_row = row[keep]
    filtered_col = col[keep]
    filtered_values = peak_values[keep]

    # minimum distance between peaks
    if len(filtered_row) > 1:
        peak_positions = np.column_stack((filtered_row, filtered_col))
        distance_matrix = squareform(pdist(peak_positions))
        keep_mask = np.ones(len(filtered_row), dtype=bool)

        for i in range(len(filtered_row)):
            for j in range(i + 1, len(filtered_row)):
                if distance_matrix[i, j] < min_dist_between_peak:
                    # remove the lower peak
                    if filtered_values[i] >= filtered_values[j]:
                        keep_mask[j] = False
                    else:
                        keep_mask[i] = False

        peaks_row = filtered_row[keep_mask]
        peaks_col = filtered_col[keep_mask]
    else:
        peaks_row, peaks_col = filtered_row, filtered_col

    # radius of each field
    n_peaks = len(peaks_row)
    coord_col, coord_row = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    radius_est = np.zeros(n_peaks)

    mask_above_prom = data > min_fr

    for i_peak in range(n_peaks):
        dist_from_peak = np.sqrt((coord_row - peaks_row[i_peak])**2 + (coord_col - peaks_col[i_peak])**2)
        tmpdist = max_radius
        while tmpdist > 0:
            pixel_percent = np.mean(mask_above_prom[(dist_from_peak < tmpdist) & (~nan_mask)])
            if pixel_percent > 0.9:
                break
            tmpdist -= 0.2
        radius_est[i_peak] = tmpdist

    if min_radius > 0:
        keep = radius_est > min_radius
        peaks_row = peaks_row[keep]
        peaks_col = peaks_col[keep]
        radius_est = radius_est[keep]

    # bin indices to coordinates
    peaks_row = row_coord[peaks_row]
    peaks_col = col_coord[peaks_col]
    radius = radius_est*binSize
    return peaks_row, peaks_col, radius


def field_visit_input(x, y, t, ev_df, peakFR, speed=None, dt=0.02):
    """Location input received during each field visit, held until the next visit.

    During visit k the instantaneous input is ``peakFR[field] * exp(-0.5 (d / radius)^2) * dt``,
    with d the distance from the field centre. Its sum over the visit is assigned to the samples
    from the exit of visit k to the exit of visit k + 1. Samples before the first visit take the
    value held at the entry of the first visit, which is 0; samples after the exit of the last
    visit take the value held at the entry of the second-to-last visit (the input of the visit
    before it).

    Parameters
    ----------
    x, y, t : ndarray
        Speed-filtered positions and times.
    ev_df : DataFrame
        Field visits from :func:`extract_field_entry`.
    peakFR : ndarray
        Rate of each field (mean rate inside the field).
    speed : ndarray, optional
        Running speed; adds the speed-weighted input ``adaptWeight_locXspeed``.
    dt : float
        Sample interval (s).

    Returns
    -------
    dict
        ``adaptWeight_loc``, ``adaptWeight_locXspeed``, ``adaptWeight_spikes`` (spike count of the
        previous visit) and ``loc_instant`` (instantaneous input during visits), one value per
        sample.
    """
    t_enter_all = ev_df['t_enter'].to_numpy()
    t_exit_all = ev_df['t_exit'].to_numpy()
    fieldId_all = ev_df['fieldId'].to_numpy()
    field_x_all = ev_df['field_x'].to_numpy()
    field_y_all = ev_df['field_y'].to_numpy()
    field_radius_all = ev_df['radius'].to_numpy()
    nspike_all = (ev_df['FR']*ev_df['dur_eff']).to_numpy()

    adaptWeight_loc = np.zeros(len(t))
    adaptWeight_locXspeed = np.zeros(len(t))
    adaptWeight_spikes = np.zeros(len(t))
    loc_instant = np.zeros(len(t))
    for event in range(len(t_enter_all)-1):
        t_enter = t_enter_all[event]
        t_exit = t_exit_all[event]
        t_exit_next = t_exit_all[event+1]
        t_current = (t >= t_enter) & (t < t_exit)
        t_untilNextVisit = (t >= t_exit) & (t < t_exit_next)
        fieldId = fieldId_all[event]
        field_x = field_x_all[event]
        field_y = field_y_all[event]
        field_radius = field_radius_all[event]
        # Gaussian of the distance from the field centre, scaled by the field rate
        distFromPeak = np.sqrt((x[t_current]-field_x)**2+(y[t_current]-field_y)**2)
        input_loc = peakFR[fieldId]*np.exp(-0.5*(distFromPeak/(field_radius))**2)
        adaptWeight_loc[t_untilNextVisit] = input_loc.sum()*dt
        loc_instant[t_current] = input_loc*dt
        if speed is not None:
            speed_tmp = np.log(speed[t_current])
            input_locXspeed = input_loc*speed_tmp
            adaptWeight_locXspeed[t_untilNextVisit] = input_locXspeed.sum()
        if nspike_all.sum() > 0:
            adaptWeight_spikes[t_untilNextVisit] = nspike_all[event]

        if event == 0:
            # before the first visit: the value held at the entry of the first visit (0)
            adaptWeight_loc[t < t_enter] = adaptWeight_loc[t_current][0]
            adaptWeight_locXspeed[t < t_enter] = adaptWeight_locXspeed[t_current][0]
            adaptWeight_spikes[t < t_enter] = adaptWeight_spikes[t_current][0]
        if event == len(t_enter_all)-2:
            # after the last exit: the value held at the entry of the second-to-last visit
            adaptWeight_loc[t >= t_exit_next] = adaptWeight_loc[t_current][0]
            adaptWeight_locXspeed[t >= t_exit_next] = adaptWeight_locXspeed[t_current][0]
            adaptWeight_spikes[t >= t_exit_next] = adaptWeight_spikes[t_current][0]
    result = {
        'adaptWeight_loc': adaptWeight_loc,
        'adaptWeight_locXspeed': adaptWeight_locXspeed,
        'adaptWeight_spikes': adaptWeight_spikes,
        'loc_instant': loc_instant,
    }

    return result


def extract_field_entry(x, y, time, peaks_x, peaks_y, radius, spikes=None, dt=0.02, tprev_rule="standard",
                        tprev_cap=10.0, recording_gap_threshold=5.0):
    """Field visits along the trajectory and t_prev at every sample.

    A visit starts at the first sample within a field's radius and ends at the first sample
    outside it. Visits to all fields are sorted by entry time.

    Parameters
    ----------
    x, y, time : ndarray
        Speed-filtered positions (cm) and times (s), T samples.
    peaks_x, peaks_y, radius : ndarray
        Field centres and radii (cm), K fields.
    spikes : ndarray, optional
        Spikes per sample; adds the in-field and preceding out-of-field rates of each visit.
    dt : float
        Sample interval (s).
    tprev_rule : {"standard", "gap_aware"}
        See the module docstring.
    tprev_cap : float or None
        Upper limit of t_prev (s) under the gap-aware rule.
    recording_gap_threshold : float
        Under the gap-aware rule, intervals between visits that contain a step in ``time``
        longer than this (s) have no defined t_prev.

    Returns
    -------
    dict
        ``ev_df`` (one row per visit: ``t_enter``, ``t_exit``, ``fieldId``, ``t_prev``,
        ``field_x``, ``field_y``, ``radius``, ``direction_prev`` (direction from the previous
        field's centre, degrees), ``dur_eff`` (s), ``FR``, ``FR_prevOutfield`` (Hz)), ``t_prev_vec``
        (T,), ``in_field`` (T,; 0 outside fields, 1 first half and 2 second half of a visit),
        ``distFromPeak`` (T,; distance from the nearest field centre) and ``t_sinceEntry_vec``.
        Under the gap-aware rule ``ev_df`` also holds ``t_prev_valid`` and
        ``spans_recording_gap``, and the dict holds ``t_prev_valid_vec``.
    """
    if tprev_rule not in TPREV_RULES:
        raise ValueError("tprev_rule must be 'standard' or 'gap_aware'")
    gap_aware = tprev_rule == "gap_aware"
    x = np.asarray(x)
    y = np.asarray(y)
    time = np.asarray(time)
    peaks_x = np.asarray(peaks_x)
    peaks_y = np.asarray(peaks_y)
    radius = np.asarray(radius)
    distSqrFromPeak = np.zeros((len(x), len(peaks_x)), dtype=float)
    events_list = []
    for k in range(len(peaks_x)):
        distSqrFromPeak[:, k] = (x - peaks_x[k]) ** 2 + (y - peaks_y[k]) ** 2
        inside = distSqrFromPeak[:, k] < (radius[k] ** 2)
        inside_pad = np.r_[False, inside, False]
        d = np.diff(inside_pad.astype(np.int8), axis=0)  # +1 at entry, -1 at exit
        on_idx = np.where(d == 1)[0]  # first sample inside
        off_idx = np.where(d == -1)[0]  # first sample outside
        events_list.append(np.column_stack([on_idx, off_idx, np.full(len(on_idx), k, dtype=int)]))
    distFromPeak = np.sqrt(np.min(distSqrFromPeak, axis=1))
    if not events_list:
        raise ValueError('no field visit event detected')

    events = np.vstack(events_list)  # (M, 3): entry index, exit index, field
    sort_idx = np.argsort(events[:, 0])
    events = events[sort_idx]
    t_enter = time[events[:, 0]]
    # a visit that lasts to the end of the recording exits at the last sample
    events[:, 1] = np.clip(events[:, 1], None, len(time)-1)
    t_exit = time[events[:, 1]]
    on_idx = events[:, 0]
    off_idx = events[:, 1]
    fieldId = events[:, 2]

    if gap_aware:
        t_prev, t_prev_vec, in_field, t_sinceEntry_vec, event_tprev_valid, spans_recording_gap = _tprev_gap_aware(
            time, t_enter, t_exit, on_idx, off_idx, dt, tprev_cap, recording_gap_threshold)
    else:
        t_prev, t_prev_vec, in_field, t_sinceEntry_vec = _tprev_standard(time, t_enter, t_exit, on_idx, off_idx, dt)
    fieldId_prev = np.full_like(fieldId, -1, dtype=int)
    if len(t_enter) >= 2:
        fieldId_prev[1:] = fieldId[:-1]

    nTimepoint = off_idx-on_idx
    # direction between consecutive field centres
    direction_23 = np.full(len(t_enter), np.nan)
    mask_tmp = (fieldId != fieldId_prev) & (fieldId_prev >= 0)
    field_x = peaks_x[fieldId]
    field_y = peaks_y[fieldId]
    field_x_prev = peaks_x[fieldId_prev]
    field_y_prev = peaks_y[fieldId_prev]
    direction_23[mask_tmp] = np.arctan2(field_y[mask_tmp]-field_y_prev[mask_tmp], field_x[mask_tmp]-field_x_prev[mask_tmp]) * 180/np.pi

    # firing rate during each visit and in the preceding out-of-field interval
    if spikes is not None:
        FR_currInfield = np.zeros(len(t_enter), dtype=float)
        FR_prevOutfield = np.zeros(len(t_enter), dtype=float)
        for i in range(1, len(t_enter)):
            # skips zero-length visits and overlapping fields
            if (off_idx[i] > on_idx[i]) & (on_idx[i] > off_idx[i-1]):
                FR_currInfield[i] = spikes[on_idx[i]:off_idx[i]].sum()/(off_idx[i]-on_idx[i])/dt
                FR_prevOutfield[i] = spikes[off_idx[i-1]:on_idx[i]].sum()/(on_idx[i]-off_idx[i-1])/dt
    columns = {"t_enter": t_enter, "t_exit": t_exit, "fieldId": fieldId, "t_prev": t_prev}
    if gap_aware:
        columns.update({"t_prev_valid": event_tprev_valid, "spans_recording_gap": spans_recording_gap})
    columns.update({
        "field_x": field_x,
        "field_y": field_y,
        "radius": radius[fieldId],
        "direction_prev": direction_23,
        "dur_eff": nTimepoint*dt,
        "FR": FR_currInfield if spikes is not None else np.nan,
        "FR_prevOutfield": FR_prevOutfield if spikes is not None else np.nan,
    })
    ev_df = pd.DataFrame(columns)
    result = {'ev_df': ev_df, 't_prev_vec': t_prev_vec}
    if gap_aware:
        result['t_prev_valid_vec'] = np.isfinite(t_prev_vec)
    result.update({'in_field': in_field, 'distFromPeak': distFromPeak, 't_sinceEntry_vec': t_sinceEntry_vec})
    return result


def _tprev_standard(time, t_enter, t_exit, on_idx, off_idx, dt):
    """t_prev of every visit and sample under the standard rule (see the module docstring)."""
    t_prev = np.zeros_like(t_enter, dtype=float)
    if len(t_enter) >= 2:
        t_prev[1:] = (t_enter[1:] - t_exit[:-1])+dt
    t_prev_vec = np.zeros_like(time, dtype=float)
    t_sinceEntry_vec = np.full_like(time, dtype=float, fill_value=np.nan)
    in_field = np.zeros_like(time, dtype=int)
    t_mid = (t_enter+t_exit)/2
    for i in range(1, len(t_enter)):
        # during the visit: constant interval since the previous exit
        t_prev_vec[on_idx[i]:off_idx[i]] = t_prev[i]
        t_sinceEntry_vec[on_idx[i]:off_idx[i]] = time[on_idx[i]:off_idx[i]] - time[on_idx[i]]
        infield_mask = np.arange(on_idx[i], off_idx[i])
        infield_mask_first = infield_mask[time[infield_mask] < t_mid[i]]
        infield_mask_second = infield_mask[time[infield_mask] >= t_mid[i]]
        in_field[infield_mask_first] = 1  # first half of the visit
        in_field[infield_mask_second] = 2  # second half of the visit
        # between the previous exit and this entry: time since the exit
        t_prev_vec[off_idx[i-1]:on_idx[i]] = time[off_idx[i-1]:on_idx[i]]-time[off_idx[i-1]]+dt
    return t_prev, t_prev_vec, in_field, t_sinceEntry_vec


def _tprev_gap_aware(time, t_enter, t_exit, on_idx, off_idx, dt, tprev_cap, recording_gap_threshold):
    """t_prev of every visit and sample under the gap-aware rule (see the module docstring)."""
    # the first visit has no previous visit
    t_prev = np.full_like(t_enter, np.nan, dtype=float)
    if len(t_enter) >= 2:
        t_prev[1:] = (t_enter[1:] - t_exit[:-1])+dt

    event_tprev_valid = np.isfinite(t_prev) & (t_prev > 0)
    spans_recording_gap = np.zeros(len(t_enter), dtype=bool)
    for i in range(1, len(t_enter)):
        interval = time[off_idx[i - 1]:on_idx[i] + 1]
        spans_recording_gap[i] = (
            len(interval) > 1
            and np.any(np.diff(interval) > recording_gap_threshold)
        )
    event_tprev_valid &= ~spans_recording_gap

    # NaN: no valid preceding interval
    t_prev_vec = np.full_like(time, fill_value=np.nan, dtype=float)
    t_sinceEntry_vec = np.full_like(time, dtype=float, fill_value=np.nan)
    in_field = np.zeros_like(time, dtype=int)
    t_mid = (t_enter+t_exit)/2
    for i in range(0, len(t_enter)):
        t_sinceEntry_vec[on_idx[i]:off_idx[i]] = time[on_idx[i]:off_idx[i]] - time[on_idx[i]]
        infield_mask = np.arange(on_idx[i], off_idx[i])
        infield_mask_first = infield_mask[time[infield_mask] < t_mid[i]]
        infield_mask_second = infield_mask[time[infield_mask] >= t_mid[i]]
        in_field[infield_mask_first] = 1  # first half of the visit
        in_field[infield_mask_second] = 2  # second half of the visit

    for i in range(1, len(t_enter)):
        if not event_tprev_valid[i]:
            continue
        visit_tprev = t_prev[i]
        if tprev_cap is not None:
            visit_tprev = min(visit_tprev, float(tprev_cap))
        t_prev_vec[on_idx[i]:off_idx[i]] = visit_tprev

        ramp = time[off_idx[i-1]:on_idx[i]]-time[off_idx[i-1]]+dt
        if tprev_cap is not None:
            ramp = np.minimum(ramp, float(tprev_cap))
        t_prev_vec[off_idx[i-1]:on_idx[i]] = ramp
    return t_prev, t_prev_vec, in_field, t_sinceEntry_vec, event_tprev_valid, spans_recording_gap


def position_binning(x, y, spatialResolution=5):
    """Linear index of square position bins.

    Bin edges start at the minimum of each coordinate in steps of ``spatialResolution``; the last
    edge is raised by 0.001 so that the maximum falls in the last bin.

    Returns
    -------
    pos_linear : ndarray of int
        ``y_idx * nBin_x + x_idx``.
    nBin_x, nBin_y : int
    x_idx, y_idx : ndarray of int
    """
    x_bins = np.arange(np.nanmin(x), np.nanmax(x)+spatialResolution, spatialResolution)
    y_bins = np.arange(np.nanmin(y), np.nanmax(y)+spatialResolution, spatialResolution)
    x_bins[-1] = x_bins[-1]+0.001  # np.digitize excludes the right edge
    y_bins[-1] = y_bins[-1]+0.001
    nBin_x = len(x_bins)-1
    nBin_y = len(y_bins)-1

    x_idx = np.digitize(x, x_bins) - 1
    y_idx = np.digitize(y, y_bins) - 1
    pos_linear = y_idx*nBin_x+x_idx  # x along columns, y along rows
    return pos_linear, nBin_x, nBin_y, x_idx, y_idx
