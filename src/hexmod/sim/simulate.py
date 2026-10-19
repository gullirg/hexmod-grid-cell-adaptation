"""Firing of simulated grid cells along a trajectory, with and without adaptation, and the
hexadirectional modulation of the summed population activity.

Model (``simulate_grid_cells``). At every sample faster than ``speedThres`` the rate of a cell is its rate
map at the current position (the field nearest to the position, ``hexmod.sim.pool``) times a speed
factor (1 in the paper, ``speedGamma_mean = 0``) times the adaptation factor

    1 - adaptEff_mean * exp(-t_prev / adaptTau),

where ``t_prev`` is the time since the animal left the previously visited field of that cell
(fields as delineated on the rate map, visits from ``hexmod.rat.fields.extract_field_entry``).
The paper uses ``adaptEff_mean = 0.1`` and ``adaptTau = 2`` s. Slower samples have rate 0.

A participant (``simulate_and_measure``) is 10,000 cells of one module and one trajectory;
the population rate is the mean over cells. Its hexadirectional modulation is estimated for
symmetries 1 to 7 with two estimators, each cross-validated by leaving one block out:

- "simple" (``calc_hm``): the grid orientation is estimated on three blocks by regressing the
  rate on sin and cos of k times the running direction (``quadratic_filter``); the coefficient is
  the slope of the rate on cos k(direction - orientation) in the left-out block;
- "multiplicative" (location-controlled, ``calc_hm_control_location``): a rank-one model
  rate(location, direction) = f(location) g(direction) is fitted to the mean rate in 5 vm x 5 deg
  bins, and the same regression is applied to the direction factor g.

Fig 7A and 7D use the rates in 12 directions relative to the true grid orientation, from all
moving samples (``noncrossResult``); Fig 7B, 7E and 7F use the cross-validated coefficients
(``crossvalResult``) averaged over the four left-out blocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats

from hexmod.rat.fields import extract_field_entry, position_binning
from hexmod.rat.poisson import fit_rank1_als
from hexmod.sim import trajectory_v1
from hexmod.sim.pool import calc_fr_from_grid_fields

__all__ = ["simulate_grid_cells", "simulate_and_measure", "draw_participant_cells", "participant_cells",
           "simulate_participant",
           "simulate_many", "quadratic_filter", "calc_hm", "calc_hm_control_location", "FOLD_SYM_LIST"]

#: Symmetries (k-fold) whose modulation is estimated.
FOLD_SYM_LIST = (1, 2, 3, 4, 5, 6, 7)


def simulate_grid_cells(gc_table, traj=None, par=None, is_adapt=True, calc_adapt_hm=True):
    """Simulate the firing rates of a set of grid cells along a trajectory.

    Accepts ``simulate_grid_cells(gc_table, traj, par)`` or a single dict
    ``simulate_grid_cells({"gcTable": ..., "traj": ..., "par": ..., "isAdapt": ..., "calcAdaptHM": ...})``.

    Parameters
    ----------
    gc_table : pandas.DataFrame
        Cells (rows of a pool): ``centers``, ``peakFR_jitter``, ``peaks_x``, ``peaks_y``, ``radius``.
    traj : pandas.DataFrame
        Trajectory with ``time``, ``x``, ``y``, ``moveSpeed``, ``moveDir``.
    par : dict
        Parameters (``hexmod.sim.pool.set_param``).
    is_adapt : bool
        Compute the adapted rates.
    calc_adapt_hm : bool
        Return the per-cell summaries of the adaptation term.

    Returns
    -------
    dict
        ``FR_adapt``, ``FR_noadapt``, ``locFactor`` (n_cells, n_samples) [Hz];
        ``ratemap_mean`` (y bins, x bins), the mean rate map of the cells; with ``calc_adapt_hm``:
        ``tprevby12dir`` (n_cells, 12), mean ``exp(-t_prev / adaptTau)`` inside fields by running
        direction relative to the grid (0, 30, ..., 330 deg), ``tprevbyAlignMis`` (n_cells, 2),
        the same for aligned (cos 6 theta > 0) and misaligned directions, ``tprev_HMness``
        (n_cells,), its slope on cos 6 theta; with ``is_adapt``: ``locFactor_HMness_mean``, the
        aligned-minus-misaligned contrast of the mean rate map along the path.
    """
    if isinstance(gc_table, dict):
        inputs = gc_table
        gc_table = inputs['gcTable']
        traj = inputs['traj']
        par = inputs['par']
        is_adapt = inputs.get('isAdapt', True)
        calc_adapt_hm = inputs.get('calcAdaptHM', True)

    # trajectory and bins (shared by all cells)
    t = np.asarray(traj["time"])
    x = np.asarray(traj["x"])
    y = np.asarray(traj["y"])
    moveSpeed = traj["moveSpeed"].to_numpy()
    mask_speed = moveSpeed > par['speedThres']

    n_cells = len(gc_table)
    T = len(t)
    if calc_adapt_hm is not None:
        moveDir = np.asarray(traj["moveDir"])
        alignness = np.cos(np.deg2rad((moveDir - par['gridOri']) * 6))
        dirRelGrid = (moveDir - par['gridOri']) % 360
        dirRelGrid_idx = np.digitize(dirRelGrid, np.arange(-30, 361, 30) + 15) - 1
        dirRelGrid_idx[dirRelGrid_idx == 12] = 0
        tprevby12dir = np.zeros((n_cells, 12))  # mean exp(-t_prev/tau) in 12 directions relative to the grid
        tprevbyAlignMis = np.zeros((n_cells, 2))  # the same for aligned and misaligned directions
        tprev_HMness = np.zeros(n_cells)  # slope of exp(-t_prev/tau) on cos(6 theta) inside fields
    x_bins = np.asarray(par['xBins'])
    y_bins = np.asarray(par['yBins'])
    nBin_x = len(par['xBins_center'])
    nBin_y = len(par['yBins_center'])

    # bin centres for the rate-map lookup (flattened y-major)
    tmpx, tmpy = np.meshgrid(par['xBins_center'], par['yBins_center'], indexing="xy")
    basePos = np.column_stack([tmpx.ravel(), tmpy.ravel()])
    rateVec_mean = np.zeros(nBin_x * nBin_y,)  # running sum of the cells' rate maps

    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    if x_min < x_bins[0] or x_max > x_bins[-1] or y_min < y_bins[0] or y_max > y_bins[-1]:
        raise ValueError(
            f"Trajectory out of bounds: x range [{x_min:.2f}, {x_max:.2f}] not within xBins [{x_bins[0]:.2f}, {x_bins[-1]:.2f}] "
            f"or y range [{y_min:.2f}, {y_max:.2f}] not within yBins [{y_bins[0]:.2f}, {y_bins[-1]:.2f}]. "
            "Adjust par.envSize_x/y or par.binSize_ratemap, or clip/translate the trajectory."
        )

    # positions to linear bin indices
    x_idx = np.digitize(x, x_bins) - 1
    y_idx = np.digitize(y, y_bins) - 1
    pos_idx = y_idx * nBin_x + x_idx

    peaks_x_all = gc_table["peaks_x"].to_numpy()
    peaks_y_all = gc_table["peaks_y"].to_numpy()
    radius_all = gc_table["radius"].to_numpy()
    centers = gc_table['centers'].to_numpy()
    peakFR_jitter = gc_table['peakFR_jitter'].to_numpy()

    # the same adaptation and speed parameters for all cells
    adapt_tau = par['adaptTau']
    adapt_beta = par['adaptEff_mean']
    speedGamma = par['speedGamma_mean']

    FR_adapt = np.zeros((n_cells, T), dtype=float)
    FR_noadapt = np.zeros_like(FR_adapt)
    locFactor = np.zeros_like(FR_adapt)
    # speed factor; used only when speedGamma_mean is non-zero (0 in all simulations of the paper).
    # Note that the expression subtracts mean/std from the log speed (operator precedence).
    if (np.nanmax(moveSpeed) > np.nanmin(moveSpeed)) & (speedGamma != 0):
        speedTransform = (np.log(moveSpeed) - np.mean(np.log(moveSpeed)) / np.std(np.log(moveSpeed)))
        speedFactor = 1.0 + speedGamma * speedTransform
    else:
        speedFactor = np.ones_like(moveSpeed)

    for cid in range(n_cells):
        rate_vec = calc_fr_from_grid_fields(centers[cid], basePos, par['fieldSigma'] * par['gridScale'], peakFR_jitter[cid])
        rateVec_mean += rate_vec
        locFactor[cid, mask_speed] = rate_vec[pos_idx[mask_speed]]
        FR_noadapt_tmp = locFactor[cid, mask_speed] * speedFactor[mask_speed]
        FR_noadapt[cid, mask_speed] = FR_noadapt_tmp

        if is_adapt:
            # field visits of this cell and the time since the previous field (t_prev). The default
            # sample interval of extract_field_entry (dt = 0.02 s, that of the recordings) is added to
            # every t_prev, as in the paper's simulations; the trajectory itself is sampled at 10 Hz.
            tmp = extract_field_entry(x, y, t, peaks_x_all[cid], peaks_y_all[cid], radius_all[cid])
            t_prev_vec, in_field = tmp['t_prev_vec'], tmp['in_field']
            t_prev_clipped = np.maximum(t_prev_vec, 0.0)  # overlapping fields can give negative t_prev
            proximity = np.exp(-t_prev_clipped / adapt_tau)
            # adaptation applies at every sample, inside and outside fields
            adaptFactor = 1.0 - adapt_beta * proximity
            FR_adapt_tmp = FR_noadapt_tmp * adaptFactor[mask_speed]
            FR_adapt[cid, mask_speed] = FR_adapt_tmp

            if calc_adapt_hm is not None:
                # adaptation term inside fields by running direction relative to the grid
                tmpTprevBy12dir = np.zeros(12)
                tmpOnesBy12dir = np.zeros(12)
                np.add.at(tmpTprevBy12dir, dirRelGrid_idx[(in_field > 0) & mask_speed], proximity[(in_field > 0) & mask_speed])
                np.add.at(tmpOnesBy12dir, dirRelGrid_idx[(in_field > 0) & mask_speed], 1)
                tprevby12dir[cid, :] = np.divide(tmpTprevBy12dir, tmpOnesBy12dir, out=np.zeros_like(tmpTprevBy12dir),
                                                 where=tmpOnesBy12dir > 0)
                tprev_HMness[cid] = scipy.stats.linregress(alignness[(in_field > 0) & mask_speed],
                                                           proximity[(in_field > 0) & mask_speed])[0]
                tprevbyAlignMis[cid, 0] = np.mean(proximity[(in_field > 0) & mask_speed & (alignness > 0)])
                tprevbyAlignMis[cid, 1] = np.mean(proximity[(in_field > 0) & mask_speed & (alignness < 0)])

    outTable = {}
    if calc_adapt_hm:
        outTable['tprevby12dir'] = tprevby12dir
        outTable['tprev_HMness'] = tprev_HMness
        outTable['tprevbyAlignMis'] = tprevbyAlignMis

    # contrast of the mean rate map between aligned and misaligned samples (location factor alone)
    rate_vec = rateVec_mean / n_cells
    locFactor_mean = np.zeros_like(moveSpeed)
    locFactor_mean[mask_speed] = rate_vec[pos_idx[mask_speed]]
    locFactor_align = np.mean(locFactor_mean[mask_speed & (alignness > 0)])
    locFactor_misalign = np.mean(locFactor_mean[mask_speed & (alignness < 0)])
    locFactor_HMness_mean = (locFactor_align - locFactor_misalign) / (locFactor_align + locFactor_misalign)
    if is_adapt and n_cells > 0:
        outTable["locFactor_HMness_mean"] = locFactor_HMness_mean

    output = {
        'FR_adapt': FR_adapt,
        'FR_noadapt': FR_noadapt,
        'locFactor': locFactor,
        'ratemap_mean': rate_vec.reshape(nBin_y, nBin_x),
        **outTable,
    }
    return output


def simulate_and_measure(par, base_seed=0, subsample=None, traj=None, gc_table=None,
                         fold_sym_list=FOLD_SYM_LIST, chunk=None):
    """Simulate one participant and estimate the hexadirectional modulation of its population rate.

    Parameters
    ----------
    par : dict
        Parameters (``hexmod.sim.pool.set_param``).
    base_seed : int
        Participant seed: seeds the trajectory (when ``traj`` is None) and the optional subsample.
    subsample : int, optional
        Also store the mean rate of this many randomly chosen cells.
    traj : pandas.DataFrame, optional
        Trajectory; generated with ``hexmod.sim.trajectory_v1`` and ``base_seed`` when omitted.
    gc_table : pandas.DataFrame
        The participant's cells.
    fold_sym_list : sequence of int
        Symmetries to estimate (1 to 7).
    chunk : int, optional
        Simulate the cells in chunks of this size and combine the chunks by cell-weighted sums
        (needs about 1 GB per 1,000 cells instead of 8 GB for 10,000). The result equals the
        single-call result to rounding (about 1e-15 relative).

    Returns
    -------
    dict
        ``FR_adapt``, ``FR_noadapt`` (moving samples) [Hz], population means over cells;
        ``FR_adapt_mean_move``, ``FR_noadapt_mean_move``; ``tprevby12dir`` (12,) and
        ``tprev_HMness``, means over cells; ``tprevbyAlignMis_nanmean`` (2,); ``ratemap_sum``
        (mean rate map of the cells); ``crossvalResult`` (one row per model, estimator, left-out
        block and symmetry: ``fold``, ``train_mask``, ``test_mask``, ``beta``, ``ori``,
        ``intercept``, ``rsq``, ``pval``, ``FRby12dir_test``, ``sub``, ``model``, ``estType``);
        ``noncrossResult`` (one row per model and estimator; rates by 12 directions relative to the
        true grid orientation); ``gridness_mean``; ``seed``; ``n_samples``; ``n_moving``;
        ``alpha``; ``tau``; ``foldSymList``.
    """
    out = {}
    if traj is None:
        traj = trajectory_v1.make_trajectory(par, base_seed=base_seed)

    n = len(gc_table)
    if chunk is None:
        outputs = simulate_grid_cells(gc_table, traj, par)
        FR_adapt_all, FR_noadapt_all = outputs['FR_adapt'], outputs['FR_noadapt']
        tprevby12dir = outputs["tprevby12dir"]
        tprev_HMness = outputs["tprev_HMness"]
        tprevbyAlignMis = outputs["tprevbyAlignMis"]
        ratemap_sum = outputs['ratemap_mean']
        if subsample is not None:  # mean rate of a random subset of cells
            rng = np.random.default_rng(base_seed)
            celllist = rng.choice(FR_adapt_all.shape[0], subsample, replace=False)
            out['FR_adapt_subsample'] = FR_adapt_all[celllist, :].mean(axis=0)
            out['FR_noadapt_subsample'] = FR_noadapt_all[celllist, :].mean(axis=0)
        FR_adapt = np.mean(FR_adapt_all, axis=0)
        FR_noadapt = np.mean(FR_noadapt_all, axis=0)
        del outputs, FR_adapt_all, FR_noadapt_all
    else:
        if subsample is not None:
            raise ValueError("subsample needs chunk=None")
        FR_adapt = FR_noadapt = ratemap_sum = None
        tprev12, tprevHM, tprevAM = [], [], []
        nseen = 0
        for c0 in range(0, n, chunk):
            o = simulate_grid_cells(gc_table.iloc[c0:c0 + chunk], traj, par)
            m = o["FR_adapt"].shape[0]
            sa = o["FR_adapt"].sum(axis=0)
            sn = o["FR_noadapt"].sum(axis=0)
            rm = o["ratemap_mean"] * m
            FR_adapt = sa if FR_adapt is None else FR_adapt + sa
            FR_noadapt = sn if FR_noadapt is None else FR_noadapt + sn
            ratemap_sum = rm if ratemap_sum is None else ratemap_sum + rm
            tprev12.append(np.asarray(o["tprevby12dir"]))
            tprevHM.append(np.asarray(o["tprev_HMness"]))
            tprevAM.append(np.asarray(o["tprevbyAlignMis"]))
            nseen += m
            del o
        FR_adapt = FR_adapt / nseen
        FR_noadapt = FR_noadapt / nseen
        ratemap_sum = ratemap_sum / nseen
        tprevby12dir = np.vstack(tprev12)
        tprev_HMness = np.concatenate(tprevHM)
        tprevbyAlignMis = np.vstack(tprevAM)

    # modulation of the population rate, adapted and not adapted, with both estimators
    crossval_rows = []
    noncross_rows = []
    for model_name, FR in zip(["adapt", "noadapt"], [FR_adapt, FR_noadapt]):
        for estType in ['simple', 'multiplicative']:
            if estType == 'simple':
                cross, noncross = calc_hm(FR, traj, par, fold_sym_list=fold_sym_list)
            if estType == 'multiplicative':
                cross, noncross = calc_hm_control_location(FR, traj, par, spatialResolution=5, fold_sym_list=fold_sym_list)
            cross['sub'] = base_seed
            cross['model'] = model_name
            cross['estType'] = estType
            noncross['sub'] = base_seed
            noncross['model'] = model_name
            noncross['estType'] = estType
            crossval_rows.append(cross)
            noncross_rows.append(noncross)
    crossvalResult = pd.concat(crossval_rows, ignore_index=True)
    noncrossResult = pd.concat(noncross_rows, ignore_index=True)

    mask_speed = traj['moveSpeed'] > par['speedThres']
    out['FR_adapt'] = FR_adapt[mask_speed]  # moving samples only (the rate is 0 elsewhere)
    out['FR_noadapt'] = FR_noadapt[mask_speed]
    out['tprev_HMness'] = np.mean(tprev_HMness)
    out['tprevby12dir'] = np.mean(tprevby12dir, axis=0)
    out['ratemap_sum'] = ratemap_sum
    out['crossvalResult'] = crossvalResult
    out['noncrossResult'] = noncrossResult
    out['gridness_mean'] = gc_table['gridness'].mean()
    out['seed'] = int(base_seed)
    out['FR_adapt_mean_move'] = FR_adapt[mask_speed].mean()
    out['FR_noadapt_mean_move'] = FR_noadapt[mask_speed].mean()
    # further summaries of the same run
    out['tprevbyAlignMis_nanmean'] = np.nanmean(tprevbyAlignMis, axis=0)
    out['n_samples'] = int(len(traj))
    out['n_moving'] = int(np.asarray(mask_speed).sum())
    out['alpha'] = par['adaptEff_mean']
    out['tau'] = par['adaptTau']
    out['foldSymList'] = list(fold_sym_list)
    return out


def draw_participant_cells(pool: pd.DataFrame, n_cells: int, seed: int) -> np.ndarray:
    """Pool rows of participant ``seed``: ``default_rng(seed).choice(len(pool), n_cells, replace=False)``."""
    rng = np.random.default_rng(seed)
    return rng.choice(len(pool), n_cells, replace=False)


def participant_cells(pool: pd.DataFrame, par: dict, seed: int) -> pd.DataFrame:
    """The ``par["nGridCells"]`` cells of participant ``seed`` (rows of ``pool``, in draw order)."""
    return pool.iloc[draw_participant_cells(pool, par['nGridCells'], seed)]


def simulate_participant(item) -> dict:
    """Simulate one participant; ``item`` is ``(par, gc_table, seed, chunk, fold_sym_list)``.

    ``gc_table`` holds the participant's cells (``participant_cells(pool, par, seed)``); the
    trajectory is ``trajectory_v1.make_trajectory(par, base_seed=seed)``. There is no other
    random draw. Defined at module level so that it can run in worker processes.
    """
    par, gc_table, seed, chunk, fold_sym_list = item
    return simulate_and_measure(par, base_seed=seed, gc_table=gc_table, fold_sym_list=fold_sym_list, chunk=chunk)


def simulate_many(par, seeds_list, pool, workers=None, chunk=1000, fold_sym_list=FOLD_SYM_LIST):
    """Simulate several participants (seeds ``seeds_list``) and return their results in seed order.

    Participant ``s`` uses the cells ``participant_cells(pool, par, s)`` and the trajectory of
    seed ``s`` (``simulate_participant``). Participants are sent to the workers in batches of
    ``workers``, so that only one batch of cell tables is held in memory.

    Parameters
    ----------
    par : dict
        Parameters with ``nGridCells`` cells per participant (10,000 in the paper).
    seeds_list : sequence of int
        Participant seeds.
    pool : pandas.DataFrame
        Cell pool.
    workers : int, optional
        Worker processes (default ``hexmod.config.workers()``).
    chunk : int, optional
        See ``simulate_and_measure``.
    fold_sym_list : sequence of int
        Symmetries to estimate.

    Returns
    -------
    list of dict
        ``simulate_and_measure`` results, one per seed, in the order of ``seeds_list``.
    """
    from hexmod import config
    from hexmod.parallel import map_ordered

    n_workers = config.workers(workers)
    seeds = [int(s) for s in seeds_list]
    results = []
    for b0 in range(0, len(seeds), n_workers):
        items = [(par, participant_cells(pool, par, s), s, chunk, tuple(fold_sym_list)) for s in seeds[b0:b0 + n_workers]]
        results.extend(map_ordered(simulate_participant, items, workers=n_workers, desc="participants"))
    return results


def quadratic_filter(dirData, neuralData, trainIdx, testIdx, foldSym=6):
    """Cross-validated k-fold directional modulation of a rate.

    On the training samples the rate is regressed on sin(k theta), cos(k theta) and a constant;
    the preferred orientation is ``atan2(b_sin, b_cos) / k`` in [0, 360 / k). On the test samples
    the rate is regressed on cos k(theta - orientation).

    Parameters
    ----------
    dirData : array_like
        Running direction [deg].
    neuralData : array_like
        Rate (same length).
    trainIdx, testIdx : array_like of bool or int
        Training and test samples.
    foldSym : int
        Symmetry k (6 for hexadirectional modulation).

    Returns
    -------
    ori : float
        Orientation [deg] from the training samples.
    beta : float
        Slope on the test samples (the modulation coefficient).
    intercept : float
        Intercept on the test samples.
    rsq : float
        R squared on the test samples.
    p_val : float
        p value of the slope on the test samples.
    """
    dirData = np.asarray(dirData)
    neuralData = np.asarray(neuralData)

    reg1 = np.sin(np.deg2rad(dirData[trainIdx] * foldSym))
    reg2 = np.cos(np.deg2rad(dirData[trainIdx] * foldSym))
    X = np.c_[reg1, reg2, np.ones_like(reg1)]
    beta, _, _, _ = np.linalg.lstsq(X, neuralData[trainIdx], rcond=None)
    b1 = beta[0]
    b2 = beta[1]
    ori = np.degrees(np.arctan2(b1, b2)) / foldSym
    if ori < 0:
        ori += 360 / foldSym

    reg3 = np.cos(np.deg2rad((dirData[testIdx] - ori) * foldSym))
    slope, intercept, r_val, p_val, stderr = scipy.stats.linregress(reg3, neuralData[testIdx])
    beta = slope
    rsq = r_val**2
    return ori, beta, intercept, rsq, p_val


def _miniblocks(blockIdx):
    """First and second half (1, 2) of every block (used by within-block cross-validation)."""
    miniBlockIdx = np.zeros_like(blockIdx)
    for b in np.unique(blockIdx):
        index_for_block = np.where(blockIdx == b)
        halfPoint = np.round(np.median(index_for_block)).astype(int)
        miniBlockIdx[(blockIdx == b) & (np.arange(len(blockIdx)) < halfPoint)] = 1
        miniBlockIdx[(blockIdx == b) & (np.arange(len(blockIdx)) >= halfPoint)] = 2
    return miniBlockIdx


def _splits(blockIdx, mask_speed, crossval_within, crossval_across):
    """Training and test masks with their labels."""
    train_masks, test_masks, train_mask_labels, test_mask_labels = [], [], [], []
    if crossval_within:
        miniBlockIdx = _miniblocks(blockIdx)
        for b in np.unique(blockIdx):
            for partIdx in [1, 2]:
                train_masks.append((blockIdx == b) & (miniBlockIdx == partIdx) & mask_speed)
                test_masks.append((blockIdx == b) & (miniBlockIdx != partIdx) & mask_speed)
                train_mask_labels.append(f'run{b}_{partIdx}')
                test_mask_labels.append(f'run{b}_{np.setdiff1d([1, 2], partIdx)[0]}')
    if crossval_across:
        # leave one block out
        for b in np.unique(blockIdx):
            train_masks.append((blockIdx != b) & mask_speed)
            test_masks.append((blockIdx == b) & mask_speed)
            train_mask_labels.append(f'leaveout_run{b}')
            test_mask_labels.append(f'run{b}')
    return train_masks, test_masks, train_mask_labels, test_mask_labels


def calc_hm(FR: np.ndarray, traj: pd.DataFrame, par, crossval_within=False, crossval_across=True,
            fold_sym_list=FOLD_SYM_LIST):
    """Hexadirectional (k-fold) modulation of a population rate, "simple" estimator.

    Parameters
    ----------
    FR : ndarray, shape (n_samples,) or (n_cells, n_samples)
        Rate (averaged over cells when 2-d).
    traj : pandas.DataFrame
        Trajectory with ``moveDir``, ``moveSpeed`` and ``blockIdx``.
    par : dict
        Parameters (``speedThres``, ``gridOri``).
    crossval_within, crossval_across : bool
        Split halves of each block, or leave one block out (the paper).
    fold_sym_list : sequence of int
        Symmetries.

    Returns
    -------
    crossvalResult : pandas.DataFrame
        One row per split and symmetry: ``fold``, ``train_mask``, ``test_mask``, ``beta``,
        ``ori``, ``intercept``, ``rsq``, ``pval``, ``FRby12dir_test`` (12 bins of the test block
        relative to the training orientation, for k = 6; None otherwise).
    noncrossResult : pandas.DataFrame
        One row: ``HM_beta_mean`` and ``intercept_mean`` (regression on cos 6(theta - gridOri)
        over all moving samples), ``FRby12dir`` and ``speedby12dir`` relative to the true grid.
    """
    if FR.ndim == 1:
        FR = FR[np.newaxis, :]
    FR_mean = FR.mean(axis=0)

    moveDir = traj["moveDir"].to_numpy()
    moveSpeed = traj["moveSpeed"].to_numpy()
    blockIdx = traj["blockIdx"].to_numpy()
    mask_speed = moveSpeed > par['speedThres']
    train_masks, test_masks, train_mask_labels, test_mask_labels = _splits(blockIdx, mask_speed, crossval_within,
                                                                           crossval_across)

    edges = np.arange(-30, 361, 30) + 15  # 12 bins centred on 0, 30, ..., 330 deg

    crossvalResult = []
    for train_mask, test_mask, train_mask_label, test_mask_label in zip(train_masks, test_masks, train_mask_labels,
                                                                        test_mask_labels):
        for foldSym in fold_sym_list:
            ori, beta, intercept, rsq, pval = quadratic_filter(moveDir, FR_mean, train_mask, test_mask, foldSym)

            if foldSym == 6:  # rate in 12 directions relative to the orientation from the training blocks
                dirRelGrid = (moveDir[test_mask] - ori) % 360.0
                dirRelGrid_idx = np.digitize(dirRelGrid, edges)
                dirRelGrid_idx[dirRelGrid_idx == 13] = 1
                sums = np.bincount(dirRelGrid_idx - 1, weights=FR_mean[test_mask], minlength=12)
                counts = np.bincount(dirRelGrid_idx - 1, minlength=12)
                FRby12dir_test = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
            else:
                FRby12dir_test = None

            crossvalResult.append({
                "fold": foldSym,
                "train_mask": train_mask_label,
                "test_mask": test_mask_label,
                "beta": beta,
                "ori": ori,
                "intercept": intercept,
                "rsq": rsq,
                "pval": pval,
                "FRby12dir_test": FRby12dir_test,
            })

    # all moving samples, relative to the true grid orientation
    gridOri = par['gridOri']
    c = np.cos(np.deg2rad(6 * (moveDir[mask_speed] - gridOri)))
    HM_beta_mean, intercept_mean, r_val, p_val, _ = scipy.stats.linregress(c, FR[:, mask_speed].mean(axis=0))

    dirRelGrid = (moveDir[mask_speed] - gridOri) % 360.0
    dirRelGrid_idx = np.digitize(dirRelGrid, edges)
    dirRelGrid_idx[dirRelGrid_idx == 13] = 1

    sums = np.bincount(dirRelGrid_idx - 1, weights=FR_mean[mask_speed], minlength=12)
    counts = np.bincount(dirRelGrid_idx - 1, minlength=12)
    FRby12dir = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)

    sums = np.bincount(dirRelGrid_idx - 1, weights=moveSpeed[mask_speed], minlength=12)
    counts = np.bincount(dirRelGrid_idx - 1, minlength=12)
    speedby12dir = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)

    noncrossResult = {
        "HM_beta_mean": HM_beta_mean,
        "intercept_mean": intercept_mean,
        "FRby12dir": FRby12dir,
        "speedby12dir": speedby12dir,
    }
    crossvalResult = pd.DataFrame(crossvalResult)
    noncrossResult = pd.DataFrame([noncrossResult])
    return crossvalResult, noncrossResult


def calc_hm_control_location(FR: np.ndarray, traj: pd.DataFrame, par, spatialResolution=5, degResolution=5,
                       crossval_within=False, crossval_across=True, fold_sym_list=FOLD_SYM_LIST):
    """Hexadirectional (k-fold) modulation of a population rate, location-controlled estimator.

    The mean rate in location bins (``spatialResolution`` vm) x direction bins (``degResolution``
    deg) is fitted by a rank-one product f(location) g(direction) (``fit_rank1_als``, weighted by
    the number of samples), separately in the training and test blocks; the k-fold regression of
    ``quadratic_filter`` is then applied to the direction factors g.

    Parameters and returns as in ``calc_hm``; ``noncrossResult`` holds only ``FRby12dir``, the
    direction factor of a rank-one fit with 12 direction bins relative to the true grid.
    """
    x, y, moveDir, moveSpeed = traj['x'], traj['y'], traj['moveDir'], traj['moveSpeed']

    if FR.ndim == 1:
        FR = FR[np.newaxis, :]
    FR = FR.mean(axis=0)

    if len(FR) != len(x):
        raise ValueError('FR and traj have different length')
    mask_speed = moveSpeed > par['speedThres']

    # location and direction bins
    pos_linear, nBin_x, nBin_y, _, _ = position_binning(x, y, spatialResolution)

    dir_bins = np.arange(-degResolution, 361, degResolution) + degResolution / 2  # bins centred on 0, 5, ... deg
    dir_bins_center = (dir_bins[:-2] + dir_bins[1:-1]) / 2
    nBin_dir = len(dir_bins) - 2
    dir_idx = np.digitize(moveDir, dir_bins)
    dir_idx[dir_idx == (len(dir_bins) - 1)] = 1  # 360 deg wraps to 0
    dir_idx = dir_idx - 1

    blockIdx = traj["blockIdx"].to_numpy()
    train_masks, test_masks, train_mask_labels, test_mask_labels = _splits(blockIdx, mask_speed, crossval_within,
                                                                           crossval_across)

    crossvalResult = []
    for train_mask, test_mask, train_mask_label, test_mask_label in zip(train_masks, test_masks, train_mask_labels,
                                                                        test_mask_labels):
        FR_locXdir = np.zeros((nBin_x * nBin_y, nBin_dir))  # summed rate per (location, direction)
        n_locXdir = np.zeros((nBin_x * nBin_y, nBin_dir))  # samples per (location, direction)
        np.add.at(FR_locXdir, (pos_linear[train_mask], dir_idx[train_mask]), FR[train_mask])
        np.add.at(n_locXdir, (pos_linear[train_mask], dir_idx[train_mask]), 1)
        meanFR_locXdir = np.divide(FR_locXdir, n_locXdir, out=np.ones_like(FR_locXdir, dtype=float) * np.nan,
                                   where=n_locXdir > 0)
        FR_loc_train, FR_dir_train = fit_rank1_als(meanFR_locXdir, n_locXdir)

        FR_locXdir = np.zeros((nBin_x * nBin_y, nBin_dir))
        n_locXdir = np.zeros((nBin_x * nBin_y, nBin_dir))
        np.add.at(FR_locXdir, (pos_linear[test_mask], dir_idx[test_mask]), FR[test_mask])
        np.add.at(n_locXdir, (pos_linear[test_mask], dir_idx[test_mask]), 1)
        meanFR_locXdir = np.divide(FR_locXdir, n_locXdir, out=np.ones_like(FR_locXdir, dtype=float) * np.nan,
                                   where=n_locXdir > 0)
        FR_loc_test, FR_dir_test = fit_rank1_als(meanFR_locXdir, n_locXdir)

        for foldSym in fold_sym_list:
            # orientation from the training direction factor
            reg1 = np.sin(np.deg2rad(dir_bins_center * foldSym))
            reg2 = np.cos(np.deg2rad(dir_bins_center * foldSym))
            X = np.c_[reg1, reg2, np.ones_like(reg1)]
            mask_tmp = np.isfinite(FR_dir_train)
            beta, _, _, _ = np.linalg.lstsq(X[mask_tmp], FR_dir_train[mask_tmp], rcond=None)
            b1 = beta[0]
            b2 = beta[1]
            ori = np.degrees(np.arctan2(b1, b2)) / foldSym
            if ori < 0:
                ori += 360 / foldSym

            # modulation of the test direction factor
            reg3 = np.cos(np.deg2rad((dir_bins_center - ori) * foldSym))
            mask_tmp = np.isfinite(FR_dir_test)
            slope, intercept, r_val, p_val, stderr = scipy.stats.linregress(reg3[mask_tmp], FR_dir_test[mask_tmp])
            beta = slope
            rsq = r_val**2

            if foldSym == 6:  # direction factor in 12 bins relative to the training orientation
                dirRelGrid = (moveDir - ori) % 360.0
                dirRelGrid_idx = np.digitize(dirRelGrid, np.arange(-30, 361, 30) + 15) - 1
                dirRelGrid_idx[dirRelGrid_idx == 12] = 0

                FR_locXdir = np.zeros((nBin_x * nBin_y, 12))
                n_locXdir = np.zeros((nBin_x * nBin_y, 12))
                np.add.at(FR_locXdir, (pos_linear[test_mask], dirRelGrid_idx[test_mask]), FR[test_mask])
                np.add.at(n_locXdir, (pos_linear[test_mask], dirRelGrid_idx[test_mask]), 1)
                meanFR_locXdir = np.divide(FR_locXdir, n_locXdir, out=np.ones_like(FR_locXdir, dtype=float) * np.nan,
                                           where=n_locXdir > 0)
                _, FR_dirRelGrid_test = fit_rank1_als(meanFR_locXdir, n_locXdir)
                FRby12dir_test = FR_dirRelGrid_test
            else:
                FRby12dir_test = None

            crossvalResult.append({
                "fold": foldSym,
                "train_mask": train_mask_label,
                "test_mask": test_mask_label,
                "beta": beta,
                "ori": ori,
                "intercept": intercept,
                "rsq": rsq,
                "pval": p_val,
                "FRby12dir_test": FRby12dir_test,
            })

    # all moving samples, 12 direction bins relative to the true grid orientation
    dirRelGrid = (moveDir - par['gridOri']) % 360.0
    dirRelGrid_idx = np.digitize(dirRelGrid, np.arange(-30, 361, 30) + 15) - 1
    dirRelGrid_idx[dirRelGrid_idx == 12] = 0

    FR_locXdir = np.zeros((nBin_x * nBin_y, 12))
    n_locXdir = np.zeros((nBin_x * nBin_y, 12))
    np.add.at(FR_locXdir, (pos_linear[mask_speed], dirRelGrid_idx[mask_speed]), FR[mask_speed])
    np.add.at(n_locXdir, (pos_linear[mask_speed], dirRelGrid_idx[mask_speed]), 1)
    meanFR_locXdir = np.divide(FR_locXdir, n_locXdir, out=np.ones_like(FR_locXdir, dtype=float) * np.nan,
                               where=n_locXdir > 0)
    _, FR_dirRelGrid = fit_rank1_als(meanFR_locXdir, n_locXdir)

    noncrossResult = {"FRby12dir": FR_dirRelGrid}

    crossvalResult = pd.DataFrame(crossvalResult)
    noncrossResult = pd.DataFrame([noncrossResult])
    return crossvalResult, noncrossResult
