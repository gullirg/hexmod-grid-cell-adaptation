"""Virtual-navigation trajectories, generator version 2 (used for Fig 6A, Fig 6D and Fig 8D).

The object-location task ("ObjLocTest") is the one of ``hexmod.sim.trajectory_v1``, with two
differences that change the random stream: candidate runs are drawn in rounds until there are at
least as many as the block needs (no candidate is discarded), and the balanced sampler only
balances across groups that still have rows. With ``set_param()`` and seed 0 the trajectory has
23,990 samples (version 1: 23,966).

Version 2 also provides "wall2wall" (Fig 8D): uninterrupted straight runs from wall to wall. Each
run starts where the previous one ended, in a random direction that leaves at least ``minLen``
to the opposite wall, and runs to that wall; ``dur_cue`` and ``dur_ITI`` set the pauses before
and after each run. Runs are added until the block duration is reached.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from hexmod.sim.trajectory_v1 import calc_speed_dir

__all__ = ["make_trajectory", "calc_speed_dir", "subsample_rows_balanced"]


def make_trajectory(par: dict, base_seed: int = 0) -> pd.DataFrame:
    """Generate the trajectory of one participant (all blocks).

    Parameters
    ----------
    par : dict
        Parameters from ``hexmod.sim.pool.set_param``; ``moveConstraint`` is "ObjLocTest" or
        "wall2wall".
    base_seed : int
        Seed of ``numpy.random.default_rng``; also seeds the start-position sampler.

    Returns
    -------
    pandas.DataFrame
        Columns as in ``hexmod.sim.trajectory_v1.make_trajectory``. For "wall2wall",
        ``trialId`` numbers the runs and ``expPhase`` is 1 during runs and -1 otherwise.
    """
    rng = np.random.default_rng(base_seed)
    dt = 0.1  # 10 Hz

    trajSub = []
    for block in range(par['nBlock']):
        moveSpeed = par['moveSpeed']
        env_w, env_h = par['envSize_x'], par['envSize_y']
        minLen, maxLen = par['minLen'], par['maxLen']

        meanTrialDur = par['dur_ITI'] + (par['minTurnDur'] + par['maxTurnDur']) / 2 + (par['minLen'] + par['maxLen']) / 2 / par['moveSpeed']
        nSegments = int(round(par['expDurBlock'] / meanTrialDur) * 1)
        dur_ITI = par['dur_ITI']
        dur_turn = par['minTurnDur'] + (par['maxTurnDur'] - par['minTurnDur']) * rng.random(nSegments)
        dur_cue = par['dur_cue']

        match par['moveConstraint']:
            case "ObjLocTest":
                objLocs = par['objLocs'][block]

                # candidate runs, drawn in rounds (three per object and 30 deg direction bin)
                # until there are at least nSegments
                tmplist = {'objID': [], 'dirID': [], 'startLoc': [], 'endLoc': []}
                while (len(tmplist['dirID']) < nSegments):
                    for objId in range(len(objLocs)):
                        for theta_base in np.tile(np.arange(0, 360, 30), 3):
                            angerror = rng.random() * 2 * np.pi
                            disterror_max = 20
                            obj = objLocs[objId] + rng.random() * disterror_max * np.array([np.cos(angerror), np.sin(angerror)])
                            margin = 0.02 * env_w
                            obj[0] = np.clip(obj[0], 0 + margin, env_w - margin)
                            obj[1] = np.clip(obj[1], 0 + margin, env_h - margin)
                            theta = theta_base + (rng.random() - 0.5) * 30
                            l_range_x = [-obj[0] / np.cos(np.deg2rad(theta)), (env_w - obj[0]) / np.cos(np.deg2rad(theta))]
                            l_range_y = [-obj[1] / np.sin(np.deg2rad(theta)), (env_h - obj[1]) / np.sin(np.deg2rad(theta))]
                            tmpmin = np.max([np.min(l_range_x), np.min(l_range_y), minLen])
                            tmpmax = np.min([np.max(l_range_x), np.max(l_range_y), maxLen])
                            if tmpmin <= tmpmax:
                                tmpLen = tmpmin + (tmpmax - tmpmin) * rng.random()
                                startLoc = obj + np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))]) * tmpLen
                                endLoc = obj
                                tmplist['startLoc'].append(startLoc)
                                tmplist['endLoc'].append(endLoc)
                                tmplist['objID'].append(objId)
                                tmplist['dirID'].append(theta_base)
                tmpdf = pd.DataFrame(tmplist)
                tmpdf = subsample_rows_balanced(tmpdf, ['objID', 'dirID'], nSegments, random_state=base_seed)
                startArr = tmpdf['startLoc']
                endArr = tmpdf['endLoc']

                feedbackLoc = objLocs[tmpdf['objID']]
                dist2recall = np.linalg.norm(np.vstack(endArr) - np.vstack(startArr), axis=1)
                dist2feedback = np.linalg.norm(np.vstack(endArr) - feedbackLoc, axis=1)
                timeVec = [0.0]
                posVec = [startArr[0]]
                trial_on, trial_off = np.zeros(tmpdf.shape[0]), np.zeros(tmpdf.shape[0])
                move_on2, move_off2 = np.zeros(tmpdf.shape[0]), np.zeros(tmpdf.shape[0])
                turn_on, move_on, move_off = np.zeros(tmpdf.shape[0]), np.zeros(tmpdf.shape[0]), np.zeros(tmpdf.shape[0])
                for i in range(tmpdf.shape[0]):
                    if par['moveSpeedVariability'] > 0:
                        moveSpeed = par['moveSpeed'] * (1 + 2 * (rng.random() - 0.5) * par['moveSpeedVariability'])
                    dur_move = dist2recall[i] / moveSpeed
                    dur_move2 = dist2feedback[i] / moveSpeed
                    trial_on[i] = timeVec[-1]
                    turn_on[i] = trial_on[i] + dur_cue
                    move_on[i] = turn_on[i] + dur_turn[i]
                    move_off[i] = move_on[i] + dur_move
                    move_on2[i] = move_off[i] + 1
                    move_off2[i] = move_on2[i] + dur_move2
                    trial_off[i] = move_off2[i] + dur_ITI
                    # cue (hold position); the extra 0.1 s point makes the jump between trials abrupt
                    timeVec.append(timeVec[-1] + 0.1)
                    posVec.append(startArr[i])
                    timeVec.append(timeVec[-1] + dur_cue - 0.1)
                    posVec.append(startArr[i])
                    # turn (hold position)
                    timeVec.append(timeVec[-1] + dur_turn[i])
                    posVec.append(startArr[i])
                    # run to the remembered location
                    timeVec.append(timeVec[-1] + dur_move)
                    posVec.append(endArr[i])
                    # turn towards the object (feedback)
                    timeVec.append(timeVec[-1] + 1)
                    posVec.append(endArr[i])
                    # run to the object
                    timeVec.append(timeVec[-1] + dur_move2)
                    posVec.append(feedbackLoc[i])
                    # inter-trial interval (hold position)
                    timeVec.append(timeVec[-1] + dur_ITI)
                    posVec.append(feedbackLoc[i])

                timeVec = np.array(timeVec)
                posVec = np.vstack(posVec)
                if np.max(timeVec) > par['expDurBlock']:
                    validIdx = timeVec <= par['expDurBlock']  # crop to the block duration
                    timeVec = timeVec[validIdx]
                    posVec = posVec[validIdx, :]
                if np.max(timeVec) < par['expDurBlock'] * 0.9:
                    warnings.warn(f"Generated trajectory is shorter than expected: {timeVec.max():.2f} s")
                t_new = np.arange(0.0, timeVec.max() + 1e-9, dt)
                x_new = np.interp(t_new, timeVec, posVec[:, 0])
                y_new = np.interp(t_new, timeVec, posVec[:, 1])
                trialId = np.zeros_like(t_new, dtype=int)
                expPhase = np.zeros_like(t_new, dtype=int)
                for i in range(len(trial_on)):
                    tidx = (t_new >= trial_on[i]) & (t_new < trial_off[i])
                    trialId[tidx] = i + 1
                    tidx = (t_new >= trial_on[i]) & (t_new < turn_on[i])
                    expPhase[tidx] = -1  # cue
                    tidx = (t_new >= turn_on[i]) & (t_new < move_on[i])
                    expPhase[tidx] = 0  # turn
                    tidx = (t_new >= move_on[i]) & (t_new < move_off[i])
                    expPhase[tidx] = 1  # run
                    tidx = (t_new >= move_off[i]) & (t_new < move_on2[i])
                    expPhase[tidx] = 0  # turn
                    tidx = (t_new >= move_on2[i]) & (t_new < move_off2[i])
                    expPhase[tidx] = 1  # run to the object
                    tidx = (t_new >= move_off2[i]) & (t_new < trial_off[i])
                    expPhase[tidx] = -1  # inter-trial interval

            case "wall2wall":
                # straight runs from wall to wall, each starting where the previous one ended
                timeVec = [0]
                posVec = [np.array([0, 0])]
                move_on = []
                move_off = []
                while timeVec[-1] < par['expDurBlock']:
                    startLoc = posVec[-1]
                    x = startLoc[0]
                    y = startLoc[1]
                    n_try = 0
                    tmpLen = 0
                    while (n_try < 100) & (tmpLen < par['minLen']):
                        n_try = n_try + 1
                        theta = rng.random() * 360
                        if np.cos(np.deg2rad(theta)) == 0:
                            tmpLen = env_w
                        elif np.sin(np.deg2rad(theta)) == 0:
                            tmpLen = env_h
                        else:
                            l_range_x = [-x / np.cos(np.deg2rad(theta)), (env_w - x) / np.cos(np.deg2rad(theta))]
                            l_range_y = [-y / np.sin(np.deg2rad(theta)), (env_h - y) / np.sin(np.deg2rad(theta))]
                            tmpLen = np.min([np.max(l_range_x), np.max(l_range_y)])  # distance to the wall ahead
                        if (tmpLen > par['minLen']):
                            endLoc = startLoc + np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))]) * tmpLen
                            moveDur = tmpLen / moveSpeed
                            posVec.append(startLoc)
                            timeVec.append(timeVec[-1] + dur_cue)
                            move_on.append(timeVec[-1])
                            posVec.append(endLoc)
                            timeVec.append(timeVec[-1] + moveDur)
                            move_off.append(timeVec[-1])
                            posVec.append(endLoc)
                            timeVec.append(timeVec[-1] + dur_ITI)
                    if n_try >= 100:
                        warnings.warn(f"Could not find a valid move after {n_try} attempts; stopping.")
                        break

                posVec = np.vstack(posVec)
                timeVec = np.array(timeVec)
                t_new = np.arange(0.0, np.max(timeVec) + 1e-9, dt)
                x_new = np.interp(t_new, timeVec, posVec[:, 0])
                y_new = np.interp(t_new, timeVec, posVec[:, 1])
                trialId = np.zeros_like(t_new, dtype=int)
                expPhase = np.ones_like(t_new, dtype=int) * -1
                for i in range(len(move_on)):
                    tidx = (t_new >= move_on[i]) & (t_new < move_off[i])
                    trialId[tidx] = i + 1
                    tidx = (t_new >= move_on[i]) & (t_new < move_off[i])
                    expPhase[tidx] = 1  # run
            case _:
                raise ValueError(f"Unknown moveConstraint: {par['moveConstraint']}")

        # 0.4 s boxcar smoothing, as for the recorded trajectories
        smooth_window = int(0.4 / dt)
        x_unsmooth = x_new.copy()
        y_unsmooth = y_new.copy()
        if smooth_window > 0:
            x_new = uniform_filter1d(x_new, size=smooth_window)
            y_new = uniform_filter1d(y_new, size=smooth_window)
        # hide cue, inter-trial and pause periods
        x_new[expPhase == -1] = np.nan
        y_new[expPhase == -1] = np.nan

        traj = {"time": t_new, "x_unsmooth": x_unsmooth, "y_unsmooth": y_unsmooth, "x": x_new, "y": y_new,
                "trialId": trialId, "expPhase": expPhase}
        traj = calc_speed_dir(traj)
        traj['moveSpeed'][expPhase == -1] = np.nan
        traj['moveDir'][expPhase == -1] = np.nan
        traj['moveDir'][traj['moveSpeed'] == 0] = np.nan
        traj['blockIdx'] = block + 1
        traj['time'] = traj['time'] + block * (par['expDurBlock'] + 1000)  # 1000 s between blocks
        trajSub.append(pd.DataFrame(traj))

    return pd.concat(trajSub, ignore_index=True)


def subsample_rows_balanced(df: pd.DataFrame, group_columns, n_rows: int, random_state=None) -> pd.DataFrame:
    """Draw ``n_rows`` rows of ``df`` balanced over the groups of ``group_columns``.

    Rows are shuffled within each group, then the group order is shuffled. Each draw takes the
    next row of a group that still has rows and has been used least so far (ties: the group with
    most rows left), so groups that run out no longer constrain the balance.

    Parameters
    ----------
    df : pandas.DataFrame
    group_columns : str or list of str
        Grouping columns (groups in order of first appearance; missing values form a group).
    n_rows : int
        Number of rows to draw (at most ``len(df)``).
    random_state : int, optional
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    pandas.DataFrame
        The drawn rows with a new RangeIndex.
    """
    if n_rows < 0:
        raise ValueError("n_rows must be non-negative.")

    rng = np.random.default_rng(random_state)

    grouped = df.groupby(group_columns, sort=False, dropna=False)
    group_rows = {
        group: rng.permutation(rows.index.to_numpy())
        for group, rows in grouped
    }

    if not group_rows or n_rows == 0:
        return df.iloc[0:0].copy().reset_index(drop=True)

    group_keys = list(group_rows)
    rng.shuffle(group_keys)

    pointers = {group: 0 for group in group_keys}  # next unused row of each group
    used_counts = {group: 0 for group in group_keys}  # rows drawn from each group
    selected_indices = []

    for _ in range(min(n_rows, len(df))):
        available = [
            group
            for group in group_keys
            if pointers[group] < len(group_rows[group])
        ]
        if not available:
            break

        # balance only across groups that can still supply a row
        min_used = min(used_counts[group] for group in available)
        candidates = [
            group for group in available
            if used_counts[group] == min_used
        ]

        # among them, the group with most rows left
        remaining_counts = {group: len(group_rows[group]) - pointers[group] for group in candidates}
        chosen_group = max(remaining_counts, key=remaining_counts.get)

        pointer = pointers[chosen_group]
        selected_indices.append(group_rows[chosen_group][pointer])

        pointers[chosen_group] += 1
        used_counts[chosen_group] += 1

    return df.loc[selected_indices].reset_index(drop=True)
