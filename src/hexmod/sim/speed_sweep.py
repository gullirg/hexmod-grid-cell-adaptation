"""One run of the running-speed sweep of Fig 8D (worker function of ``scripts/step26_speed_sweep.py``).

A module of perfect grid cells is simulated with adaptation along straight wall-to-wall runs at one
running speed, and the run is reduced to the population rate by direction relative to the grid and
the in-field proximity for aligned and misaligned running.
"""

from __future__ import annotations

import numpy as np

__all__ = ["RUN_LENGTH", "sweep_run"]

#: Length of each straight wall-to-wall run (vm).
RUN_LENGTH = 70


def sweep_run(item) -> dict:
    """Simulate one run of the sweep and reduce it to the values of Fig 8D.

    Parameters
    ----------
    item : tuple
        ``(run, cells, par)``: the run description (``scale``, ``scaleLevel``, ``speedLevel``, ``moveSpeed``;
        the trajectory seed is ``speedLevel``), the pool of the module and its simulation parameters.

    Returns
    -------
    dict
        The run description plus ``FRby12dir`` (12,), ``FR_align``, ``FR_misalign`` (Hz), ``proximity_align``,
        ``proximity_misalign`` and ``n_time`` (samples).
    """
    from . import simulate, trajectory_v2
    from .pool import set_param

    run, cells, par = item
    par_traj = set_param(dict(minLen=RUN_LENGTH, maxLen=RUN_LENGTH, moveSpeed=run["moveSpeed"], dur_ITI=0, dur_cue=0,
                              moveConstraint="wall2wall"))
    traj = trajectory_v2.make_trajectory(par_traj, base_seed=run["speedLevel"])
    res = simulate.simulate_grid_cells({"gcTable": cells, "traj": traj, "par": par})
    _, noncross = simulate.calc_hm(res["FR_adapt"].mean(axis=0), traj, par, crossval_across=False)
    rate_by_dir = np.asarray(noncross["FRby12dir"][0], float)
    prox = np.asarray(res["tprevbyAlignMis"], float).mean(axis=0)
    return dict(run, FRby12dir=rate_by_dir, FR_align=float(rate_by_dir[::2].mean()),
                FR_misalign=float(rate_by_dir[1::2].mean()), proximity_align=float(prox[0]),
                proximity_misalign=float(prox[1]), n_time=int(len(traj)))
