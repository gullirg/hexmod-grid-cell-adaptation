"""Simulated grid cells with firing rate adaptation (Figs 1, 6, 7 and 8, Supp Figs 3E and 5).

Modules:

``pool``
    Simulation parameters (``set_param``, ``MODULE_PARAMS``) and pools of simulated grid cells
    (``set_gridcells_pool``; the small-module pool, 100,000 cells, base seed 0).
``pool_large``
    Builder of the shipped large-module pool (70 vm, 100,000 cells, base seed 0).
``trajectory_v1``, ``trajectory_v2``
    Virtual-navigation trajectories. The two versions draw random numbers differently; version 1
    gives the Fig 7 participants, version 2 gives Fig 6A, Fig 6D and the speed sweep of Fig 8D.
``simulate``
    Firing rates along a trajectory with and without adaptation (``simulate_grid_cells``), simulated
    participants (``simulate_and_measure``, ``simulate_many``) and the two estimators of
    hexadirectional modulation (``calc_hm``, ``calc_hm_control_location``).
``analytic``
    Closed forms: proximity against running speed (Fig 8B, 8C) and the six-fold component of
    summed von Mises tuning curves (Fig 1, Supp Fig 5).
``speed_sweep``
    One run of the running-speed sweep of Fig 8D (``sweep_run``).
``io_pool``
    Pool tables to npz files and back, and column checksums of a pool.

Units: virtual metres (vm), seconds, Hz, degrees. Random numbers come from
``numpy.random.default_rng`` with the seeds given in each function.
"""
