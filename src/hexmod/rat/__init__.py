"""Analyses of the grid cell recordings.

Modules:

- ``gridcell``: speed-filtered samples of one cell (``GridCellAnalysis``) and its rate map.
- ``maps``: autocorrelograms, gridness, scale and orientation, direction tuning, lattice fits.
- ``fields``: field delineation, field visits and the time since the last field visit (t_prev).
- ``characterise``: grid measures, fields and direction tuning of one cell.
- ``adaptation``: non-parametric and parametric fits of firing rate against t_prev.
- ``hexmod_fit``: hexadirectional modulation controlling for location and speed.
- ``prospective``: spatial information with time-shifted spikes.
- ``poisson``: multiplicative Poisson model fits used by the modules above.
- ``visit_sequences``: firing rate on successive field visits (Fig 3F).
- ``timescales``: adaptation timescales (Supp Fig 4) and numbers quoted in the Methods.
- ``conjunctive``: preferred movement direction relative to the grid axes (Fig 2B to 2E).
- ``same_cell``: repeated recordings of the same unit.
- ``examples``: arrays of the example cells.
- ``paper_stats``, ``figure_tables``: printed statistics and figure tables of the rat figures.

Per-cell functions take one dictionary (``inputs``) holding ``cell_data`` (see
``gridcell.make_cell_data``) and the settings, so that they can be run over many cells with
``hexmod.parallel.map_ordered``. The options ``rate_map_spike_mode`` and ``tprev_rule`` select
the visit extraction of Fig 3F (see ``visit_sequences``), and ``mask_valid`` the sample mask of
Supp Fig 4 (see ``timescales``); their defaults are the settings of the other analyses.
"""
