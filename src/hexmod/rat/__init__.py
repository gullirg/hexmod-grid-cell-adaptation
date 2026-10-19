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

Per-cell functions take one dictionary (``inputs``) holding ``cell_data`` (see
``gridcell.make_cell_data``) and the settings, so that they can be run over many cells with
``hexmod.parallel.map_ordered``. Defaults reproduce the published analyses; the documented
options (``rate_map_spike_mode``, ``tprev_rule``, ``mask_valid``) select the variants named in
docs/PIPELINE.md for Fig 3F and Supp Fig 4.
"""
