"""Builder of the large-module (70 vm) pool used in the paper.

The 100,000-cell large-module pool (Fig 6C, Fig 7, Supp Fig 3E) is shipped as
``inputs/simulation/large_module_pool_70cm.npz``. This module rebuilds it. The cells follow the
same model as ``hexmod.sim.pool.set_gridcells_pool`` (same lattice, jitter and rate distributions,
same gridness criterion), but the builder differs in ways that change its random stream, so the
two builders give different pools for the same seed:

- grid offsets and rate-map bins come from ``meshgrid(indexing="xy")``, and the rate map is laid
  out (y, x); fields are delineated on that map;
- a cell is redrawn until its gridness reaches the threshold, without an iteration limit, and the
  number of draws is stored (``cellcount``);
- after each cell two more standard normal values are drawn, for the per-cell adaptation
  strength and speed gain (``adaptEff``, ``speedGamma``; both have s.d. 0 here, so the columns are
  constant and the simulations do not use them).

With ``large_pool_param()`` and base seed 0 (``LARGE_POOL_N_CELLS`` cells), ``set_gridcells_pool_large``
reproduces the shipped pool: delineated peaks and radii, draw counts and grid-scale estimates
exactly, field centres to 3e-14 vm, field rates to 2e-16 Hz and gridness to 1e-11. These last-digit
differences come from the floating-point rounding of the numerical libraries of the machine that
built the shipped pool. The grid-orientation estimate ``gridori_est``, which no analysis uses,
differs by up to 2.5 deg in 11% of the cells.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hexmod.rat.fields import delineate_fields
from hexmod.rat.maps import compute_gridness_scale_ori, spatial_autocorrelogram_fft2d
from hexmod.sim.pool import calc_fr_from_grid_fields, generate_rotated_triangular_lattice, set_param

__all__ = ["LARGE_POOL_OVERRIDES", "LARGE_POOL_N_CELLS", "LARGE_POOL_BASE_SEED", "large_pool_param",
           "set_gridcells_pool_large", "LARGE_POOL_COLUMNS"]

#: Columns of a pool built by ``set_gridcells_pool_large``, in order.
LARGE_POOL_COLUMNS = ("centers", "peaks_x", "peaks_y", "radius", "adaptEff", "speedGamma", "adaptTau", "gridness",
                       "cellcount", "gridscale_est", "gridori_est", "peakFR_jitter")

#: Parameters of the shipped large-module pool (other keys take the ``set_param`` defaults).
#: The rate-map and field parameters equal ``hexmod.sim.pool.MODULE_PARAMS["large"]``; adaptEff_mean
#: and speedGamma_mean only fill the constant ``adaptEff`` (0.04) and ``speedGamma`` (0.5) columns.
LARGE_POOL_OVERRIDES = dict(nSub=1, nGridCells=32 ** 2, gridScale=70, peakCoords_shape=2.1, peakCoords_scale=0.04,
                              FR_acrossCell_shape=3.4, FR_acrossCell_scale=1.1, FR_withinCell_shape=7.2,
                              FR_withinCell_scale=0.10, adaptEff_mean=0.04, speedGamma_mean=0.5, maxLen=100 * 0.7)

#: Size and seed of the shipped pool.
LARGE_POOL_N_CELLS = 100000
LARGE_POOL_BASE_SEED = 0


def large_pool_param() -> dict:
    """Parameters (``set_param`` dict) of the shipped large-module pool."""
    return set_param(LARGE_POOL_OVERRIDES)


def set_gridcells_pool_large(par: dict, n_cells: int, base_seed: int, progress: bool = False) -> pd.DataFrame:
    """Build a pool of simulated grid cells with the large-module builder.

    Parameters
    ----------
    par : dict
        Parameters (``large_pool_param()`` for the shipped pool).
    n_cells : int
        Number of cells (100,000 for the shipped pool).
    base_seed : int
        Seed of ``numpy.random.default_rng`` (0 for the shipped pool).
    progress : bool
        Show a progress bar (tqdm).

    Returns
    -------
    pandas.DataFrame
        One row per cell with the columns of ``LARGE_POOL_COLUMNS``; ``centers``,
        ``peakFR_jitter``, ``peaks_x``, ``peaks_y``, ``radius``, ``gridness``, ``gridscale_est``
        and ``gridori_est`` have the meaning of ``hexmod.sim.pool.set_gridcells_pool``.
    """
    rng = np.random.default_rng(base_seed)
    tmpx, tmpy = np.meshgrid(par['xBins_center'], par['yBins_center'], indexing="xy")
    basePos = np.column_stack([tmpx.ravel(), tmpy.ravel()])  # (N, 2)

    nOffset = int(np.ceil(np.sqrt(n_cells)))
    u = np.linspace(0, 1, nOffset, endpoint=False)
    offx, offy = np.meshgrid(u, u, indexing="xy")
    gridOffsets = np.column_stack([offx.ravel(), offy.ravel()])
    meanFR_all = rng.gamma(shape=par['FR_acrossCell_shape'], scale=par['FR_acrossCell_scale'], size=n_cells)
    rows = []

    cells = range(n_cells)
    if progress:
        from tqdm import tqdm
        cells = tqdm(cells, desc="cell", total=n_cells)
    for cid in cells:
        peakCoords = generate_rotated_triangular_lattice(par['envSize_x'], par['envSize_y'], par['gridScale'],
                                                         par['gridOri'], tuple(gridOffsets[cid]))
        K = peakCoords.shape[0]

        gridness = -1.0
        gridscale_est = np.nan
        gridori_est = np.nan
        cellcount = 0
        while (gridness < par['minGridness']):
            cellcount += 1
            match par['gridtype']:
                case "perfectGrid":
                    jitter_vec = 0
                    peakFR_jitter = None
                    gridness = 10
                case "inhomogeneousFR":
                    jitter_vec = 0
                    peakFR_jitter = meanFR_all[cid] * rng.gamma(shape=par['FR_withinCell_shape'],
                                                                scale=par['FR_withinCell_scale'], size=(K,))
                    gridness = 10
                case "spatialDeviation":
                    peakFR_jitter = None
                    peakCoords_jitter_mag = par['gridScale'] * rng.gamma(shape=par['peakCoords_shape'],
                                                                         scale=par['peakCoords_scale'], size=(K,))
                    angles = rng.uniform(0, 2 * np.pi, size=K)
                    jitter_vec = (peakCoords_jitter_mag * 1)[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])
                    gridness = 10
                case "realistic":
                    peakCoords_jitter_mag = par['gridScale'] * rng.gamma(shape=par['peakCoords_shape'],
                                                                         scale=par['peakCoords_scale'], size=(K,))
                    angles = rng.uniform(0, 2 * np.pi, size=K)
                    jitter_vec = (peakCoords_jitter_mag * 1)[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])

                    peakFR_jitter = meanFR_all[cid] * rng.gamma(shape=par['FR_withinCell_shape'],
                                                                scale=par['FR_withinCell_scale'], size=(K,))
                    gridness = 10
                case _:
                    raise ValueError(f"Unknown gridtype: {par['gridtype']}")
            grid_sigma = par['fieldSigma'] * par['gridScale']

            centers = peakCoords + jitter_vec
            FR_vec = calc_fr_from_grid_fields(centers, basePos, grid_sigma, peakFR_jitter)
            ratemap = FR_vec.reshape(len(par['yBins_center']), len(par['xBins_center']))  # (y, x)

            if par['gridtype'] == "realistic" or par['gridtype'] == "spatialDeviation":
                autocorr = spatial_autocorrelogram_fft2d(ratemap[::2, ::2])
                try:
                    gridness, gridscale_est, gridori_est = compute_gridness_scale_ori(
                        autocorr, rate_map_bin_size=par['binSize_ratemap'] * 2)
                except Exception:
                    # kept with the placeholder gridness 10 (no cell of the shipped pool took this path)
                    pass

        gridscaleInBins = par['gridScale'] / par['binSize_ratemap']
        min_dist_between_peak = gridscaleInBins * 1 / 2
        max_radius = min_dist_between_peak / 2
        min_radius = gridscaleInBins / 8
        peaks_y, peaks_x, radius = delineate_fields(ratemap, min_dist_between_peak, max_radius, min_radius,
                                                   row_coord=par['yBins_center'], col_coord=par['xBins_center'])

        # the two normal draws are made in this order (adaptEff, then speedGamma) for every cell
        adaptEff = par['adaptEff_mean'] + par['adaptEff_std'] * rng.normal()
        speedGamma = par['speedGamma_mean'] + par['speedGamma_std'] * rng.normal()
        rows.append(
            {
                "centers": centers,
                "peaks_x": peaks_x,
                "peaks_y": peaks_y,
                "radius": radius,
                "adaptEff": adaptEff,
                "speedGamma": speedGamma,
                "adaptTau": par['adaptTau'],
                "gridness": gridness,
                "cellcount": cellcount,
                "gridscale_est": gridscale_est,
                "gridori_est": gridori_est,
                "peakFR_jitter": peakFR_jitter,
            }
        )
    gcTable = pd.DataFrame(rows)
    return gcTable
