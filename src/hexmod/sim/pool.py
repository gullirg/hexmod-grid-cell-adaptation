"""Simulation parameters and pools of simulated grid cells.

A simulated grid cell is a set of Gaussian firing fields placed on a triangular lattice with
spacing ``gridScale`` (virtual metres, vm) in a 100 x 100 vm environment. In the "realistic" cells
used in the paper every field centre is displaced from the lattice by a distance drawn from a
gamma distribution (``peakCoords_shape``, ``peakCoords_scale``, in units of the grid scale) in a
uniformly random direction, and every field has its own peak rate: the cell's mean rate (gamma
``FR_acrossCell_*``) times a field factor (gamma ``FR_withinCell_*``). The gamma parameters come
from fits to the recorded grid cells (Supp Fig 3B to 3D; see ``MODULE_PARAMS``). A cell is redrawn
until the gridness of its rate map reaches ``minGridness`` (0.3).

``set_gridcells_pool`` builds a pool of such cells (100,000 per module in the paper, base seed 0);
each simulated participant draws 10,000 cells from it (``hexmod.sim.simulate``). The grid phase of
cell ``i`` is fixed: the offsets form a regular ``ceil(sqrt(n)) x ceil(sqrt(n))`` lattice over one
unit cell, so only field jitter and rates are random. The pipeline rebuilds the small-module pool
(40 vm) with this function. The large-module pool (70 vm) is shipped in ``inputs/simulation`` and
can be rebuilt with ``hexmod.sim.pool_large``, a variant of this builder with a different
random stream.

Rate maps follow the convention of the recorded data: first axis x, second axis y (3 vm bins).
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

from hexmod.rat.fields import delineate_fields
from hexmod.rat.maps import compute_gridness_scale_ori, spatial_autocorrelogram_fft2d

__all__ = ["set_param", "MODULE_PARAMS", "module_param", "set_gridcells_pool",
           "generate_rotated_triangular_lattice", "calc_fr_from_grid_fields", "ratemap_from_centers",
           "POOL_COLUMNS"]

#: Columns of a pool table, in order.
POOL_COLUMNS = ("peaks_x", "peaks_y", "radius", "gridness", "gridscale_est", "gridori_est", "centers",
                "peakFR_jitter")

#: Field-placement and firing-rate parameters of the two simulated modules: gamma shape and scale
#: from fits to the recorded cells of small and large grid scale (Supp Fig 3B to 3D). For the large
#: module the field-rate factor uses shape 7.2 and scale 0.10 (mean 0.72), the values of the
#: simulations in the paper; the fit gives 7.10 and 0.141 (mean 1). Both have a coefficient of
#: variation of 0.37, so the difference mainly rescales the field rates of the module.
MODULE_PARAMS = {
    "small": dict(gridScale=40, peakCoords_shape=2.0, peakCoords_scale=0.07, FR_acrossCell_shape=3.2,
                  FR_acrossCell_scale=1.4, FR_withinCell_shape=10.3, FR_withinCell_scale=0.10),
    "large": dict(gridScale=70, peakCoords_shape=2.1, peakCoords_scale=0.04, FR_acrossCell_shape=3.4,
                  FR_acrossCell_scale=1.1, FR_withinCell_shape=7.2, FR_withinCell_scale=0.10),
}


def set_param(overrides: Optional[Mapping] = None) -> dict:
    """Return the full parameter dict of a simulation, with defaults for every key.

    Parameters
    ----------
    overrides : mapping, optional
        Values that replace the defaults. Unknown keys raise ``ValueError``.

    Returns
    -------
    dict
        Parameters (environment, trajectory, grid cells, adaptation) plus the derived bin edges
        and centres (``xBins``, ``yBins``, ``xBins_center``, ``yBins_center``, ``dirBins``,
        ``dirBins_center``) and the object locations of the four trajectory blocks (``objLocs``).
    """
    overrides = {} if overrides is None else overrides
    param = {}
    param['nSub'] = overrides.get('nSub', 30)
    param['nBlock'] = overrides.get('nBlock', 4)
    param['envSize_x'] = overrides.get('envSize_x', 100.0)
    param['envSize_y'] = overrides.get('envSize_y', 100.0)
    # trajectory
    param['expDurBlock'] = overrides.get('expDurBlock', 600)  # block duration (s)
    param['dur_ITI'] = overrides.get('dur_ITI', 2.0)
    param['dur_cue'] = overrides.get('dur_cue', 1.0)
    param['moveSpeed'] = overrides.get('moveSpeed', 100.0 / 6.0)  # vm/s: crossing the environment takes 6 s
    param['moveSpeedVariability'] = overrides.get('moveSpeedVariability', 0)  # +/- fraction of the speed per trial
    param['minLen'] = overrides.get('minLen', 100.0 * 0.4)
    param['maxLen'] = overrides.get('maxLen', 100.0 * 0.7)
    param['minTurnDur'] = overrides.get('minTurnDur', 0.8)
    param['maxTurnDur'] = overrides.get('maxTurnDur', 3.2)
    param['moveConstraint'] = overrides.get('moveConstraint', "ObjLocTest")
    param['objLocs'] = overrides.get('objLocs', None)
    # grid cells
    param['gridScale'] = overrides.get('gridScale', 0.4 * 100.0)
    param['nGridCells'] = overrides.get('nGridCells', 32 ** 2)
    param['gridOri'] = overrides.get('gridOri', 0.0)
    param['binSize_ratemap'] = overrides.get('binSize_ratemap', 3.0)
    param['minGridness'] = overrides.get('minGridness', 0.3)
    param['fieldSigma'] = overrides.get('fieldSigma', 1.0 / 6.0)  # field s.d. as a fraction of the grid scale
    param['gridtype'] = overrides.get('gridtype', "realistic")
    # field jitter and firing-rate variability of "realistic" cells (gamma shape and scale)
    param['FR_acrossCell_shape'] = overrides.get('FR_acrossCell_shape', 3.26)  # mean rate of a cell
    param['FR_acrossCell_scale'] = overrides.get('FR_acrossCell_scale', 1.70)
    param['FR_withinCell_shape'] = overrides.get('FR_withinCell_shape', 8.75)  # field rate / mean rate of the cell
    param['FR_withinCell_scale'] = overrides.get('FR_withinCell_scale', 0.11)
    param['peakCoords_shape'] = overrides.get('peakCoords_shape', 1.9)  # field displacement / grid scale
    param['peakCoords_scale'] = overrides.get('peakCoords_scale', 0.07)
    # adaptation and speed modulation
    param['adaptEff_mean'] = overrides.get('adaptEff_mean', 0.10)
    param['adaptEff_std'] = overrides.get('adaptEff_std', 0)
    param['adaptTau'] = overrides.get('adaptTau', 2.0)
    param['speedGamma_mean'] = overrides.get('speedGamma_mean', 0)
    param['speedGamma_std'] = overrides.get('speedGamma_std', 0)
    param['speedThres'] = overrides.get('speedThres', 5.0)  # samples faster than this are analysed (vm/s)
    # other cell types (not used in the paper's simulations)
    param['nHDCells'] = overrides.get('nHDCells', 256)
    param['nSpatialCells'] = overrides.get('nSpatialCells', 256)
    param['dirBinSize'] = overrides.get('dirBinSize', 6.0)  # degrees
    param['relFR_HD'] = overrides.get('relFR_HD', 0.5)
    param['relFR_spatial'] = overrides.get('relFR_spatial', 0.5)
    param['noiseLevel'] = overrides.get('noiseLevel', 2.0)

    param['xBins'] = np.arange(0.0, param['envSize_x'] + param['binSize_ratemap'], param['binSize_ratemap'])
    param['yBins'] = np.arange(0.0, param['envSize_y'] + param['binSize_ratemap'], param['binSize_ratemap'])
    param['yBins'][-1] = param['yBins'][-1] + 0.01  # include the upper edge
    param['xBins'][-1] = param['xBins'][-1] + 0.01
    param['xBins_center'] = (param['xBins'][:-1] + param['xBins'][1:]) / 2.0
    param['yBins_center'] = (param['yBins'][:-1] + param['yBins'][1:]) / 2.0
    param['dirBins'] = np.arange(0.0, 360.0 + param['dirBinSize'], param['dirBinSize'])
    param['dirBins_center'] = (param['dirBins'][:-1] + param['dirBins'][1:]) / 2.0
    param['dirBins'][-1] = param['dirBins'][-1] + 0.01
    if param['objLocs'] is None:
        # object locations (fractions of the environment) of the four trajectory blocks
        size = np.array([param['envSize_x'], param['envSize_y']])[None, :]
        newObjLocs = []
        newObjLocs.append(np.array([[0.2, 0.2], [0.2, 0.8], [0.5, 0.5], [0.9, 0.3], [0.8, 0.8]]) * size)
        newObjLocs.append(np.array([[0.1, 0.3], [0.5, 0.2], [0.6, 0.4], [0.3, 0.8], [0.9, 0.8]]) * size)
        newObjLocs.append(np.array([[0.3, 0.1], [0.6, 0.6], [0.8, 0.5], [0.3, 0.8], [0.9, 0.8]]) * size)
        newObjLocs.append(np.array([[0.4, 0.9], [0.9, 0.6], [0.4, 0.2], [0.3, 0.8], [0.8, 0.3]]) * size)
        param['objLocs'] = newObjLocs

    extra_keys = set(overrides) - set(param)
    if extra_keys:
        raise ValueError(f"Unexpected keys found for param: {sorted(extra_keys)}")
    return param


def module_param(module: str, **overrides) -> dict:
    """Return ``set_param`` for the "small" (40 vm) or "large" (70 vm) module plus ``overrides``."""
    values = dict(MODULE_PARAMS[module])
    values.update(overrides)
    return set_param(values)


def set_gridcells_pool(par: dict, n_cells: int, base_seed: int, progress: bool = False) -> pd.DataFrame:
    """Build a pool of simulated grid cells.

    Parameters
    ----------
    par : dict
        Parameters from ``set_param``.
    n_cells : int
        Number of cells.
    base_seed : int
        Seed of ``numpy.random.default_rng``; the paper's pools use 0.
    progress : bool
        Show a progress bar (tqdm).

    Returns
    -------
    pandas.DataFrame
        One row per cell with the columns of ``POOL_COLUMNS``: ``centers`` (k, 2) true field
        centres [vm] and ``peakFR_jitter`` (k,) their peak rates [Hz] (these define the cell);
        ``gridness``, ``gridscale_est`` [vm] and ``gridori_est`` [deg] from the autocorrelogram of
        the rate map; ``peaks_x``, ``peaks_y`` and ``radius`` [vm] of the fields delineated on the
        rate map (used to find field visits).
    """
    rng = np.random.default_rng(base_seed)
    # positions of the rate-map bins (first axis x, second axis y, as for the recorded cells)
    tmpx, tmpy = np.meshgrid(par['xBins_center'], par['yBins_center'], indexing="ij")
    basePos = np.column_stack([tmpx.ravel(), tmpy.ravel()])  # (N, 2)

    # fixed grid phases: a regular lattice of offsets over one unit cell
    nOffset = int(np.ceil(np.sqrt(n_cells)))
    u = np.linspace(0, 1, nOffset, endpoint=False)
    offx, offy = np.meshgrid(u, u, indexing="ij")
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
        gridscale_est = par['gridScale']
        gridori_est = par['gridOri']
        n_iter = 0
        maxiter = 100
        while (gridness < par['minGridness']) & (n_iter < maxiter):
            n_iter += 1
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
                    angles = rng.uniform(0, 2 * np.pi, size=K)  # direction of each field displacement
                    jitter_vec = (peakCoords_jitter_mag * 1)[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])
                    gridness = 10
                case "realistic":
                    peakCoords_jitter_mag = par['gridScale'] * rng.gamma(shape=par['peakCoords_shape'],
                                                                         scale=par['peakCoords_scale'], size=(K,))
                    angles = rng.uniform(0, 2 * np.pi, size=K)  # direction of each field displacement
                    jitter_vec = (peakCoords_jitter_mag * 1)[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])

                    peakFR_jitter = meanFR_all[cid] * rng.gamma(shape=par['FR_withinCell_shape'],
                                                                scale=par['FR_withinCell_scale'], size=(K,))
                    gridness = 10
                case _:
                    raise ValueError(f"Unknown gridtype: {par['gridtype']}")
            grid_sigma = par['fieldSigma'] * par['gridScale']

            centers = peakCoords + jitter_vec
            FR_vec = calc_fr_from_grid_fields(centers, basePos, grid_sigma, peakFR_jitter)
            ratemap = FR_vec.reshape(len(par['xBins_center']), len(par['yBins_center']))

            if par['gridtype'] == "realistic" or par['gridtype'] == "spatialDeviation":
                # gridness of the rate map downsampled by 2 (the simulated maps are smooth)
                autocorr = spatial_autocorrelogram_fft2d(ratemap[::2, ::2])
                try:
                    gridness, gridscale_est, gridori_est = compute_gridness_scale_ori(
                        autocorr, rate_map_bin_size=par['binSize_ratemap'] * 2)
                except Exception:
                    # A map whose autocorrelogram has no ring of peaks raises here; the cell is then
                    # kept with the placeholder gridness 10. No cell of the paper's pools took this path.
                    pass
        if n_iter == maxiter:
            raise ValueError(f"Max iterations reached for cell {cid} with gridness {gridness:.2f}. "
                             "Consider adjusting parameters to achieve higher gridness more easily.")

        # delineate the fields on the rate map (fields with very low rates are not detected)
        gridscaleInBins = par['gridScale'] / par['binSize_ratemap']
        min_dist_between_peak = gridscaleInBins * 1 / 2
        max_radius = min_dist_between_peak / 2
        min_radius = gridscaleInBins / 8
        peaks_row, peaks_col, radius = delineate_fields(ratemap, min_dist_between_peak, max_radius, min_radius,
                                                       row_coord=par['xBins_center'], col_coord=par['yBins_center'])

        rows.append(
            {
                "peaks_x": peaks_row,  # delineated field centres, x
                "peaks_y": peaks_col,  # delineated field centres, y
                "radius": radius,  # delineated field radii
                "gridness": gridness,
                "gridscale_est": gridscale_est,
                "gridori_est": gridori_est,
                "centers": centers,  # true field centres (lattice plus jitter)
                "peakFR_jitter": peakFR_jitter,  # true peak rate of every field
            }
        )
    gcTable = pd.DataFrame(rows)
    return gcTable


def generate_rotated_triangular_lattice(width, height, dx, rotAng, startOffset=(0.0, 0.0)):
    """Coordinates of a triangular lattice covering a rectangle.

    Parameters
    ----------
    width, height : float
        Size of the rectangle.
    dx : float
        Lattice spacing (side of the equilateral triangles).
    rotAng : float
        Rotation in degrees (counterclockwise).
    startOffset : tuple of float
        Offset of the lattice in units of the spacing, values in [0, 1).

    Returns
    -------
    ndarray, shape (N, 2)
        [x, y] of the lattice points within the rectangle extended by ``dx / 2`` on each side.
    """
    dy = dx * np.sqrt(3) / 2  # vertical step between rows
    theta = np.deg2rad(rotAng)

    # pad so that the rotated lattice still covers the rectangle
    pad = max(width, height)
    x_max = width + 2 * pad
    y_max = height + 2 * pad
    n_cols = int(np.ceil(x_max / dx))
    n_rows = int(np.ceil(y_max / dy))

    x_list = []
    y_list = []
    for row in range(-n_rows // 2, n_rows // 2 + 1):
        y = row * dy
        x_offset = (row % 2) * (dx / 2)
        for col in range(-n_cols // 2, n_cols // 2 + 1):
            x = col * dx + x_offset
            x_list.append(x)
            y_list.append(y)

    coords = np.column_stack((x_list, y_list))
    coords += np.array([startOffset[0], np.sqrt(3) / 2 * startOffset[1]]) * dx

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    coords_rot = coords @ R.T

    mask = (
        (coords_rot[:, 0] >= -dx / 2) & (coords_rot[:, 0] <= width + dx / 2) &
        (coords_rot[:, 1] >= -dx / 2) & (coords_rot[:, 1] <= height + dx / 2)
    )
    coords = coords_rot[mask]
    return coords


def calc_fr_from_grid_fields(peakCoords, pos, radius, peakFR=None):
    """Firing rate of a grid cell at given positions.

    The rate at a position is set by the nearest field only: ``peakFR * exp(-d^2 / (2 radius^2))``
    with ``d`` the distance to the nearest field centre.

    Parameters
    ----------
    peakCoords : ndarray, shape (N, 2)
        Field centres.
    pos : ndarray, shape (M, 2)
        Positions.
    radius : float
        Field s.d.
    peakFR : ndarray, shape (N,), optional
        Peak rate of each field; all fields have peak 1 when omitted.

    Returns
    -------
    ndarray, shape (M,)
        Rate at each position.
    """
    distVec = np.linalg.norm(pos[:, np.newaxis, :] - peakCoords[np.newaxis, :, :], axis=2)  # (M, N)
    dist_min = np.min(distVec, axis=1)
    minidx = np.argmin(distVec, axis=1)
    FR = np.exp(-dist_min**2 / (2 * radius**2))
    if peakFR is not None and len(peakFR) > 0:
        FR = FR * peakFR[minidx]
    return FR


def ratemap_from_centers(centers, peakFR_jitter, par):
    """Rate map (x bins, y bins) of a cell given its field centres and peak rates."""
    tmpx, tmpy = np.meshgrid(par['xBins_center'], par['yBins_center'], indexing="ij")
    basePos = np.column_stack([tmpx.ravel(), tmpy.ravel()])
    rate_vec = calc_fr_from_grid_fields(centers, basePos, par['fieldSigma'] * par['gridScale'], peakFR_jitter)
    rate_map = rate_vec.reshape(len(par['xBins_center']), len(par['yBins_center']))
    return rate_map
