"""Rate maps, spatial autocorrelograms, grid measures and movement-direction tuning.

- ``spatial_autocorrelogram_fft2d``: Pearson autocorrelogram of a 2D map at every spatial lag,
  computed with FFTs; bins without data (NaN) are excluded.
- ``compute_gridness_scale_ori``: gridness score (correlation of the autocorrelogram with its
  60 and 120 degree rotations minus the largest correlation at 30, 90 and 150 degrees, inside an
  annulus around the six central peaks), grid scale (mean distance of the six peaks closest to
  the centre) and grid orientation (circular mean, modulo 60 degrees, of the autocorrelogram
  peaks on the ring at the grid scale).
- ``compute_ratemap``: rate map from position samples and spikes (samples with spikes count
  once), optionally smoothed.
- ``compute_directional_rate_map_and_rayleigh_vector``: movement-direction tuning curve (spike
  counts and dwell per direction bin, circular boxcar smoothing) and its Rayleigh vector.
- ``quantify_fr_inhomogeneity``: mean rate of the rate map inside each grid field.
- ``quantify_field_deviation`` and ``fit_lattice_and_assign``: fit of a perfect triangular
  lattice (given scale and orientation) to the field centres and the distance of each centre from
  its lattice point, in units of the grid scale.
"""

from __future__ import annotations

import numpy as np
import scipy.stats
from scipy.ndimage import convolve1d, gaussian_filter
from skimage.feature import peak_local_max
from skimage.transform import rotate

__all__ = ["quantify_fr_inhomogeneity", "quantify_field_deviation", "fit_lattice_and_assign",
           "spatial_autocorrelogram_fft2d", "compute_ratemap", "compute_gridness_scale_ori",
           "compute_directional_rate_map_and_rayleigh_vector"]


def quantify_fr_inhomogeneity(rate_map, row_coords, col_coords, fieldLoc):
    """Mean firing rate of the rate map inside each grid field.

    Parameters
    ----------
    rate_map : (m, n) ndarray
        Firing-rate map.
    row_coords, col_coords : ndarray
        Coordinates of the bin centres along the rows (x) and the columns (y).
    fieldLoc : (K, 3) ndarray
        Field centres and radii: ``[x, y, radius]`` per field.

    Returns
    -------
    regionFR : (K,) ndarray
        Mean rate inside each field (bins within the field radius).
    gfit_shape, gfit_scale : float
        Gamma fit (location 0) of ``regionFR`` when there are more than three fields, else NaN.
    """
    regionFR = []
    for (peak_row, peak_col, radius) in fieldLoc:
        col_indices, row_indices = np.meshgrid(col_coords, row_coords)
        mask = (col_indices - peak_col)**2 + (row_indices - peak_row)**2 <= radius**2
        regionFR.append(np.nanmean(rate_map[mask]))
    if len(fieldLoc) > 3:
        gfit_shape, gfit_loc, gfit_scale = scipy.stats.gamma.fit(regionFR, floc=0)
    else:
        gfit_shape, gfit_loc, gfit_scale = np.nan, np.nan, np.nan
    return np.asarray(regionFR), gfit_shape, gfit_scale


def quantify_field_deviation(fieldLoc, lattice_spacing, lattice_orientation):
    """Deviation of the field centres from a perfect triangular lattice.

    Parameters
    ----------
    fieldLoc : (K, 2) ndarray
        Field centres ``[x, y]`` (cm).
    lattice_spacing : float
        Grid scale (cm).
    lattice_orientation : float
        Grid orientation (rad).

    Returns
    -------
    optimal_origin : (2,) ndarray
        Lattice offset that minimises the squared distances.
    assigned_lattice_coords : (K, 2) ndarray
        Nearest lattice point of each field centre.
    distFromPerfectGrid : (K,) ndarray
        Distance of each centre from its lattice point divided by the grid scale.
    gfit_shape, gfit_scale : float
        Gamma fit (location 0) of ``distFromPerfectGrid`` when there are more than three fields,
        else NaN.
    """
    optimal_origin, assigned_lattice_coords, _ = fit_lattice_and_assign(fieldLoc, lattice_spacing, lattice_orientation)
    distFromPerfectGrid = np.linalg.norm(assigned_lattice_coords - fieldLoc, axis=1)
    distFromPerfectGrid = distFromPerfectGrid / lattice_spacing
    if len(fieldLoc) > 3:
        gfit_shape, gfit_loc, gfit_scale = scipy.stats.gamma.fit(distFromPerfectGrid, floc=0)
    else:
        gfit_shape, gfit_loc, gfit_scale = np.nan, np.nan, np.nan

    return optimal_origin, assigned_lattice_coords, distFromPerfectGrid, gfit_shape, gfit_scale


def fit_lattice_and_assign(observed_coords, lattice_spacing, lattice_orientation, steps=50):
    """Fit the origin of a triangular lattice and assign points to lattice sites.

    The origin is found by a grid search over ``steps x steps`` offsets within one unit cell;
    each point is assigned to its nearest site (rounded fractional coordinates).

    Parameters
    ----------
    observed_coords : (N, 2) array_like
        Point coordinates.
    lattice_spacing : float
        Distance between neighbouring sites.
    lattice_orientation : float
        Angle (rad) of the first lattice vector.
    steps : int
        Resolution of the origin search (default 50).

    Returns
    -------
    optimal_origin : (2,) ndarray
    assigned_lattice_coords : (N, 2) ndarray
    min_error : float
        Sum of squared distances at the optimal origin.
    """
    obs = np.asarray(observed_coords, dtype=float)
    a1 = lattice_spacing * np.array([np.cos(lattice_orientation), np.sin(lattice_orientation)])
    a2 = lattice_spacing * np.array([np.cos(lattice_orientation + np.pi/3),
                                     np.sin(lattice_orientation + np.pi/3)])
    lattice_basis = np.column_stack((a1, a2))  # (2, 2)

    u_vals = np.linspace(0, 1, steps)
    v_vals = np.linspace(0, 1, steps)

    min_error = np.inf
    optimal_origin = np.zeros(2)

    for u in u_vals:
        for v in v_vals:
            origin_candidate = u * a1 + v * a2
            rel_coords = obs - origin_candidate
            # fractional coordinates in the lattice basis
            frac_coords = rel_coords @ np.linalg.inv(lattice_basis).T  # (N, 2)
            lattice_indices = np.round(frac_coords)
            assigned_coords = origin_candidate + lattice_indices @ lattice_basis.T
            diffs = obs - assigned_coords
            total_error = np.sum(diffs**2)
            if total_error < min_error:
                min_error = total_error
                optimal_origin = origin_candidate.copy()

    rel_coords = obs - optimal_origin
    frac_coords = rel_coords @ np.linalg.inv(lattice_basis).T
    lattice_indices = np.round(frac_coords)
    assigned_lattice_coords = optimal_origin + lattice_indices @ lattice_basis.T

    return optimal_origin, assigned_lattice_coords, min_error


def spatial_autocorrelogram_fft2d(x, presmoothing=None, aftersmoothing=None, nodwell=None, tol=1e-12):
    """Spatial autocorrelogram of a 2D map computed with FFTs.

    At every lag the value is the Pearson correlation between the map and its shifted copy over
    the overlapping bins with data.

    Parameters
    ----------
    x : (m, n) array_like
        Map; NaN marks bins without data.
    presmoothing, aftersmoothing : float, optional
        Standard deviation (bins) of a Gaussian filter applied to the map before, and to the
        autocorrelogram after, the computation.
    nodwell : (m, n) bool array_like, optional
        Bins without data (default: NaN bins of ``x``); they are set to zero.
    tol : float
        Values below this are set to zero to suppress numerical noise.

    Returns
    -------
    acorr : (2m - 1, 2n - 1) ndarray
    """
    x = np.asarray(x, dtype=np.float64)
    if nodwell is None:
        nodwell = np.isnan(x)
    nodwell = np.asarray(nodwell, dtype=bool)
    if nodwell.shape != x.shape:
        raise ValueError("nodwell must have the same shape as x.")

    m, n = x.shape
    s = (2*m - 1, 2*n - 1)

    xx = x.copy()
    xx[nodwell] = 0.0

    if presmoothing is not None:
        xx = gaussian_filter(xx, sigma=presmoothing)
    # FFTs padded to (2m - 1, 2n - 1)
    Fx = np.fft.fft2(xx, s=s)
    Fxx = np.fft.fft2(xx*xx, s=s)
    Fn = np.fft.fft2((~nodwell).astype(np.float64), s=s)

    # correlations by inverse FFT, shifted so that lag zero is in the centre
    rawCorr = np.fft.ifft2(Fx * np.conj(Fx)).real
    rawCorr = np.fft.fftshift(rawCorr)

    sums_x = np.fft.ifft2(Fn * np.conj(Fx)).real
    sums_x = np.fft.fftshift(sums_x)

    sumOfSquares_x = np.fft.ifft2(Fn * np.conj(Fxx)).real
    sumOfSquares_x = np.fft.fftshift(sumOfSquares_x)

    N = np.fft.ifft2(Fn * np.conj(Fn)).real
    N = np.fft.fftshift(N)

    rawCorr[np.abs(rawCorr) < tol] = 0.0
    sums_x[np.abs(sums_x) < tol] = 0.0
    sumOfSquares_x[sumOfSquares_x < tol] = 0.0
    N = np.round(N)
    N[N <= 1] = np.nan  # too few overlapping bins for a correlation

    # mapStd = sqrt(max(N sum(x^2) - (sum x)^2, 0))
    var_term = N * sumOfSquares_x - sums_x**2
    var_term[var_term < 0] = 0.0
    mapStd = np.sqrt(var_term)

    # mapCovar = N rawCorr - sums_x * reversed(sums_x)
    sums_x_rev = sums_x[::-1, ::-1]
    mapCovar = N * rawCorr - sums_x * sums_x_rev

    denom = mapStd * mapStd[::-1, ::-1]
    with np.errstate(invalid='ignore', divide='ignore'):
        acorr = mapCovar / denom
    acorr[np.isnan(acorr)] = 0.0

    if aftersmoothing is not None:
        acorr = gaussian_filter(acorr, sigma=aftersmoothing)

    return acorr


def compute_ratemap(x, y, spikes, x_edges, y_edges, smooth=None, dt=1):
    """Rate map from position samples and spikes.

    Samples with one or more spikes count once in the spike map.

    Parameters
    ----------
    x, y : ndarray
        Positions of the samples.
    spikes : ndarray
        Spikes per sample.
    x_edges, y_edges : ndarray
        Bin edges.
    smooth : float, optional
        Standard deviation (bins) of a Gaussian filter applied to the rate and occupancy maps.
    dt : float
        Factor applied to both returned maps.

    Returns
    -------
    rate_map, occupancy_map : ndarray
        Spike map divided by occupancy (samples), and occupancy, each multiplied by ``dt``.
    """
    occupancy_map, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    spike_indices = np.where(spikes > 0)[0]
    spike_x = x[spike_indices]
    spike_y = y[spike_indices]
    spike_map, _, _ = np.histogram2d(spike_x, spike_y, bins=[x_edges, y_edges])
    rate_map = spike_map / (occupancy_map + 1e-10)  # avoids division by zero
    if smooth is not None:
        rate_map = gaussian_filter(rate_map, sigma=smooth)
        occupancy_map = gaussian_filter(occupancy_map, sigma=smooth)
    return rate_map*dt, occupancy_map*dt


def compute_gridness_scale_ori(autocorrelogram, rate_map_bin_size=1, gridness_thres=0.3):
    """Gridness score, grid scale and grid orientation from a spatial autocorrelogram.

    NaN values of ``autocorrelogram`` are set to zero in place. Peaks are searched in the
    autocorrelogram with negative values clipped to zero; the central peak (within 5 bins) is
    excluded and the six closest peaks define an annulus (0.7 to 1.3 times their mean distance)
    in which the autocorrelogram is correlated with its rotations by 0 to 150 degrees in steps
    of 30 degrees. Scale and orientation are computed only when the gridness exceeds
    ``gridness_thres``.

    Parameters
    ----------
    autocorrelogram : (m, n) ndarray
    rate_map_bin_size : float
        Bin size of the rate map (cm), to express the scale in cm.
    gridness_thres : float

    Returns
    -------
    gridness : float
    gridscale : float
        Mean distance of the six closest peaks (cm), or NaN.
    gridori_deg : float
        Circular mean, modulo 60 degrees, of the peak angles on the ring at the grid scale, or
        NaN.
    """
    autocorrelogram[np.isnan(autocorrelogram)] = 0
    # local maxima are searched where the correlation is positive
    autocorr_clipped = autocorrelogram.copy()
    autocorr_clipped[autocorrelogram <= 0] = 0

    center = np.array(autocorrelogram.shape) // 2

    peaks = peak_local_max(autocorr_clipped, min_distance=2, exclude_border=False)

    # remove the central peak and sort the others by distance from the centre
    peaks = peaks[np.linalg.norm(peaks - center, axis=1) > 5]
    distances = np.linalg.norm(peaks - center, axis=1)
    sorted_peaks = peaks[np.argsort(distances)]

    closest_peaks = sorted_peaks[:6]
    closest_peaks_distances = distances[np.argsort(distances)[:6]]

    # not used further; raises ValueError when the autocorrelogram has no peak outside the centre
    outermost_peak = closest_peaks[np.argmax(np.linalg.norm(closest_peaks - center, axis=1))]  # noqa: F841
    outer_radius = np.mean(closest_peaks_distances) * 1.3
    inner_radius = np.mean(closest_peaks_distances) * 0.7

    x, y = np.ogrid[:autocorrelogram.shape[0], :autocorrelogram.shape[1]]
    mask_outer = (x - center[0])**2 + (y - center[1])**2 > outer_radius**2
    mask_inner = (x - center[0])**2 + (y - center[1])**2 < inner_radius**2
    mask = mask_outer | mask_inner

    masked_autocorrelogram = np.ma.masked_array(autocorrelogram, mask)

    rotation_step = 30  # degrees
    correlations = []
    for angle in range(0, 180, rotation_step):
        rotated_autocorrelogram = rotate(autocorrelogram, angle, resize=False, order=1, mode='wrap', preserve_range=True)
        rotated_masked_autocorrelogram = np.ma.masked_array(rotated_autocorrelogram, mask)
        correlation = np.corrcoef(masked_autocorrelogram.compressed(), rotated_masked_autocorrelogram.compressed())[0, 1]
        correlations.append(correlation)

    # 60 and 120 degree correlations minus the 30, 90 and 150 degree correlations
    gridness = (min([correlations[60 // rotation_step], correlations[120 // rotation_step]]) -
                max([correlations[30 // rotation_step], correlations[90 // rotation_step], correlations[150 // rotation_step]]))
    gridscale = np.nan
    gridori_deg = np.nan

    if gridness > gridness_thres:
        gridscale_bin = np.mean(closest_peaks_distances)
        gridscale = gridscale_bin*rate_map_bin_size
        # autocorrelogram values on the ring at the grid scale
        ang = np.linspace(0, 2*np.pi, 360)
        coord_ring_x = center[0]+np.cos(ang)*gridscale_bin
        coord_ring_y = center[1]+np.sin(ang)*gridscale_bin
        coord_x, coord_y = np.round([coord_ring_x, coord_ring_y]).astype(int)
        values_on_ring = autocorrelogram[np.clip(coord_x, 0, autocorrelogram.shape[0]-1), np.clip(coord_y, 0, autocorrelogram.shape[1]-1)]
        # pad the ring to find peaks near 0 and 360 degrees
        values_on_ring_padded = np.concatenate([values_on_ring[-15:], values_on_ring, values_on_ring[:15]])
        ang_padded = np.concatenate([ang[-15:], ang, ang[:15]])
        peaks_angs_idx = peak_local_max(values_on_ring_padded, min_distance=30)
        peaks_angs_idx[peaks_angs_idx < 0] = []
        peaks_angs_idx[peaks_angs_idx >= 360] = []
        peaks_angs = ang_padded[peaks_angs_idx]
        gridori_deg = scipy.stats.circmean(peaks_angs, high=2*np.pi/6, low=0)*180/np.pi
    return gridness, gridscale, gridori_deg


def compute_directional_rate_map_and_rayleigh_vector(direction_deg, spikes, bin_size_deg=5, smooth_nBins=3, dt=1):
    """Movement-direction tuning curve and its Rayleigh vector.

    Spike counts and dwell (``dt`` per sample) are accumulated in direction bins of
    ``bin_size_deg`` and both are smoothed with a circular boxcar of ``smooth_nBins`` bins before
    division. Bins without dwell after smoothing get a rate of zero.

    Parameters
    ----------
    direction_deg : ndarray
        Movement direction of each sample, degrees in [0, 360).
    spikes : ndarray
        Spikes per sample.
    bin_size_deg : float
    smooth_nBins : int
        Width of the boxcar (odd, to avoid a half-bin shift).
    dt : float
        Dwell per sample (1: rates in spikes per sample).

    Returns
    -------
    rayleigh_vector_magnitude : float
        Resultant length of the tuning curve (0 to 1).
    pref_angle_deg : float
        Direction of the resultant, degrees in [0, 360).
    firing_rate : ndarray
        Tuning curve (one value per bin, bin centres at ``bin_size_deg / 2`` and on).
    smoothed_dwell_times : ndarray
        Smoothed dwell per bin.
    """
    bins = np.arange(0, 360 + bin_size_deg, bin_size_deg)
    direction_idx = np.digitize(direction_deg, bins) - 1
    direction_idx = np.where(direction_idx == len(bins) - 1, 0, direction_idx)  # wrap around

    n_bins = len(bins) - 1
    dwell_times = np.zeros(n_bins)
    spike_counts = np.zeros(n_bins)

    np.add.at(dwell_times, direction_idx, dt)
    np.add.at(spike_counts, direction_idx, spikes)

    kernel = np.ones(smooth_nBins) / smooth_nBins
    smoothed_spike_counts = convolve1d(spike_counts, kernel, mode='wrap')
    smoothed_dwell_times = convolve1d(dwell_times, kernel, mode='wrap')

    # bins without dwell are set to zero explicitly (out=), not left uninitialised
    firing_rate = np.divide(smoothed_spike_counts, smoothed_dwell_times, out=np.zeros_like(smoothed_spike_counts),
                            where=smoothed_dwell_times != 0)
    firing_rate = np.nan_to_num(firing_rate, nan=0.0)

    angles_mid = (bins[:-1] + bins[1:]) / 2
    sum_sin = np.sum(firing_rate * np.sin(angles_mid*np.pi/180))
    sum_cos = np.sum(firing_rate * np.cos(angles_mid * np.pi/180))
    rayleigh_vector_magnitude = np.sqrt(sum_sin**2 + sum_cos**2) / np.sum(firing_rate)
    rayleigh_vector_angle_rad = np.arctan2(sum_sin, sum_cos)
    pref_angle_deg = np.rad2deg(rayleigh_vector_angle_rad) % 360

    return rayleigh_vector_magnitude, pref_angle_deg, firing_rate, smoothed_dwell_times
