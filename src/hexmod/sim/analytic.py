"""Closed-form results: proximity against running speed (Fig 8B, 8C) and the six-fold
component of summed direction tuning curves (Fig 1, Supp Fig 5).

Proximity. With adaptation that decays as ``exp(-t_prev / tau)``, a run along a grid axis meets
the next field after one grid scale, and a run between axes after sqrt(3) grid scales. At running
speed v the proximities are ``exp(-scale / v / tau)`` (aligned) and
``exp(-sqrt(3) scale / v / tau)`` (misaligned). Their difference is largest at
``v = (sqrt(3) - 1) / ln(sqrt(3)) * scale / tau = 1.3327 scale / tau``, for example 20.0 vm/s
for a 30 vm grid and 46.6 vm/s for a 70 vm grid with tau = 2 s.

Six-fold component. A direction tuning curve ``exp(kappa cos(theta - mu)) / I0(kappa)`` has a
``cos 6(theta - mu)`` component of relative amplitude ``2 I6(kappa) / I0(kappa)``
(``hexmod.circular``). Six such cells, one per grid axis, sum to a curve with the same relative
six-fold modulation; six cosine-tuned cells (``1 + cos(theta - mu)``, circular SD 67.46 deg) sum
to a flat line.
"""

from __future__ import annotations

import numpy as np
from scipy.special import i0

from hexmod.circular import kappa_from_sd, sixfold_component, sixfold_from_sd

__all__ = ["SPEED_OPT_FACTOR", "FIG8BC_SPEEDS", "FIG8BC_SCALES", "proximity_aligned", "proximity_misaligned",
           "proximity_difference", "optimal_speed", "max_proximity_difference", "kappa_from_sd",
           "sixfold_component", "sixfold_from_sd", "von_mises_cell", "population_sum", "cosine_cell",
           "population_sum_cosine", "population_summary", "beta_from_bars", "SIXFOLD_SDS", "sixfold_curve"]

#: (sqrt(3) - 1) / ln(sqrt(3)) = 1.33268...: optimal speed in units of scale / tau.
SPEED_OPT_FACTOR = 1 / np.log(np.sqrt(3)) * (np.sqrt(3) - 1)

#: Running speeds (vm/s) and grid scales (vm) drawn in Fig 8B and 8C.
FIG8BC_SPEEDS = np.arange(5, 150, 2)
FIG8BC_SCALES = (30, 70)

#: Circular SDs (deg) of the Supp Fig 5 curve.
SIXFOLD_SDS = np.linspace(2, 95, 940)


def proximity_aligned(scale, speed, tau):
    """``exp(-scale / speed / tau)``: proximity of the next field on a run along a grid axis."""
    return np.exp(-scale / speed / tau)


def proximity_misaligned(scale, speed, tau):
    """``exp(-scale * sqrt(3) / speed / tau)``: proximity of the next field on a run between axes."""
    return np.exp(-scale / speed * np.sqrt(3) / tau)


def proximity_difference(scale, speed, tau):
    """Aligned minus misaligned proximity (Fig 8C)."""
    return -(proximity_misaligned(scale, speed, tau) - proximity_aligned(scale, speed, tau))


def optimal_speed(scale, tau):
    """Running speed at which the aligned-minus-misaligned proximity is largest."""
    return SPEED_OPT_FACTOR * scale / tau


def max_proximity_difference():
    """Largest aligned-minus-misaligned proximity, ``exp(-a) - exp(-sqrt(3) a)`` with
    ``a = ln(sqrt(3)) / (sqrt(3) - 1)``; it does not depend on scale or tau."""
    a = np.log(np.sqrt(3)) / (np.sqrt(3) - 1)
    return np.exp(-a) - np.exp(-np.sqrt(3) * a)


def von_mises_cell(theta, mu_deg, kappa):
    """von Mises direction tuning curve with mean 1, ``exp(kappa cos(theta - mu)) / I0(kappa)``.

    ``theta`` in radians, ``mu_deg`` in degrees; computed as
    ``exp(kappa (cos(theta - mu) - 1)) / (I0(kappa) exp(-kappa))`` to avoid overflow.
    """
    return np.exp(kappa * (np.cos(theta - np.deg2rad(mu_deg)) - 1.0)) / (i0(kappa) * np.exp(-kappa))


def population_sum(theta, kappa):
    """Mean of six von Mises cells preferring the six grid axes (0, 60, ..., 300 deg)."""
    return sum(von_mises_cell(theta, 60.0 * j, kappa) for j in range(6)) / 6.0


def cosine_cell(theta, mu_deg):
    """Cosine tuning curve ``1 + cos(theta - mu)`` (mean 1, zero opposite the preferred direction)."""
    return 1.0 + np.cos(theta - np.deg2rad(mu_deg))


def population_sum_cosine(theta):
    """Mean of six cosine-tuned cells preferring the six grid axes (exactly 1)."""
    return sum(cosine_cell(theta, 60.0 * j) for j in range(6)) / 6.0


def population_summary(sd_deg, theta=None):
    """Six-fold modulation of the summed tuning of six von Mises cells of circular SD ``sd_deg``.

    Parameters
    ----------
    sd_deg : float
        Circular SD of each cell [deg].
    theta : ndarray, optional
        Directions [rad] at which the sum is evaluated (default 0 to 360 deg in 0.25 deg steps).

    Returns
    -------
    dict
        ``kappa``; ``sixfold_formula`` (2 I6 / I0); ``sixfold_numeric`` (2 mean(sum cos 6 theta));
        ``aligned_over_mean`` and ``misaligned_over_mean``, the mean of the sum within 15 deg of a
        grid axis and further away.
    """
    if theta is None:
        theta = np.deg2rad(np.arange(0.0, 360.0, 0.25))
    k = kappa_from_sd(sd_deg)
    s = population_sum(theta, k)
    dev = np.rad2deg(np.abs(np.angle(np.exp(1j * 6 * theta)))) / 6.0  # distance to the nearest axis [deg]
    return {
        "kappa": k,
        "sixfold_formula": sixfold_component(k),
        "sixfold_numeric": 2.0 * np.mean(s * np.cos(6 * theta)),
        "aligned_over_mean": s[dev <= 15.0].mean(),
        "misaligned_over_mean": s[dev > 15.0].mean(),
    }


def beta_from_bars(aligned_over_mean, misaligned_over_mean):
    """Six-fold coefficient implied by aligned and misaligned mean rates.

    A rate ``1 + beta cos 6(theta - phi)`` has mean ``1 + 2 beta / pi`` within 15 deg of the grid
    axes and ``1 - 2 beta / pi`` elsewhere, so ``beta = (pi / 4) (aligned - misaligned)``.
    """
    return np.pi / 4.0 * (aligned_over_mean - misaligned_over_mean)


def sixfold_curve(sds=None):
    """Six-fold component (fraction of the mean) of a von Mises curve for each circular SD [deg]."""
    sds = SIXFOLD_SDS if sds is None else np.asarray(sds, float)
    return np.array([sixfold_from_sd(s) for s in sds])
