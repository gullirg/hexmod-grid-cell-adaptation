"""Closed-form results: proximity against running speed (Fig 8B, 8C) and the six-fold
component of direction tuning curves (Fig 1 legend, Supp Fig 5).

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

from hexmod.circular import kappa_from_sd, sixfold_component, sixfold_from_sd

__all__ = ["SPEED_OPT_FACTOR", "FIG8BC_SPEEDS", "FIG8BC_SCALES", "proximity_aligned", "proximity_misaligned",
           "proximity_difference", "optimal_speed", "max_proximity_difference", "kappa_from_sd",
           "sixfold_component", "sixfold_from_sd", "SIXFOLD_SDS", "sixfold_curve"]

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


def sixfold_curve(sds=None):
    """Six-fold component (fraction of the mean) of a von Mises curve for each circular SD [deg]."""
    sds = SIXFOLD_SDS if sds is None else np.asarray(sds, float)
    return np.array([sixfold_from_sd(s) for s in sds])
