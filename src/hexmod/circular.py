"""Circular helpers: wrapping, circular standard deviation and von Mises harmonics.

- ``wrap60`` maps an angle difference (degrees) to [-30, 30), the deviation of a preferred
  direction from the nearest grid axis when the difference is taken modulo 60 degrees.
- ``circular_sd_deg`` is the circular standard deviation ``sqrt(-2 ln R)`` of a direction
  tuning curve with resultant (Rayleigh vector) length R, in degrees.
- A von Mises tuning curve ``exp(kappa cos(theta - mu)) / I0(kappa)`` contains a six-fold
  component ``cos 6(theta - mu)`` whose amplitude relative to the mean is
  ``2 I6(kappa) / I0(kappa)`` (``sixfold_component``). The concentration matching a circular
  SD is found from ``I1(kappa) / I0(kappa) = exp(-sd^2 / 2)`` (``kappa_from_sd``).
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
from scipy import optimize, special

ArrayLike = Union[float, np.ndarray]

__all__ = ["wrap", "wrap60", "circular_sd_deg", "resultant_from_sd", "kappa_from_sd",
           "harmonic_amplitude", "sixfold_component", "sixfold_from_sd"]


def wrap(x: ArrayLike, period: float) -> np.ndarray:
    """Wrap values to [-period/2, period/2): ``(x + period/2) mod period - period/2``."""
    return (np.asarray(x, float) + period / 2) % period - period / 2


def wrap60(x: ArrayLike) -> np.ndarray:
    """Wrap angles in degrees to [-30, 30): ``(x + 30) mod 60 - 30``."""
    return (np.asarray(x, float) + 30.0) % 60.0 - 30.0


def circular_sd_deg(R: ArrayLike) -> np.ndarray:
    """Circular standard deviation in degrees, ``degrees(sqrt(-2 ln R))``, for resultant length R."""
    return np.degrees(np.sqrt(-2 * np.log(np.asarray(R, float))))


def resultant_from_sd(sd_deg: float) -> float:
    """Resultant length ``exp(-sd^2 / 2)`` (sd in radians) of a circular SD given in degrees."""
    sr = math.radians(sd_deg)
    return math.exp(-sr * sr / 2)


def kappa_from_sd(sd_deg: float) -> float:
    """von Mises concentration whose circular SD equals ``sd_deg`` degrees.

    Solves ``I1(kappa) / I0(kappa) = exp(-sd^2 / 2)`` with Brent's method on [1e-6, 1e4], using
    exponentially scaled Bessel functions.
    """
    target = resultant_from_sd(sd_deg)
    return float(optimize.brentq(lambda k: special.i1e(k) / special.i0e(k) - target, 1e-6, 1e4))


def harmonic_amplitude(k: int, kappa: float) -> float:
    """Relative amplitude ``2 I_k(kappa) / I_0(kappa)`` of the k-th harmonic of a von Mises curve."""
    return float(2 * special.ive(k, kappa) / special.i0e(kappa))


def sixfold_component(kappa: float) -> float:
    """Six-fold component ``2 I6(kappa) / I0(kappa)`` of a von Mises tuning curve."""
    return harmonic_amplitude(6, kappa)


def sixfold_from_sd(sd_deg: float) -> float:
    """Six-fold component of a von Mises tuning curve with circular SD ``sd_deg`` degrees.

    For example 0.5929 at 15 degrees (kappa = 15.1055).
    """
    return sixfold_component(kappa_from_sd(sd_deg))
