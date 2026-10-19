"""Multiplicative Poisson models of firing rate.

- ``fit_model``: maximum-likelihood fit of a rate that is the product of one, two or three
  factors, each a free value per bin of a discrete variable (for example location, speed and
  t_prev bins), with optional cross-validation. The factors are found by alternating updates
  (``fit_two_factor``, ``fit_three_factor``), as in "Characterizing multiple independent
  behavioral correlates of cell firing in freely moving animals" (Hippocampus 15:149-153, 2005).
- ``fit_location_parametric_poisson``: fit of
  ``mu_t = f[x_t] (1 + gamma var1_t) (1 + beta var2_t) dt``, alternating a closed-form update
  of the spatial map ``f`` with Newton (Fisher scoring) steps for ``gamma`` and ``beta``.
- ``fit_rank1_als``: weighted least-squares rank-1 factorisation ``M ~ f g^T`` by alternating
  least squares (used by the simulations).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np
from scipy.special import gammaln

__all__ = ["poisson_loglikelihood", "fit_model", "fit_three_factor", "fit_two_factor",
           "fit_location_parametric_poisson", "fit_rank1_als"]


def poisson_loglikelihood(spikes, expected, tol=1e-6):
    """Poisson log-likelihood of spike counts given expected counts.

    Samples with NaN expectation are excluded; expectations are floored at ``tol``.

    Returns
    -------
    fit : float
        ``sum(spikes log(expected) - expected - log(spikes!))``.
    n_exclude : int
        Number of excluded samples.
    """
    expected = np.array(expected, dtype=float)
    spikes = np.array(spikes, dtype=float)

    mask = ~np.isnan(expected)
    expected = expected[mask]
    spikes = spikes[mask]

    expected = np.maximum(expected, tol)
    fit = np.sum(spikes * np.log(expected) - expected - gammaln(spikes + 1))
    n_exclude = np.sum(~mask)
    return fit, n_exclude


def fit_model(inp: Dict[str, Any]) -> Dict[str, Any]:
    """Fit multiplicative Poisson models with one to three discrete factors.

    Parameters
    ----------
    inp : dict
        ``spikes`` (1D spike counts), ``factors`` (list of integer bin indices, one array per
        variable), ``modelsToCompare`` (list of lists of factor indices, one per model),
        ``maxBins`` (number of possible bins of each factor) and optional ``max_iter`` (100),
        ``accuracy`` (1e-6, relative change of the log-likelihood at convergence), ``tol``
        (1e-6) and ``crossvalIdx`` (fold number of each sample, 1 to n; 0 or absent: no
        cross-validation).

    Returns
    -------
    dict
        ``model`` (per model: ``A_ml``, ``B_ml``, ``C_ml``, the rate per bin of each factor, NaN
        for bins without samples), ``aic``, ``bic``, ``TotSpk``, ``nParams``, ``trainFit``,
        ``trainLogLik``, ``alpha`` (unused, NaN) and, with cross-validation, ``crossvalLogLik``,
        ``crossvalExclusion``, ``crossvalTotSpk``, ``crossvalNum``.
    """
    max_iter = inp.get("max_iter", 100)
    accuracy = inp.get("accuracy", 1e-6)
    tol = inp.get("tol", 1e-6)

    n_models = len(inp["modelsToCompare"])
    trainLogLik = np.full(n_models, np.nan)
    trainFit = np.full(n_models, np.nan)
    nParams = np.full(n_models, np.nan)
    alpha = np.full(n_models, np.nan)

    # bins with samples, re-indexed from 0
    uniqVals, newFactors, nLevelFactors = [], [], []
    for factor in inp["factors"]:
        uniqValsOld, uniqIdx = np.unique(factor, return_inverse=True)
        newFactors.append(uniqIdx)
        uniqVals.append(uniqValsOld)
        nLevelFactors.append(len(uniqValsOld))
    nLevelFactors = np.array(nLevelFactors, int)

    out = {"model": [{} for _ in range(n_models)]}

    for modelIdx, factors in enumerate(inp["modelsToCompare"]):
        numFactors = len(factors)
        spikes = inp["spikes"]

        if numFactors == 1:
            fA = factors[0]
            dwell_A = np.bincount(newFactors[fA], minlength=nLevelFactors[fA])
            spikes_A = np.bincount(newFactors[fA], weights=spikes, minlength=nLevelFactors[fA])
            A_ml = np.divide(spikes_A, dwell_A, out=np.zeros_like(spikes_A, dtype=float), where=dwell_A > 0)
            fr_expected = A_ml[newFactors[fA]]
            nParams[modelIdx] = nLevelFactors[fA]

            oldA = np.full(inp["maxBins"][fA], np.nan)
            oldA[uniqVals[fA]] = A_ml
            out["model"][modelIdx]["A_ml"] = oldA

            fit = 0

        elif numFactors == 2:
            fA, fB = factors
            shape = (nLevelFactors[fA], nLevelFactors[fB])
            idx = np.vstack((newFactors[fA], newFactors[fB])).T

            dwell = np.zeros(shape)
            np.add.at(dwell, tuple(idx.T), 1)
            spikes_mat = np.zeros(shape)
            np.add.at(spikes_mat, tuple(idx.T), spikes)

            A_ml, B_ml, converged, fit, scaleA, scaleB = fit_two_factor(spikes_mat, dwell, max_iter, accuracy, tol)
            fr_expected = (A_ml[newFactors[fA]] * B_ml[newFactors[fB]]) / (scaleA * scaleB)
            nParams[modelIdx] = nLevelFactors[fA] + nLevelFactors[fB]

            oldA = np.full(inp["maxBins"][fA], np.nan)
            oldA[uniqVals[fA]] = A_ml
            out["model"][modelIdx]["A_ml"] = oldA

            oldB = np.full(inp["maxBins"][fB], np.nan)
            oldB[uniqVals[fB]] = B_ml
            out["model"][modelIdx]["B_ml"] = oldB

        elif numFactors == 3:
            fA, fB, fC = factors
            shape = (nLevelFactors[fA], nLevelFactors[fB], nLevelFactors[fC])
            idx = np.vstack((newFactors[fA], newFactors[fB], newFactors[fC])).T

            dwell = np.zeros(shape)
            np.add.at(dwell, tuple(idx.T), 1)
            spikes_mat = np.zeros(shape)
            np.add.at(spikes_mat, tuple(idx.T), spikes)

            A_ml, B_ml, C_ml, converged, fit, scaleA, scaleB, scaleC = fit_three_factor(
                spikes_mat, dwell, max_iter, accuracy, tol
            )
            fr_expected = (
                A_ml[newFactors[fA]] * B_ml[newFactors[fB]] * C_ml[newFactors[fC]]
            ) / (scaleA * scaleB * scaleC)
            nParams[modelIdx] = nLevelFactors[fA] + nLevelFactors[fB] + nLevelFactors[fC]

            oldA = np.full(inp["maxBins"][fA], np.nan)
            oldA[uniqVals[fA]] = A_ml
            out["model"][modelIdx]["A_ml"] = oldA

            oldB = np.full(inp["maxBins"][fB], np.nan)
            oldB[uniqVals[fB]] = B_ml
            out["model"][modelIdx]["B_ml"] = oldB

            oldC = np.full(inp["maxBins"][fC], np.nan)
            oldC[uniqVals[fC]] = C_ml
            out["model"][modelIdx]["C_ml"] = oldC

        else:
            raise ValueError(f"Model {modelIdx+1} has more than 3 factors")

        trainFit[modelIdx] = fit
        trainLogLik[modelIdx], _ = poisson_loglikelihood(spikes, fr_expected, tol)

    nObs = len(inp["spikes"])
    aic = -2 * trainLogLik + 2 * nParams
    bic = -2 * trainLogLik + np.log(nObs) * nParams

    out.update({
        "aic": aic,
        "bic": bic,
        "TotSpk": np.sum(inp["spikes"]),
        "nParams": nParams,
        "trainFit": trainFit,
        "trainLogLik": trainLogLik,
        "alpha": alpha,
    })

    # cross-validation
    if "crossvalIdx" in inp and inp["crossvalIdx"] is not None and np.any(inp["crossvalIdx"]):
        cvIdx = np.array(inp["crossvalIdx"])
        nFold = int(np.nanmax(cvIdx))
        out["crossvalLogLik"] = np.full((nFold, n_models), np.nan)
        out["crossvalExclusion"] = np.full((nFold, n_models), np.nan)
        crossvalTotSpk = np.full(nFold, np.nan)
        crossvalNumIdx = np.full((nFold, 2), np.nan)

        for modelIdx, factors in enumerate(inp["modelsToCompare"]):
            numFactors = len(factors)
            testLogLik = np.full(nFold, np.nan)
            nExclude = np.full(nFold, np.nan)

            for iFold in range(1, nFold + 1):
                testIdx = cvIdx == iFold
                trainIdx = ~testIdx

                spikes_train, spikes_test = inp["spikes"][trainIdx], inp["spikes"][testIdx]

                if numFactors == 1:
                    fA = factors[0]
                    dwell_A = np.bincount(newFactors[fA][trainIdx], minlength=nLevelFactors[fA])
                    spikes_A = np.bincount(newFactors[fA][trainIdx], weights=spikes_train, minlength=nLevelFactors[fA])
                    A_ml = np.divide(spikes_A, dwell_A, out=np.zeros_like(spikes_A, float), where=dwell_A > 0)
                    fr_expected = A_ml[newFactors[fA][testIdx]]

                elif numFactors == 2:
                    fA, fB = factors
                    shape = (nLevelFactors[fA], nLevelFactors[fB])
                    idx_train = np.vstack((newFactors[fA][trainIdx], newFactors[fB][trainIdx])).T
                    dwell = np.zeros(shape)
                    np.add.at(dwell, tuple(idx_train.T), 1)
                    spikes_mat = np.zeros(shape)
                    np.add.at(spikes_mat, tuple(idx_train.T), spikes_train)
                    A_ml, B_ml, *_ = fit_two_factor(spikes_mat, dwell, max_iter, accuracy, tol)
                    fr_expected = (A_ml[newFactors[fA][testIdx]] * B_ml[newFactors[fB][testIdx]]) / (A_ml.mean() * B_ml.mean())

                elif numFactors == 3:
                    fA, fB, fC = factors
                    shape = (nLevelFactors[fA], nLevelFactors[fB], nLevelFactors[fC])
                    idx_train = np.vstack((newFactors[fA][trainIdx], newFactors[fB][trainIdx], newFactors[fC][trainIdx])).T
                    dwell = np.zeros(shape)
                    np.add.at(dwell, tuple(idx_train.T), 1)
                    spikes_mat = np.zeros(shape)
                    np.add.at(spikes_mat, tuple(idx_train.T), spikes_train)
                    A_ml, B_ml, C_ml, *_ = fit_three_factor(spikes_mat, dwell, max_iter, accuracy, tol)
                    fr_expected = (
                        A_ml[newFactors[fA][testIdx]] *
                        B_ml[newFactors[fB][testIdx]] *
                        C_ml[newFactors[fC][testIdx]]
                    ) / (A_ml.mean() * B_ml.mean() * C_ml.mean())

                fit, nExc = poisson_loglikelihood(spikes_test, fr_expected, tol)
                testLogLik[iFold - 1] = fit
                nExclude[iFold - 1] = nExc
                crossvalTotSpk[iFold - 1] = np.sum(spikes_test)
                crossvalNumIdx[iFold - 1, :] = [np.sum(trainIdx), np.sum(testIdx)]

            out["crossvalLogLik"][:, modelIdx] = testLogLik
            out["crossvalExclusion"][:, modelIdx] = nExclude

        out["crossvalTotSpk"] = crossvalTotSpk
        out["crossvalNum"] = crossvalNumIdx

    return out


def fit_three_factor(spikes, times, max_iter=100, accuracy=1e-6, tol=1e-3):
    """Fit ``spikes[x, y, z] ~ f1[x] f2[y] f3[z] times[x, y, z]`` by maximum likelihood.

    The three factors are updated in turn until the relative change of the Poisson
    log-likelihood falls below ``accuracy``. Each factor is then rescaled so that it predicts the
    total spike count with the dwell of its own bins, and set to NaN in bins without dwell.

    Returns
    -------
    f1, f2, f3 : ndarray
    converged : bool
    fit : float
        Final log-likelihood.
    scale_f1, scale_f2, scale_f3 : float
        Rescaling factors.
    """

    def f1_estimate(f2, f3, spikes, times, tol):
        f2_full = np.broadcast_to(f2[np.newaxis, :, np.newaxis], spikes.shape)
        f3_full = np.broadcast_to(f3[np.newaxis, np.newaxis, :], spikes.shape)
        denom = np.sum(f2_full * f3_full * times, axis=(1, 2))
        denom[denom < tol] = 1 / tol
        return np.sum(spikes, axis=(1, 2)) / denom

    def f2_estimate(f1, f3, spikes, times, tol):
        f1_full = np.broadcast_to(f1[:, np.newaxis, np.newaxis], spikes.shape)
        f3_full = np.broadcast_to(f3[np.newaxis, np.newaxis, :], spikes.shape)
        denom = np.sum(f1_full * f3_full * times, axis=(0, 2))
        denom[denom < tol] = 1 / tol
        return np.sum(spikes, axis=(0, 2)) / denom

    def f3_estimate(f1, f2, spikes, times, tol):
        f1_full = np.broadcast_to(f1[:, np.newaxis, np.newaxis], spikes.shape)
        f2_full = np.broadcast_to(f2[np.newaxis, :, np.newaxis], spikes.shape)
        denom = np.sum(f1_full * f2_full * times, axis=(0, 1))
        denom[denom < tol] = 1 / tol
        return np.sum(spikes, axis=(0, 1)) / denom

    def loglikelihood(f1, f2, f3, spikes, times, tol, spk_factorial):
        f1_full = np.broadcast_to(f1[:, np.newaxis, np.newaxis], spikes.shape)
        f2_full = np.broadcast_to(f2[np.newaxis, :, np.newaxis], spikes.shape)
        f3_full = np.broadcast_to(f3[np.newaxis, np.newaxis, :], spikes.shape)
        expected = np.maximum(f1_full * f2_full * f3_full * times, tol)
        return np.sum(spikes * np.log(expected) - expected - spk_factorial)

    spk_factorial = gammaln(spikes + 1)
    n1, n2, n3 = spikes.shape
    f1, f2, f3 = np.ones(n1), np.ones(n2), np.ones(n3)
    fit = loglikelihood(f1, f2, f3, spikes, times, tol, spk_factorial)
    converged = False

    for iter in range(max_iter):
        f1 = f1_estimate(f2, f3, spikes, times, tol)
        f2 = f2_estimate(f1, f3, spikes, times, tol)
        f3 = f3_estimate(f1, f2, spikes, times, tol)

        prev_fit = fit
        fit = loglikelihood(f1, f2, f3, spikes, times, tol, spk_factorial)

        frac_change = abs(fit - prev_fit) / (abs(prev_fit) + 1e-12)
        if frac_change < accuracy:
            converged = True
            break

    if not converged:
        warnings.warn(f"fit_three_factor did not converge after {max_iter} iterations "
                      f"(relative change {frac_change:.3e})", RuntimeWarning)

    # rescale so that each factor predicts the total spike count
    tot_spikes = np.sum(spikes)

    dwell_f1 = np.sum(times, axis=(1, 2))
    pred_spikes_f1 = np.sum(f1 * dwell_f1)
    scale_f1 = tot_spikes / pred_spikes_f1
    f1 *= scale_f1
    f1[dwell_f1 == 0] = np.nan

    dwell_f2 = np.sum(times, axis=(0, 2))
    pred_spikes_f2 = np.sum(f2 * dwell_f2)
    scale_f2 = tot_spikes / pred_spikes_f2
    f2 *= scale_f2
    f2[dwell_f2 == 0] = np.nan

    dwell_f3 = np.sum(times, axis=(0, 1))
    pred_spikes_f3 = np.sum(f3 * dwell_f3)
    scale_f3 = tot_spikes / pred_spikes_f3
    f3 *= scale_f3
    f3[dwell_f3 == 0] = np.nan

    return f1, f2, f3, converged, fit, scale_f1, scale_f2, scale_f3


def fit_two_factor(spikes, times, max_iter=100, accuracy=1e-6, tol=1e-6):
    """Fit ``spikes[x, y] ~ f1[x] f2[y] times[x, y]`` by maximum likelihood.

    As :func:`fit_three_factor` with two factors.

    Returns
    -------
    f1, f2 : ndarray
    converged : bool
    fit : float
    scale_f1, scale_f2 : float
    """

    def f1_estimate(f2, spikes, times, tol):
        f2_full = np.broadcast_to(f2[np.newaxis, :], spikes.shape)
        denom = np.sum(f2_full * times, axis=1)
        denom[denom < tol] = 1 / tol
        return np.sum(spikes, axis=1) / denom

    def f2_estimate(f1, spikes, times, tol):
        f1_full = np.broadcast_to(f1[:, np.newaxis], spikes.shape)
        denom = np.sum(f1_full * times, axis=0)
        denom[denom < tol] = 1 / tol
        return np.sum(spikes, axis=0) / denom

    def loglikelihood(f1, f2, spikes, times, tol, spk_factorial):
        f1_full = np.broadcast_to(f1[:, np.newaxis], spikes.shape)
        f2_full = np.broadcast_to(f2[np.newaxis, :], spikes.shape)
        expected = np.maximum(f1_full * f2_full * times, tol)
        return np.sum(spikes * np.log(expected) - expected - spk_factorial)

    spk_factorial = gammaln(spikes + 1)
    n1, n2 = spikes.shape
    f1 = np.ones(n1)
    f2 = np.ones(n2)
    fit = loglikelihood(f1, f2, spikes, times, tol, spk_factorial)
    converged = False

    for it in range(max_iter):
        f1 = f1_estimate(f2, spikes, times, tol)
        f2 = f2_estimate(f1, spikes, times, tol)

        prev_fit = fit
        fit = loglikelihood(f1, f2, spikes, times, tol, spk_factorial)
        frac_change = abs(fit - prev_fit) / (abs(prev_fit) + 1e-12)

        if frac_change < accuracy:
            converged = True
            break

    if not converged:
        warnings.warn(f"fit_two_factor did not converge after {max_iter} iterations "
                      f"(relative change {frac_change:.3e})", RuntimeWarning)

    tot_spikes = np.sum(spikes)

    dwell_f1 = np.sum(times, axis=1)
    pred_spikes_f1 = np.sum(f1 * dwell_f1)
    scale_f1 = tot_spikes / pred_spikes_f1
    f1 *= scale_f1
    f1[dwell_f1 == 0] = np.nan

    dwell_f2 = np.sum(times, axis=0)
    pred_spikes_f2 = np.sum(f2 * dwell_f2)
    scale_f2 = tot_spikes / pred_spikes_f2
    f2 *= scale_f2
    f2[dwell_f2 == 0] = np.nan

    return f1, f2, converged, fit, scale_f1, scale_f2


def fit_rank1_als(M, W, n_iter=100, tol=1e-6):
    """Weighted rank-1 factorisation ``M ~ f g^T`` by alternating least squares.

    Minimises ``sum(W_ij (M_ij - f_i g_j)^2)``. NaN entries of ``M`` are set to zero in place.
    ``f`` and ``g`` are finally scaled so that their means equal the dwell-weighted mean of
    ``M``, and set to NaN for rows or columns without weight.

    Parameters
    ----------
    M : (n_loc, n_dir) ndarray
        Mean firing rate per location and direction bin.
    W : (n_loc, n_dir) ndarray
        Weights (dwell).
    n_iter : int
    tol : float
        Relative change of the loss at convergence.

    Returns
    -------
    f : (n_loc,) ndarray
    g : (n_dir,) ndarray
    """
    n_loc, n_dir = M.shape

    M[np.isnan(M)] = 0  # missing data count as zero

    f = np.ones(n_loc)
    g = np.ones(n_dir)

    prev_loss = np.inf
    for it in range(n_iter):
        numerator = (W*M)@g
        denominator = W @ (g**2)
        valid = denominator > 0
        f[valid] = numerator[valid]/denominator[valid]

        numerator = (W * M).T @ f
        denominator = W.T @ (f**2)
        valid = denominator > 0
        g[valid] = numerator[valid] / denominator[valid]

        pred = f[:, None] * g[None, :]
        residual = M - pred
        loss = np.nansum(W * residual**2)

        if it > 0:
            rel_change = abs(prev_loss - loss) / (abs(prev_loss) + 1e-12)
        else:
            rel_change = np.inf
        if rel_change < tol:
            break
        prev_loss = loss

    FR_mean = np.nansum(M*W/W.sum())

    g *= FR_mean/np.nanmean(g)
    f *= FR_mean/np.nanmean(f)
    f[np.sum(W, axis=1) == 0] = np.nan
    g[np.sum(W, axis=0) == 0] = np.nan

    return f, g


def fit_location_parametric_poisson(spikes, x_bin, n_bins, var1, var2=None, gamma_bounds=(-np.inf, np.inf),
                                    beta_bounds=(-np.inf, np.inf), delta_t=1.0, max_iter=100, tol_loss=1e-6):
    """Fit ``mu_t = f[x_t] (1 + gamma var1_t) (1 + beta var2_t) delta_t`` to spike counts.

    Each iteration updates the spatial map ``f`` in closed form given ``gamma`` and ``beta``,
    then takes one Newton step with the expected Fisher information for ``gamma`` and one for
    ``beta`` (each clipped to its bounds). Writing ``mu_t = A_t + gamma B_t`` with
    ``B_t = f[x_t] (1 + beta var2_t) var1_t delta_t``, the step is
    ``score / fisher = sum(B (spikes - mu) / mu) / sum(B^2 / mu)``. Iterations stop when the
    relative change of ``sum(mu - spikes log mu)`` falls below ``tol_loss``.

    Parameters
    ----------
    spikes : ndarray
        Spike counts per sample.
    x_bin : ndarray of int
        Spatial bin of each sample.
    n_bins : int
        Number of spatial bins.
    var1 : ndarray
        First modulating variable.
    var2 : ndarray, optional
        Second modulating variable; without it ``beta`` stays 0.
    gamma_bounds, beta_bounds : sequence of two floats
    delta_t : float
    max_iter : int
    tol_loss : float

    Returns
    -------
    f : ndarray
        Spatial map (NaN for bins without samples).
    gamma, beta : float
    is_converged : bool or int
        True when the tolerance was reached (0 otherwise).
    """
    f = np.ones(n_bins)
    gamma = 0.0
    beta = 0.0

    prev_loss = np.inf
    is_converged = 0

    if var2 is None:
        skipBetaFit = True
        var2 = 0
    else:
        skipBetaFit = False
    for it in range(max_iter):

        # 1) spatial map f
        A = (1 + gamma * var1) * (1 + beta * var2)

        numerator = np.zeros(n_bins)
        denominator = np.zeros(n_bins)

        np.add.at(numerator, x_bin, spikes)
        np.add.at(denominator, x_bin, A * delta_t)

        valid = denominator > 0
        f[valid] = numerator[valid] / denominator[valid]

        # 2) Newton step for gamma
        C = (1 + beta * var2)
        A0 = f[x_bin] * C * delta_t
        B = f[x_bin] * C * var1 * delta_t

        mu = A0 + gamma * B
        mu = np.clip(mu, 1e-12, None)

        score = np.sum(B * (spikes - mu) / mu)
        fisher = np.sum(B**2 / mu)

        if fisher > 0:
            gamma += score / fisher
        gamma = np.clip(gamma, gamma_bounds[0], gamma_bounds[1])

        # 3) Newton step for beta
        if skipBetaFit:
            beta = 0
        else:
            D = (1 + gamma * var1)
            A0 = f[x_bin] * D * delta_t
            B = f[x_bin] * D * var2 * delta_t

            mu = A0 + beta * B
            mu = np.clip(mu, 1e-12, None)

            score = np.sum(B * (spikes - mu) / mu)
            fisher = np.sum(B**2 / mu)

            if fisher > 0:
                beta += score / fisher
            beta = np.clip(beta, beta_bounds[0], beta_bounds[1])

        # 4) negative log-likelihood (without the constant term)
        mu_full = (
            f[x_bin]
            * (1 + gamma * var1)
            * (1 + beta * var2)
            * delta_t
        )
        mu_full = np.clip(mu_full, 1e-12, None)

        loss = np.sum(mu_full - spikes * np.log(mu_full))

        if it > 0:
            rel_change = abs(prev_loss - loss) / (abs(prev_loss) + 1e-12)
        else:
            rel_change = np.inf

        if rel_change < tol_loss:
            is_converged = True
            break

        prev_loss = loss
    # bins without samples
    occupancy = np.zeros(n_bins)
    np.add.at(occupancy, x_bin, 1)
    f[occupancy == 0] = np.nan

    if rel_change >= tol_loss:
        warnings.warn(f"fit_location_parametric_poisson did not converge after {max_iter} iterations "
                      f"(relative loss change {rel_change:.3e})", RuntimeWarning)
    return f, gamma, beta, is_converged
