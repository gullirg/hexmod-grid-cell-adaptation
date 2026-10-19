"""Supplementary Figure 4: firing rate change against the time since the previous field visit, with exponential fits.

One panel: mean firing rate change (percent of each cell's mean rate) in nine logarithmically spaced bins of t_prev,
with 95% intervals (mean plus or minus 1.96 s.e.m. across cells), and the weighted least-squares fits of one exponential
(teal, solid) and of two exponentials (orange, dashed). The key gives the fitted time constants.

Inputs (``rat/timescales/`` in the results folder): ``supp4_points.csv`` (bin means and s.e.m.) and
``supp4_fits.csv`` (fit parameters). The y range is the automatic range of the curves on a logarithmic grid of
t_prev from 0.04 to 25 s and of the error bars (data range plus 5% margins), with ticks every 10%. Figure size
85 x 66 mm; positions in mm from the top left corner of the figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl

from .. import io as hio
from . import common
from . import style as hs

FIGURE = "SuppFig4"
W, H = 85.0, 66.0                 # one column, one box, no letter
KEY_UP = 4.29                     # the key's words start at the top margin (1.5 mm)
BOX_W, BOX_H = 60.0, 40.0 + KEY_UP
BOX_TOP = 13.0 - KEY_UP           # 8.71 mm; the box ends at 53 mm
KEY_ROW_TOP = hs.LETTER_TOP + hs.TITLE_PITCH - KEY_UP
X_MAX, X_TICKS = 16.0, [0, 5, 10, 15]
RANGE_GRID = np.geomspace(0.04, 25, 300)     # t_prev grid (s) that sets the y range
MARGIN = 0.05


def exp1(t, a, tau, c):
    """One exponential: c + a exp(-t / tau)."""
    return c + a * np.exp(-t / tau)


def exp2(t, a1, t1, a2, t2, c):
    """Two exponentials: c + a1 exp(-t / t1) + a2 exp(-t / t2)."""
    return c + a1 * np.exp(-t / t1) + a2 * np.exp(-t / t2)


def load_inputs(results_dir):
    """Read the bin means and the fit parameters of Supplementary Figure 4."""
    pts = hio.read_csv(common.find_input(results_dir, "rat/timescales/supp4_points.csv"))
    fits = hio.read_csv(common.find_input(results_dir, "rat/timescales/supp4_fits.csv"))
    return pts, fits


def build(results_dir, out_dir):
    """Draw Supplementary Figure 4 from ``results_dir`` into ``out_dir``."""
    inputs = load_inputs(results_dir)
    with mpl.rc_context():
        hs.use_style()
        _draw(out_dir, *inputs)


def y_range(f, D, y, e):
    """Return the y limits (data range of the curves on RANGE_GRID, the error bars and 0, plus 5% margins) and ticks."""
    vals = np.concatenate([exp1(RANGE_GRID, f["a"], f["tau"], f["c"]),
                           exp2(RANGE_GRID, D["a_fast"], D["tau_fast"], D["a_slow"], D["tau_slow"], D["c"]),
                           y - 1.96 * e, y + 1.96 * e, y, [0.0]])
    lo, hi = float(vals.min()), float(vals.max())
    pad = (hi - lo) * MARGIN
    ylim = (lo - pad, hi + pad)
    yticks = [float(v) for v in np.arange(np.ceil(ylim[0] / 10.0), np.floor(ylim[1] / 10.0) + 1) * 10.0]
    return ylim, yticks


def _draw(out_dir, PTS, FITS):
    PAL, VAR, LAB = hs.PALETTE, hs.VAR, hs.LABEL
    x = PTS["x_s"].to_numpy(float)
    y = PTS["mean_pct_change"].to_numpy(float)
    e = PTS["sem_pct_change"].to_numpy(float)
    par = {(m, p): v for m, p, v in zip(FITS["model"], FITS["parameter"], FITS["value"])}
    f = {"a": par[("one exponential", "a_pct")], "tau": par[("one exponential", "tau_s")],
         "c": par[("one exponential", "c_pct")], "chi2": par[("one exponential", "chi2_weighted")]}
    D = {"a_fast": par[("two exponentials", "a_fast_pct")], "tau_fast": par[("two exponentials", "tau_fast_s")],
         "a_slow": par[("two exponentials", "a_slow_pct")], "tau_slow": par[("two exponentials", "tau_slow_s")],
         "c": par[("two exponentials", "c_pct")], "chi2": par[("two exponentials", "chi2_weighted")]}
    ylim, yticks = y_range(f, D, y, e)

    fig = hs.new_figure(W, H)
    r = fig.canvas.get_renderer()
    # horizontal place of the box: the span from the left edge of the y label to the right edge of the box is centred
    probe = hs.axes_mm(fig, 20.0, BOX_TOP, BOX_W, BOX_H)
    probe.set_ylim(*ylim)
    probe.set_yticks(yticks)
    probe.yaxis.set_major_formatter(hs.NUM)
    probe.set_ylabel(LAB["fr_change"])
    fig.canvas.draw()
    lab_left_mm = probe.yaxis.label.get_window_extent(r).x0 / fig.dpi * 25.4
    reach = 20.0 - lab_left_mm
    probe.remove()
    box_left = round((W - (reach + BOX_W)) / 2.0 + reach, 2)
    ax = common.tag(hs.axes_mm(fig, box_left, BOX_TOP, BOX_W, BOX_H), panel="", name="curves and points")
    tt = np.linspace(0.0, X_MAX, 4001)
    hs.zero_line(ax)
    C_ONE, LS_ONE, C_TWO, LS_TWO = PAL["aligned"], "-", PAL["misaligned"], (0, (4, 2))   # teal solid, orange dashed
    l1, = ax.plot(tt, exp1(tt, f["a"], f["tau"], f["c"]), color=C_ONE, linestyle=LS_ONE, linewidth=hs.LW_REF, zorder=2)
    l2, = ax.plot(tt, exp2(tt, D["a_fast"], D["tau_fast"], D["a_slow"], D["tau_slow"], D["c"]), color=C_TWO,
                  linestyle=LS_TWO, linewidth=hs.LW_REF, zorder=3)
    eb = hs.errorbars(ax, x, y, 1.96 * e)            # black points, no caps, 0.8 pt bars
    eb[0].set_markersize(2.5)
    eb[0].set_markeredgewidth(0.5)
    eb[0].set_markeredgecolor(PAL["ink"])
    eb[0].set_clip_on(False)                          # the first point lies next to the left axis: drawn whole
    for c in eb[2]:
        c.set_clip_on(False)
    ax.set_xlim(0.0, X_MAX)
    ax.set_xticks(X_TICKS)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.xaxis.set_major_formatter(hs.NUM)
    ax.yaxis.set_major_formatter(hs.NUM)
    ax.set_xlabel(LAB["t_prev"])
    ax.set_ylabel(LAB["fr_change"])
    k1 = f"{VAR['tau']} = {f['tau']:.2f} s"
    k2 = f"{VAR['tau1']} = {D['tau_fast']:.2f} s, {VAR['tau2']} = {D['tau_slow']:.1f} s"
    # the key carries subscripts: 10 pt text puts them at 7 pt; it sits above the box, flush right
    hs.line_key(fig, [(k1, C_ONE, LS_ONE), (k2, C_TWO, LS_TWO)], right=box_left + BOX_W,
                row_top=KEY_ROW_TOP, fontsize=hs.FS_SCRIPT_TEXT, lw=hs.LW_REF)

    common.save_outputs(fig, out_dir, FIGURE)

    # ---- source data
    src = pd.DataFrame({"bin": PTS["bin"].values, "bin_low_s": PTS["edge_low_s"].values,
                        "bin_high_s": PTS["edge_high_s"].values, "t_prev_s_plotted": x,
                        "cells": PTS["cells_in_bin"].values, "mean_firing_rate_change_pct": y, "sem_pct": e,
                        "bar_low_pct_mean_minus_1.96sem": y - 1.96 * e, "bar_high_pct_mean_plus_1.96sem": y + 1.96 * e})
    common.write_source_data(out_dir, FIGURE, "", "points", src)
    n = len(x)
    fits = pd.DataFrame([("one exponential (teal, solid)", "c + a exp(-t/tau)", f["a"], f["tau"], f["c"], np.nan,
                          np.nan,
                          f["chi2"], n - 3),
                         ("two exponentials (orange, dashed)", "c + a1 exp(-t/tau1) + a2 exp(-t/tau2)", D["a_fast"],
                          D["tau_fast"], D["c"], D["a_slow"], D["tau_slow"], D["chi2"], n - 5)],
                        columns=["model", "formula", "a_or_a1_pct", "tau_or_tau1_s", "c_pct", "a2_pct", "tau2_s",
                                 "weighted_chi_square", "residual_df"])
    common.write_source_data(out_dir, FIGURE, "", "fits", fits)
    common.write_source_data(out_dir, FIGURE, "", "curves", {"t_prev_s": tt, "one_exponential_pct": l1.get_ydata(),
                                                             "two_exponentials_pct": l2.get_ydata()})
