"""Supplementary Figure 5: six-fold component of a von Mises tuning curve against its circular standard deviation.

Panels
    A  Linear scale: the six-fold component (amplitude of the cos 6(theta - mu) term over the mean, 2 I6(kappa) /
       I0(kappa), in percent) falls steeply as tuning broadens; the narrow tuning of Fig 1C (15 deg) and cosine tuning
       (circular SD 67.5 deg, no six-fold component) are marked.
    B  Log scale, with the narrowest and the median conjunctive cell marked, and below it the distribution of the
       circular SD of the conjunctive cells (resultant length above 0.3).

Inputs: the closed form of ``hexmod.sim.analytic`` (circular SD 2 to 95 deg, 940 points) and
``rat/figure_tables/supp5_widths.csv`` (circular SD of each conjunctive cell). Figure size 114 x 62 mm; positions in mm
from the top left corner of the figure.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

from .. import io as hio
from ..sim import analytic
from . import common
from . import style as hs

FIGURE = "SuppFig5"
H = 62.0
LINE = hs.LETTER_TOP                   # top of the capitals of letters A and B (mm)
TOP = hs.panels_top(LINE)              # top of the frames of A and of the upper plot of B, style.LETTER_ABOVE lower
A_BOTTOM, B_BOTTOM = 49.0, 39.0        # bottoms of the frames of A and of the upper plot of B
# both y labels at 10 pt and at the same height, centred 31.5 mm from the top, so that their upper ends lie 1.78 mm
# under the letters' baseline
YLABEL_MID = 31.5
COSINE_SD = 67.46                      # circular SD of cosine tuning (deg)
NARROW_SD = 15.0                       # narrow tuning of Fig 1C (deg)
LOG_TICK_LABELS = {1e-4: "0.0001", 1e-2: "0.01", 1: "1", 100: "100"}


def load_inputs(results_dir):
    """Read the circular SD of the conjunctive cells."""
    return hio.read_csv(common.find_input(results_dir, "rat/figure_tables/supp5_widths.csv"))


def build(results_dir, out_dir):
    """Draw Supplementary Figure 5 from ``results_dir`` into ``out_dir``."""
    widths = load_inputs(results_dir)
    with mpl.rc_context():
        hs.use_style()
        _draw(out_dir, widths)


def _sixfold_pct(sd_deg):
    return analytic.sixfold_from_sd(sd_deg) * 100


def _draw(out_dir, widths):
    P = hs.PALETTE
    SD = analytic.SIXFOLD_SDS
    Y = analytic.sixfold_curve(SD) * 100                 # percent of the mean
    W = widths["circular_sd_deg"].to_numpy(float)        # circular SD of each conjunctive cell
    has_cells = W.size > 0                               # synthetic test data may have no conjunctive cell
    med = float(np.median(W)) if has_cells else np.nan
    wmin = float(W.min()) if has_cells else np.nan
    marks = {s: _sixfold_pct(s) for s in ((NARROW_SD, wmin, med) if has_cells else (NARROW_SD,))}
    if not has_cells:
        warnings.warn(f"{FIGURE}B: no conjunctive cells; the cell widths are left out")

    fig = hs.new_figure("onehalf", H)
    # ---- A: linear scale
    let_a = hs.panel_letter(fig, "A", row_top=LINE, not_above=LINE, row=1)
    axA = common.tag(hs.axes_mm(fig, 13.0, TOP, 40.0, A_BOTTOM - TOP), panel="A", name="A linear scale")
    axA.plot(SD, Y, color=P["ink"], lw=hs.LW_DATA)
    axA.set_xlim(0, 90)
    axA.set_ylim(0, 200)
    axA.xaxis.set_major_locator(FixedLocator([0, 30, 60, 90]))
    axA.xaxis.set_major_formatter(hs.DEG)
    axA.yaxis.set_major_locator(FixedLocator([0, 50, 100, 150, 200]))
    axA.yaxis.set_major_formatter(hs.NUM)
    axA.set_xlabel("Circular SD of tuning")
    axA.set_ylabel("Six-fold component (% of mean)", y=(A_BOTTOM - YLABEL_MID) / (A_BOTTOM - TOP))
    axA.plot([NARROW_SD], [marks[NARROW_SD]], "o", ms=4, color=P["ink"], zorder=4)
    axA.annotate(f"{NARROW_SD:.0f}°: {marks[NARROW_SD]:.0f}%\n(Fig 1C, narrow)", (NARROW_SD, marks[NARROW_SD]),
                 xytext=(24, 120), fontsize=hs.FS_SMALL,
                 arrowprops=dict(arrowstyle="-", lw=hs.LW_GUIDE, color=P["guide"]))
    axA.plot([COSINE_SD], [0], "s", ms=4, mfc="white", mec=P["ink"], mew=hs.LW_FRAME, zorder=5, clip_on=False)
    axA.annotate(f"Cosine tuning\n({COSINE_SD:.1f}°): 0", (COSINE_SD, 0), xytext=(52, 45), fontsize=hs.FS_SMALL,
                 arrowprops=dict(arrowstyle="-", lw=hs.LW_GUIDE, color=P["guide"]))

    # ---- B: log scale, with the observed widths
    let_b = hs.panel_letter(fig, "B", x=62.0, row_top=LINE, not_above=LINE, row=1)
    axB = common.tag(hs.axes_mm(fig, 75.0, TOP, 36.0, B_BOTTOM - TOP), panel="B", name="B log scale")
    axB.semilogy(SD, Y, color=P["ink"], lw=hs.LW_DATA)
    axB.set_xlim(0, 90)
    axB.set_ylim(1e-4, 300)
    axB.yaxis.set_major_locator(FixedLocator([1e-4, 1e-2, 1, 100]))
    axB.yaxis.set_major_formatter(FuncFormatter(lambda v, _: LOG_TICK_LABELS.get(v, "")))
    axB.yaxis.set_minor_formatter(NullFormatter())
    axB.yaxis.set_minor_locator(NullLocator())           # no minor ticks on the log axis
    axB.xaxis.set_major_locator(FixedLocator([0, 30, 60, 90]))
    axB.xaxis.set_major_formatter(NullFormatter())
    # the same size and height as the y label of A; it runs beside the histogram strip, which has no y label
    axB.set_ylabel("Six-fold component (% of mean)", y=(B_BOTTOM - YLABEL_MID) / (B_BOTTOM - TOP))
    if has_cells:
        for s0 in (wmin, med):
            axB.plot([s0], [marks[s0]], "o", ms=3.5, color=P["ink"], zorder=4)
        axB.axvline(med, color=P["guide"], lw=hs.LW_GUIDE, ls=(0, (3, 2)), zorder=1)
        axB.annotate(f"Narrowest cell\n{wmin:.0f}°: {marks[wmin]:.1f}%", (wmin, marks[wmin]), xytext=(40, 6),
                     fontsize=hs.FS_SMALL, bbox=dict(fc="white", ec="none", pad=0.6), zorder=6,
                     arrowprops=dict(arrowstyle="-", lw=hs.LW_GUIDE, color=P["guide"]))
        axB.annotate(f"Median {med:.0f}°:\n{marks[med]:.3f}%", (med, marks[med]), xytext=(6, 1.2e-3),
                     fontsize=hs.FS_SMALL, arrowprops=dict(arrowstyle="-", lw=hs.LW_GUIDE, color=P["guide"]))
    axH = common.tag(hs.axes_mm(fig, 75.0, 41.0, 36.0, 8.0, sides=("bottom",)), panel="B",
                     name="B histogram of cell widths")                    # the strip has no y scale: bottom spine only
    counts, edges, _ = axH.hist(W, bins=np.arange(0, 91, 2.5), color=P["neutral_fill"], edgecolor="white",
                                linewidth=0.3)
    if has_cells:
        axH.axvline(med, color=P["guide"], lw=hs.LW_GUIDE, ls=(0, (3, 2)))
    axH.set_xlim(0, 90)
    axH.set_yticks([])
    axH.xaxis.set_major_locator(FixedLocator([0, 30, 60, 90]))
    axH.xaxis.set_major_formatter(hs.DEG)
    axH.set_xlabel("Circular SD of tuning")
    hs.add_clearance(let_a, axA)                         # each letter clears its plot's y label and y tick labels
    hs.add_clearance(let_b, axB)

    common.save_outputs(fig, out_dir, FIGURE)

    # ---- source data
    common.write_source_data(out_dir, FIGURE, "AB", "sixfold_component_curve",
                             {"circular_sd_deg": SD, "sixfold_percent_of_mean": Y})
    common.write_source_data(out_dir, FIGURE, "B", "conjunctive_cell_widths",
                             {"row": widths["row"].values, "Rayleigh_mag": widths["Rayleigh_mag"].values,
                              "circular_sd_deg": W})
    common.write_source_data(out_dir, FIGURE, "B", "width_histogram",
                             {"bin_left_deg": edges[:-1], "bin_right_deg": edges[1:], "cells": counts.astype(int)})
    points = [dict(panel="A", point="narrow tuning of Fig 1C", circular_sd_deg=NARROW_SD,
                   sixfold_percent_of_mean=marks[NARROW_SD]),
              dict(panel="A", point="cosine tuning", circular_sd_deg=COSINE_SD, sixfold_percent_of_mean=0.0)]
    if has_cells:
        points += [dict(panel="B", point="narrowest conjunctive cell", circular_sd_deg=wmin,
                        sixfold_percent_of_mean=marks[wmin]),
                   dict(panel="B", point="median conjunctive cell", circular_sd_deg=med,
                        sixfold_percent_of_mean=marks[med])]
    common.write_source_data(out_dir, FIGURE, "", "marked_points", pd.DataFrame(points))
