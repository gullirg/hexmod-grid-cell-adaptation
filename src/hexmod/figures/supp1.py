"""Supplementary Figure 1: adaptation coefficient fitted separately to the odd and the even minutes of each recording.

Panels
    A  Box plots of the adaptation coefficient alpha (model with location input) in odd and in even minutes; asterisks:
       one-sample t-tests against 0, drawn below the lower whisker ends.
    B  Odd against even minutes, one point per cell, with the least-squares line.

Input (``rat/`` in the results folder): ``adaptation/param_cells.csv`` (columns ``alpha_tprevXloc_oddMinute`` and
``alpha_tprevXloc_evenMinute``). The statistics (one-sample and paired t-tests, Pearson r, fitted line) are computed
from the plotted arrays. Figure size 114 x 48 mm; positions in mm from the top left corner of the figure.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy import stats

from .. import io as hio
from . import common
from . import style as hs

FIGURE = "SuppFig1"
W, H = 114.0, 48.0
COLB = 57.0                                     # left edge of the second column (mm)
ROW1 = hs.LETTER_TOP                            # top of the capitals of letters A and B (mm)
TOP, BOXH = hs.panels_top(ROW1), 36.0           # top (style.LETTER_ABOVE under the letters), height of the boxes
AX_A = dict(left=19.0, top=TOP, width=32.0, height=BOXH)
AX_B = dict(left=75.0, top=TOP, width=BOXH, height=BOXH)    # square box; equal limits on both axes give equal scales
COL_ODD, COL_EVEN = "alpha_tprevXloc_oddMinute", "alpha_tprevXloc_evenMinute"
LIM_B = (-0.22, 0.22)


def load_inputs(results_dir):
    """Read the per-cell adaptation coefficients of the odd and the even minutes."""
    return hio.read_csv(common.find_input(results_dir, "rat/adaptation/param_cells.csv"))


def build(results_dir, out_dir):
    """Draw Supplementary Figure 1 from ``results_dir`` into ``out_dir``."""
    cells = load_inputs(results_dir)
    with mpl.rc_context():
        hs.use_style()
        _draw(out_dir, cells)


def _draw(out_dir, cells):
    odd = cells[COL_ODD].to_numpy(float)
    even = cells[COL_EVEN].to_numpy(float)

    # statistics of the plotted arrays
    one = [stats.ttest_1samp(g, 0) for g in (odd, even)]
    t_pair, p_pair = stats.ttest_rel(odd, even)
    r_, p_r = stats.pearsonr(odd, even)
    slope, intercept = np.polyfit(odd, even, 1)          # least-squares line

    fig = hs.new_figure(W, H)
    P = hs.PALETTE

    # ---- A: box plots of the two halves, asterisks for the one-sample tests against zero
    axA = hs.axes_mm(fig, **AX_A)
    hs.panel_letter(fig, "A", x=hs.LETTER_X, row_top=ROW1, clear_of=[axA], not_above=ROW1, row=1)
    axA.set_xlim(-0.55, 1.55)
    axA.set_ylim(-0.26, 0.16)
    ends = hs.boxplot(axA, [odd, even], positions=[0, 1], width=hs.box_width_data(axA))   # boxes 7.6 mm wide
    hs.zero_line(axA)
    axA.set_xticks([0, 1])
    axA.set_xticklabels([s.replace(" ", "\n") for s in hs.TICKS["halves"]])
    axA.set_yticks([-0.2, -0.1, 0, 0.1])
    axA.yaxis.set_major_formatter(hs.NUM)
    axA.set_ylabel("Adaptation\ncoefficient " + hs.VAR["alpha"])
    p_one = [r.pvalue for r in one]
    low_A = min(lo for lo, hi in ends)              # both marks on one line, just below the lower whisker end
    for x, p in zip([0, 1], p_one):
        hs.sig_mark(axA, x, low_A, p, side="below")

    # ---- B: scatter of the two halves and the fitted line
    axB = hs.axes_mm(fig, **AX_B)
    hs.panel_letter(fig, "B", x=COLB + hs.LETTER_X, row_top=ROW1, clear_of=[axB], not_above=ROW1, row=1)
    axB.scatter(odd, even, s=3.4, c=P["dot"], alpha=0.6, linewidths=0, zorder=2)
    x_fit = np.linspace(odd.min(), odd.max(), 100)
    y_fit = slope * x_fit + intercept
    axB.plot(x_fit, y_fit, color=P["ink"], linewidth=hs.LW_REF, zorder=3, solid_capstyle="butt")
    axB.set_xlim(LIM_B)
    axB.set_ylim(LIM_B)
    for a in (axB.xaxis, axB.yaxis):
        a.set_major_formatter(hs.NUM)
    axB.set_xticks([-0.2, 0, 0.2])
    axB.set_yticks([-0.2, 0, 0.2])
    axB.set_xlabel(hs.LABEL["alpha_odd"])
    axB.set_ylabel(hs.LABEL["alpha_even"])
    outside = int(((np.abs(odd) >= LIM_B[1]) | (np.abs(even) >= LIM_B[1])).sum())
    if outside:
        warnings.warn(f"{FIGURE}B: {outside} cells lie outside the axes limits {LIM_B} and are not shown")

    common.save_outputs(fig, out_dir, FIGURE)

    # ---- source data
    src = pd.DataFrame({"row": cells["row"].values, "animal_id": cells["animal_id"].values,
                        "session_id": cells["session_id"].values, "cell_id": cells["cell_id"].values,
                        "alpha_odd_minutes": odd, "alpha_even_minutes": even})
    common.write_source_data(out_dir, FIGURE, "", "adaptation_coefficient_odd_even_minutes", src)
    box_rows = [dict(panel="A", group=g, **s) for g, s in zip(hs.TICKS["halves"], hs.box_stats([odd, even]))]
    common.write_source_data(out_dir, FIGURE, "A", "box_numbers", pd.DataFrame(box_rows))
    common.write_source_data(out_dir, FIGURE, "B", "fitted_line", {"alpha_odd_minutes": x_fit,
                                                                    "alpha_even_minutes": y_fit})
    p_holm = hs.holm(p_one)                     # over the two halves (the marks show the uncorrected p, both < 0.001)
    rows = []

    def addrow(panel, comparison, test, n, statistic, df, p, p_holm=np.nan, mean=np.nan, median=np.nan,
               fraction_negative=np.nan, mark_drawn="not drawn"):
        rows.append(dict(panel=panel, comparison=comparison, test=test, n=n, statistic=statistic, df=df, p=p,
                         p_holm=p_holm, mean=mean, median=median, fraction_negative=fraction_negative,
                         mark_drawn=mark_drawn))

    for k, (half, g, r) in enumerate(zip(("odd minutes", "even minutes"), (odd, even), one)):
        addrow("A", f"{half} against 0", "one-sample t", g.size, r.statistic, g.size - 1, r.pvalue, p_holm[k],
               g.mean(), np.median(g), (g < 0).mean(), hs.sig_stars(r.pvalue) or "none")
    addrow("A", "odd against even minutes", "paired t", odd.size, t_pair, odd.size - 1, p_pair,
           mean=(odd - even).mean())
    addrow("B", "odd against even minutes", "Pearson r", odd.size, r_, odd.size - 2, p_r)
    addrow("B", "slope of the fitted line", "least squares", odd.size, slope, np.nan, np.nan)
    addrow("B", "intercept of the fitted line", "least squares", odd.size, intercept, np.nan, np.nan)
    addrow("B", "cells with alpha below 0 in both halves", "fraction", odd.size, ((odd < 0) & (even < 0)).mean(),
           np.nan, np.nan)
    addrow("B", "cells with alpha at or above 0 in both halves", "fraction", odd.size,
           ((odd >= 0) & (even >= 0)).mean(), np.nan, np.nan)
    common.write_source_data(out_dir, FIGURE, "", "statistics", pd.DataFrame(rows))
