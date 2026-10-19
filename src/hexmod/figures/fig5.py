"""Figure 5: hexadirectional modulation (HexMod beta) of recorded grid cells.

Panels
    A  HexMod beta of all cells; one-sample t-test against 0.
    B  Cells in the top third of the proximity difference (aligned minus misaligned), split by thirds of the
       adaptation coefficient (Strong, Weak, No adaptation); asterisks: one-sample t-tests against 0; brackets:
       two-sample t-tests between groups with Holm correction within the panel.
    C  Cells with strong adaptation, top against bottom third of the proximity difference; asterisks: one-sample
       t-tests against 0; bracket: two-sample t-test.

Inputs (``rat/`` in the results folder): ``figure_tables/cells_thirds.csv`` (beta and the thirds of the adaptation
coefficient and of the proximity difference) and ``cohort/cells.csv``. All boxes are 7.6 mm wide and the three
panels share one y range. Positions in mm from the top left corner of the figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy import stats

from .. import io as hio
from . import common
from . import style as hs

FIGURE = "Fig5"
W, H = hs.WIDTH["onehalf"], 63.0
YLIM, YTICKS = (-0.15, 0.15), [-0.1, 0, 0.1]
ROW_TOP = hs.LETTER_TOP                           # top of the capitals of the letters (1.5 mm)
FRAME_CAP = hs.LW_FRAME * hs.PT / 2.0             # half the frame line, which projects above the box top
# The frame lines of B and C pass just right of letters B and C, so the boxes start where the frame lines are
# style.LETTER_CLEAR below the letters' baseline; the boxes end at 50 mm.
BOX_TOP = ROW_TOP + hs.CAP12 + hs.LETTER_CLEAR + FRAME_CAP     # 5.17 mm
BOX_BOTTOM = 50.0
BOX_H = BOX_BOTTOM - BOX_TOP                      # 44.83 mm
BOX_MM = 7.6                                      # every box is 7.6 mm wide, whatever the panel width
COL = {"A": 0.0, "B": 28.0, "C": 74.5}            # left edge of each column (letter at +1.5 mm)
BOX = {"A": (12.5, 14.0), "B": (33.0, 40.0), "C": (79.5, 33.0)}     # box of each panel (left, width), mm
XLIM = {"A": (-0.5, 0.5), "B": (-0.5, 2.5), "C": (-0.5, 1.5)}


def load_inputs(results_dir):
    """Read the result files of Figure 5."""
    thirds = hio.read_csv(common.find_input(results_dir, "rat/figure_tables/cells_thirds.csv"))
    cells = hio.read_csv(common.find_input(results_dir, "rat/cohort/cells.csv"))
    if not np.array_equal(thirds["row"].values, cells["row"].values):
        raise ValueError("cells_thirds.csv is not row-aligned with cells.csv")
    return thirds, cells


def build(results_dir, out_dir):
    """Draw Figure 5 from ``results_dir`` into ``out_dir``."""
    inputs = load_inputs(results_dir)
    with mpl.rc_context():
        hs.use_style()
        _draw(out_dir, *inputs)


def _draw(out_dir, thirds, cells):
    beta = thirds["beta"].to_numpy()
    adapt_cat = thirds["adaptation_third"].to_numpy()        # 1 strong, 2 weak, 3 no adaptation
    prox_cat = thirds["proximity_third"].to_numpy()          # 1 bottom to 3 top
    B_mask1, B_mask2, B_mask3 = ((prox_cat == 3) & (adapt_cat == k) for k in (1, 2, 3))
    C_mask1, C_mask2, C_mask3 = ((prox_cat == k) & (adapt_cat == 1) for k in (3, 2, 1))
    groups = {"A": [beta], "B": [beta[B_mask1], beta[B_mask2], beta[B_mask3]], "C": [beta[C_mask1], beta[C_mask3]]}

    # statistics
    tA, pA = stats.ttest_1samp(beta, 0)
    F, pF = stats.f_oneway(*groups["B"])
    oneB = [stats.ttest_1samp(g, popmean=0, nan_policy="omit") for g in groups["B"]]
    pairs_B = [(0, 1), (0, 2), (1, 2)]
    p_pair_B = [stats.ttest_ind(groups["B"][i], groups["B"][j]).pvalue for i, j in pairs_B]
    t_pair_B = [stats.ttest_ind(groups["B"][i], groups["B"][j]).statistic for i, j in pairs_B]
    p_holm_B = hs.holm(p_pair_B)
    rC = stats.ttest_ind(groups["C"][0], groups["C"][1])
    oneC = [stats.ttest_1samp(g, popmean=0, nan_policy="omit") for g in (beta[C_mask1], beta[C_mask2], beta[C_mask3])]

    fig = hs.new_figure("onehalf", H)
    axes = {}
    for key in "ABC":
        left, width = BOX[key]
        ax = hs.axes_mm(fig, left, BOX_TOP, width, BOX_H)
        axes[key] = ax
        # the letter must clear the panel's y label, y tick labels and frame line by style.LETTER_CLEAR
        hs.panel_letter(fig, key, COL[key] + hs.LETTER_X, ROW_TOP, clear_of=[ax, ax.spines["left"]], not_above=ROW_TOP)
        ax.set_xlim(*XLIM[key])
        ax.set_ylim(*YLIM)
        unit_mm = width / (XLIM[key][1] - XLIM[key][0])
        hs.zero_line(ax)
        ax._ends = hs.boxplot(ax, groups[key], width=BOX_MM / unit_mm)
        ax.set_yticks(YTICKS)
        ax.yaxis.set_major_formatter(hs.NUM)
    axA, axB, axC = axes["A"], axes["B"], axes["C"]
    axA.set_ylabel(hs.LABEL["hexmod_beta"])
    axA.set_xticks([])                                            # the legend names the group
    for k in "BC":
        axes[k].tick_params(axis="y", labelleft=False)            # one y range for the row: numbers and label on A only
    axB.set_xticks([0, 1, 2])
    axB.set_xticklabels(hs.TICKS["adaptation_groups"])
    axB.set_xlabel(hs.LABEL["adaptation_strength"])
    axC.set_xticks([0, 1])
    axC.set_xticklabels(hs.TICKS["proximity_groups"])
    axC.set_xlabel(hs.LABEL["proximity_diff"], linespacing=1.1)

    # significance against zero: asterisks beyond the whisker, on the side away from zero
    for key, rs in (("A", [stats.ttest_1samp(beta, 0)]), ("B", oneB), ("C", [oneC[0], oneC[2]])):
        for i, r in enumerate(rs):
            lo, hi = axes[key]._ends[i]
            side = "below" if r.statistic < 0 else "above"
            hs.sig_mark(axes[key], i, lo if side == "below" else hi, r.pvalue, side=side)
    # between groups: B pairwise two-sample t-tests with Holm within the panel; C one two-sample t-test
    yB = max(axB._ends[i][1] for i in range(3)) + hs.pt_to_data(axB, 3.0)
    hs.bracket_stack(axB, pairs_B, p_holm_B, yB)
    yC = max(axC._ends[i][1] for i in range(2)) + hs.pt_to_data(axC, 3.0)
    hs.sig_bracket(axC, 0, 1, yC, rC.pvalue)

    common.save_outputs(fig, out_dir, FIGURE)

    # ---- source data
    names_a = np.array(["", *hs.TICKS["adaptation_groups"]])
    names_p = np.array(["", "Bottom third", "Middle third", "Top third"])
    src = pd.DataFrame({"unique_cell_id": cells["unique_cell_id"].values, "HexMod_beta": beta,
                        "adaptation_coefficient_alpha": thirds["alpha"].to_numpy(),
                        "proximity_aligned_minus_misaligned": thirds["proximity_diff"].to_numpy(),
                        "adaptation_third": adapt_cat, "proximity_third": prox_cat,
                        "adaptation_third_name": names_a[adapt_cat], "proximity_third_name": names_p[prox_cat]})
    src["in_panel_A"] = True
    src["panel_B_group"] = np.where(B_mask1, names_a[1],
                                    np.where(B_mask2, names_a[2], np.where(B_mask3, names_a[3], "")))
    src["panel_C_group"] = np.where(C_mask1, "Top third", np.where(C_mask3, "Bottom third", ""))
    common.write_source_data(out_dir, FIGURE, "", "cells", src)
    rows = []

    def addrow(panel, test, grp, n, stat, df, p, p_holm=np.nan, drawn=""):
        rows.append(dict(panel=panel, test=test, groups=grp, n=str(n), statistic=stat, df=str(df), p=p,
                         p_holm_within_panel=p_holm, mark_drawn=drawn))

    gB, gC = groups["B"], groups["C"]
    addrow("A", "one-sample t-test against 0", "All cells", beta.size, tA, beta.size - 1, pA,
           drawn=hs.sig_stars(pA) or "none (p > 0.05)")
    addrow("B", "one-way ANOVA", ", ".join(hs.TICKS["adaptation_groups"]), ", ".join(str(g.size) for g in gB), F,
           "2, %d" % (sum(g.size for g in gB) - 3), pF, drawn="none (stated in legend)")
    for i, (nm, r) in enumerate(zip(hs.TICKS["adaptation_groups"], oneB)):
        addrow("B", "one-sample t-test against 0", nm, gB[i].size, r.statistic, int(r.df), r.pvalue,
               drawn=hs.sig_stars(r.pvalue) or "none")
    names = hs.TICKS["adaptation_groups"]
    for k, (i, j) in enumerate(pairs_B):
        addrow("B", "two-sample t-test (equal variances)", f"{names[i]} vs {names[j]}", f"{gB[i].size}, {gB[j].size}",
               t_pair_B[k], gB[i].size + gB[j].size - 2, p_pair_B[k], p_holm_B[k],
               drawn=hs.sig_stars(p_holm_B[k]) or "none")
    for nm, r, g in (("Top third", oneC[0], gC[0]), ("Bottom third", oneC[2], gC[1])):
        addrow("C", "one-sample t-test against 0", nm, g.size, r.statistic, int(r.df), r.pvalue,
               drawn=hs.sig_stars(r.pvalue) or "none")
    addrow("C", "two-sample t-test (equal variances)", "Top third vs Bottom third", f"{gC[0].size}, {gC[1].size}",
           rC.statistic, gC[0].size + gC[1].size - 2, rC.pvalue, drawn=hs.sig_stars(rC.pvalue))
    addrow("C (not drawn)", "one-sample t-test against 0", "Middle third of proximity, strong adaptation",
           int(C_mask2.sum()), oneC[1].statistic, int(oneC[1].df), oneC[1].pvalue, drawn="not in the figure")
    common.write_source_data(out_dir, FIGURE, "", "statistics", pd.DataFrame(rows))
    box_rows = []
    for key in "ABC":
        labels = {"A": ["All cells"], "B": list(hs.TICKS["adaptation_groups"]),
                  "C": list(hs.TICKS["proximity_groups"])}[key]
        for lab, s in zip(labels, hs.box_stats(groups[key])):
            box_rows.append(dict(panel=key, group=lab, **s))
    common.write_source_data(out_dir, FIGURE, "", "box_numbers", pd.DataFrame(box_rows))
