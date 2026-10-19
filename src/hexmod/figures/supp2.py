"""Supplementary Figure 2: prospective coding control, spike times shifted against the position samples.

Panels
    A  Spikes (dots) and field outlines of the example cell with its spike times shifted by minus its best shift, 0 and
       plus its best shift (120 ms for the published cell).
    B  Spatial information of the example cell against the time shift; the three shifts of A are marked.
    C  Best time shift (shift of maximal spatial information) of every cell, with the median.
    D  Firing rate change in four bins of t_prev after shifting each cell's spikes by its best shift (cells that remain
       grid cells); brackets: paired t-tests of the first bin against each other bin, Holm correction over the six
       pairs of bins.
    E  Adaptation coefficient alpha with the original and with the shifted spike times; asterisks: one-sample t-tests
       against 0; bracket: paired t-test.

Inputs (``rat/`` in the results folder): ``examples/supp2a_shifted_maps.npz`` (A, B), ``prospective/peak_shift.csv``
(C), ``prospective/fr_percent_change_shifted.npy`` and ``prospective/shift_cells.csv`` (D, E). The statistics are
computed from the plotted arrays. Figure size 174 x 104.14 mm; positions in mm from the top left corner of the
figure.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.patches import Circle
from scipy import stats

from .. import io as hio
from .. import stats as hstats
from . import common
from . import style as hs

FIGURE = "SuppFig2"
W, H = hs.WIDTH["full"], 104.141
# Each row of panels has one line of letters (top of the capital); the topmost element of the row's panels (row 1: the
# headings of the maps; row 2: the boxes) stands style.LETTER_ABOVE under it. Row 2's line lies 2.70 mm under the
# lowest text of row 1.
ROW1, ROW2 = 1.5, 46.641                    # letter lines of row 1 (A, B) and row 2 (C, D, E)
MAP_W = 28.0
MAP_GAP = 2.6
MAP_L = 11.5
MAP_TOP = 6.641                             # top of the maps and of the box of B
RIGHT = 170.5                               # right edge shared by the last box of each row
B_LEFT, B_LETTER = 118.0, 104.0
BOX2_TOP, BOX2_H = hs.panels_top(ROW2), 44.0    # row 2: C, D and E share top and bottom
C_L, C_W = 15.0, 56.0
D_LETTER, D_L, D_W = 75.0, 89.0, 38.0
E_LETTER, E_L, E_W = 132.0, 146.0, RIGHT - 146.0
BIN_LABELS = ["0 to 1", "1 to 2", "2 to 3", "> 3"]
BIN_COLUMNS = ["t_prev_0_to_1_s", "t_prev_1_to_2_s", "t_prev_2_to_3_s", "t_prev_above_3_s"]
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
DRAWN_PAIRS = [(0, 1), (0, 2), (0, 3)]      # the first bin against each other bin


def _shift_prefixes(names):
    """Return [(prefix, shift in ms, map slot)] for the shifts mapped in the example-cell file.

    The file holds the shifts -s, 0 and +s (s: the best shift of the cell), drawn in slots 0, 1 and 2. When the cell is
    not a grid cell at its best shift (possible with synthetic smoke-test data), only shift 0 is mapped (slot 1).
    """
    found = {}
    for name in names:
        m = re.match(r"^shift_(m|p)?(\d+)ms__", name)
        if m:
            sign = -1 if m.group(1) == "m" else 1
            found[int(m.group(2)) * sign] = name.split("__")[0]
    shifts = sorted(found)
    if shifts == [0]:
        return [(found[0], 0, 1)]
    if len(shifts) != 3 or shifts[1] != 0 or shifts[0] != -shifts[2]:
        raise ValueError(f"expected the shifts -s, 0 and +s in supp2a_shifted_maps.npz, found {shifts}")
    return [(found[s], s, i) for i, s in enumerate(shifts)]


def _shift_label(s):
    """Map heading of a shift: '−120 ms', '0 ms (original)', '+120 ms' (true minus sign)."""
    return "0 ms (original)" if s == 0 else f"{s:+d} ms".replace("-", "−")


def load_inputs(results_dir):
    """Read the result files of Supplementary Figure 2."""
    ex = hio.load_npz(common.find_input(results_dir, "rat/examples/supp2a_shifted_maps.npz"))
    peak = hio.read_csv(common.find_input(results_dir, "rat/prospective/peak_shift.csv"))
    frc = hio.load_npy(common.find_input(results_dir, "rat/prospective/fr_percent_change_shifted.npy"))
    kept = hio.read_csv(common.find_input(results_dir, "rat/prospective/shift_cells.csv"))
    if frc.shape != (len(kept), 4):
        raise ValueError(f"fr_percent_change_shifted.npy {frc.shape} is not row-aligned with shift_cells.csv")
    return ex, peak, frc, kept


def build(results_dir, out_dir):
    """Draw Supplementary Figure 2 from ``results_dir`` into ``out_dir``."""
    inputs = load_inputs(results_dir)
    with mpl.rc_context():
        hs.use_style()
        _draw(out_dir, *inputs)


def _draw(out_dir, ex, peak, FRC, kept):
    P = hs.PALETTE
    shifts = _shift_prefixes(ex.keys())
    shift_ms = np.asarray(ex["shift_ms"])
    curve = np.asarray(ex["spatial_information"], float)
    best_shift = peak["peak_shift_ms"].to_numpy()
    ALPHA = kept[["alpha_original", "alpha_tprevXloc_all"]].to_numpy(float)

    # statistics of the plotted arrays
    pair_t = [stats.ttest_rel(FRC[:, i], FRC[:, j]) for i, j in PAIRS]
    pair_p = np.array([r.pvalue for r in pair_t])
    pair_holm = hs.holm(pair_p)
    anova = hstats.rm_anova(FRC)
    one_E = [stats.ttest_1samp(ALPHA[:, j], 0) for j in range(2)]
    t_E, p_E = stats.ttest_rel(ALPHA[:, 0], ALPHA[:, 1])

    fig = hs.new_figure("full", H)

    # ================================================================ A: maps
    key0 = [key for key, sh, slot in shifts if sh == 0][0]
    xe, ye = ex[key0 + "__x_edges"], ex[key0 + "__y_edges"]
    ratio = (ye[-1] - ye[0]) / (xe[-1] - xe[0])
    MAP_H = MAP_W * ratio
    map_axes = []
    for k, (key, sh, i) in enumerate(shifts):
        ax = hs.axes_mm(fig, MAP_L + i * (MAP_W + MAP_GAP), MAP_TOP, MAP_W, MAP_H, full=True)   # maps keep the full box
        common.tag(ax, panel="A", name=f"A map {_shift_label(sh)}")
        x, y, fl = ex[key + "__spike_x"], ex[key + "__spike_y"], ex[key + "__field_locations"]
        if not (np.array_equal(ex[key + "__x_edges"], xe) and np.array_equal(ex[key + "__y_edges"], ye)):
            raise ValueError(f"the maps of {key} and {key0} have different bins")
        ax.plot(x, y, linestyle="", marker="o", ms=1.0, mfc=P["dot"], mec="none", alpha=0.6, zorder=2, rasterized=False)
        for cx, cy, r in fl[:, :3]:
            ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=P["ink"], linewidth=hs.LW_FRAME, zorder=4))
        ax.set_xlim(xe[0], xe[-1])
        ax.set_ylim(ye[0], ye[-1])
        ax.set_aspect("equal", adjustable="box")
        for sp in ax.spines.values():
            sp.set_zorder(6)                       # the frame stays above the outlines of fields cut by it
        ax.set_xticks([0, 50, 100, 150])
        ax.set_yticks([0, 50, 100, 150])
        ax.xaxis.set_major_formatter(hs.NUM)
        ax.yaxis.set_major_formatter(hs.NUM)
        if k:
            ax.tick_params(labelleft=False)
        else:
            ax.set_xlabel(hs.LABEL["x_cm"])
            ax.set_ylabel(hs.LABEL["y_cm"])
        ax.set_title(_shift_label(sh), fontsize=hs.FS_SUBHEAD, pad=3.0)
        map_axes.append(ax)
    hs.panel_letter(fig, "A", row_top=ROW1, clear_of=[map_axes[0]], not_above=ROW1, row=1)

    # ================================================================ B: spatial information against the shift
    # letter B stands on the end of the rotated y label, at least 1.25 mm above it
    axb = common.tag(hs.axes_mm(fig, B_LEFT, MAP_TOP, RIGHT - B_LEFT, MAP_H), panel="B", name="B curve")
    hs.panel_letter(fig, "B", x=B_LETTER, row_top=ROW1, clear_of=[axb], not_above=ROW1, row=1)
    for key, sh, _ in shifts:
        axb.axvline(sh, color=P["guide"], linewidth=hs.LW_GUIDE, zorder=1)
    axb.plot(shift_ms, curve, color=P["ink"], linewidth=hs.LW_DATA, zorder=3, solid_joinstyle="round")
    axb.set_xlim(-400, 400)
    axb.set_ylim(0.2, 1.0)
    axb.set_xticks([-400, 0, 400])
    axb.set_yticks([0.2, 0.6, 1.0])
    axb.xaxis.set_major_formatter(hs.NUM)
    axb.yaxis.set_major_formatter(hs.NUM)
    axb.set_xlabel(hs.LABEL["time_shift"])
    axb.set_ylabel("Spatial information\n(bits per spike)", linespacing=1.1)
    for key, sh, _ in shifts:                   # the three shifts of the maps, named above the box
        axb.annotate(f"{sh:+d}".replace("-", "−").replace("+0", "0"), (sh, 1.0), xycoords=("data", "axes fraction"),
                     xytext=(0, hs.NOTE_UP / hs.PT), textcoords="offset points", ha="center", va="bottom",
                     fontsize=hs.FS_SMALL)

    # ================================================================ C: best time shift of each cell
    axc = common.tag(hs.axes_mm(fig, C_L, BOX2_TOP, C_W, BOX2_H), panel="C", name="C histogram")
    hs.panel_letter(fig, "C", row_top=ROW2, clear_of=[axc], not_above=ROW2, row=2)
    n_c, edges_c, _ = axc.hist(best_shift, bins=shift_ms, histtype="stepfilled", facecolor=P["neutral_fill"],
                               edgecolor="none", linewidth=0, zorder=2)     # fill only, as in Supp Fig 3B to D
    med = float(np.median(best_shift))
    axc.axvline(med, color=P["ink"], linewidth=hs.LW_REF, zorder=4)
    axc.annotate(f"Median {med:.0f} ms", (med, 1.0), xycoords=("data", "axes fraction"), xytext=(-3, -3),
                 textcoords="offset points", ha="right", va="top", fontsize=hs.FS_SMALL)
    axc.set_xlim(-400, 400)
    axc.set_ylim(0, 350)
    axc.set_xticks([-400, 0, 400])
    axc.set_yticks([0, 100, 200, 300])
    axc.xaxis.set_major_formatter(hs.NUM)
    axc.yaxis.set_major_formatter(hs.NUM)
    axc.set_xlabel(hs.LABEL["best_shift"])
    axc.set_ylabel(hs.LABEL["cells"])

    # ================================================================ D: firing rate change after the shift
    axD = common.tag(hs.axes_mm(fig, D_L, BOX2_TOP, D_W, BOX2_H), panel="D", name="D box plots")
    hs.panel_letter(fig, "D", x=D_LETTER, row_top=ROW2, clear_of=[axD], not_above=ROW2, row=2)
    axD.set_xlim(-0.6, 3.6)
    axD.set_ylim(-30, 62)
    groups_D = [FRC[:, j] for j in range(4)]
    hs.boxplot(axD, groups_D, width=hs.box_width_data(axD, 6.5))   # 6.5 mm, as in Fig 3C (four bins of t_prev)
    hs.zero_line(axD)
    axD.set_xticks(range(4))
    axD.set_xticklabels(hs.TICKS["t_prev_bins"])
    axD.set_yticks([-20, 0, 20, 40])
    axD.yaxis.set_major_formatter(hs.NUM)
    axD.set_xlabel(hs.LABEL["t_prev"])
    axD.set_ylabel("Firing rate\nchange (%)", linespacing=1.1)
    holm_of = dict(zip(PAIRS, pair_holm))
    top_whisker = max(s["whisker_high"] for s in hs.box_stats(groups_D))
    drawn_D, _ = hs.bracket_stack(axD, DRAWN_PAIRS, [holm_of[p] for p in DRAWN_PAIRS],
                                  y0=top_whisker + hs.pt_to_data(axD, 5.0), order="span")

    # ================================================================ E: adaptation coefficient, original and shifted
    axE = common.tag(hs.axes_mm(fig, E_L, BOX2_TOP, E_W, BOX2_H), panel="E", name="E box plots")
    hs.panel_letter(fig, "E", x=E_LETTER, row_top=ROW2, clear_of=[axE], not_above=ROW2, row=2)
    axE.set_xlim(-0.6, 1.6)
    axE.set_ylim(-0.22, 0.2)
    ends_E = hs.boxplot(axE, [ALPHA[:, 0], ALPHA[:, 1]], width=hs.box_width_data(axE))   # 7.6 mm, as in Fig 3D
    hs.zero_line(axE)
    axE.set_xticks([0, 1])
    axE.set_xticklabels(hs.TICKS["shift"])
    axE.set_yticks([-0.2, -0.1, 0, 0.1])
    axE.yaxis.set_major_formatter(hs.NUM)
    axE.set_ylabel("Adaptation\ncoefficient " + hs.VAR["alpha"], linespacing=1.1)
    low_E = min(e[0] for e in ends_E)           # both marks on one line, just below the lower of the two whisker ends
    for j in range(2):
        hs.sig_mark(axE, j, low_E, one_E[j].pvalue, side="below")
    hs.sig_bracket(axE, 0, 1, max(e[1] for e in ends_E) + hs.pt_to_data(axE, 5.0), p_E)

    common.save_outputs(fig, out_dir, FIGURE)

    # ================================================================ source data
    rows = [(sh, xx, yy) for key, sh, _ in shifts for xx, yy in zip(ex[key + "__spike_x"], ex[key + "__spike_y"])]
    common.write_source_data(out_dir, FIGURE, "A", "spike_positions",
                             pd.DataFrame(rows, columns=["shift_ms", "x_cm", "y_cm"]))
    rows = [(sh, k, cx, cy, r) for key, sh, _ in shifts
            for k, (cx, cy, r) in enumerate(ex[key + "__field_locations"][:, :3])]
    common.write_source_data(out_dir, FIGURE, "A", "fields",
                             pd.DataFrame(rows, columns=["shift_ms", "field", "centre_x_cm", "centre_y_cm",
                                                        "radius_cm"]))
    common.write_source_data(out_dir, FIGURE, "B", "spatial_information_example_cell",
                             {"shift_ms": shift_ms, "spatial_information_bits_per_spike": curve})
    bdf = peak[["row", "animal_id", "session_id", "cell_id"]].copy()
    bdf["best_time_shift_ms"] = best_shift
    common.write_source_data(out_dir, FIGURE, "C", "best_time_shift", bdf)
    common.write_source_data(out_dir, FIGURE, "C", "histogram",
                             {"bin_left_ms": edges_c[:-1], "bin_right_ms": edges_c[1:], "cells": n_c.astype(int)})
    ddf = kept[["row", "animal_id", "session_id", "cell_id"]].copy()
    for j, c in enumerate(BIN_COLUMNS):
        ddf[c] = FRC[:, j]
    common.write_source_data(out_dir, FIGURE, "D", "firing_rate_change_pct", ddf)
    edf = kept[["row", "animal_id", "session_id", "cell_id"]].copy()
    edf["best_time_shift_ms"] = kept["peakShift_SI"].values
    edf["alpha_original"] = ALPHA[:, 0]
    edf["alpha_shifted"] = ALPHA[:, 1]
    common.write_source_data(out_dir, FIGURE, "E", "adaptation_coefficient", edf)
    drawn = {tuple(d[0]) for d in drawn_D}
    st = [dict(panel="D", comparison=f"{BIN_LABELS[i]} s against {BIN_LABELS[j]} s", test="paired t", t=r.statistic,
               df=FRC.shape[0] - 1, p=r.pvalue, p_holm=ph, bracket=(i, j) in drawn, F=np.nan, df1=np.nan, df2=np.nan)
          for (i, j), r, ph in zip(PAIRS, pair_t, pair_holm)]
    st.append(dict(panel="D", comparison="four bins", test="one-way repeated measures ANOVA", t=np.nan, df=np.nan,
                   p=anova["p"], p_holm=np.nan, bracket=False, F=anova["F"], df1=anova["df1"], df2=anova["df2"]))
    for j, nm in enumerate(("original", "shifted")):
        st.append(dict(panel="E", comparison=f"{nm} against 0", test="one-sample t", t=one_E[j].statistic,
                       df=ALPHA.shape[0] - 1, p=one_E[j].pvalue, p_holm=np.nan, bracket=False, F=np.nan, df1=np.nan,
                       df2=np.nan))
    st.append(dict(panel="E", comparison="original against shifted", test="paired t", t=t_E, df=ALPHA.shape[0] - 1,
                   p=p_E, p_holm=np.nan, bracket=bool(hs.sig_stars(p_E)), F=np.nan, df1=np.nan, df2=np.nan))
    common.write_source_data(out_dir, FIGURE, "", "statistics", pd.DataFrame(st))
    box_rows = [dict(panel="D", group=f"{lab} s", **s) for lab, s in zip(BIN_LABELS, hs.box_stats(groups_D))]
    box_rows += [dict(panel="E", group=lab, **s)
                 for lab, s in zip(hs.TICKS["shift"], hs.box_stats([ALPHA[:, 0], ALPHA[:, 1]]))]
    common.write_source_data(out_dir, FIGURE, "", "box_numbers", pd.DataFrame(box_rows))
