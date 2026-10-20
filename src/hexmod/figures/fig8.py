"""Figure 8: prediction for the effect of running speed on hexadirectional modulation.

Panels
    A  Field outlines of an ideal small (30 vm) and large (70 vm) grid, with one aligned run (along a grid axis, one
       grid scale between field centres) and one misaligned run (between axes, sqrt(3) grid scales) on each.
    B  Proximity ``exp(-t_prev / tau)`` at the next field against running speed, aligned and misaligned, both grids
       (closed form, tau = 2 s): ``exp(-scale / v / tau)`` and ``exp(-sqrt(3) scale / v / tau)``.
    C  Proximity, aligned minus misaligned, against running speed (closed form); largest at
       ``v = 1.3327 scale / tau``.
    D  Simulated firing rate, misaligned minus aligned, in percent of the mean rate, against running speed (ideal grid
       cells on straight runs from wall to wall; every second speed of the sweep).

Inputs (``sim/fig8/`` in the results folder, else the shipped ``results_precomputed/sim/fig8/``):
``fig8a_ideal_grids.npz`` and ``fig8d_sweep.csv``. B and C are computed here (``hexmod.sim.analytic``). Positions in
the code are in mm from the top left corner of the figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.lines import Line2D

from .. import io as hio
from ..sim import analytic
from . import common
from . import style as hs

FIGURE = "Fig8"
W = hs.WIDTH["full"]
TAU = 2.0                                             # adaptation time constant (s) of B and C
GRIDS = (("small", 30), ("large", 70))                # drawn grids (name, grid scale in vm)
GRID_WORD = {30: "small", 50: "medium", 70: "large"}  # the three grids of the speed sweep
# runs of A between field centres (rows of the field table): aligned (from, to), misaligned (from, to)
RUNS = {"small": ((9, 6), (9, 1)), "large": ((0, 3), (1, 3))}
# The four letters share one line at the top margin; the boxes of B to D start style.LETTER_ABOVE below it and end
# level with the maps of A. The key of A lies under letter A, its words style.LETTER_CLEAR below the letter's baseline.
LINE = hs.LETTER_TOP                                  # top of the capitals of the letters (1.5 mm)
BOX_TOP, BOX_W = hs.panels_top(LINE), 21.0            # boxes of B to D
COL = {"A": 0.0, "B": 61.5, "C": 97.5, "D": 134.75}   # left edges of the panel columns (letters at COL + 1.5)
AX_LEFT = {"B": 72.0, "C": 109.5, "D": 151.0}         # left edges of the boxes of B to D
MAP = 20.5                                            # maps of A
DASH = hs.LINESTYLE["module_large"]
LINESTYLE = {"small": "-", "large": DASH}
LABEL_D = "Firing rate,\nmisaligned −\naligned (%)"


# ------------------------------------------------------------------------------------------------------------ inputs
def load_inputs(results_dir):
    """Read the result files of Figure 8.

    Returns
    -------
    dict
        ``fields`` (per drawn grid an array (k, 3): centre x, centre y, radius in vm) and ``sweep`` (the speed sweep
        table, rows ordered by grid scale and speed).
    """
    z = hio.load_npz(common.find_input(results_dir, "sim/fig8/fig8a_ideal_grids.npz"))
    fields = {name: np.asarray(z[f"{name}__field_locations"], float) for name, _ in GRIDS}
    sweep = hio.read_csv(common.find_input(results_dir, "sim/fig8/fig8d_sweep.csv"))
    sweep = sweep.sort_values(["scale", "speedLevel"], kind="stable").reset_index(drop=True)
    return dict(fields=fields, sweep=sweep)


def analytic_curves(speeds=None, tau=TAU):
    """Closed-form proximities of B and C at the drawn speeds (default 5 to 149 vm/s in steps of 2)."""
    v = analytic.FIG8BC_SPEEDS if speeds is None else np.asarray(speeds)
    out = dict(moveSpeed=v)
    for name, scale in GRIDS:
        out[f"proxy_align_{name}"] = analytic.proximity_aligned(scale, v, tau)
        out[f"proxy_misalign_{name}"] = analytic.proximity_misaligned(scale, v, tau)
        out[f"proxy_diff_{name}"] = analytic.proximity_difference(scale, v, tau)
    return out


def rate_difference_pct(sweep):
    """Misaligned minus aligned mean rate in percent of the mean of the two, per row of the sweep."""
    fr = sweep[["FR_align", "FR_misalign"]].to_numpy(float)
    return -(fr[:, 0] - fr[:, 1]) / fr.mean(axis=1) * 100


# ----------------------------------------------------------------------------------------------------------- drawing
def layout():
    """Rows of the key of A, bottom of the panels and figure height (mm).

    Returns
    -------
    dict
        ``top2``, ``top3`` (row tops of the two key lines), ``box_h`` (height of the boxes of B to D; their bottom is
        the bottom of the maps of A) and ``height`` (of the figure). Call after ``style.use_style`` (the key lines
        depend on the height of the key words in the typeface).
    """
    probe = hs.new_figure("full", 58.5)
    words = list(hs.TICKS["aligned"][:2]) + list(hs.TICKS["modules"][:2])
    ascent = max(30.0 - hs.extent_mm(probe, hs.text_mm(probe, 8.0, 30.0, wd, fontsize=hs.FS_TITLE))[1] for wd in words)
    top2 = hs.LETTER_TOP + hs.LETTER_CLEAR + ascent    # the key words start LETTER_CLEAR below the letter's baseline
    top3 = top2 + hs.TITLE_PITCH
    rise = (hs.LETTER_TOP + hs.TITLE_PITCH) - top2     # how far panel A stands above a key one line below the letters
    return dict(top2=top2, top3=top3, box_h=(15.0 + 33.0 - rise) - BOX_TOP, height=round(58.5 - rise, 3))


def build(results_dir, out_dir):
    """Draw Figure 8 from ``results_dir`` into ``out_dir``."""
    inputs = load_inputs(results_dir)
    with mpl.rc_context():
        hs.use_style()
        _draw(out_dir, inputs)


def _frame_ticks(ax):
    ax.xaxis.set_major_formatter(hs.NUM); ax.yaxis.set_major_formatter(hs.NUM)


def _key_item(fig, word, colour, ls, left, row_top, arts):
    """A line sample (as long as a key swatch) with its word on a key line; returns the width of the word (mm).

    The line and the word are appended to ``arts``.
    """
    base = row_top + hs.CAP12; ym = base - hs.KEY["swatch_h"] / 2.0
    ln = Line2D([hs.fx(fig, left), hs.fx(fig, left + hs.KEY["swatch_w"])], [hs.fy(fig, ym), hs.fy(fig, ym)],
                transform=fig.transFigure, color=colour, linestyle=ls, linewidth=hs.LW_DATA,
                solid_capstyle="butt", dash_capstyle="butt")
    fig.add_artist(ln)
    t = hs.text_mm(fig, left + hs.KEY["swatch_w"] + hs.KEY["gap_text"], base, word, fontsize=hs.FS_TITLE, ha="left")
    arts.extend([ln, t])
    return t.get_window_extent(fig.canvas.get_renderer()).width * 25.4 / fig.dpi


def _curve_axes(fig, key, label, box_h):
    ax = hs.axes_mm(fig, AX_LEFT[key], BOX_TOP, BOX_W, box_h)
    ax.set_xlim(0, 160); ax.set_xticks([0, 50, 100, 150]); ax.set_xlabel(hs.LABEL["speed_vm"]); _frame_ticks(ax)
    ax.set_ylabel(label, linespacing=1.1)
    hs.panel_letter(fig, key, COL[key] + hs.LETTER_X, LINE, clear_of=[ax], not_above=LINE, row=1)
    return ax


def _draw(out_dir, inp):
    P = hs.PALETTE
    fields, sweep = inp["fields"], inp["sweep"]
    L = layout()
    fig = hs.new_figure("full", L["height"])

    # ---- A: ideal grids with runs
    map_left = [COL["A"] + 11.5, COL["A"] + 11.5 + MAP + 5.0]; map_top = BOX_TOP + L["box_h"] - MAP
    axA = []; runs = {}
    for k, (name, scale) in enumerate(GRIDS):
        ls = LINESTYLE[name]
        ax = hs.axes_mm(fig, map_left[k], map_top, MAP, MAP, full=True); axA.append(ax)      # maps keep the full box
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
        fl = fields[name]
        hs.field_outlines(ax, fl, colour=P["ink"], lw=hs.LW_FRAME, zorder=2)
        (a0, a1), (m0, m1) = RUNS[name]
        al = (fl[a0, :2], fl[a1, :2]); mis = (fl[m0, :2], fl[m1, :2])
        runs[name] = dict(aligned=np.r_[al[0], al[1]], misaligned=np.r_[mis[0], mis[1]])
        for (p0, p1), c in ((al, P["aligned"]), (mis, P["misaligned"])):
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=c, linewidth=hs.LW_DATA, linestyle=ls, solid_capstyle="round",
                    dash_capstyle="butt", zorder=4)
        ax.set_xticks([0, 50, 100]); ax.set_yticks([0, 50, 100]); _frame_ticks(ax)
        ax.set_xlabel(hs.LABEL["x_vm"])
        if k == 0: ax.set_ylabel(hs.LABEL["y_vm"], labelpad=1.0)
        else: ax.tick_params(axis="y", labelleft=False)
        cx = map_left[k] + MAP / 2
        hs.text_mm(fig, cx, map_top - 5.0, hs.TICKS["modules"][k], fontsize=hs.FS_SUBHEAD, ha="center")
        hs.text_mm(fig, cx, map_top - 1.3, f"({scale} vm)", fontsize=hs.FS_SUBHEAD, ha="center")
    # lengths of the two runs in grid scales, small grid only
    axA[0].text(44.0, 44.5, "1", color=P["aligned"], fontsize=hs.FS_SMALL, ha="center", va="top", zorder=5)
    axA[0].text(31.0, 66.0, hs.VAR["sqrt3"], color=P["misaligned"], fontsize=hs.FS_SMALL, ha="center", va="center",
                zorder=5)
    # one key for the figure in two columns on the two lines under the letter A, from its left edge: direction
    # colours (Aligned above Misaligned) and module line styles (Small module above Large module)
    key_left = COL["A"] + 1.5
    key = []
    directions = ((hs.TICKS["aligned"][0], P["aligned"], L["top2"]),
                  (hs.TICKS["aligned"][1], P["misaligned"], L["top3"]))
    w1 = max(_key_item(fig, w, c, "-", key_left, row, key) for w, c, row in directions)
    key_col2 = key_left + hs.KEY["swatch_w"] + hs.KEY["gap_text"] + w1 + hs.KEY["gap_item"]
    for w, ls, row in ((hs.TICKS["modules"][0], "-", L["top2"]), (hs.TICKS["modules"][1], DASH, L["top3"])):
        _key_item(fig, w, P["ink"], ls, key_col2, row, key)
    hs.panel_letter(fig, "A", COL["A"] + hs.LETTER_X, LINE, clear_of=key + [axA[0]], not_above=LINE, row=1)

    # ---- B: proximity, closed form
    AN = analytic_curves()
    v = AN["moveSpeed"]
    axB = _curve_axes(fig, "B", hs.LABEL["proximity"], L["box_h"])
    for name, _ in GRIDS:
        for d, c in (("align", P["aligned"]), ("misalign", P["misaligned"])):
            axB.plot(v, AN[f"proxy_{d}_{name}"], color=c, linestyle=LINESTYLE[name], linewidth=hs.LW_DATA,
                     dash_capstyle="butt", solid_capstyle="butt", zorder=3)
    axB.set_ylim(0, 1.0); axB.set_yticks([0, 0.5, 1])

    # ---- C: proximity difference, closed form
    axC = _curve_axes(fig, "C", hs.LABEL["proximity_diff"], L["box_h"])
    for name, _ in GRIDS:
        axC.plot(v, AN[f"proxy_diff_{name}"], color=P["ink"], linestyle=LINESTYLE[name], linewidth=hs.LW_DATA,
                 dash_capstyle="butt", solid_capstyle="butt", zorder=3)
    axC.set_ylim(0, 0.22); axC.set_yticks([0, 0.1, 0.2])

    # ---- D: simulated rate difference, every second speed of the sweep
    axD = _curve_axes(fig, "D", LABEL_D, L["box_h"])
    hs.zero_line(axD)
    diff = rate_difference_pct(sweep)
    for name, scale in GRIDS:
        sel = (sweep["scale"] == scale).to_numpy() & sweep["plotted"].to_numpy(bool)
        axD.plot(sweep["moveSpeed"].to_numpy()[sel], diff[sel], color=P["ink"], linestyle=LINESTYLE[name],
                 linewidth=hs.LW_DATA, dash_capstyle="butt", solid_joinstyle="round", zorder=3)
    axD.set_ylim(-0.1, 1.2); axD.set_yticks([0, 0.5, 1])

    common.save_outputs(fig, out_dir, FIGURE)
    _write_source_data(out_dir, fields, runs, AN, sweep, diff)


# ------------------------------------------------------------------------------------------------------- source data
def _write_source_data(out_dir, fields, runs, AN, sweep, diff):
    common.write_source_data(out_dir, FIGURE, "A", "fields", pd.concat(
        [pd.DataFrame({"grid": name, "grid_scale_vm": scale, "field": np.arange(len(fields[name])),
                       "x_vm": fields[name][:, 0], "y_vm": fields[name][:, 1], "radius_vm": fields[name][:, 2]})
         for name, scale in GRIDS], ignore_index=True))
    common.write_source_data(out_dir, FIGURE, "A", "runs", pd.DataFrame(
        [dict(grid=name, grid_scale_vm=scale, run=run, x_start_vm=p[0], y_start_vm=p[1], x_end_vm=p[2], y_end_vm=p[3])
         for name, scale in GRIDS for run, p in runs[name].items()]))
    rows = []
    for i in range(len(AN["moveSpeed"])):
        for name, scale in GRIDS:
            rows.append(dict(speed_vm_s=AN["moveSpeed"][i], grid=name, grid_scale_vm=scale, tau_s=TAU,
                             proximity_aligned=AN[f"proxy_align_{name}"][i],
                             proximity_misaligned=AN[f"proxy_misalign_{name}"][i],
                             proximity_aligned_minus_misaligned=AN[f"proxy_diff_{name}"][i]))
    common.write_source_data(out_dir, FIGURE, "BC", "proximity", pd.DataFrame(rows))
    drawn = sweep["scale"].isin([s for _, s in GRIDS]).to_numpy() & sweep["plotted"].to_numpy(bool)
    common.write_source_data(out_dir, FIGURE, "D", "speed_sweep", pd.DataFrame({
        "grid": sweep["scale"].map(GRID_WORD).to_numpy(), "grid_scale_vm": sweep["scale"].to_numpy(),
        "speed_vm_s": sweep["moveSpeed"].to_numpy(), "mean_rate_aligned": sweep["FR_align"].to_numpy(float),
        "mean_rate_misaligned": sweep["FR_misalign"].to_numpy(float), "rate_misaligned_minus_aligned_pct": diff,
        "simulated_proximity_aligned": sweep["proximity_align"].to_numpy(float),
        "simulated_proximity_misaligned": sweep["proximity_misalign"].to_numpy(float), "plotted_in_D": drawn}))
    opt = "speed of the largest proximity difference"
    stat = [("B, C", f"{opt}, in units of grid scale / tau", analytic.SPEED_OPT_FACTOR, ""),
            ("B, C", f"{opt}, small grid (30 vm)", analytic.optimal_speed(30, TAU), "vm/s"),
            ("B, C", f"{opt}, large grid (70 vm)", analytic.optimal_speed(70, TAU), "vm/s"),
            ("C", "largest proximity difference (any grid scale and tau)", analytic.max_proximity_difference(), "")]
    for scale, word in GRID_WORD.items():
        sel = (sweep["scale"] == scale).to_numpy() & sweep["plotted"].to_numpy(bool)
        q = sweep[sel]; i = int(np.argmax(diff[sel]))
        note = "" if scale in dict(GRIDS).values() else " (not in the figure)"
        what = f"largest rate difference at the plotted speeds, {word} grid ({scale} vm){note}"
        stat += [("D", what, float(diff[sel][i]), "%"),
                 ("D", f"speed of the {what}", float(q["moveSpeed"].iloc[i]), "vm/s"),
                 ("D", f"speed of the {what}, in units of grid scale / tau",
                  float(q["moveSpeed"].iloc[i] / (scale / TAU)), "")]
    common.write_source_data(out_dir, FIGURE, "", "statistics",
                             pd.DataFrame(stat, columns=["panel", "quantity", "value", "unit"]))
