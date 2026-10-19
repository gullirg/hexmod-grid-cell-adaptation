"""Figure 4: proximity of field visits depends on the running direction relative to the grid axes.

Panels
    A  Rate maps of a small-scale and a large-scale grid cell recorded in one session, with 12 s of path coloured by
       running speed; fields visited in the window are numbered 1, 2, ... in the order of their first visit.
    B  For the same 12 s and both cells: time since the previous field visit (t_prev), movement direction (shown from
       -90 to 270 deg, so that the line does not wrap in this window) and grid alignment cos[6(theta - phi)] (teal above
       zero, orange below); shaded, field visits.
    C  Proximity (mean adaptation regressor exp(-t_prev / 2 s) of the in-field samples) for running within 15 deg of a
       grid axis (aligned) and further from the axes (misaligned), all cells; bracket: paired t-test.
    D  Proximity difference (aligned minus misaligned) by thirds of grid scale; asterisks: one-sample t-tests against
       0; brackets: two-sample t-tests between thirds with Holm correction within the panel.

Inputs (``rat/`` in the results folder): ``examples/fig4ab_small.npz``, ``examples/fig4ab_large.npz`` and their
``*_field_visits.csv``, ``adaptation/param_cells.csv``, ``cohort/cells.csv`` and ``figure_tables/cells_thirds.csv``.
The grid alignment and the movement direction in (-pi, pi] of panel B are the arrays ``grid_alignment`` and
``movedir_rad_pm_pi`` written by step 11 (computed there from the movement directions as stored, in single precision
for this recording). Positions in mm from the top left corner of the figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from scipy import stats

from .. import io as hio
from . import common
from . import style as hs

FIGURE = "Fig4"
W, H = hs.WIDTH["full"], 105.14
CELLS = ("small", "large")
B_LEFT = (37.0, 84.0)                  # the two columns of B (small, large)
B_W = 36.0
# Each row of panels has one line of letters (top of the capital); the topmost element of the row's panels stands
# style.LETTER_ABOVE under it. Row 2's line lies 2.27 mm under the lowest text of row 1.
LINE1 = hs.LETTER_TOP                  # 1.5 mm: letters A and C
TOP1 = hs.panels_top(LINE1)            # 3.0 mm: top of the map headings and of the box of C
LINE2 = 45.64                          # letters B and D
ROW2 = hs.panels_top(LINE2)            # 47.14 mm: top of the second row (the box of D; the key and time courses of B)
MAP_W, MAP_TOP = 28.0, 7.14            # the map headings (baseline MAP_TOP - 1.6) start at TOP1
MAP_LEFT = (16.0, 64.0)                # maps with x, y axes in cm, each followed by its rate bar
RIGHT = 172.0                          # right edge of the boxes of C and D
COL3 = 124.0                           # left edge of the right-hand column (C, D)
BOX_H, GAP = 12.0, 1.8                 # B: height of each time-course box and space between boxes
B_TOP = ROW2 + 9.6
SAMPLE_GAP = np.inf                    # lines are not broken at gaps between samples (as in the analysis plots)
SPEED_RANGE = (5, 45)                  # colour range of the running speed (cm/s)


def with_breaks(t, y, extra=None):
    """The points unchanged, with a NaN after each time step longer than SAMPLE_GAP (and after the indices in extra)."""
    cut = set(np.where(np.diff(t) > SAMPLE_GAP)[0].tolist()) | set([] if extra is None else list(extra))
    cut = np.array(sorted(cut), int)
    return np.insert(np.asarray(t, float), cut + 1, np.nan), np.insert(np.asarray(y, float), cut + 1, np.nan)


def place_numbers(fig, ax, field_loc, ids, path_xy, gap_mm=0.6, size=hs.FS_SMALL, labels=None):
    """Field numbers just outside their outlines (white, 8 pt), each at the angle (steps of 15 deg) that keeps it inside
    the frame and farthest from the path, from the numbers already placed and from the other outlines."""
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    bb = ax.get_position()
    Wm, Hm = fig.get_size_inches() * 25.4
    mm_x = (x1 - x0) / (bb.width * Wm)
    mm_y = (y1 - y0) / (bb.height * Hm)
    half_w = 0.5 * 0.6 * size * hs.PT * 2
    half_h = 0.5 * 0.72 * size * hs.PT
    placed = []
    arts = []
    for k in ids:
        cx, cy, r = field_loc[k, :3]
        best = None
        for a in np.arange(0, 360, 15):
            ux, uy = np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))
            px = cx + ux * (r + gap_mm * mm_x + half_w * mm_x * abs(ux))
            py = cy + uy * (r + gap_mm * mm_y + half_h * mm_y * abs(uy))
            if not (x0 + half_w * mm_x <= px <= x1 - half_w * mm_x and y0 + half_h * mm_y <= py <= y1 - half_h * mm_y):
                continue
            dist = np.min(np.hypot((path_xy[:, 0] - px) / mm_x, (path_xy[:, 1] - py) / mm_y))
            if placed:
                dist = min(dist, min(np.hypot((qx - px) / mm_x, (qy - py) / mm_y) - 2.0 for qx, qy in placed))
            for j, (ox, oy, orad) in enumerate(field_loc[:, :3]):
                if j == k:
                    continue
                dist = min(dist, abs(np.hypot((ox - px) / mm_x, (oy - py) / mm_y) - orad / mm_x) - 0.6)
            if best is None or dist > best[0] + 1e-9:
                best = (dist, px, py)
        placed.append(best[1:])
        arts.append(ax.text(best[1], best[2], f"{(labels or {}).get(k, k):d}", ha="center", va="center", fontsize=size,
                            color=hs.PALETTE["white"], zorder=6, clip_on=False))
    return arts


def load_inputs(results_dir):
    """Read the result files of Figure 4."""
    ex = {}
    for which in CELLS:
        d = hio.load_npz(common.find_input(results_dir, f"rat/examples/fig4ab_{which}.npz"))
        for key in ("grid_alignment", "movedir_rad_pm_pi"):
            if key not in d:
                raise ValueError(f"fig4ab_{which}.npz lacks {key!r}; run step 11 again")
        ev = hio.read_csv(common.find_input(results_dir, f"rat/examples/fig4ab_{which}_field_visits.csv"))
        ex[which] = dict(d=d, ev=ev)
    param = hio.read_csv(common.find_input(results_dir, "rat/adaptation/param_cells.csv"))
    cells = hio.read_csv(common.find_input(results_dir, "rat/cohort/cells.csv"))
    thirds = hio.read_csv(common.find_input(results_dir, "rat/figure_tables/cells_thirds.csv"))
    for name, tab in (("param_cells.csv", param), ("cells_thirds.csv", thirds)):
        if not np.array_equal(tab["row"].values, cells["row"].values):
            raise ValueError(f"{name} is not row-aligned with cells.csv")
    return ex, param, cells, thirds


def build(results_dir, out_dir):
    """Draw Figure 4 from ``results_dir`` into ``out_dir``."""
    inputs = load_inputs(results_dir)
    with mpl.rc_context():
        hs.use_style()
        _draw(out_dir, *inputs)


def _draw(out_dir, ex, param, cells, thirds):
    P, LAB, TK = hs.PALETTE, hs.LABEL, hs.TICKS
    a = param["align_tprev_infield"].to_numpy()
    m = param["misalign_tprev_infield"].to_numpy()
    s = cells["scale_cm"].to_numpy()
    diff = a - m
    cat = thirds["scale_third"].to_numpy()
    groups = [diff[cat == i] for i in (1, 2, 3)]
    p_c = stats.ttest_rel(a, m).pvalue
    pair_idx = [(0, 1), (0, 2), (1, 2)]
    p_pairs = [stats.ttest_ind(groups[i], groups[j])[1] for i, j in pair_idx]
    p_pairs_holm = np.asarray(hs.holm(p_pairs))

    fig = hs.new_figure("full", H)
    TW = tuple(float(v) for v in ex["small"]["d"]["time_window"])
    speed_cmap = ListedColormap(mpl.colormaps[hs.CMAP["speed"]](np.linspace(0.0, 0.9, 256)), name="plasma_0_0.9")
    speed_norm = Normalize(*SPEED_RANGE)
    drawn = {}

    # ---- A: maps with 12 s of path
    for k, which in enumerate(CELLS):
        d = ex[which]["d"]
        ev = ex[which]["ev"]
        xe, ye = d["x_edges"], d["y_edges"]
        ext = (xe.min(), xe.max(), ye.min(), ye.max())
        lim = (min(0.0, xe.min()), xe.max(), min(0.0, ye.min()), ye.max())     # axes from 0 cm
        map_h = MAP_W * (lim[3] - lim[2]) / (lim[1] - lim[0])
        left = MAP_LEFT[k]
        ax = hs.axes_mm(fig, left, MAP_TOP, MAP_W, map_h, full=True)          # maps keep the full box
        bg = d["rate_map"].T                                                  # rate maps are indexed [x, y]
        lo, hi = np.percentile(bg[~np.isnan(bg)], 1), np.percentile(bg[~np.isnan(bg)], 99)   # colour limits
        im = ax.imshow(bg, origin="lower", extent=ext, interpolation="nearest", aspect="auto",
                       cmap=hs.CMAP["rate_under_colour"], vmin=lo, vmax=hi, zorder=1)
        ax.set_xticks([0, 50, 100, 150])
        ax.set_yticks([0, 50, 100])                               # ticks first: setting ticks can widen the limits
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
        ax.xaxis.set_major_formatter(hs.NUM)
        ax.yaxis.set_major_formatter(hs.NUM)
        ax.set_xlabel(LAB["x_cm"])
        if k == 0:
            ax.set_ylabel(LAB["y_cm"])
        for sp in ax.spines.values():
            sp.set_zorder(7)
        # path of the window, coloured by speed per segment (mean of the two ends), on a white casing
        t = d["t"]
        idx = (t >= TW[0]) & (t <= TW[1])
        xw, yw, cw = d["x"][idx].astype(np.float32), d["y"][idx].astype(np.float32), d["speed"][idx]
        pts = np.column_stack((xw, yw)).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        cvals = 0.5 * (cw[:-1] + cw[1:])
        ax.plot(xw, yw, color=P["white"], linewidth=2.4, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        lc = LineCollection(segs, cmap=speed_cmap, norm=speed_norm, linewidths=1.4, capstyle="round", joinstyle="round",
                            zorder=4)
        lc.set_array(cvals)
        ax.add_collection(lc)
        vis = ev[(ev["t_exit"] > TW[0]) & (ev["t_enter"] < TW[1])]
        seq = [int(v) for v in vis.sort_values("t_enter", kind="mergesort")["field"]]
        order = {f: i + 1 for i, f in enumerate(dict.fromkeys(seq))}          # numbered by first visit
        hs.field_outlines(ax, d["field_locations"], colour=P["white"], lw=hs.LW_FRAME, numbers=None)
        place_numbers(fig, ax, d["field_locations"], sorted(set(seq)), np.column_stack((xw, yw)), labels=order)
        hs.text_mm(fig, left + MAP_W / 2.0, MAP_TOP - 1.6, TK["grid_scale_rat"][k], ha="center", fontsize=hs.FS_SUBHEAD)
        tk = [v for v in mpl.ticker.MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi) if lo <= v <= hi]
        cbr = hs.colour_bar(fig, im, left + MAP_W + 1.5, MAP_TOP, 1.6, map_h, label=LAB["fr"], ticks=tk)
        cbr.ax.yaxis.set_major_formatter(hs.NUM)
        drawn[which] = dict(map_h=map_h, seq=seq, order=order, vis=vis, ax_map=ax)
    sm = mpl.cm.ScalarMappable(norm=speed_norm, cmap=speed_cmap)                # one speed bar for both paths
    hs.colour_bar(fig, sm, 108.0, MAP_TOP, 1.6, drawn["small"]["map_h"], label=LAB["speed_cm"],
                  ticks=[SPEED_RANGE[0], (SPEED_RANGE[0] + SPEED_RANGE[1]) // 2, SPEED_RANGE[1]])
    hs.panel_letter(fig, "A", hs.LETTER_X, LINE1, clear_of=[drawn["small"]["ax_map"]], not_above=LINE1, row=1)

    # ---- B: time courses for the same 12 s
    row_labels = [LAB["t_prev"], "Movement\ndirection, " + hs.VAR["theta"], LAB["alignment_formula"]]
    for k, which in enumerate(CELLS):
        d = ex[which]["d"]
        vis = drawn[which]["vis"]
        order = drawn[which]["order"]
        visits = [(int(r.field), float(r.t_enter), float(r.t_exit)) for r in vis.itertuples(index=False)]
        direction_deg = np.degrees(d["movedir_rad_pm_pi"])                   # -180 to 180 deg
        direction_deg[direction_deg < -90] += 360                            # shown from -90 to 270 deg
        S = hs.stacked_traces(fig, B_LEFT[k], B_TOP, B_W, [BOX_H] * 3, d["t"], [None, None, None], [None, None, None],
                              TW, visits=[(order[f], a_, b_) for f, a_, b_ in visits], spikes_t=None, gap=GAP,
                              xticks=list(np.linspace(TW[0], TW[1], 4)), xlabel=LAB["time"])
        a0, a1, a2 = S["axes"]
        for ax in (a0, a1):
            hs.box_axes(ax, sides=("left",))                                  # rows 1 and 2: left spine only
        msk = (d["t"] >= TW[0]) & (d["t"] <= TW[1])
        tt = d["t"][msk]
        tp_t, tp_y = with_breaks(tt, d["t_prev"][msk])
        a0.plot(tp_t, tp_y, color=P["ink"], linewidth=hs.LW_DATA, zorder=3, solid_joinstyle="round")
        a0.set_ylim(0, 5.2)
        a0.set_yticks([0, 2.5, 5])
        a0.yaxis.set_major_formatter(hs.NUM)
        dd = direction_deg[msk]
        jump = np.where(np.abs(np.diff(dd)) > 180)[0]                         # a wrap, if any, breaks the line
        tt_b, dd_b = with_breaks(tt, dd, extra=jump)
        a1.plot(tt_b, dd_b, color=P["ink"], linewidth=hs.LW_DATA, zorder=3, solid_joinstyle="round")
        a1.set_ylim(-100, 260)
        a1.set_yticks([0, 180])
        a1.yaxis.set_major_formatter(hs.DEG)
        # grid alignment: aligned colour above zero, misaligned colour below; the curve is split at each zero crossing
        # (crossing points by linear interpolation), so that neither colour runs along the zero line
        mo = (d["t"] > TW[0]) & (d["t"] < TW[1])
        tg = d["t"][mo]
        g = d["grid_alignment"][mo]
        xs, ys = [tg[0]], [g[0]]
        for i in range(len(g) - 1):
            if tg[i + 1] - tg[i] > SAMPLE_GAP:
                xs.append(np.nan)
                ys.append(np.nan)
            elif (g[i] > 0) != (g[i + 1] > 0) and g[i] != g[i + 1]:
                tc = tg[i] + (0 - g[i]) * (tg[i + 1] - tg[i]) / (g[i + 1] - g[i])
                xs.append(tc)
                ys.append(0.0)
            xs.append(tg[i + 1])
            ys.append(g[i + 1])
        xs, ys = np.asarray(xs), np.asarray(ys)
        pos = np.where(ys >= 0, ys, np.nan)
        neg = np.where(ys <= 0, ys, np.nan)
        hs.zero_line(a2)
        a2.plot(xs, pos, color=P["aligned"], linewidth=hs.LW_DATA, zorder=3, solid_joinstyle="round")
        a2.plot(xs, neg, color=P["misaligned"], linewidth=hs.LW_DATA, zorder=3, solid_joinstyle="round")
        a2.set_ylim(-1.25, 1.25)
        a2.set_yticks([-1, 0, 1])
        a2.yaxis.set_major_formatter(hs.NUM)
        drawn[which]["visits"] = visits
        drawn[which]["trace_axes"] = S["axes"]
    row_label_texts = []
    for i, s_ in enumerate(row_labels):                                       # row labels once, left of column 1
        yc = B_TOP + i * (BOX_H + GAP) + BOX_H / 2.0
        row_label_texts.append(hs.text_mm(fig, B_LEFT[0] - 10.0, yc, s_, ha="right", va="center", fontsize=hs.FS_TITLE,
                                          linespacing=1.15))
    key = hs.line_key(fig, [(TK["aligned"][0], P["aligned"], "-"), (TK["aligned"][1], P["misaligned"], "-")],
                      B_LEFT[1] + B_W, ROW2)                                  # at the top of the panel, flush right
    hs.panel_letter(fig, "B", hs.LETTER_X, LINE2, clear_of=list(drawn["small"]["trace_axes"]) + row_label_texts + key,
                    not_above=LINE2, row=2)

    # ---- C: proximity, aligned and misaligned; from the row top to the bottom of the maps of A
    a_c = hs.axes_mm(fig, 140.0, TOP1, RIGHT - 140.0, MAP_TOP - TOP1 + drawn["small"]["map_h"])
    hs.panel_letter(fig, "C", COL3 + hs.LETTER_X, LINE1, clear_of=[a_c], not_above=LINE1, row=1)
    a_c.set_xlim(-0.6, 1.6)
    a_c.set_ylim(0, 0.8)
    box_w = 7.6 * (2.2 / (RIGHT - 140.0))                                     # every box 7.6 mm wide
    ends_c = hs.boxplot(a_c, [a, m], colours=[P["aligned_fill"], P["misaligned_fill"]], width=box_w)
    a_c.set_xticks([0, 1])
    a_c.set_xticklabels(TK["aligned"])
    a_c.set_yticks([0, 0.2, 0.4, 0.6])
    a_c.yaxis.set_major_formatter(hs.NUM)
    a_c.set_ylabel(LAB["proximity"])
    yb_c = max(e[1] for e in ends_c) + hs.pt_to_data(a_c, 4.0)
    hs.sig_bracket(a_c, 0, 1, yb_c, p_c)

    # ---- D: proximity difference by thirds of grid scale; from the row top to the bottom of the boxes of B
    a_d = hs.axes_mm(fig, 140.0, ROW2, RIGHT - 140.0, B_TOP - ROW2 + 3 * BOX_H + 2 * GAP)
    hs.panel_letter(fig, "D", COL3 + hs.LETTER_X, LINE2, clear_of=[a_d], not_above=LINE2, row=2)
    a_d.set_xlim(-0.6, 2.6)
    a_d.set_ylim(-0.035, 0.07)
    box_w_d = 7.6 * (3.2 / (RIGHT - 140.0))
    ends_d = hs.boxplot(a_d, groups, width=box_w_d)
    hs.zero_line(a_d)
    a_d.set_xticks([0, 1, 2])
    a_d.set_xticklabels(TK["scale_groups"])
    a_d.set_xlabel(LAB["grid_scale"])
    a_d.set_yticks([-0.02, 0, 0.02, 0.04])
    a_d.yaxis.set_major_formatter(hs.NUM)
    a_d.set_ylabel(LAB["proximity_diff"], linespacing=1.1)
    p_one = [stats.ttest_1samp(g_, 0)[1] for g_ in groups]
    for i, (lo_, hi_) in enumerate(ends_d):
        hs.sig_mark(a_d, i, hi_, p_one[i], side="above")
    star_top = max(e[1] for e in ends_d) + hs.pt_to_data(a_d, 1.5 + (0.74 - 0.40) * 10 + 1.5)   # above the asterisks
    hs.bracket_stack(a_d, pair_idx, list(p_pairs_holm), star_top + hs.pt_to_data(a_d, 2.5))

    common.save_outputs(fig, out_dir, FIGURE)

    # ---- source data
    rows = []
    for which in CELLS:
        d = ex[which]["d"]
        msk = (d["t"] >= TW[0]) & (d["t"] <= TW[1])
        md = np.degrees(d["movedir_rad"][msk].astype(np.float32)).astype(np.float64)   # in single precision
        rows.append(pd.DataFrame(dict(
            cell=which, row=int(d["row"]), time_s=d["t"][msk], x_cm=d["x"][msk].astype(np.float32).astype(np.float64),
            y_cm=d["y"][msk].astype(np.float32).astype(np.float64), speed_cm_s=d["speed"][msk],
            t_prev_s=d["t_prev"][msk],
            movement_direction_deg=md - np.where(md > 180, 360.0, 0.0),
            grid_alignment=d["grid_alignment"][msk],
            inside_field=d["in_field"][msk].astype(np.int64))))
    common.write_source_data(out_dir, FIGURE, "AB", "paths", pd.concat(rows, ignore_index=True))
    fr = []
    for which in CELLS:
        fl = ex[which]["d"]["field_locations"]
        order = drawn[which]["order"]
        fr.append(pd.DataFrame(dict(cell=which, field=np.arange(len(fl)), centre_x_cm=fl[:, 0], centre_y_cm=fl[:, 1],
                                    radius_cm=fl[:, 2],
                                    visited_in_window=[int(k in drawn[which]["seq"]) for k in range(len(fl))],
                                    number_in_figure=pd.array([order.get(k) for k in range(len(fl))], dtype="Int64"))))
    common.write_source_data(out_dir, FIGURE, "A", "fields", pd.concat(fr, ignore_index=True))
    mp = []
    for which in CELLS:
        d = ex[which]["d"]
        xe, ye, M = d["x_edges"], d["y_edges"], d["rate_map"]
        ii, jj = np.meshgrid(np.arange(M.shape[0]), np.arange(M.shape[1]), indexing="ij")
        mp.append(pd.DataFrame(dict(cell=which, x_bin_from_cm=xe[ii.ravel()], x_bin_to_cm=xe[ii.ravel() + 1],
                                    y_bin_from_cm=ye[jj.ravel()], y_bin_to_cm=ye[jj.ravel() + 1], rate_hz=M.ravel())))
    common.write_source_data(out_dir, FIGURE, "A", "rate_maps", pd.concat(mp, ignore_index=True))
    vv = [dict(cell=which, field=f, number_in_figure=drawn[which]["order"][f], t_enter_s=a_, t_exit_s=b_)
          for which in CELLS for f, a_, b_ in drawn[which]["visits"]]
    common.write_source_data(out_dir, FIGURE, "B", "field_visits", pd.DataFrame(vv))
    names = np.array(["", "Small", "Medium", "Large"])
    common.write_source_data(out_dir, FIGURE, "CD", "cells", dict(
        row=cells["row"].values, session_id=cells["session_id"].values, cell_id=cells["cell_id"].values,
        proximity_aligned=a, proximity_misaligned=m, proximity_difference=diff, grid_scale_cm=s,
        grid_scale_third=names[cat]))
    st = [dict(panel="C", test="paired t-test, aligned against misaligned", n=a.size, df=str(a.size - 1),
               statistic=stats.ttest_rel(a, m)[0], p=p_c, p_holm=np.nan, mark=hs.sig_stars(p_c))]
    for i in range(3):
        t1, p1 = stats.ttest_1samp(groups[i], 0)
        st.append(dict(panel="D", test=f"one-sample t-test against zero, {TK['scale_groups'][i].lower()} grid scale",
                       n=groups[i].size, df=str(groups[i].size - 1), statistic=t1, p=p1, p_holm=np.nan,
                       mark=hs.sig_stars(p1)))
    F, pF = stats.f_oneway(*groups)
    n_all = sum(g_.size for g_ in groups)
    st.append(dict(panel="D", test="one-way ANOVA across the three thirds", n=n_all, df="2, %d" % (n_all - 3),
                   statistic=F, p=pF, p_holm=np.nan, mark="(none drawn)"))
    for k, (i, j) in enumerate(pair_idx):
        tt_, pp_ = stats.ttest_ind(groups[i], groups[j])
        st.append(dict(panel="D", test=f"two-sample t-test, {TK['scale_groups'][i].lower()} against "
                                       f"{TK['scale_groups'][j].lower()}", n=groups[i].size + groups[j].size,
                       df=str(groups[i].size + groups[j].size - 2), statistic=tt_, p=pp_, p_holm=p_pairs_holm[k],
                       mark=hs.sig_stars(p_pairs_holm[k])))
    common.write_source_data(out_dir, FIGURE, "", "statistics", pd.DataFrame(st))
    bn = []
    for nm, gs in (("C", [a, m]), ("D", groups)):
        for pos_, s_ in enumerate(hs.box_stats(gs)):
            bn.append(dict(panel=nm, position=float(pos_), **{k: s_[k] for k in ("n", "median", "q1", "q3",
                                                                                  "whisker_low", "whisker_high")}))
    common.write_source_data(out_dir, FIGURE, "", "box_numbers", pd.DataFrame(bn))
