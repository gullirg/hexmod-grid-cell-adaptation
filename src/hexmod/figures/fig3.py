"""Figure 3: firing rate adaptation of grid cells.

Panels
    A  Rate map of an example cell with 12 s of path (white), its spikes (black dots) and the field outlines; the
       fields visited in the window are numbered 1, 2, ... in the order of their first visit.
    B  Time courses along that path: time since the previous field visit (t_prev), the location input and the
       adaptation variable; shaded, field visits (numbered as in A); ticks above, spikes.
    C  Change of firing rate from each cell's mean by t_prev bin (box plots, all cells); brackets: first bin against
       each other bin, paired t-tests with Holm correction within the panel.
    D  Adaptation coefficient of the parametric model (all cells); asterisks: one-sample t-test against 0.
    E  A schematic that the code does not draw; its space is left empty.
    F  Firing rate ratio of the third visit when the interval between visits 2 and 3 is shorter (T23 < T12) or
       longer (T23 > T12) than the interval between visits 1 and 2; asterisks: one-sample t-tests against 0;
       bracket: paired t-test.

Inputs (``rat/`` in the results folder): ``examples/fig3ab.npz``, ``examples/fig3ab_field_visits.csv``,
``adaptation/fr_percent_change.npy``, ``adaptation/param_cells.csv``, ``cohort/cells.csv`` and
``visits/paired_cells.csv``. Box plots follow the house convention (``style.boxplot``). Positions in mm from the top
left corner of the figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy import stats

from .. import io as hio
from .. import stats as hstats
from . import common
from . import style as hs

FIGURE = "Fig3"
W, H = hs.WIDTH["full"], 125.5
# Each row of panels has one line of letters (top of the capital); the topmost element of the row's panels stands
# style.LETTER_ABOVE under it. Row 2's line lies 2.79 mm under the lowest text of row 1.
LINE1 = hs.LETTER_TOP                  # 1.5 mm: letters A, C, D
TOP1 = hs.panels_top(LINE1)            # 3.0 mm: top of the map, its colour bar and the boxes of C and D
LINE2 = 49.5                           # letters B, F
TOP2 = hs.panels_top(LINE2)            # 51.0 mm: top of the field numbers of B and the box of F
XL, XC, XD = 0.0, 76.0, 136.0          # column lefts: A and B; C (and the empty space of E); D and F
T_PREV_BINS = ("0_to_1_s", "1_to_2_s", "2_to_3_s", "above_3_s")


def head(fig, letter, x, line, row, clear_of):
    """Panel letter of a panel whose column starts at x (mm), on the letter line of its row."""
    return hs.panel_letter(fig, letter, x + hs.LETTER_X, line, clear_of=list(clear_of), not_above=line, row=row)


def first_visit_order(visits):
    """Numbers 1, 2, ... of the fields in the order of their first visit (visits: (field, t_enter, t_exit))."""
    return {f: i + 1 for i, f in enumerate(dict.fromkeys(v[0] for v in sorted(visits, key=lambda v: v[1])))}


def load_inputs(results_dir):
    """Read the result files of Figure 3."""
    Z = hio.load_npz(common.find_input(results_dir, "rat/examples/fig3ab.npz"))
    ev = hio.read_csv(common.find_input(results_dir, "rat/examples/fig3ab_field_visits.csv"))
    FR = hio.load_npy(common.find_input(results_dir, "rat/adaptation/fr_percent_change.npy"))
    param = hio.read_csv(common.find_input(results_dir, "rat/adaptation/param_cells.csv"))
    cells = hio.read_csv(common.find_input(results_dir, "rat/cohort/cells.csv"))
    quad = hio.read_csv(common.find_input(results_dir, "rat/visits/paired_cells.csv"))
    return Z, ev, FR, param, cells, quad


def build(results_dir, out_dir):
    """Draw Figure 3 from ``results_dir`` into ``out_dir``."""
    inputs = load_inputs(results_dir)
    with mpl.rc_context():
        hs.use_style()
        _draw(out_dir, *inputs)


def _draw(out_dir, Z, ev_df, FR, param, cells, quad):
    P = hs.PALETTE

    x, y, t, spikes = Z["x"], Z["y"], Z["t"], Z["spikes"]
    t_prev, loc_in, adapt = Z["t_prev"], Z["location_input"], Z["adaptation_variable"]
    rate_map, x_edges, y_edges, field_loc = Z["rate_map"], Z["x_edges"], Z["y_edges"], Z["field_locations"]
    FR = np.asarray(FR, float)
    alpha = param["alpha_tprevXloc_all"].to_numpy()
    f_short = quad["frr_T23_lt_T12"].to_numpy(float)
    f_long = quad["frr_T23_gt_T12"].to_numpy(float)
    ok = ~np.isnan(f_short) & ~np.isnan(f_long)
    f_short, f_long = f_short[ok], f_long[ok]

    TW = (float(Z["time_window"][0]), float(Z["time_window"][1]))
    idx = (t >= TW[0]) & (t <= TW[1])                                   # samples of the window (map)
    spk_map = np.where(spikes[idx])[0]
    xw, yw = x[idx], y[idx]
    spk_t = t[(spikes > 0) & (t > TW[0]) & (t < TW[1])]                 # spike times of the window (time courses)
    sub = ev_df[(ev_df["t_exit"] > TW[0]) & (ev_df["t_enter"] < TW[1])]
    visits = [(int(r.field), float(r.t_enter), float(r.t_exit)) for r in sub.itertuples(index=False)]
    bg = rate_map.T
    vmin, vmax = np.percentile(bg[~np.isnan(bg)], 1), np.percentile(bg[~np.isnan(bg)], 99)   # colour limits

    # statistics drawn
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    p_raw = np.array([stats.ttest_rel(FR[:, i], FR[:, j]).pvalue for i, j in pairs])
    p_holm = hs.holm(p_raw)
    anova = hstats.rm_anova(FR)
    rD = stats.ttest_1samp(alpha, 0.0)
    rFs, rFl = stats.ttest_1samp(f_short, 0.0), stats.ttest_1samp(f_long, 0.0)
    rFp = stats.ttest_1samp(f_short - f_long, 0.0)

    fig = hs.new_figure("full", H)

    # ---- A: rate map, path, spikes, fields
    XLO, YLO = min(0.0, float(x_edges.min())), min(0.0, float(y_edges.min()))   # axes from 0 cm (ticks include 0)
    map_h = 34.0                                                    # shares top and bottom with the boxes of C and D
    map_w = map_h * (x_edges.max() - XLO) / (y_edges.max() - YLO)
    axA = hs.axes_mm(fig, 17.0, TOP1, map_w, map_h)
    im = hs.rate_map(axA, rate_map, (x_edges.min(), x_edges.max(), y_edges.min(), y_edges.max()), vmin=vmin, vmax=vmax)
    axA.set_aspect("auto")                                          # the box is already in proportion
    axA.plot(xw, yw, color=P["white"], linewidth=hs.LW_REF, zorder=3, solid_joinstyle="round", solid_capstyle="round")
    axA.set_xlim(XLO, x_edges.max())
    axA.set_ylim(YLO, y_edges.max())
    visited = [v[0] for v in visits]
    ORDER = first_visit_order(visits)
    hs.field_outlines(axA, field_loc, numbers=None)
    # numbers of the visited fields just outside the outline, on the side farthest from the path, the other fields and
    # the numbers already placed (angles in steps of 15 deg, with a mild preference for the upper right)
    mm_per_cm = map_w / (x_edges.max() - XLO)
    half_w, half_h = 1.1 / mm_per_cm, 1.0 / mm_per_cm                # half size of one 8 pt digit (cm)
    pts = np.column_stack([xw, yw])
    pts = pts[np.all(np.isfinite(pts), 1)]
    num_pos = {}
    for k in visited:
        cx, cy, rr = field_loc[k, :3]
        best = None
        for ang in range(0, 360, 15):
            ux, uy = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
            px = cx + ux * (rr + 0.5 / mm_per_cm + half_w * abs(ux))
            py = cy + uy * (rr + 0.5 / mm_per_cm + half_h * abs(uy))
            if not (XLO + half_w + 0.3 / mm_per_cm <= px <= x_edges.max() - half_w - 0.3 / mm_per_cm and
                    YLO + half_h + 0.3 / mm_per_cm <= py <= y_edges.max() - half_h - 0.3 / mm_per_cm):
                continue
            d_path = np.min(np.hypot(pts[:, 0] - px, pts[:, 1] - py)) - max(half_w, half_h)
            d_field = min([np.hypot(field_loc[j, 0] - px, field_loc[j, 1] - py) - field_loc[j, 2] - max(half_w, half_h)
                           for j in range(len(field_loc)) if j != k] + [99])
            d_other = min([np.hypot(q[0] - px, q[1] - py) - 2 * max(half_w, half_h) for q in num_pos.values()] + [99])
            sc = min(d_path, 4.0) + min(d_field, 4.0) + min(d_other, 4.0) - 0.02 * abs(((ang - 45 + 180) % 360) - 180)
            if best is None or sc > best[0]:
                best = (sc, px, py, ang)
        num_pos[k] = (best[1], best[2])
        axA.text(best[1], best[2], f"{ORDER[k]:d}", ha="center", va="center", fontsize=hs.FS_SMALL, color=P["white"],
                 zorder=6)
    hs.spikes_on_map(axA, xw[spk_map], yw[spk_map])
    axA.set_xticks([0, 50, 100, 150])
    axA.set_yticks([0, 50, 100, 150])
    axA.set_xlabel(hs.LABEL["x_cm"])
    axA.set_ylabel(hs.LABEL["y_cm"])
    hs.colour_bar(fig, im, 17.0 + map_w + 2.5, TOP1, 2.2, map_h, label=hs.LABEL["fr"], ticks=[0, 5, 10, 15])
    head(fig, "A", XL, LINE1, 1, [axA])

    # ---- B: time courses; the field numbers start at the row top, the last box ends level with the box of F
    B_TOP, B_LEFT, B_W, B_GAP = TOP2 + 3.54, 26.0, 44.0, 2.0
    B_BOTTOM = TOP2 + 63.3
    B_H = ((B_BOTTOM - (B_TOP + 2.2 + 0.6) - 2 * B_GAP) / 3.0,) * 3     # spike row 2.2 mm, 0.6 mm above box 1
    st = hs.stacked_traces(fig, B_LEFT, B_TOP, B_W, B_H, t, [t_prev, loc_in, adapt], [None, None, None], TW,
                           visits=[(ORDER[f], a, b) for f, a, b in visits], spikes_t=spk_t, gap=B_GAP,
                           xticks=list(np.linspace(TW[0], TW[1], 4)), xlabel=hs.LABEL["time"])
    axB = st["axes"]
    for a in axB[:2]:
        hs.box_axes(a, sides=("left",))                            # rows 1 and 2: left spine only
    limits = [(-0.1, 2.1), (-0.006, 0.126), (-0.016, 0.33)]
    for a, lim, tk in zip(axB, limits, [[0, 1, 2], [0, 0.1], [0, 0.1, 0.2, 0.3]]):
        a.set_ylim(*lim)
        a.set_yticks(tk)
        a.yaxis.set_major_formatter(hs.NUM)
    sa = st["spike_ax"]
    B_LABEL_X = B_LEFT - 7.0                                        # row labels horizontal, right-aligned (as Fig 4B)
    row_labels = []
    for a, s_ in zip(axB, [hs.LABEL["t_prev"], "Location\ninput", "Adaptation\nvariable"]):
        bb = a.get_position()
        row_labels.append(hs.text_mm(fig, B_LABEL_X, (1.0 - (bb.y0 + bb.y1) / 2.0) * H, s_, ha="right", va="center",
                                     fontsize=hs.FS_TITLE, linespacing=1.15))
    bbs = sa.get_position()
    row_labels.append(hs.text_mm(fig, B_LABEL_X, (1.0 - (bbs.y0 + bbs.y1) / 2.0) * H, "Spikes", ha="right", va="center",
                                 fontsize=hs.FS_TITLE))
    head(fig, "B", XL, LINE2, 2, list(axB) + row_labels)

    # ---- C: firing rate change by t_prev bin
    axC = hs.axes_mm(fig, 96.0, TOP1, 38.0, 34.0)
    head(fig, "C", XC, LINE1, 1, [axC])
    axC.set_xlim(-0.6, 3.6)
    ends = hs.boxplot(axC, [FR[:, i] for i in range(4)], width=hs.box_width_data(axC, 6.5))
    hs.zero_line(axC)
    axC.set_xticks(range(4))
    axC.set_xticklabels(hs.TICKS["t_prev_bins"])
    axC.set_xlabel(hs.LABEL["t_prev"])
    axC.set_ylabel("Firing rate\nchange (%)")
    axC.set_yticks([-20, 0, 20, 40])
    axC.yaxis.set_major_formatter(hs.NUM)
    lo = min(e[0] for e in ends) - 4.0
    top = 60.0
    for _ in range(20):                                             # y limit that holds three brackets
        axC.set_ylim(lo, top)
        y0 = max(e[1] for e in ends) + hs.pt_to_data(axC, 4.0)
        need = y0 + 2 * hs.pt_to_data(axC, 9.5) + hs.pt_to_data(axC, 1.0 + 0.74 * hs.FS_TITLE + 2.0)
        if abs(need - top) < 1e-6:
            break
        top = need
    axC.set_ylim(lo, top)
    hs.bracket_stack(axC, [(0, 1), (0, 2), (0, 3)], p_holm[:3], y0)

    # ---- D: adaptation coefficient
    axD = hs.axes_mm(fig, 156.0, TOP1, 17.0, 34.0)
    head(fig, "D", XD, LINE1, 1, [axD])
    axD.set_xlim(-0.6, 0.6)
    endsD = hs.boxplot(axD, [alpha], width=hs.box_width_data(axD))
    hs.zero_line(axD)
    axD.set_xticks([])
    axD.set_ylim(-0.25, 0.15)
    axD.set_yticks([-0.2, -0.1, 0, 0.1])
    axD.yaxis.set_major_formatter(hs.NUM)
    axD.set_ylabel("Adaptation\ncoefficient " + hs.VAR["alpha"])
    hs.sig_mark(axD, 0, endsD[0][0], rD.pvalue, side="below")

    # ---- F: firing rate ratio by interval order; from the row top to the bottom of the last box of B
    f_top = TOP2
    f_h = B_BOTTOM - TOP2
    axF = hs.axes_mm(fig, 141.0, f_top, 32.0, f_h)
    axF.set_xlim(-0.5, 1.5)
    endsF = hs.boxplot(axF, [f_short, f_long], width=hs.box_width_data(axF))
    hs.zero_line(axF)
    axF.set_xticks([0, 1])
    axF.set_xticklabels(["Shorter", "Longer"])
    axF.set_xlabel(hs.VAR["T23"] + " against " + hs.VAR["T12"])
    axF.set_ylabel(hs.LABEL["fr_ratio"])
    axF.set_yticks([-0.2, 0, 0.2])
    axF.yaxis.set_major_formatter(hs.NUM)
    axF.set_ylim(-0.4, 0.45)
    for _ in range(20):
        lo = endsF[1][0] - hs.pt_to_data(axF, 1.5 + 0.74 * hs.FS_TITLE + 3.0)
        yb = max(endsF[0][1] + hs.pt_to_data(axF, 1.5 + 0.74 * hs.FS_TITLE - 0.4 * hs.FS_TITLE + 5.0),
                 endsF[1][1] + hs.pt_to_data(axF, 3.0))
        hi = yb + hs.pt_to_data(axF, 1.0 + 0.74 * hs.FS_TITLE + 3.0)
        if np.allclose(axF.get_ylim(), (lo, hi), atol=1e-9):
            break
        axF.set_ylim(lo, hi)
    hs.sig_mark(axF, 0, endsF[0][1], rFs.pvalue, side="above")
    hs.sig_mark(axF, 1, endsF[1][0], rFl.pvalue, side="below")
    hs.sig_bracket(axF, 0, 1, yb, rFp.pvalue)
    head(fig, "F", XD, LINE2, 2, [axF])

    common.save_outputs(fig, out_dir, FIGURE)

    # ---- source data
    common.write_source_data(out_dir, FIGURE, "A", "rate_map_hz", common.map_table(rate_map, x_edges, y_edges,
                                                                                  "rate_hz", "x_cm", "y_cm"))
    common.write_source_data(out_dir, FIGURE, "A", "path_and_spikes", {"t_s": t[idx], "x_cm": xw, "y_cm": yw,
                                                                       "spikes_in_bin": spikes[idx]})
    common.write_source_data(out_dir, FIGURE, "A", "fields", {
        "field": np.arange(len(field_loc)), "centre_x_cm": field_loc[:, 0], "centre_y_cm": field_loc[:, 1],
        "radius_cm": field_loc[:, 2], "field_rate_hz": Z["field_rate_hz"],
        "visited_in_window": [k in visited for k in range(len(field_loc))],
        "number_in_figure": pd.array([ORDER.get(k) for k in range(len(field_loc))], dtype="Int64")})
    common.write_source_data(out_dir, FIGURE, "B", "time_courses", {
        "t_s": t[idx], "t_prev_s": t_prev[idx], "location_input": loc_in[idx], "adaptation_variable": adapt[idx],
        "spikes_in_bin": spikes[idx]})
    common.write_source_data(out_dir, FIGURE, "B", "field_visits", pd.DataFrame(
        visits, columns=["field", "t_enter_s", "t_exit_s"]).assign(number_in_figure=[ORDER[f] for f, _, _ in visits]))
    dC = pd.DataFrame(FR, columns=[f"pct_change_{b}" for b in T_PREV_BINS])
    dC.insert(0, "cell", cells["unique_cell_id"].values)
    common.write_source_data(out_dir, FIGURE, "C", "firing_rate_change_pct", dC)
    common.write_source_data(out_dir, FIGURE, "D", "adaptation_coefficient", {"cell": cells["unique_cell_id"].values,
                                                                              "adaptation_coefficient": alpha})
    common.write_source_data(out_dir, FIGURE, "F", "firing_rate_ratio", quad.loc[ok, [
        "unique_cell_id", "frr_T23_lt_T12", "frr_T23_gt_T12", "n_events_T23_lt_T12", "n_events_T23_gt_T12"]].rename(
        columns={"unique_cell_id": "cell"}))
    n_c, n_f = FR.shape[0], len(f_short)
    rows = [dict(panel="C", test="one-way repeated-measures ANOVA over the four bins",
                 statistic=f"F({anova['df1']},{anova['df2']}) = {anova['F']:.1f}", p=anova["p"], p_holm=np.nan, n=n_c)]
    for k, (i, j) in enumerate(pairs):
        rows.append(dict(panel="C", test=f"paired t-test, bin {i + 1} against bin {j + 1}",
                         statistic=f"t({n_c - 1}) = {stats.ttest_rel(FR[:, i], FR[:, j]).statistic:.2f}", p=p_raw[k],
                         p_holm=p_holm[k], n=n_c))
    rows.append(dict(panel="D", test="one-sample t-test against 0",
                     statistic=f"t({n_c - 1}) = {rD.statistic:.2f}; "
                               f"Cohen's d = {alpha.mean() / alpha.std(ddof=1):.2f}; "
                               f"{np.mean(alpha < 0) * 100:.1f}% of cells below 0", p=rD.pvalue, p_holm=np.nan, n=n_c))
    for nm, r, a in (("T23 < T12, one-sample t-test against 0", rFs, f_short),
                     ("T23 > T12, one-sample t-test against 0", rFl, f_long),
                     ("paired t-test, T23 < T12 against T23 > T12", rFp, f_short - f_long)):
        rows.append(dict(panel="F", test=nm, statistic=f"t({n_f - 1}) = {r.statistic:.2f}; mean {a.mean():.4f}; "
                                                       f"Cohen's d = {a.mean() / a.std(ddof=1):.2f}",
                         p=r.pvalue, p_holm=np.nan, n=n_f))
    common.write_source_data(out_dir, FIGURE, "", "statistics", pd.DataFrame(rows))
