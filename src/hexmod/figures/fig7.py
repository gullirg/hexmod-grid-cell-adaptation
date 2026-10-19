"""Figure 7: detection of hexadirectional modulation in simulated participants.

Each simulated participant is the summed activity of 10,000 grid cells with firing rate adaptation; participants are
numbered by their seed, from 0. Left column, simple method; right column, location-controlled method. In every panel
the small module is on the left and the large module on the right.

Panels
    A, D  Firing rate in 12 bins of movement direction relative to the grid (0 deg = along a grid axis; aligned bins
          even, misaligned bins odd), percent change from each participant's mean, participants 0 to 29.
    B, E  Cross-validated HexMod beta for 1- to 7-fold symmetry (percent of the mean moving rate), participants 0 to
          29; asterisks: two-sided one-sample t-test against 0 (uncorrected).
    C     Estimated six-fold grid orientation of participants 0 to 29 (orientation times 6, 30 deg bins), both
          methods; bar heights relative to the fullest bin.
    F     Detection rate: share of repeated experiments of 30 participants in which the one-sided one-sample t-test
          of the coefficient against 0 gives p < 0.05. By default 49 experiments (participants 30 to 1,499), or 50
          (participants 0 to 1,499) when step 25 was run with ``--experiments 50``.

The statistics table is computed from the plotted arrays. The tests of A and D compare the mean of the aligned bins
with the mean of the misaligned bins of the summed rate in Hz (the values from which the plotted percentages are
derived), read from the participant tables; their effect size ``cohens_dz_ddof0`` is the mean paired difference over
its standard deviation computed with ddof 0, as in the published analysis.

Inputs (``sim/fig7/`` in the results folder, else the shipped ``results_precomputed/sim/fig7/``):
``fig7ad_direction_bins.csv``, ``fig7be_beta.csv``, ``fig7c_orientation.csv``, ``fig7f_p_per_experiment.csv``,
``fig7f_detection.csv``, ``participants_small.csv.gz``, ``participants_large.csv.gz`` and ``fig7_values.json``
(number of experiments of panel F). Positions in the code are in mm from the top left corner of the figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.ticker import FixedLocator, MultipleLocator
from scipy import stats

from .. import io as hio
from . import common
from . import style as hs

FIGURE = "Fig7"
W = hs.WIDTH["full"]
H_MAX = 146.0                       # height of the figure (mm), never exceeded
MODULES = ("small", "large")
METHODS = ("simple", "location-controlled")
MODULE_WORD = dict(zip(MODULES, hs.TICKS["modules"]))
N_PER_EXPERIMENT = 30               # participants per experiment; panels A to E show the first experiment
DIRS = np.arange(0, 360, 30)        # direction bins (deg from a grid axis)
FOLDS = range(1, 8)                 # symmetries
ALPHA = 0.05

COLX = {"L": 0.0, "R": 87.0}
BOX_L, BOX_W, BOX_GAP = 17.0, 32.5, 3.0
# Rows (mm from the top of the figure). Each row has one line of letters (top of the capital); its panels start
# style.LETTER_ABOVE below it. The first line of rows 1 and 3 holds the keys (their swatch tops at the top of the row),
# then the module headings and the boxes follow; in row 2 the module headings start at the top of the row. Rows 2 and
# 3 have their letter line G_ROW below the lowest text of the row above. Every Cartesian box has one height, chosen so
# that the figure is at most H_MAX high; the polar plots of C are 22 mm across.
G_ROW = 1.93                        # white from the lowest text of a row to the letter line of the next row
BOTTOM = 1.16                       # white below the lowest text
DEPTH = {1: 8.537, 2: 8.834, 3: 8.833}      # tick labels and axis label below the boxes of each row
KEYED = {1: True, 2: False, 3: True}        # rows whose first line holds a key
KEY_UP = hs.TITLE_PITCH + hs.CAP12 - hs.KEY["swatch_h"]
POLAR_CX = {"small": 30.0, "large": 70.0}     # centres of the polar plots of C, from the left edge of the column
POL_D = 22.0                                  # diameter of the polar plots
WHERE = {"A": (1, "L"), "B": (2, "L"), "C": (3, "L"), "D": (1, "R"), "E": (2, "R"), "F": (3, "R")}
PANEL = {"A": ("AD", "simple"), "D": ("AD", "location-controlled"), "B": ("BE", "simple"),
         "E": ("BE", "location-controlled")}
YL = {"A": (-1.0, 0.8), "D": (-0.75, 0.5), "B": (-0.6, 0.75), "E": (-0.2, 0.5)}
YT = {"A": [-1, -0.5, 0, 0.5], "D": [-0.5, 0, 0.5], "B": [-0.5, 0, 0.5], "E": [-0.2, 0, 0.2, 0.4]}


def box_x(col, m):
    """Left edge (mm) of the box of module m in column col."""
    return COLX[col] + BOX_L + (0.0 if m == "small" else BOX_W + BOX_GAP)


def heading_ascent():
    """Height (mm) of a 10 pt module heading above its baseline, as matplotlib measures it (the higher of the two)."""
    probe = hs.new_figure("full", 100.0)
    return max(50.0 - hs.extent_mm(probe, hs.text_mm(probe, 33.25, 50.0, wd, fontsize=hs.FS_TITLE, ha="center"))[1]
               for wd in MODULE_WORD.values())


def rows_layout(box_h, head_asc):
    """Letter line, top of the panels, baseline of the module headings and top of the boxes of each row (mm), and the
    figure height, for boxes ``box_h`` mm high."""
    line, rt, hb, bt = {}, {}, {}, {}
    for r in (1, 2, 3):
        line[r] = hs.LETTER_TOP if r == 1 else bt[r - 1] + box_h + DEPTH[r - 1] + G_ROW
        rt[r] = hs.panels_top(line[r])
        hb[r] = (rt[r] + hs.KEY["swatch_h"] + 4.5) if KEYED[r] else (rt[r] + head_asc)
        bt[r] = hb[r] + 1.5
    return line, rt, hb, bt, bt[3] + box_h + DEPTH[3] + BOTTOM


def layout():
    """The rows of the figure (``rows_layout``) with the box height that keeps the figure at most H_MAX high.

    Each of the three rows of boxes gives up the same part of a starting height of 26 mm plus KEY_UP.
    Call after ``style.use_style`` (the heading height depends on the typeface).
    """
    head_asc = heading_ascent()
    box_h0 = 26.0 + KEY_UP
    box_h = box_h0 - max(0.0, (rows_layout(box_h0, head_asc)[-1] - H_MAX) / 3.0)
    line, rt, hb, bt, height = rows_layout(box_h, head_asc)
    return dict(line=line, row_top=rt, head_base=hb, box_top=bt, box_h=box_h, height=round(height, 3))


# ------------------------------------------------------------------------------------------------------------ inputs
def _rows(table, m, method, participants=None, order="participant"):
    """Rows of one module and method, sorted by ``order``; checks that they are exactly ``participants`` if given."""
    d = table[(table["module"] == m) & (table["method"] == method)].sort_values(order, kind="stable")
    if participants is not None and list(d[order]) != list(participants):
        raise ValueError(f"{m}, {method}: expected participants {participants[0]} to {participants[-1]}")
    return d


def load_inputs(results_dir):
    """Read the result files of Figure 7.

    Returns
    -------
    dict
        Arrays keyed by (kind, module, method): ``AD`` (30 x 12, percent change by direction), ``BE`` (30 x 7,
        coefficients), ``C`` (30, six-fold orientation modulo 60 deg), ``Fp`` (experiments x 7, one-sided p values),
        ``F50`` (7, detection rates over all 50 experiments), ``rate_hz`` (30 x 12, summed rate by direction in Hz);
        ``n_experiments`` (49 or 50) and ``source`` (label of the experiments of panel F).
    """
    first = list(range(N_PER_EXPERIMENT))
    rd = lambda name: hio.read_csv(common.find_input(results_dir, f"sim/fig7/{name}"))
    ad, be, c, fp, fd = (rd(n) for n in ("fig7ad_direction_bins.csv", "fig7be_beta.csv", "fig7c_orientation.csv",
                                         "fig7f_p_per_experiment.csv", "fig7f_detection.csv"))
    part = {m: rd(f"participants_{m}.csv.gz") for m in MODULES}
    values = hio.read_json(common.find_input(results_dir, "sim/fig7/fig7_values.json"))
    if "fig7f.n_experiments" not in values:
        raise ValueError("fig7_values.json lacks 'fig7f.n_experiments'; run step 25 again")
    n_exp = int(values["fig7f.n_experiments"])
    source = {49: "rest_49_experiments", 50: "all_50_experiments"}[n_exp]
    V = dict(n_experiments=n_exp, source=source)
    for m in MODULES:
        for w in METHODS:
            V[("AD", m, w)] = _rows(ad, m, w, first)[[f"dir_{d}" for d in DIRS]].to_numpy(float)
            V[("BE", m, w)] = _rows(be, m, w, first)[[f"fold_{k}" for k in FOLDS]].to_numpy(float)
            V[("C", m, w)] = _rows(c, m, w, first)["orientation_deg_mod60"].to_numpy(float)
            p = _rows(fp[fp["source"] == source], m, w, order="experiment")
            V[("Fp", m, w)] = p[[f"p_fold_{k}" for k in FOLDS]].to_numpy(float)
            f50 = fd[(fd["module"] == m) & (fd["method"] == w) & (fd["source"] == "all_50_experiments")]
            V[("F50", m, w)] = f50[[f"rate_fold_{k}" for k in FOLDS]].to_numpy(float)[0]
            t = part[m][(part[m]["method"] == w) & (part[m]["participant"].isin(first))].sort_values("participant",
                                                                                                    kind="stable")
            if list(t["participant"]) != first:
                raise ValueError(f"participants_{m}.csv.gz lacks participants 0 to 29 of the {w} method")
            # C order: row-wise means of row-stacked arrays, as in the published analysis
            V[("rate_hz", m, w)] = np.ascontiguousarray(t[[f"rate_dir_{d}_hz" for d in DIRS]].to_numpy(dtype=float))
    return V


def orientation_hist(orientation_deg, fold=6, n_bins=12):
    """Counts of participants 0 to 29 in 30 deg bins of six times the orientation (bin centres 0, 30, ..., 330 deg).

    Returns the counts, the bin centres (deg) and the bin of each participant.
    """
    X = orientation_deg[range(N_PER_EXPERIMENT)] * fold
    dir_idx = np.digitize(X, np.arange(-360 / n_bins, 361, 360 / n_bins) + 360 / n_bins / 2) - 1
    dir_idx[dir_idx == n_bins] = 0
    bin_count = np.bincount(dir_idx, minlength=n_bins)
    dir_center = np.arange(n_bins) * 360 / n_bins
    return bin_count, dir_center, dir_idx


# ----------------------------------------------------------------------------------------------------------- drawing
def build(results_dir, out_dir):
    """Draw Figure 7 from ``results_dir`` into ``out_dir``."""
    V = load_inputs(results_dir)
    with mpl.rc_context():
        hs.use_style()
        _draw(out_dir, V)


def _draw(out_dir, V):
    P = hs.PALETTE
    L = layout()
    BOX_H, box_top = L["box_h"], L["box_top"]
    fig = hs.new_figure("full", L["height"])
    for k, (r, c) in WHERE.items():
        for m in MODULES:
            x = (COLX[c] + POLAR_CX[m]) if k == "C" else box_x(c, m) + BOX_W / 2.0
            hs.text_mm(fig, x, L["head_base"][r], MODULE_WORD[m], fontsize=hs.FS_TITLE, ha="center")
    col_right = {"L": box_x("L", "large") + BOX_W, "R": box_x("R", "large") + BOX_W}
    # keys flush right at the top of their row (swatch tops at the row top): aligned and misaligned (A); the two
    # methods in their line tones (C, F)
    key_row_top = {r: L["row_top"][r] + hs.KEY["swatch_h"] - hs.CAP12 for r in (1, 3)}
    keys = {"A": hs.colour_key(fig, [(hs.TICKS["aligned"][0], P["aligned"]), (hs.TICKS["aligned"][1], P["misaligned"])],
                               col_right["L"], key_row_top[1])}
    for k, col in (("C", "L"), ("F", "R")):
        keys[k] = hs.solid_key(fig, [("Simple", P["method_simple"]), ("Location-controlled", P["method_location"])],
                               col_right[col], key_row_top[3])

    AX, GROUPS = {}, {}
    for p in "ADBE":
        r, c = WHERE[p]; kind, w = PANEL[p]
        for m in MODULES:
            ax = hs.axes_mm(fig, box_x(c, m), box_top[r], BOX_W, BOX_H); AX[(p, m)] = ax
            common.tag(ax, panel=p, name=MODULE_WORD[m])
            A = V[(kind, m, w)]; groups = [A[:, j] for j in range(A.shape[1])]; GROUPS[(p, m)] = groups
            if kind == "AD":
                ax.set_xlim(-18, 348); ax.set_ylim(*YL[p]); hs.zero_line(ax)
                cols = [P["aligned_fill"] if j % 2 == 0 else P["misaligned_fill"] for j in range(12)]
                hs.boxplot(ax, groups, positions=DIRS, colours=cols, width=21.0, points=True)
                ax.xaxis.set_major_locator(FixedLocator([0, 90, 180, 270]))
                ax.xaxis.set_minor_locator(MultipleLocator(30))
                ax.xaxis.set_major_formatter(hs.DEG); ax.set_yticks(YT[p])
                ax.set_xlabel(hs.LABEL["movement_direction_deg"])
                if m == "small": ax.set_ylabel("Firing rate\nchange (%)", linespacing=1.1)
            else:
                ax.set_xlim(0.4, 7.6); ax.set_ylim(*YL[p]); hs.zero_line(ax)
                ends = hs.boxplot(ax, groups, positions=range(1, 8), width=0.6, points=True)
                ax.set_xticks(range(1, 8)); ax.set_yticks(YT[p]); ax.set_xlabel(hs.LABEL["symmetry"])
                if m == "small": ax.set_ylabel(r"$\beta$ (% of" + "\nmean rate)", linespacing=1.1)
                # asterisks: two-sided one-sample t-test against 0, beyond the whisker on the side away from 0
                for j, g in enumerate(groups):
                    t = stats.ttest_1samp(g, 0)
                    side = "above" if t.statistic > 0 else "below"; lo, hi = ends[j]
                    hs.sig_mark(ax, j + 1, hi if side == "above" else lo, t.pvalue, side=side)
            ax.yaxis.set_major_formatter(hs.NUM)
            if m == "large": ax.tick_params(axis="y", labelleft=False)

    # C: polar histograms of six times the orientation (one 60 deg period around the circle), both methods overlaid
    r, c = WHERE["C"]
    for m in MODULES:
        cx = COLX[c] + POLAR_CX[m]; cy = box_top[r] + 3.6 + POL_D / 2.0
        ax = hs.polar_mm(fig, cx, cy, POL_D); AX[("C", m)] = ax
        common.tag(ax, panel="C", name=MODULE_WORD[m])
        hs.polar_style(ax, labels=False, cross=True)
        ax.spines["polar"].set_zorder(5)     # the frame stays on top where a bar or the outline reaches it
        ax.set_xticks(np.deg2rad([0, 90, 180, 270])); ax.set_xticklabels(["0°", "15°", "30°", "45°"])
        for w in METHODS:
            bc, dc, _ = orientation_hist(V[("C", m, w)]); h = bc / bc.max()
            for th, hh in zip(dc, h):
                if w == "simple":
                    ax.bar(np.deg2rad(th), hh, width=np.deg2rad(30), align="center", facecolor=P["method_simple_fill"],
                           edgecolor="none", linewidth=0, zorder=2)
                else:
                    ax.bar(np.deg2rad(th), hh, width=np.deg2rad(30), align="center",
                           facecolor=P["method_location_fill"], edgecolor=P["method_location"] if hh > 0 else "none",
                           linewidth=hs.LW_MINOR, zorder=3)
            if w == "simple":
                h_simple = (dc.copy(), h.copy())
        # the simple histogram again as a step outline in its line tone, over the location-controlled bars, so that it
        # stays visible where both methods have the same height
        dc_s, h_s = h_simple; tt, rr = [], []
        for th, hh in zip(dc_s, h_s):
            a = np.linspace(th - 15.0, th + 15.0, 31); tt += list(a); rr += [hh] * len(a)
        tt.append(tt[0]); rr.append(rr[0])
        ax.plot(np.deg2rad(tt), rr, color=P["method_simple"], linewidth=hs.LW_FRAME, zorder=4, solid_joinstyle="miter",
                clip_on=False)
        ax.set_ylim(0, 1.0); ax.set_axisbelow(True)
        hs.even_polar_labels(fig, ax)

    # F: detection rate, bars in the line tones of the two methods, solid grey line at 0.05
    r, c = WHERE["F"]
    for m in MODULES:
        ax = hs.axes_mm(fig, box_x(c, m), box_top[r], BOX_W, BOX_H); AX[("F", m)] = ax
        common.tag(ax, panel="F", name=MODULE_WORD[m])
        ax.set_xlim(0.4, 7.6); ax.set_ylim(0, 1.05)     # 1.05: a bar at 1 stays below the frame
        hs.zero_line(ax, ALPHA)
        wbar = 0.36
        for w, dx in (("simple", -wbar / 2), ("location-controlled", wbar / 2)):
            rate = np.mean(V[("Fp", m, w)] < ALPHA, axis=0)
            ax.bar(np.arange(1, 8) + dx, rate, width=wbar,
                   color=P["method_simple"] if w == "simple" else P["method_location"], linewidth=0, zorder=3)
        ax.set_xticks(range(1, 8)); ax.set_yticks([0, 0.5, 1]); ax.yaxis.set_major_formatter(hs.NUM)
        ax.set_xlabel(hs.LABEL["symmetry"])
        if m == "small": ax.set_ylabel(hs.LABEL["detection"])
        else: ax.tick_params(axis="y", labelleft=False)

    # panel letters on the letter line of their row, clear of their panel's y labels, y tick labels and key
    for k, (r, c) in WHERE.items():
        hs.panel_letter(fig, k, COLX[c] + hs.LETTER_X, L["line"][r],
                        clear_of=[AX[(k, m)] for m in MODULES] + keys.get(k, []), not_above=L["line"][r], row=r)

    common.save_outputs(fig, out_dir, FIGURE)
    _write_source_data(out_dir, V, GROUPS)


# ------------------------------------------------------------------------------------------------------- source data
def statistics_table(V):
    """The statistics of every panel, computed from the plotted arrays (A and D from the rates in Hz behind them)."""
    n_exp = V["n_experiments"]
    rows = []

    def row(**kw):
        base = dict(panel="", module="", method="", test="", symmetry="", n="", statistic="", df="", p="",
                    p_holm_7_within_box="", cohens_dz_ddof0="", aligned_minus_misaligned_pct="", count="",
                    rate_all_50_experiments="", mark_drawn="")
        base.update(kw); rows.append(base)

    for p in "AD":
        w = PANEL[p][1]
        for m in MODULES:
            rate = V[("rate_hz", m, w)]
            aligned, misaligned = np.mean(rate[:, 0::2], axis=1), np.mean(rate[:, 1::2], axis=1)
            tt = stats.ttest_rel(aligned, misaligned)
            dd = aligned - misaligned
            A = V[("AD", m, w)]
            pct = A[:, 0::2].mean(1) - A[:, 1::2].mean(1)          # aligned minus misaligned, % of the mean rate
            row(panel=p, module=MODULE_WORD[m], method=w,
                test="paired t-test, mean of aligned bins against mean of misaligned bins (summed rate)",
                n=int(rate.shape[0]), statistic=float(tt.statistic), df=int(rate.shape[0] - 1), p=float(tt.pvalue),
                cohens_dz_ddof0=float(np.mean(dd) / np.std(dd)), aligned_minus_misaligned_pct=float(pct.mean()),
                mark_drawn="none (stated in legend)")
    for p in "BE":
        w = PANEL[p][1]
        for m in MODULES:
            B = V[("BE", m, w)]; tt = [stats.ttest_1samp(B[:, j], 0) for j in range(7)]
            pv = [float(r.pvalue) for r in tt]; ph = hs.holm(pv)
            for j in range(7):
                row(panel=p, module=MODULE_WORD[m], method=w, test="two-sided one-sample t-test against 0",
                    symmetry=j + 1, n=int(B.shape[0]), statistic=float(tt[j].statistic), df=int(B.shape[0] - 1),
                    p=pv[j], p_holm_7_within_box=float(ph[j]), mark_drawn=hs.sig_stars(pv[j]) or "none")
    for m in MODULES:
        for w in METHODS:
            o = V[("C", m, w)]; cnt, _, _ = orientation_hist(o)
            cm = stats.circmean(o, high=60, low=0); cs = stats.circstd(o, high=60, low=0)
            nw = int((np.abs(o - 30) <= 7.5).sum())
            row(panel="C", module=MODULE_WORD[m], method=w,
                test=f"orientation: circular mean {cm:.2f} deg, circular SD {cs:.2f} deg; {nw} of {o.size} within "
                     f"7.5 deg of 30 deg; counts in 12 bins {cnt.tolist()}", n=int(o.size))
            Pv = V[("Fp", m, w)]; rate = np.mean(Pv < ALPHA, axis=0); cnt_f = (Pv < ALPHA).sum(0)
            for j in range(7):
                row(panel="F", module=MODULE_WORD[m], method=w,
                    test=f"detection rate: fraction of {n_exp} experiments (30 participants each) with one-sided "
                         "one-sample t-test p < 0.05", symmetry=j + 1, n=int(Pv.shape[0]), statistic=float(rate[j]),
                    count=int(cnt_f[j]), rate_all_50_experiments=float(V[("F50", m, w)][j]))
    return pd.DataFrame(rows)


def _write_source_data(out_dir, V, GROUPS):
    rows = []
    for p in "ADBE":
        kind, w = PANEL[p]
        ad = kind == "AD"
        for m in MODULES:
            A = V[(kind, m, w)]
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    rows.append(dict(panel=p, module=MODULE_WORD[m], method=w, participant=i,
                                     quantity="firing rate change (% of mean)" if ad else "beta (% of mean rate)",
                                     x_kind="direction_deg" if ad else "symmetry_n_fold",
                                     x=int(DIRS[j]) if ad else j + 1,
                                     direction_class=("aligned" if j % 2 == 0 else "misaligned") if ad else "",
                                     value=A[i, j]))
    columns = ["panel", "module", "method", "participant", "quantity", "x_kind", "x", "direction_class", "value"]
    src = pd.DataFrame(rows)[columns]
    common.write_source_data(out_dir, FIGURE, "AD", "firing_rate_by_direction", src[src["panel"].isin(["A", "D"])])
    common.write_source_data(out_dir, FIGURE, "BE", "beta_by_symmetry", src[src["panel"].isin(["B", "E"])].drop(
        columns="direction_class"))
    crow = []
    for m in MODULES:
        for w in METHODS:
            o = V[("C", m, w)]; _, _, idx = orientation_hist(o)
            for i in range(N_PER_EXPERIMENT):
                crow.append(dict(panel="C", module=MODULE_WORD[m], method=w, participant=i,
                                 orientation_deg_modulo_60=o[i], bin=int(idx[i]),
                                 bin_centre_orientation_deg=float(idx[i] * 5.0)))
    common.write_source_data(out_dir, FIGURE, "C", "orientation", pd.DataFrame(crow))
    frow = []
    for m in MODULES:
        for w in METHODS:
            Pv = V[("Fp", m, w)]
            for x in range(Pv.shape[0]):
                for j in range(7):
                    frow.append(dict(panel="F", module=MODULE_WORD[m], method=w, experiment=x, symmetry_n_fold=j + 1,
                                     p_one_sided=Pv[x, j], significant=bool(Pv[x, j] < ALPHA)))
    common.write_source_data(out_dir, FIGURE, "F", "p_per_experiment", pd.DataFrame(frow))
    common.write_source_data(out_dir, FIGURE, "", "statistics", statistics_table(V))
    brow = []
    for p in "ADBE":
        for m in MODULES:
            labs = [f"{d} deg" for d in DIRS] if p in "AD" else [f"{j}-fold" for j in FOLDS]
            for lab, s in zip(labs, hs.box_stats(GROUPS[(p, m)])):
                brow.append(dict(panel=p, module=MODULE_WORD[m], group=lab, **s))
    common.write_source_data(out_dir, FIGURE, "", "box_numbers", pd.DataFrame(brow))
