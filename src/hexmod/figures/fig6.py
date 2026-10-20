"""Figure 6: simulated grid cells with firing rate adaptation.

Panels
    A  Example path of a simulated participant in the object-location task (first 600 s of the trajectory of seed
       0) with the five object locations of the first block.
    B  Small module: six example rate maps (cells of the pool drawn with ``default_rng(1)``) and the summed rate maps
       of the 10,000 cells of participants 1 and 2 (cells drawn with seeds 0 and 1; percent of the mean). Two
       straight example runs from one start point are drawn on every map: aligned, along a grid axis (the simulated
       grid orientation is 0 deg), and misaligned, 30 deg away from it.
    C  The same for the large module.
    D  Proximity ``exp(-t_prev / tau)`` of the 10,000 cells of participant 1 (seed 0) when running aligned (within
       15 deg of a grid axis) and misaligned, both modules; paired t-tests, Holm correction over the two modules.

Inputs (``sim/fig6/`` in the results folder, else the shipped ``results_precomputed/sim/fig6/``):
``trajectory_seed0.csv``, ``trajectory_seed0_objects.json``, ``maps.npz``, ``fig6d_small.csv`` and
``fig6d_large.csv`` (or their ``.csv.gz`` copies). Positions in the code are in mm from the top left corner of the
figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.patches import Ellipse, FancyArrowPatch
from scipy import stats

from .. import io as hio
from . import common
from . import style as hs

FIGURE = "Fig6"
W, H = hs.WIDTH["full"], 93.58
# Each row of panels has one line of letters (top of the capital); the topmost element of the row's panels stands
# style.LETTER_ABOVE under it. Row 2's line lies 3.52 mm under the lowest text of row 1.
LINE1 = hs.LETTER_TOP              # 1.5 mm: letters A and B
LINE2 = 48.5                       # letters C and D
ROW1_TOP = hs.panels_top(LINE1)    # 3.0 mm: top of the box of A and of the maps of B
ROW2_TOP = hs.panels_top(LINE2)    # 50.0 mm: top of the maps of C
MAP, MGAP = 15.0, 1.2              # single-cell maps and the gap between them
SUM, SGAP = 21.0, 2.5              # summed maps
ARROW_W = 16.0                     # zone of the sum arrow
CB_GAP, CB_H = 2.0, 1.5            # colour bars: gap below the maps, height
ENV = (0.0, 100.0, 0.0, 100.0)     # arena (vm)
WIN_A = (0, 600)                   # time window of the path in A (s)
RUN_START, RUN_LEN = (20.0, 30.0), 50.0      # example runs on the maps of B and C (vm)
RUN_ANGLES = (("aligned", 0.0), ("misaligned", 30.0))
X0_B = 50.0                        # left edge of the column of B
DOT_RX, DOT_RY = 0.32, 0.32 * 107.0 / W      # half width and half height of the three dots (mm)
MODULES = ("small", "large")


# ------------------------------------------------------------------------------------------------------------ inputs
def _read_table(results_dir, relpath):
    """Read a result table, also accepting its gzip-compressed copy (``relpath + '.gz'``)."""
    return hio.read_csv(common.find_input(results_dir, relpath, alternatives=[relpath + ".gz"]))


def load_inputs(results_dir):
    """Read the result files of Figure 6.

    Returns
    -------
    dict
        ``traj`` (time, x, y, moveSpeed arrays), ``objects`` (5, 2), ``maps`` and ``cells`` (example maps and pool
        indices per module), ``summed`` (summed maps of participants 1 and 2 per module), ``extent`` and ``prox``
        (per module a table with ``cell_index``, ``aligned``, ``misaligned``).
    """
    t = hio.read_csv(common.find_input(results_dir, "sim/fig6/trajectory_seed0.csv"))
    traj = {k: t[k].to_numpy(float) for k in ("time", "x", "y", "moveSpeed")}
    objects = hio.read_json(common.find_input(results_dir, "sim/fig6/trajectory_seed0_objects.json"))["objLocs"]
    objects = np.asarray(objects, float)
    z = hio.load_npz(common.find_input(results_dir, "sim/fig6/maps.npz"))
    maps = {m: np.asarray(z[f"{m}__example_maps"], float) for m in MODULES}
    cells = {m: np.asarray(z[f"{m}__example_cells"]) for m in MODULES}
    summed = {f"seed{s}__{m}": np.asarray(z[f"seed{s}__{m}"], float) for m in MODULES for s in (0, 1)}
    prox = {m: _read_table(results_dir, f"sim/fig6/fig6d_{m}.csv") for m in MODULES}
    return dict(traj=traj, objects=objects, maps=maps, cells=cells, summed=summed,
                extent=np.asarray(z["extent"], float), prox=prox)


def proximity_statistics(prox):
    """Paired t-test of aligned against misaligned proximity per module, with Holm correction over the modules.

    Returns
    -------
    dict
        Per module: n, means and SDs (ddof 1), t, df, p, Cohen's d_z (mean difference over its SD, ddof 1), p_holm.
    """
    st = {}
    for m in MODULES:
        a = np.c_[prox[m]["aligned"].to_numpy(float), prox[m]["misaligned"].to_numpy(float)]
        t_, p_ = stats.ttest_rel(a[:, 0], a[:, 1])
        dz = np.mean(a[:, 0] - a[:, 1]) / np.std(a[:, 0] - a[:, 1], ddof=1)
        st[m] = dict(n=int(a.shape[0]), mean_aligned=float(a[:, 0].mean()), mean_misaligned=float(a[:, 1].mean()),
                     sd_aligned=float(a[:, 0].std(ddof=1)), sd_misaligned=float(a[:, 1].std(ddof=1)), t=float(t_),
                     df=int(a.shape[0] - 1), p=float(p_), cohen_dz=float(dz))
    for m, ph in zip(MODULES, hs.holm([st["small"]["p"], st["large"]["p"]])):
        st[m]["p_holm"] = float(ph)
    return st


# ----------------------------------------------------------------------------------------------------------- drawing
def build(results_dir, out_dir):
    """Draw Figure 6 from ``results_dir`` into ``out_dir``."""
    inputs = load_inputs(results_dir)
    with mpl.rc_context():
        hs.use_style()
        _draw(out_dir, inputs)


def _run_ends():
    """Start and end points (vm) of the two example runs drawn on the maps of B and C."""
    out = []
    for name, ang in RUN_ANGLES:
        x1 = RUN_START[0] + RUN_LEN * np.cos(np.deg2rad(ang)); y1 = RUN_START[1] + RUN_LEN * np.sin(np.deg2rad(ang))
        out.append((name, ang, x1, y1))
    return out


def _draw_runs(ax):
    """The two example runs, each a 0.9 pt line in its direction colour over a 1.6 pt white casing."""
    P = hs.PALETTE
    for (name, ang, x1, y1), col in zip(_run_ends(), (P["aligned"], P["misaligned"])):
        for lw, c, z in ((1.6, P["white"], 3), (0.9, col, 4)):
            ax.plot([RUN_START[0], x1], [RUN_START[1], y1], color=c, linewidth=lw, solid_capstyle="round", zorder=z)


def _module_block(fig, letter, x0, maps_top, line, row, module, seed_keys, maps, summed, extra=0.0):
    """Six single-cell maps (two rows of three) with their colour bar, the sum arrow, two summed maps with their colour
    bar and three dots. ``maps_top``: top of the six maps; ``line``: letter line of the row (mm). ``extra`` (mm)
    lengthens the sum arrow and moves the summed maps right by that much."""
    P = hs.PALETTE
    mx0 = x0 + 7.0; my0 = maps_top
    vmax = float(max(np.max(m_) for m_ in maps[module]))
    axs = []; ims = []
    for i in range(6):
        r_, c_ = divmod(i, 3)
        a_ = hs.axes_mm(fig, mx0 + c_ * (MAP + MGAP), my0 + r_ * (MAP + MGAP), MAP, MAP)
        im = hs.rate_map(a_, maps[module][i], ENV, vmin=0.0, vmax=vmax)
        _draw_runs(a_)
        a_.set_xticks([]); a_.set_yticks([]); a_.set_xlim(0, 100); a_.set_ylim(0, 100)
        axs.append(a_); ims.append(im)
    hs.panel_letter(fig, letter, x0 + 1.5, line, clear_of=[axs[0]], not_above=line, row=row)
    grid_w = 3 * MAP + 2 * MGAP; grid_bottom = my0 + 2 * MAP + MGAP
    ticks = {"small": [0, 5, 10], "large": [0, 2, 4]}[module]
    cb1 = hs.colour_bar(fig, ims[0], mx0, grid_bottom + CB_GAP, grid_w, CB_H, label=hs.LABEL["fr"],
                        orientation="horizontal", ticks=ticks)
    cb1.ax.xaxis.set_major_formatter(hs.NUM)
    # sum arrow with its words
    ax0 = mx0 + grid_w + 1.0; ax1 = ax0 + ARROW_W + extra - 1.0; sy0 = grid_bottom - SUM; yc = sy0 + SUM / 2.0
    fig.add_artist(FancyArrowPatch((hs.fx(fig, ax0), hs.fy(fig, yc)), (hs.fx(fig, ax1), hs.fy(fig, yc)),
                                   transform=fig.transFigure, arrowstyle="-|>,head_length=2.4,head_width=1.2",
                                   mutation_scale=1.0, color=P["ink"], linewidth=hs.LW_FRAME, shrinkA=0, shrinkB=0))
    xm = 0.5 * (ax0 + ax1)
    hs.text_mm(fig, xm, yc - 4.6, "Sum of", fontsize=hs.FS_SMALL, ha="center")
    hs.text_mm(fig, xm, yc - 1.6, "10,000 cells", fontsize=hs.FS_SMALL, ha="center")
    sx0 = ax1 + 1.5; sums = []
    for j, (k, who) in enumerate(zip(seed_keys, hs.TICKS["participants"])):
        a_ = hs.axes_mm(fig, sx0 + j * (SUM + SGAP), sy0, SUM, SUM)
        im = hs.rate_map(a_, summed[k], ENV, vmin=95.0, vmax=105.0)
        _draw_runs(a_)
        a_.set_xticks([]); a_.set_yticks([]); a_.set_xlim(0, 100); a_.set_ylim(0, 100)
        hs.note_above(a_, who, ha="left").set_fontsize(hs.FS_SUBHEAD)
        sums.append((a_, im, k))
    # three dots: further participants
    dx = sx0 + 2 * SUM + SGAP + 1.6
    for q in range(3):
        fig.add_artist(Ellipse((hs.fx(fig, dx + q * 1.3), hs.fy(fig, yc)), width=2 * DOT_RX / W, height=2 * DOT_RY / H,
                               transform=fig.transFigure, facecolor=P["ink"], edgecolor="none"))
    sw = 2 * SUM + SGAP
    cb2 = hs.colour_bar(fig, sums[0][1], sx0, grid_bottom + CB_GAP, sw, CB_H, label=hs.LABEL["summed_rate"],
                        orientation="horizontal", ticks=[95, 100, 105])
    cb2.ax.xaxis.set_major_formatter(hs.NUM)
    return dict(maps=axs, sums=sums, right=dx + 2 * 1.3 + 0.32, vmax=vmax)


def _draw(out_dir, inp):
    P = hs.PALETTE
    traj, objects, maps, summed, prox = inp["traj"], inp["objects"], inp["maps"], inp["summed"], inp["prox"]
    st = proximity_statistics(prox)
    fig = hs.new_figure("full", H)

    # ---- A: example path, 0 to 600 s, in one grey tone; object locations as open black circles
    xa, ya = 12.0, ROW1_TOP
    sa = 2 * MAP + MGAP                # as tall as the grid of six maps, so the boxes of row 1 share their bottom
    axA = hs.axes_mm(fig, xa, ya, sa, sa, full=True)       # the arena wall keeps the full box
    hs.panel_letter(fig, "A", 1.5, LINE1, clear_of=[axA], not_above=LINE1, row=1)
    mA = (traj["time"] >= WIN_A[0]) & (traj["time"] <= WIN_A[1])
    axA.plot(traj["x"][mA], traj["y"][mA], color=P["neutral"], linewidth=0.4, solid_joinstyle="round", zorder=2)
    axA.plot(objects[:, 0], objects[:, 1], linestyle="", marker="o", ms=5.5, mfc="none", mec=P["ink"], mew=1.0,
             zorder=4, clip_on=False)
    axA.set_xlim(0, 100); axA.set_ylim(0, 100); axA.set_aspect("equal")
    axA.set_xticks([0, 50, 100]); axA.set_yticks([0, 50, 100])
    axA.xaxis.set_major_formatter(hs.NUM); axA.yaxis.set_major_formatter(hs.NUM)
    axA.set_xlabel(hs.LABEL["x_vm"]); axA.set_ylabel(hs.LABEL["y_vm"])

    # ---- B and C: the dots after the B summed maps end at the right edge of the content (W - 1.5 mm)
    block_right_0 = X0_B + 7.0 + (3 * MAP + 2 * MGAP) + ARROW_W + 1.5 + 2 * SUM + SGAP + 1.6 + 2 * 1.3 + 0.32
    extra_b = (W - 1.5) - block_right_0
    _module_block(fig, "B", X0_B, ROW1_TOP, LINE1, 1, "small", ("seed0__small", "seed1__small"), maps, summed,
                  extra=extra_b)
    blk_c = _module_block(fig, "C", 0.0, ROW2_TOP, LINE2, 2, "large", ("seed0__large", "seed1__large"), maps, summed)

    # ---- D: proximity by movement direction, both modules on one axis. The key starts under letter D: its swatches
    # start style.LETTER_CLEAR plus a capital below the row top, and the box (as tall as the grid of six maps) starts
    # 4.62 mm below the key's line.
    xD0 = blk_c["right"] + 2.0
    dl = xD0 + 11.5; dr = W - 1.5
    key_row_top = ROW2_TOP + hs.LETTER_CLEAR + hs.KEY["swatch_h"]
    key_to_box = 10.0 - 1.5 - hs.TITLE_PITCH
    dtop = key_row_top + key_to_box; dh = 2 * MAP + MGAP
    axD = hs.axes_mm(fig, dl, dtop, dr - dl, dh)
    pos_d = [0, 1, 2.5, 3.5]
    groups = [prox[m][c].to_numpy(float) for m in MODULES for c in ("aligned", "misaligned")]
    hs.boxplot(axD, groups, positions=pos_d, colours=[P["aligned_fill"], P["misaligned_fill"]] * 2)
    axD.set_xlim(-0.6, 4.1); axD.set_ylim(0, 0.34)
    axD.set_yticks([0, 0.1, 0.2, 0.3]); axD.yaxis.set_major_formatter(hs.NUM)
    axD.set_xticks([0.5, 3.0]); axD.set_xticklabels(hs.TICKS["modules"]); axD.tick_params(axis="x", length=0, pad=3.0)
    axD.set_ylabel(hs.LABEL["proximity"])
    bxs = hs.box_stats(groups)
    gap = hs.pt_to_data(axD, 4.0)
    for (a_, b_), m in (((0, 1), "small"), ((2.5, 3.5), "large")):
        i0 = pos_d.index(a_); y = max(bxs[i0]["whisker_high"], bxs[i0 + 1]["whisker_high"]) + gap
        hs.sig_bracket(axD, a_, b_, y, st[m]["p_holm"])
    # colour key flush right to the D box, above it
    key = hs.colour_key(fig, [(hs.TICKS["aligned"][0], P["aligned"]), (hs.TICKS["aligned"][1], P["misaligned"])],
                        right=W - 1.5, row_top=key_row_top)
    hs.panel_letter(fig, "D", xD0 + 1.5, LINE2, clear_of=list(key) + [axD], not_above=LINE2, row=2)

    common.save_outputs(fig, out_dir, FIGURE)
    _write_source_data(out_dir, inp, mA, st, bxs)


# ------------------------------------------------------------------------------------------------------- source data
def _write_source_data(out_dir, inp, mA, st, bxs):
    keys = ("traj", "objects", "maps", "cells", "summed", "prox", "extent")
    traj, objects, maps, cells, summed, prox, ext = (inp[k] for k in keys)
    common.write_source_data(out_dir, FIGURE, "A", "path", pd.DataFrame(
        {"time_s": traj["time"][mA], "x_vm": traj["x"][mA], "y_vm": traj["y"][mA],
         "speed_vm_per_s": traj["moveSpeed"][mA]}))
    common.write_source_data(out_dir, FIGURE, "A", "objects", pd.DataFrame(
        {"object": np.arange(1, len(objects) + 1), "x_vm": objects[:, 0], "y_vm": objects[:, 1]}))
    frames = []
    for m in MODULES:
        series = [(f"single cell {i + 1} (pool row {int(cells[m][i])})", maps[m][i], "Hz") for i in range(6)]
        series += [(f"summed, {who} (cells drawn with seed {s})", summed[f"seed{s}__{m}"], "% of mean")
                   for s, who in zip((0, 1), hs.TICKS["participants"])]
        for name, a, unit in series:
            nx, ny = a.shape
            ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
            frames.append(pd.DataFrame({"module": m, "map": name, "x_bin": ix.ravel(), "y_bin": iy.ravel(),
                                        "x_centre_vm": ext[0] + (ix.ravel() + 0.5) * (ext[1] - ext[0]) / nx,
                                        "y_centre_vm": ext[2] + (iy.ravel() + 0.5) * (ext[3] - ext[2]) / ny,
                                        "value": a.ravel(), "unit": unit}))
    common.write_source_data(out_dir, FIGURE, "BC", "maps", pd.concat(frames, ignore_index=True))
    rows = []
    for m in MODULES:
        vm = float(max(np.max(x) for x in maps[m]))
        for i in range(6):
            rows.append(dict(module=m, map=f"single cell {i + 1}", value_min=float(maps[m][i].min()),
                             value_max=float(maps[m][i].max()), colour_min=0.0, colour_max=vm, unit="Hz"))
        for s, who in zip((0, 1), hs.TICKS["participants"]):
            a = summed[f"seed{s}__{m}"]
            rows.append(dict(module=m, map=f"summed, {who}", value_min=float(a.min()), value_max=float(a.max()),
                             colour_min=95.0, colour_max=105.0, unit="% of mean"))
    common.write_source_data(out_dir, FIGURE, "BC", "colour_scales", pd.DataFrame(rows))
    common.write_source_data(out_dir, FIGURE, "BC", "example_runs", pd.DataFrame(
        [dict(run=name, angle_to_grid_axis_deg=ang, x_start_vm=RUN_START[0], y_start_vm=RUN_START[1], x_end_vm=x1,
              y_end_vm=y1) for name, ang, x1, y1 in _run_ends()]))
    common.write_source_data(out_dir, FIGURE, "D", "proximity", pd.concat(
        [pd.DataFrame({"module": m, "pool_row": prox[m]["cell_index"].to_numpy(),
                       "proximity_aligned": prox[m]["aligned"].to_numpy(float),
                       "proximity_misaligned": prox[m]["misaligned"].to_numpy(float)}) for m in MODULES],
        ignore_index=True))
    common.write_source_data(out_dir, FIGURE, "", "statistics", pd.DataFrame(
        [dict(panel="D", test="paired t test, aligned against misaligned", module=m, n=st[m]["n"],
              mean_aligned=st[m]["mean_aligned"], mean_misaligned=st[m]["mean_misaligned"],
              sd_aligned=st[m]["sd_aligned"], sd_misaligned=st[m]["sd_misaligned"], t=st[m]["t"],
              df=st[m]["df"], p=st[m]["p"], p_holm_within_panel=st[m]["p_holm"], cohen_dz=st[m]["cohen_dz"],
              mark=hs.sig_stars(st[m]["p_holm"]))
         for m in MODULES]))
    common.write_source_data(out_dir, FIGURE, "", "box_numbers", pd.DataFrame(
        [dict(module=m, direction=d, **b)
         for (m, d), b in zip([(m, d) for m in MODULES for d in ("aligned", "misaligned")], bxs)]))
