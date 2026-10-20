"""Supplementary Figure 3: recorded and simulated grid cells of small and large grid scale.

Panels
    A  Rate maps of 12 recorded grid cells: rows small and large grid scale, columns two cells each of low, medium and
       high gridness. White circles: delineated fields; white lines: the lattice fitted to the field centres. Above
       each map: gridness and peak firing rate. Colour limits: 1st to 99th percentile of each map.
    B  Spatial deviation of the field centres from the fitted lattice (in grid scales), small and large grid scale.
    C  Mean field rate of each cell (mean rate inside its fields, Hz), across cells.
    D  Firing rate across the fields of each cell (normalised to the cell's mean field rate).
       B to D: probability density histograms (30 bins over each group's range) with gamma fits (location 0).
    E  Rate maps of 12 simulated grid cells: small and large module, two cells each of low, medium and high gridness.
       Above each map: gridness and peak firing rate. Colour limits: minimum to maximum of each map.

Inputs: ``rat/examples/supp3a_cells.csv`` and ``.npz`` (A), ``rat/figure_tables/supp3bcd_arrays.npz`` and
``supp3bcd_fits.csv`` (B to D), ``sim/fig6/maps.npz`` (E; taken from ``results_precomputed/`` when the simulation steps
have not been run). Figure size 174 x 196 mm; positions in mm from the top left corner of the figure.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy import stats

from .. import io as hio
from . import common
from . import style as hs

FIGURE = "SuppFig3"
W, H = 174.0, 196.0
RIGHT = 171.0                      # right edge shared by the last map column and the last histogram box
MAP_L0 = 16.5                      # left edge of the first map of each row (row label, y label and ticks to its left)
MAP_GAP = 7.6                      # between two maps (room for the 7 pt y tick labels of the next map)
MAP_W = (RIGHT - MAP_L0 - 5 * MAP_GAP) / 6.0
# Each row of panels has one line of letters (top of the capital); the topmost element of the row's panels (rows A and
# E: the word 'Firing rate' of the colour key; B to D: the histogram frames) stands style.LETTER_ABOVE under it. The
# lines of rows BCD and E lie 4.33 mm under the lowest text of the row above.
ROW_A, ROW_BCD, ROW_E = 1.5, 76.467, 123.033                 # letter lines of the rows
KEY_WORD_DROP = 0.533                                        # top of the key word 'Firing rate' under its row origin
ORG_A = round(hs.panels_top(ROW_A) - KEY_WORD_DROP, 3)       # origin of the key, maps and row labels of A (2.467 mm)
ORG_E = round(hs.panels_top(ROW_E) - KEY_WORD_DROP, 3)       # the same for E (124.0 mm)
MAP_TOP_OFF = 9.5                                            # from the row origin to the top of the first map row
MAP_PITCH = 32.7                                             # from one map row to the next
BOX_TOP_OFF, BOX_H, BOX_W, BOX_L0 = hs.LETTER_ABOVE, 32.076, 38.5, 17.0
BOX_PITCH = (RIGHT - BOX_W - BOX_L0) / 2.0
FS_MAPTICK = hs.FS_MIN                                       # 7 pt: tick labels of the map grids only
MODULES = ("small", "large")
# panel, name in the legend, array name, distribution name in supp3bcd_fits.csv, x label, x limits, x ticks, y limit,
# y ticks
SPEC = (("B", "Spatial deviation", "deviation", "deviation", hs.LABEL["spatial_deviation"], (0, 0.6),
         [0, 0.2, 0.4, 0.6], 9.0, [0, 4, 8]),
        ("C", "Firing rate across cells", "mean_rate", "rate_across_cells", hs.LABEL["mean_fr"], (0, 15),
         [0, 5, 10, 15], 0.27, [0, 0.1, 0.2]),
        ("D", "Firing rate across fields", "rate_normalised", "rate_within_cell", hs.LABEL["fr_norm"], (0, 3.5),
         [0, 1, 2, 3], 1.8, [0, 0.5, 1, 1.5]))


def map_label(gridness, peak):
    """Gridness and peak rate above a map, '0.7, 9 Hz'; one decimal for a peak below 1 Hz (else it would read 0 Hz)."""
    return f"{gridness:.1f}, {peak:.0f} Hz" if peak >= 1 else f"{gridness:.1f}, {peak:.1f} Hz"


def load_inputs(results_dir):
    """Read the result files of Supplementary Figure 3."""
    tab_a = hio.read_csv(common.find_input(results_dir, "rat/examples/supp3a_cells.csv"))
    z_a = hio.load_npz(common.find_input(results_dir, "rat/examples/supp3a_cells.npz"))
    z_b = hio.load_npz(common.find_input(results_dir, "rat/figure_tables/supp3bcd_arrays.npz"))
    fits = hio.read_csv(common.find_input(results_dir, "rat/figure_tables/supp3bcd_fits.csv"))
    z_e = hio.load_npz(common.find_input(results_dir, "sim/fig6/maps.npz"))
    return tab_a, z_a, z_b, fits, z_e


def build(results_dir, out_dir):
    """Draw Supplementary Figure 3 from ``results_dir`` into ``out_dir``."""
    inputs = load_inputs(results_dir)
    with mpl.rc_context():
        hs.use_style()
        _draw(out_dir, *inputs)


def _draw(out_dir, tab_a, z_a, z_b, fits, z_e):
    P, LAB = hs.PALETTE, hs.LABEL
    fig = hs.new_figure(W, H)
    letters = {}

    def head(letter, x, line, row):
        letters[letter] = hs.panel_letter(fig, letter, x, line, clear_of=[], not_above=line, row=row)

    def rate_key(row_top, right=RIGHT):
        """Colour key of the maps above them, flush right: 'Firing rate  Low [viridis bar] High'.

        The key's baseline lies one capital height under ``row_top``.
        """
        base = row_top + hs.CAP12
        r = fig.canvas.get_renderer()
        t_hi = hs.text_mm(fig, right, base, "High", fontsize=hs.FS_SMALL, ha="right")
        w_hi = t_hi.get_window_extent(r).width * 25.4 / fig.dpi
        bw, bh = 9.0, 2.4
        bx = right - w_hi - 1.0 - bw
        cax = fig.add_axes([bx / W, 1 - base / H, bw / W, bh / H])
        cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=hs.CMAP["rate"], extent=(0, 1, 0, 1),
                   interpolation="bilinear")
        cax.set_xticks([])
        cax.set_yticks([])
        for s in cax.spines.values():
            s.set_visible(False)
        t_lo = hs.text_mm(fig, bx - 1.0, base, "Low", fontsize=hs.FS_SMALL, ha="right")
        w_lo = t_lo.get_window_extent(r).width * 25.4 / fig.dpi
        return hs.text_mm(fig, bx - 1.0 - w_lo - 2.0, base, "Firing rate", fontsize=hs.FS_TITLE, ha="right")

    def map_axes(col, top, xmax, ymax):
        """The box of a map: the nominal square MAP_W shrunk to the environment's aspect, left and bottom edges kept."""
        left = MAP_L0 + col * (MAP_W + MAP_GAP)
        w, h = (MAP_W, MAP_W * ymax / xmax) if xmax >= ymax else (MAP_W * xmax / ymax, MAP_W)
        ax = hs.axes_mm(fig, left, top + (MAP_W - h), w, h)
        return ax, left + w / 2.0

    def map_ticks(ax, xt, yt):
        ax.set_xticks(xt)
        ax.set_yticks(yt)
        ax.xaxis.set_major_formatter(hs.NUM)
        ax.yaxis.set_major_formatter(hs.NUM)
        ax.tick_params(labelsize=FS_MAPTICK, pad=1.5)

    def row_label(text, colour, top):
        """Row label in the colour of the scale or module, 10 pt, upright along the left edge, centred on the row."""
        return hs.text_mm(fig, hs.LETTER_X, top + MAP_W / 2.0, text, fontsize=hs.FS_TITLE, color=colour, rotation=90,
                          ha="left", va="center")

    # ================================================================ A recorded maps
    head("A", hs.LETTER_X, ROW_A, 1)
    rate_key(ORG_A)
    maps_a = []
    for i, rr in enumerate(tab_a.itertuples(index=False)):
        p = f"p{i + 1:02d}"
        row, col = divmod(i, 6)
        rate_map = z_a[p + "__rate_map"]
        x_edges, y_edges = z_a[p + "__x_edges"], z_a[p + "__y_edges"]
        top = ORG_A + MAP_TOP_OFF + row * MAP_PITCH
        ax, xc = map_axes(col, top, x_edges[-1], y_edges[-1])
        maps_a.append(ax)
        bg = rate_map.T
        lo, hi = np.percentile(bg[~np.isnan(bg)], 1), np.percentile(bg[~np.isnan(bg)], 99)
        hs.rate_map(ax, rate_map, (np.min(x_edges), np.max(x_edges), np.min(y_edges), np.max(y_edges)),
                    vmin=lo, vmax=hi)
        common.draw_triangular_lattice_lines(dx=rr.scale_cm, theta_deg=rr.orientation_deg, env_size_x=x_edges[-1],
                                             env_size_y=y_edges[-1], origin=z_a[p + "__origin"], ax=ax,
                                             line_colour=P["white"])          # white lines, 0.5 pt
        hs.field_outlines(ax, z_a[p + "__field_locations"], lw=hs.LW_FRAME)   # white outlines, 0.8 pt
        ax.set_xlim(0, x_edges[-1])
        ax.set_ylim(0, y_edges[-1])
        ax.set_xlabel("")
        ax.set_ylabel("")
        map_ticks(ax, np.arange(0, x_edges[-1], 50), np.arange(0, y_edges[-1], 50))
        hs.text_mm(fig, xc, top - 1.2, map_label(rr.gridness, np.nanmax(rate_map)), fontsize=hs.FS_SMALL,
                   ha="center")
        if col == 0:
            ax.set_xlabel(LAB["x_cm"], labelpad=1.5)
            ax.set_ylabel(LAB["y_cm"], labelpad=1.5)
    label_a = row_label(hs.TICKS["grid_scale_rat"][0], P["module_small"], ORG_A + MAP_TOP_OFF)
    row_label(hs.TICKS["grid_scale_rat"][1], P["module_large"], ORG_A + MAP_TOP_OFF + MAP_PITCH)
    hs.add_clearance(letters["A"], maps_a[0], label_a)     # the first map's y labels and the row label

    # ================================================================ B to D histograms with gamma fits
    groups = (("small", "module_small", hs.TICKS["grid_scale_rat"][0]),
              ("large", "module_large", hs.TICKS["grid_scale_rat"][1]))
    fit_of = {(r.module, r.distribution): r for r in fits.itertuples(index=False)}
    hist, curves = {}, {}
    for k, (letter, name, key, dist, xlabel, xlim, xt, ytop, yt) in enumerate(SPEC):
        left = BOX_L0 + k * BOX_PITCH
        head(letter, left - (BOX_L0 - hs.LETTER_X), ROW_BCD, 2)
        ax = hs.axes_mm(fig, left, ROW_BCD + BOX_TOP_OFF, BOX_W, BOX_H)
        hs.add_clearance(letters[letter], ax)                # its y label and y tick labels
        ymax = 0
        for g, ckey, word in groups:
            x = z_b[f"{key}_{g}"]
            if len(x) == 0:                       # guard against an empty group
                warnings.warn(f"{FIGURE}{letter}: no values for the {g} grid scale; histogram and fit left out")
                continue
            n, e, _ = ax.hist(x, bins=30, density=True, histtype="stepfilled", facecolor=P[ckey + "_fill"], alpha=0.75,
                              edgecolor="none", linewidth=0, zorder=2)        # 30 bins over each group's own range
            f = fit_of[(g, dist)]
            if int(f.n_values) != len(x):
                raise ValueError(f"supp3bcd_fits.csv ({g}, {dist}) has {f.n_values} values, the arrays {len(x)}")
            xtmp = np.linspace(min(x), max(x), 100)
            ytmp = stats.gamma.pdf(xtmp, a=f.shape, loc=0, scale=f.scale)
            ax.plot(xtmp, ytmp, color=P[ckey], linestyle=hs.LINESTYLE[ckey], linewidth=hs.LW_DATA, zorder=3)
            hist[(letter, g)] = (n, e)
            curves[(letter, g)] = (f.shape, f.scale, xtmp, ytmp)
            ymax = max(ymax, n.max(), ytmp.max())
        ax.set_xlim(*xlim)
        ax.set_xticks(xt)
        ax.xaxis.set_major_formatter(hs.NUM)
        ax.set_ylim(0, ytop)
        ax.set_yticks(yt)
        ax.yaxis.set_major_formatter(hs.NUM)
        if ymax > ytop:
            warnings.warn(f"{FIGURE}{letter}: the highest bar or curve ({ymax:.3g}) is above the axes ({ytop})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability\ndensity", linespacing=1.1)
        if letter == "B":
            ax.legend(handles=[Line2D([], [], color=P[c], linestyle=hs.LINESTYLE[c], linewidth=hs.LW_DATA, label=w)
                               for _, c, w in groups],
                      loc="upper right", bbox_to_anchor=(1.0, 1.0), handlelength=2.0, fontsize=hs.FS_SMALL)

    # ================================================================ E simulated maps
    head("E", hs.LETTER_X, ROW_E, 3)
    rate_key(ORG_E)
    ext = tuple(float(v) for v in z_e["extent"])
    for r_i, mod in enumerate(MODULES):
        top = ORG_E + MAP_TOP_OFF + r_i * MAP_PITCH
        first_map = None
        maps = z_e[f"supp3e__{mod}__maps"]
        for j in range(maps.shape[0]):
            m = maps[j]
            peak = float(z_e[f"supp3e__{mod}__peak_rate_hz"][j])
            if abs(peak - float(m.max())) > 1e-9:
                raise ValueError(f"sim/fig6/maps.npz: the peak rate of {mod} map {j + 1} ({peak}) is not the maximum "
                                 f"of the map ({m.max()}); labels and maps must come from the same run")
            ax, xc = map_axes(j, top, ext[1], ext[3])
            hs.rate_map(ax, m, ext)                       # colour limits: the map's minimum and maximum
            first_map = ax if j == 0 else first_map
            map_ticks(ax, np.arange(0, ext[1] + 1, 50), np.arange(0, ext[3] + 1, 50))
            hs.text_mm(fig, xc, top - 1.2, map_label(float(z_e[f"supp3e__{mod}__gridness"][j]), peak),
                       fontsize=hs.FS_SMALL, ha="center")
            if j == 0:
                ax.set_xlabel(LAB["x_vm"], labelpad=1.5)
                ax.set_ylabel(LAB["y_vm"], labelpad=1.5)
        label = row_label(hs.TICKS["modules"][r_i], P["module_small" if mod == "small" else "module_large"], top)
        if r_i == 0:
            hs.add_clearance(letters["E"], first_map, label)

    common.save_outputs(fig, out_dir, FIGURE)

    # ================================================================ source data
    rows_m, rows_f, rows_c = [], [], []
    for i, rr in enumerate(tab_a.itertuples(index=False)):
        p = f"p{i + 1:02d}"
        m, xe, ye = z_a[p + "__rate_map"], z_a[p + "__x_edges"], z_a[p + "__y_edges"]
        a_idx, b_idx = np.meshgrid(np.arange(m.shape[0]), np.arange(m.shape[1]), indexing="ij")
        rows_m.append(pd.DataFrame({"map": i + 1, "session_id": rr.session_id, "cell_id": rr.cell_id,
                                    "x_bin_left_cm": xe[a_idx.ravel()], "x_bin_right_cm": xe[a_idx.ravel() + 1],
                                    "y_bin_bottom_cm": ye[b_idx.ravel()], "y_bin_top_cm": ye[b_idx.ravel() + 1],
                                    "rate_hz": m.ravel()}))
        for k, (cx, cy, r) in enumerate(z_a[p + "__field_locations"][:, :3]):
            rows_f.append(dict(map=i + 1, field=k, centre_x_cm=cx, centre_y_cm=cy, radius_cm=r))
        rows_c.append(dict(map=i + 1, session_id=rr.session_id, cell_id=rr.cell_id, gridness=rr.gridness,
                           peak_rate_hz=rr.rate_map_max_hz, grid_scale_cm=rr.scale_cm,
                           orientation_deg=rr.orientation_deg,
                           lattice_origin_x_cm=z_a[p + "__origin"][0], lattice_origin_y_cm=z_a[p + "__origin"][1],
                           environment_cm=rr.env_size_cm, grid_scale_over_environment=rr.scale_cm / rr.env_size_cm))
    common.write_source_data(out_dir, FIGURE, "A", "rate_maps", pd.concat(rows_m, ignore_index=True))
    common.write_source_data(out_dir, FIGURE, "A", "fields", pd.DataFrame(rows_f))
    common.write_source_data(out_dir, FIGURE, "A", "cells_and_lattices", pd.DataFrame(rows_c))
    rows_v = []
    for letter, name, key, *_ in SPEC:
        for g in MODULES:
            vals = z_b[f"{key}_{g}"]
            rows_v.append(pd.DataFrame({"panel": letter, "group": f"{g} scale", "value": vals}))
    common.write_source_data(out_dir, FIGURE, "BCD", "values", pd.concat(rows_v, ignore_index=True))
    rows_g = [dict(panel=letter, group=f"{g} scale", n=int(z_b[f"{key}_{g}"].size), gamma_shape=curves[(letter, g)][0],
                   gamma_scale=curves[(letter, g)][1])
              for letter, name, key, *_ in SPEC for g in MODULES if (letter, g) in curves]
    common.write_source_data(out_dir, FIGURE, "BCD", "gamma_fits", pd.DataFrame(rows_g))
    rows_h = []
    for letter, name, key, *_ in SPEC:
        for g in [g for g in MODULES if (letter, g) in hist]:
            n, e = hist[(letter, g)]
            rows_h.append(pd.DataFrame({"panel": letter, "group": f"{g} scale", "bin_left": e[:-1], "bin_right": e[1:],
                                        "density": n}))
    common.write_source_data(out_dir, FIGURE, "BCD", "histograms", pd.concat(rows_h, ignore_index=True))
    rows_e = []
    for mod in MODULES:
        maps = z_e[f"supp3e__{mod}__maps"]
        for j in range(maps.shape[0]):
            m = maps[j]
            bx = (ext[1] - ext[0]) / m.shape[0]
            by = (ext[3] - ext[2]) / m.shape[1]
            a_idx, b_idx = np.meshgrid(np.arange(m.shape[0]), np.arange(m.shape[1]), indexing="ij")
            rows_e.append(pd.DataFrame({"module": mod, "map": j + 1,
                                        "pool_index": int(z_e[f"supp3e__{mod}__cells"][j]),
                                        "gridness": float(z_e[f"supp3e__{mod}__gridness"][j]),
                                        "peak_rate_hz": float(z_e[f"supp3e__{mod}__peak_rate_hz"][j]),
                                        "x_bin": a_idx.ravel(), "y_bin": b_idx.ravel(),
                                        "x_left_vm": ext[0] + a_idx.ravel() * bx,
                                        "y_bottom_vm": ext[2] + b_idx.ravel() * by,
                                        "rate_hz": m.ravel()}))
    common.write_source_data(out_dir, FIGURE, "E", "rate_maps", pd.concat(rows_e, ignore_index=True))
