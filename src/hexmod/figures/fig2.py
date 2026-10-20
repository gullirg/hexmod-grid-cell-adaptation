"""Figure 2: movement-direction tuning of conjunctive grid cells relative to the grid axes.

Panels
    A  Tuning curves of five conjunctive cells recorded in one session, over the sectors within 15 deg of a grid axis
       (aligned, teal) and between the axes (misaligned, orange); dashed line: preferred direction.
    B  Deviation of the preferred direction of every conjunctive cell from the nearest grid axis: counts in 5 deg bins
       over one 60 deg period, drawn around the circle (angles multiplied by 6), with the numbers of aligned and
       misaligned cells below.
    C  Mean tuning curve across animals. Each cell's curve is rotated so that the bin nearest its preferred direction is
       at 0 deg and divided by its own mean; curves are averaged over the cells of each animal and then over the animals
       with at least 5 conjunctive cells (line); band, mean plus or minus s.e.m. across animals.
    D  Preferred direction modulo 60 deg against grid orientation, as in Fig 1b of Doeller et al. (2010), for the cells
       with a directional tuning curve (earliest recording of each cell); the six animals with most cells have colours
       of their own, the others are grey.
    E  Clustering of preferred directions at 60 deg intervals in each animal of the Vollan et al. (2025) data
       (resultant length of six times the preferred direction) against its number of conjunctive cells, with the 95th
       percentile of that length for uniform directions (grey area: Rayleigh test, p > 0.05).

Inputs (``rat/`` in the results folder): ``examples/fig2a_cells.csv`` and ``.npz``, ``conjunctive/fig2b_bins.csv``,
``conjunctive/fig2c_mean.csv``, ``conjunctive/fig2c_per_animal_curves.csv``, ``conjunctive/fig2d_cells.csv``,
``conjunctive/fig2e_per_animal.csv`` and ``conjunctive/fig2e_band.csv``.

Panels A to C are drawn in the plotting style of the analysis (polar frames 0.77 inch across, curves 1.5 pt, wedge
outlines 1.0 pt, frames 0.8 pt); D and E in the style of the Cartesian panels (``CARTESIAN_RC``). Type: letters 12 pt
bold, axis labels and key 10 pt, tick labels and counts 8 pt. Panels carry no titles (the figure legend names them).
Positions are in mm from the top left corner of the figure (``LAYOUT``): the four panels of row 2 are spaced evenly and
the last example of A, the box of E and the colour key end on one vertical line. The letters of a row share one line,
their capitals at least 1.5 mm above the topmost element of the row (row 1: the colour key; row 2: the end of the y
label of E, which the letter E clears by 1.25 mm, with the angle labels of B and C level with it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from .. import io as hio
from . import common, style

FIGURE = "Fig2"
MM = 1 / 25.4                                     # inches per mm
COLOUR_ALIGN = "#009FB9"                          # aligned: within 15 deg of a grid axis
COLOUR_MISALIGN = "#F18D00"                       # misaligned
DIAM = 0.77 * 25.4                                # polar frame, mm
AXW = 0.775 * 25.4                                # width of the polar axes (the frame is the inscribed circle), mm
CAP12 = 12 * 0.716 / 72 * 25.4                    # height of a 12 pt capital, mm
HALF_FRAME = 0.8 / 72 * 25.4 / 2                  # half of a 0.8 pt frame line, mm
ANGLE_LABEL_PAD = -2.0                            # pad of the angle labels of the polar plots (pt)

#: Positions (mm from the top left corner): letters (xl, yl: left edge and top of the capital), centres of the examples
#: of A (cxA, cyA) and of the polar plots of B and C, boxes of D and E (left, top, width, height), right edge of row 1
#: (R), top of the colour key (yk) and anchors of the two counts of B (cnt).
LAYOUT = {
    "W": 165.0, "H": 90.88246666666664,
    "xl": {"A": 1.5, "B": 1.5, "C": 36.067459920634896, "D": 72.20654484126982, "E": 117.31354166666662},
    "yl": {"A": 1.516933333333327, "B": 43.06314027777776, "C": 43.06314027777776, "D": 43.06314027777776,
           "E": 43.06314027777776},
    "cxA": [25.5, 57.519972222222215, 89.53994444444443, 121.55991666666665, 153.57988888888886],
    "cyA": 23.99659999999999,
    "cxB": 17.195965277777773, "cxC": 53.33505019841267, "cyB": 61.4068111111111, "cyC": 61.4068111111111,
    "Dl": 86.02908373015867, "El": 127.35888888888886, "Dt": 52.29347777777777, "Et": 52.29347777777777,
    "R": 163.35888888888886, "yk": 2.6175999999999937,
    "DE_H": 25.0, "D_W": 25.0, "E_W": 36.0,
    "cnt": {"x_al": 18.295965277777775, "x_mis": 16.095965277777772, "y": 76.15442222222221},
}
KEY_WORDS = ("Aligned", "Misaligned")

# Cartesian panels D and E
INK, DOT, CHANCE, EDGE = "#000000", "#3C3C3C", "#DCDCDC", "#8C8C8C"
DEG = FuncFormatter(lambda v, _: f"{v:g}°".replace("-", "−"))
NUM = FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−"))
RAT = dict(marker="o", linestyle="", ms=3.0, mfc=to_rgba(DOT, 0.65), mec=DOT, mew=0.5, zorder=4, clip_on=False)
PAD_D = 1.5                                       # margin around 0 to 60 deg in D
DOT_AREA = 3.4                                    # dot area in D (pt^2)
DOT_ALPHA = 0.85
XLIM_E, YLIM_E = (5, 300), (-0.035, 1.0)
CARTESIAN_RC = {
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6, "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0, "xtick.direction": "out", "ytick.direction": "out", "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0, "axes.labelpad": 2.5, "axes.edgecolor": "#000000", "axes.labelcolor": "#000000",
    "xtick.color": "#000000", "ytick.color": "#000000", "text.color": "#000000", "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "legend.frameon": False, "axes.unicode_minus": True}

# Colours of the six animals with most cells in D; the other animals are grey.
ANIMAL_COLOURS = {"V18": "#C61E58", "V15": "#0FA626", "V10": "#581378", "V12": "#0F4BF7", "V11": "#B775FE",
                  "V14": "#354E03"}
OTHER_ANIMALS = "#969696"
# The coloured dots of D are drawn in one random order, so that no animal covers another systematically; this seed
# gives the order of the published panel.
PUBLISHED_DRAW_ORDER_SEED = 20261013


# ------------------------------------------------------------------------------------------------------------- style
def polar_panel_rc(font):
    """Return the rcParams of panels A to C (all sizes at 1:1); ``font`` is the typeface."""
    return {"image.interpolation": "none", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
            "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "figure.dpi": 100, "savefig.dpi": 300,
            "figure.figsize": (4, 3), "font.family": "sans-serif", "font.sans-serif": [font], "pdf.fonttype": 42,
            "ps.fonttype": 42, "svg.fonttype": "none", "lines.linewidth": 1.5, "patch.linewidth": 1.0,
            "axes.linewidth": 0.8, "grid.linewidth": 0.8, "xtick.major.pad": 3.5, "ytick.major.pad": 3.5,
            "xtick.major.size": 3.5, "xtick.major.width": 0.8, "ytick.major.size": 3.5, "ytick.major.width": 0.8,
            "mathtext.fontset": "custom", "mathtext.rm": font, "mathtext.it": font + ":italic",
            "mathtext.bf": font + ":bold", "mathtext.cal": font}


class _Canvas:
    """A figure addressed in mm from its top left corner."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig = Figure(figsize=(w * MM, h * MM), dpi=600)
        FigureCanvasAgg(self.fig)
        self.even = []

    def X(self, x):
        return x / self.w

    def Y(self, y):
        return 1 - y / self.h

    def text(self, x, y, s, **kw):
        return self.fig.text(self.X(x), self.Y(y), s, **kw)

    def letter(self, name, x, top):
        return self.text(x, top + CAP12, name, fontsize=12, fontweight="bold", ha="left", va="baseline")

    def polar(self, cx, cy):
        ax = self.fig.add_axes([self.X(cx - AXW / 2), self.Y(cy + DIAM / 2), AXW / self.w, DIAM / self.h],
                               projection="polar")
        return ax

    def cart(self, left, top, w, h):
        ax = self.fig.add_axes([self.X(left), self.Y(top + h), w / self.w, h / self.h])
        return ax


def _box(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(INK)


def _despine(ax):
    """Left and bottom spines only (both axes of D wrap at 60 deg; its frame follows the house style)."""
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    ax.tick_params(which="both", top=False, right=False)


# ------------------------------------------------------------------------------------------------------- panels A-C
def draw_example(ax, angles_padded, curve, orientation_deg, preferred_deg, first):
    """Tuning curve of one cell over the aligned and misaligned sectors of its grid; angle labels on the first only."""
    for i in range(6):
        ax.bar(np.deg2rad(orientation_deg + 60 * i), curve.max(), width=np.deg2rad(30), color=COLOUR_ALIGN, alpha=0.6)
        ax.bar(np.deg2rad(30 + orientation_deg + 60 * i), curve.max(), width=np.deg2rad(30), color=COLOUR_MISALIGN,
               alpha=0.6)
    ax.plot(np.deg2rad(angles_padded), curve, color="blue", label="original")
    ax.plot(np.deg2rad([preferred_deg, preferred_deg]), [0, curve.max()], linestyle="--", color="blue",
            label="preferred direction")
    ax.set_yticks([])
    ax.set_ylim(0, curve.max())
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5)[:-1])
    if not first:
        ax.set_xticks([])
    ax.grid(False, axis="x")
    ax.tick_params(axis="x", pad=ANGLE_LABEL_PAD)


def draw_deviation_histogram(ax, bin_edges, bin_count):
    """Counts of the deviation from the nearest grid axis in 5 deg bins of one 60 deg period (angles times 6).

    Returns the numbers of cells, aligned cells and misaligned cells.
    """
    bin_resolution = 30 / 6
    bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2
    mask_align = (bin_center < 15) | (bin_center > 45)
    ax.bar(6 * np.deg2rad(bin_center[mask_align]), bin_count[mask_align], width=6 * np.deg2rad(bin_resolution),
           align="center", facecolor=COLOUR_ALIGN, edgecolor="white", alpha=0.6)
    ax.bar(6 * np.deg2rad(bin_center[~mask_align]), bin_count[~mask_align], width=6 * np.deg2rad(bin_resolution),
           align="center", facecolor=COLOUR_MISALIGN, edgecolor="white", alpha=0.6)
    angle_tick = np.array([0, 90, 180, 270])
    ax.set_xticks(angle_tick * np.pi / 180)
    ax.set_xticklabels(["0°", "15°", "30°", "−15°"])
    ax.set_yticks([])
    ax.tick_params(axis="x", pad=ANGLE_LABEL_PAD)
    ax.set_ylim(0, bin_count.max())
    return int(bin_count.sum()), int(bin_count[mask_align].sum()), int(bin_count[~mask_align].sum())


def count_texts(fig, cx_in, cy_in, r_in, n, n_al, n_mis, fontsize=8.0):
    """The two coloured counts of B (aligned on the right, misaligned on the left); positions in inches."""
    W, H = fig.get_size_inches()
    rho = 1.12 * r_in
    dx = rho * np.cos(np.deg2rad(45.0))
    dy = rho * np.sin(np.deg2rad(45.0))
    t_al = "%s\n%d (%d%%)" % (KEY_WORDS[0], n_al, int(round(100.0 * n_al / n)))
    t_mis = "%s\n%d (%d%%)" % (KEY_WORDS[1], n_mis, int(round(100.0 * n_mis / n)))
    a = fig.text((cx_in + dx) / W, (cy_in - dy) / H, t_al, color=COLOUR_ALIGN, fontsize=fontsize, ha="left", va="top",
                 multialignment="left", linespacing=1.15)
    b = fig.text((cx_in - dx) / W, (cy_in - dy) / H, t_mis, color=COLOUR_MISALIGN, fontsize=fontsize, ha="right",
                 va="top", multialignment="right", linespacing=1.15)
    return a, b


def draw_mean_curve(ax, angles_padded, mean, sem, line_colour="#1F77B4", band_alpha=0.3, line_width=1.5):
    """Mean curve across animals (line) and mean plus or minus s.e.m. (one closed band that follows the curve).

    The radial limit is 0 to the top of the band plus 5% of the band's range.
    """
    up, lo = mean + sem, mean - sem
    rmax = up.max() + 0.05 * (up.max() - lo.min())
    th = np.deg2rad(np.r_[angles_padded[:-1], angles_padded[0] + 360.0])   # 72 directions and the first one again
    n = len(th) - 1
    outer = np.c_[th[:n], up[:n]]
    inner = np.c_[th[:n][::-1], lo[:n][::-1]]
    verts = np.r_[outer, outer[:1], inner, inner[:1]]
    codes = np.r_[[MplPath.MOVETO], [MplPath.LINETO] * (n - 1), [MplPath.CLOSEPOLY], [MplPath.MOVETO],
                  [MplPath.LINETO] * (n - 1), [MplPath.CLOSEPOLY]].astype(np.uint8)
    band = PathPatch(MplPath(verts, codes), facecolor=line_colour, alpha=band_alpha, edgecolor="none", linewidth=0,
                     zorder=1)
    ax.add_patch(band)
    ax.plot(th, mean, color=line_colour, label="mean across animals", zorder=2, linewidth=line_width)
    angle_tick = np.arange(4) * 360 / 4
    ax.set_xticks(angle_tick * np.pi / 180)
    ax.set_xticklabels([f"{x:.0f}°" for x in angle_tick])
    ax.set_yticks([])
    ax.set_ylim(0, rmax)
    ax.tick_params(axis="x", pad=ANGLE_LABEL_PAD)
    return up, lo


def colour_key(cv, right, top):
    """Colour key of A, its top at ``top``, ending at x = right: swatch, 'Aligned', swatch, 'Misaligned' (10 pt)."""
    sw_w, sw_h, gap_txt, gap_item = 4.8, 2.6, 1.6, 4.5
    base = top + CAP12
    r = cv.fig.canvas.get_renderer()
    x = right
    for lab, col in ((KEY_WORDS[1], COLOUR_MISALIGN), (KEY_WORDS[0], COLOUR_ALIGN)):
        t = cv.text(x, base, lab, fontsize=10, ha="right", va="baseline")
        wtxt = t.get_window_extent(r).width * 25.4 / cv.fig.dpi
        x0 = x - wtxt - gap_txt - sw_w
        p = Rectangle((cv.X(x0), cv.Y(base)), sw_w / cv.w, sw_h / cv.h, transform=cv.fig.transFigure, facecolor=col,
                      alpha=0.6, edgecolor="none", linewidth=0)
        cv.fig.add_artist(p)
        x = x0 - gap_item


def white_after_last_stroke(fig, word, fontsize=10):
    """White space between the last stroke of a word and the right end of its text box (mm)."""
    t = fig.text(0, 0, word, fontsize=fontsize, ha="right", va="baseline")
    box_w = t.get_window_extent(fig.canvas.get_renderer()).width * 72.0 / fig.dpi
    ink_w = TextPath((0, 0), word, size=t.get_fontsize(), prop=t.get_fontproperties()).get_extents().x1
    t.remove()
    return (box_w - ink_w) / 72.0 * 25.4


def even_side_labels(cv):
    """Left angle label of the polar plots in ``cv.even`` ('180°' or '30°'), right-aligned at the distance from the
    frame that the right label ('0°') has (centred labels would leave the wider left label almost on the frame)."""
    cv.fig.canvas.draw()
    r = cv.fig.canvas.get_renderer()
    todo = []
    for ax in cv.even:
        labs = ax.get_xticklabels()
        locs = np.asarray(ax.get_xticks())
        strings = [t.get_text() for t in labs]
        i0 = int(np.argmin(np.abs(locs)))
        i180 = int(np.argmin(np.abs(locs - np.pi)))
        fb = ax.spines["polar"].get_window_extent(r)
        b0 = labs[i0].get_window_extent(r)
        gap_px = b0.x0 - fb.x1
        todo.append((ax, i180, strings, gap_px, labs[i180].get_fontsize()))
    for ax, i180, strings, gap_px, size in todo:
        new = list(strings)
        new[i180] = ""
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(new)
        ax.annotate(strings[i180], xy=(np.pi, ax.get_ylim()[1]), xycoords="data",
                    xytext=(-gap_px * 72.0 / cv.fig.dpi, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=size, annotation_clip=False)
    cv.fig.canvas.draw()


# ---------------------------------------------------------------------------------------------------------- panel D
def draw_direction_vs_orientation(ax, cells):
    """Preferred direction (mod 60 deg) against grid orientation, dots coloured by animal (``ANIMAL_COLOURS``)."""
    ax.plot([0, 60], [0, 60], color=COLOUR_ALIGN, linewidth=1.2, zorder=2, solid_capstyle="butt")
    own = cells.animal_id.isin(list(ANIMAL_COLOURS)).values
    grey = cells[~own]
    ax.scatter(grey.orientation_deg, grey.dir_mod60, s=DOT_AREA, color=OTHER_ANIMALS, alpha=DOT_ALPHA, linewidths=0,
               zorder=3)
    col = cells[own].sample(frac=1, random_state=PUBLISHED_DRAW_ORDER_SEED)
    ax.scatter(col.orientation_deg, col.dir_mod60, s=DOT_AREA, c=[ANIMAL_COLOURS[a] for a in col.animal_id],
               alpha=DOT_ALPHA, linewidths=0, zorder=4)
    ax.set_xlim(-PAD_D, 60 + PAD_D)
    ax.set_ylim(-PAD_D, 60 + PAD_D)
    ax.set_xticks([0, 30, 60])
    ax.set_yticks([0, 30, 60])
    ax.xaxis.set_major_formatter(DEG)
    ax.yaxis.set_major_formatter(DEG)
    ax.set_aspect("equal")
    _box(ax)
    ax.set_xlabel("Grid orientation")
    ax.set_ylabel("Preferred direction\n(mod 60°)", linespacing=1.1)


# ---------------------------------------------------------------------------------------------------------- panel E
def draw_clustering(ax, animals, band):
    """Resultant length of six times the preferred direction per animal against its number of conjunctive cells."""
    vol = animals.sort_values("n", kind="mergesort").reset_index(drop=True)
    bn, bv = band.n.values, band.R_chance95.values
    ax.fill_between(bn, 0, bv, color=CHANCE, linewidth=0, zorder=1)
    ax.plot(bn, bv, color=EDGE, linewidth=0.6, zorder=2, solid_joinstyle="round")
    ax.plot(vol.n, vol.R6, **RAT)
    ax.set_xscale("log")
    ax.set_xlim(*XLIM_E)
    ax.set_xticks([10, 100])
    ax.xaxis.set_major_formatter(NUM)
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(*YLIM_E)
    ax.set_yticks([0, 0.5, 1])
    ax.yaxis.set_major_formatter(NUM)
    _box(ax)
    ax.set_xlabel("Conjunctive cells\nper animal", linespacing=1.1)
    ax.set_ylabel("Rayleigh vector length")
    key = Patch(facecolor=CHANCE, edgecolor=EDGE, linewidth=0.6, label="Rayleigh test, $p$ > 0.05")
    ax.legend(handles=[key], loc="upper right", bbox_to_anchor=(1.0, 1.0), handlelength=1.0, handleheight=0.8,
              handletextpad=0.5, borderaxespad=0.4, borderpad=0)
    return vol


# ------------------------------------------------------------------------------------------------------------ inputs
def load_inputs(results_dir):
    """Read the result files of Figure 2."""
    rd = lambda rel: hio.read_csv(common.find_input(results_dir, rel))
    ex_cells = rd("rat/examples/fig2a_cells.csv")
    ex_maps = hio.load_npz(common.find_input(results_dir, "rat/examples/fig2a_cells.npz"))
    bins = rd("rat/conjunctive/fig2b_bins.csv")
    mean = rd("rat/conjunctive/fig2c_mean.csv")
    per_animal = rd("rat/conjunctive/fig2c_per_animal_curves.csv")
    d_cells = rd("rat/conjunctive/fig2d_cells.csv")
    e_animals = rd("rat/conjunctive/fig2e_per_animal.csv")
    band = rd("rat/conjunctive/fig2e_band.csv")
    return ex_cells, ex_maps, bins, mean, per_animal, d_cells, e_animals, band


def histogram_mod60(bins):
    """Bin edges (0 to 60 deg) and counts of the deviation histogram in the order of the angle modulo 60 deg."""
    b = bins.copy()
    b["from_mod60"] = np.mod(b.deviation_from_deg.values, 60)
    b = b.sort_values("from_mod60", kind="mergesort").reset_index(drop=True)
    edges = np.arange(len(b) + 1) * 5.0
    if not np.array_equal(b.from_mod60.values.astype(float), edges[:-1]):
        raise ValueError("fig2b_bins.csv does not hold 5 deg bins covering one 60 deg period")
    return edges, b["count"].values.astype(np.int64), b


# ------------------------------------------------------------------------------------------------------------- build
def build(results_dir, out_dir):
    """Draw Figure 2 from ``results_dir`` into ``out_dir``."""
    ex_cells, ex_maps, bins, mean_tab, per_animal, d_cells, e_animals, band = load_inputs(results_dir)
    G = LAYOUT
    font = style.house_font()
    with mpl.rc_context():
        mpl.rcParams.update(mpl.rcParamsDefault)
        mpl.rcParams.update(polar_panel_rc(font))

        # ---- data
        order = [int(np.flatnonzero(np.asarray(ex_maps["rows"]) == r)[0]) for r in ex_cells["row"]]
        maps = np.asarray(ex_maps["dir_maps"], float)[order]
        n_bins = maps.shape[1]
        angles_padded = np.r_[np.arange(n_bins) * (360 / n_bins) + 360 / n_bins / 2, 360 / n_bins / 2]
        curves = [np.r_[m, m[0]] for m in maps]
        edges, counts, bins_sorted = histogram_mod60(bins)
        form = mean_tab[mean_tab.form == "normalised_to_cell_mean"].sort_values("bin")
        ang_c = form.direction_deg.values.astype(float)
        mean_c, sem_c = form.mean_across_animals.values, form.sem_across_animals.values
        d = d_cells.copy()
        d["dir_mod60"] = d.direction_deg % 60

        # ---- figure
        cv = _Canvas(G["W"], G["H"])
        right_key = G["R"] + white_after_last_stroke(cv.fig, KEY_WORDS[1]) + HALF_FRAME
        # row 1: panel A
        cv.letter("A", G["xl"]["A"], G["yl"]["A"])
        for j in range(len(curves)):
            row = ex_cells.iloc[j]
            ax = cv.polar(G["cxA"][j], G["cyA"])
            draw_example(ax, angles_padded, curves[j], row["orientation_deg"], row["Rayleigh_angle"], first=(j == 0))
            if j == 0:
                cv.even.append(ax)
        colour_key(cv, right_key, G["yk"])
        # row 2: panels B and C
        for P in "BC":
            cv.letter(P, G["xl"][P], G["yl"][P])
        axB = cv.polar(G["cxB"], G["cyB"])
        n, n_al, n_mis = draw_deviation_histogram(axB, edges, counts)
        cv.even.append(axB)
        a, b = count_texts(cv.fig, G["cxB"] * MM, (cv.h - G["cyB"]) * MM, DIAM / 2 * MM, n, n_al, n_mis)
        a.set_position((cv.X(G["cnt"]["x_al"]), cv.Y(G["cnt"]["y"])))
        b.set_position((cv.X(G["cnt"]["x_mis"]), cv.Y(G["cnt"]["y"])))
        axC = cv.polar(G["cxC"], G["cyC"])
        up_c, lo_c = draw_mean_curve(axC, ang_c, mean_c, sem_c)
        cv.even.append(axC)
        # row 2: panels D and E
        with mpl.rc_context(CARTESIAN_RC):
            for P in "DE":
                cv.letter(P, G["xl"][P], G["yl"][P])
            axD = cv.cart(G["Dl"], G["Dt"], G["D_W"], G["DE_H"])
            draw_direction_vs_orientation(axD, d)
            axE = cv.cart(G["El"], G["Et"], G["E_W"], G["DE_H"])
            vol = draw_clustering(axE, e_animals, band)
        for ax in (axD, axE):
            _despine(ax)
        even_side_labels(cv)
        common.save_outputs(cv.fig, out_dir, FIGURE)

    # ---- source data (the values drawn)
    rows = []
    for j, c in enumerate(curves):
        r = ex_cells.iloc[j]
        rows.append(pd.DataFrame({"example": j + 1, "row": int(r["row"]), "session_id": r["session_id"],
                                  "cell_id": int(r["cell_id"]), "direction_deg": angles_padded, "rate_per_sample": c}))
    common.write_source_data(out_dir, FIGURE, "A", "tuning_curves", pd.concat(rows, ignore_index=True))
    cells_a = ex_cells[["row", "session_id", "cell_id", "orientation_deg", "Rayleigh_angle", "Rayleigh_mag"]].copy()
    cells_a.insert(0, "example", np.arange(1, len(cells_a) + 1))
    cells_a = cells_a.rename(columns={"Rayleigh_angle": "preferred_direction_deg", "Rayleigh_mag": "vector_length"})
    common.write_source_data(out_dir, FIGURE, "A", "cells", cells_a)
    centre = (edges[:-1] + edges[1:]) / 2
    common.write_source_data(out_dir, FIGURE, "B", "deviation_histogram", {
        "bin_centre_mod60_deg": centre, "angle_drawn_deg": 6 * centre,
        "deviation_from_deg": bins_sorted.deviation_from_deg.values,
        "deviation_to_deg": bins_sorted.deviation_to_deg.values,
        "half": np.where((centre < 15) | (centre > 45), "aligned", "misaligned"), "count": counts})
    pa = per_animal[(per_animal.form == "normalised_to_cell_mean") & per_animal.in_mean_across_animals]
    piv = pa.pivot(index="bin", columns="animal_id", values="mean_curve")
    nmap = pa.groupby("animal_id").n_cells.first()
    k = len(ang_c) - 1
    cols = {"direction_deg": ang_c[:k]}
    for an in piv.columns:
        cols[f"animal_{an}_n{int(nmap[an])}"] = piv[an].to_numpy()[:k]
    cols.update(mean_across_animals=mean_c[:k], sem_across_animals=sem_c[:k], band_lower=lo_c[:k], band_upper=up_c[:k])
    common.write_source_data(out_dir, FIGURE, "C", "per_animal_curves_and_mean", cols)
    dd = d[["row", "animal_id", "session_id", "cell_id", "orientation_deg", "direction_deg", "dir_mod60"]].copy()
    dd = dd.rename(columns={"direction_deg": "preferred_direction_deg", "dir_mod60": "preferred_direction_mod60_deg"})
    dd["colour"] = dd.animal_id.map(ANIMAL_COLOURS).fillna(OTHER_ANIMALS)
    common.write_source_data(out_dir, FIGURE, "D", "cells", dd)
    common.write_source_data(out_dir, FIGURE, "E", "animals", vol[["animal_id", "n", "R6"]].rename(
        columns={"n": "n_conjunctive_cells", "R6": "rayleigh_vector_length_6x"}))
    common.write_source_data(out_dir, FIGURE, "E", "chance_band", {"n_conjunctive_cells": band.n.values,
                                                                    "rayleigh_vector_length_95th_percentile":
                                                                        band.R_chance95.values})
