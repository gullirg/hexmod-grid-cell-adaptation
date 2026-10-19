"""Figure 1: grid cell properties and the two hypotheses for hexadirectional modulation (HexMod).

A schematic drawn at final size (174 mm wide), all in vector form, in mm from the top left corner of the figure. The
letters of a row of panels share one line, their capitals 1.5 mm (``style.LETTER_ABOVE``) above the topmost element of
the row; panels carry no titles (the figure legend names them).

Panels
    A  Grid cell properties: three tiles (Orientation, Scale, Offset) of grid fields.
    B  HexMod: the wheel of movement directions, aligned (within 15 deg of a grid axis) and misaligned, with the colour
       key to its right.
    C  Conjunctive hypothesis: a cell preferring phi, a cell preferring phi + 60 deg and the sum of six such cells, for
       narrow and for broad tuning; each row ends in the two prediction charts (mean firing rate for aligned and
       misaligned movement, and the HexMod coefficient beta).
    D  Adaptation hypothesis: firing is suppressed after a recent field visit (row 1) and aligned runs give short
       inter-field intervals more often (row 2); together they give the prediction, drawn with the same two charts,
       columns and scales as in C.

What is computed and what is schematic
    Panel C is computed (``hexmod.sim.analytic``). Narrow tuning: von Mises curves with unit mean,
    ``exp(kappa cos(theta - mu)) / I0(kappa)``, of circular SD 15 deg (kappa from ``hexmod.circular``); six cells with
    mu = phi + j 60 deg are summed, and the six-fold modulation of the sum relative to its mean is
    ``2 I6(kappa) / I0(kappa)``. Broad tuning: cosine curves ``1 + cos(theta - mu)`` (circular SD 67.46 deg), whose sum
    is exactly uniform (beta = 0). The bars of the rate charts are the mean of the sum within 15 deg of a grid axis
    and further away, relative to its mean.
    Panels A, B and D are schematics. The bar heights of D (``D_RATIO``) are illustrative constants, not results;
    the beta bar of D follows from its two firing rate bars by the rule that links the bars of C
    (``hexmod.sim.analytic.beta_from_bars``), so that C and D can be compared. Nothing is read from result files.

Source data: the fields drawn in the tiles of A, the tuning curves and prediction values of C, and the schematic bar
values of D.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, PathPatch, Polygon, Rectangle
from matplotlib.path import Path

from .. import circular
from ..sim import analytic
from . import common
from . import style as hs

FIGURE = "Fig1"
X_B = 102.0                  # left edge of the letter B on the first row
LABEL_SHORT = "Short\nintervals (%)"
KEY_WORDS = ("Aligned", "Misaligned")

# ------------------------------------------------------------------ model of C
SD_NARROW = 15.0     # deg, circular SD of the narrow cells (half-width of the aligned zone)
SUM_RLIM = 1.70      # outer circle of both sum plots = 1.70 x mean of the sum
CELL_RLIM = 1.10     # outer circle of each example cell = 1.10 x its own peak

# Bar ratios of D: illustrative constants of the schematic (not results)
D_RATIO = {
    "fr_aligned_over_misaligned": 101.0 / 158.0,              # prediction: firing rate, aligned over misaligned
    "fr_fields_over_first": (1.0, 128.0 / 162.0, 28.0 / 162.0),   # row 1: firing rate at fields X, Y, Z
    "short_misaligned_over_aligned": 105.0 / 158.0,           # row 2: share of short intervals
}

NAME = "HexMod"
PHI = "φ"
BETA = "β"

# ------------------------------------------------------------------ colours
C_ALIGN, C_MISAL = "#66C5D5", "#F6BA66"      # fills of aligned and misaligned
T_ALIGN, T_MISAL = "#009FB9", "#F18D00"      # text and arrows
C_BAR = "#858585"                # neutral bars (HexMod beta; firing rate at one field)
C_AXIS = "#929292"               # grid axes (tiles, wheel, polar plots)
CELL1 = ("#F3ABA2", "#D93A2B")   # C, cell preferring phi: fill, outline
CELL2 = ("#B5DDAD", "#3E9E38")   # C, cell preferring phi + 60 deg
SUMC = ("#DDDDDD", "#000000")
FIELD = ("#D2D2D2", "#4D4D4D")   # grid fields: reference cell of A; the cell of the wheel and of D (fill, outline)
FIELD_B = ("#B9A6E8", "#6F52C0") # A, second cell

# ------------------------------------------------------------------ type and lines
FS = 9.0             # pt, all labels
FS_SMALL = 8.0       # pt, letters X, Y, Z in the field circles; "long", "short"
FS_LETTER = 12.0     # pt, bold panel letters
FS_SYMBOL = 13.0     # pt, "+", the three dots, "="
LW_THIN = 0.5        # pt, grid axes and the dashed mean circle
LW = 0.8             # pt, frames, chart axes, field outlines
LW_BOLD = 1.2        # pt, tuning curves; the solid and dashed lines of A
PT = 25.4 / 72.0     # mm per pt
HEAD_W = 2.2         # mm, width of the head of every coloured arrow (shaft 1.0 mm)

# ------------------------------------------------------------------ page, mm
W_MM = 174.0
X_LETTER = 1.5
X0, X1 = 8.0, 172.0          # left and right edge of the content
Y_TITLE = 4.55               # baseline of the letters below the top of a section (capital top 1.52 mm)
GAP = 3.5                    # between the sections of A and B and of C
GAP_CD = 3.4                 # between the sections of C and D (5.46 mm of white under the lowest text of each section)

# the two prediction charts: one size, one scale, the same x in C and D
X_FR, X_HM = 126.5, 163.0    # the two vertical axes
X_PRED = (X_FR + X1) / 2.0   # centre of the heading "Prediction"
X_WHEEL = 125.6              # centre of the wheel of B, so that the stacked colour key fits to its right
KEY_PITCH = 3.88             # between the baselines of the two rows of the colour key of B
KEY_GAP_MIN = 3.0            # least white between the rightmost phi label of the wheel and the key (in Arial)
AX_UP, AX_DOWN = 10.0, 9.0   # arrow tip above and baseline below the centre of a chart
ZERO_DY = 1.5                # zero line of HexMod beta below the centre of the chart
FR_MM_PER_MEAN = 10.0        # bar height of a firing rate equal to the mean
HM_MM_PER_UNIT = 12.0 * FR_MM_PER_MEAN / 9.5   # bar height of a coefficient of 1
BAR_W, BAR_GAP, BAR_LEAD = 10.0, 6.0, 1.0
BETA_W = 7.0
LABEL_DY = 4.3               # baseline of the bar labels below the chart baseline
YLAB_DX = 2.7                # centre of the vertical axis label left of the axis
C_FULL = 0.80                # tallest bar of a schematic chart as a fraction of the axis

# A
TILE = 24.0
TILE_TOP = 4.62             # row 1's top (label "phi + 60°" of the wheel) lies style.LETTER_ABOVE under the letters
TILE_X = tuple(X0 + i * (95.0 - X0 - TILE) / 2.0 for i in range(3))    # three tiles from x 8 to 95
TILE_A = 9.0                 # grid spacing in the tiles
TILE_RF = 0.25 * TILE_A      # field radius
A_ROT = 20.0                 # deg, the cell of the Orientation tile (angle of its nearest grid axis)
A_ARC_R = 12.0               # mm, radius of the arc of phi, from the vertex
A_PHI_GAP = 2.3              # mm, centre of the label phi beyond the arc, on the bisector of the angle
A_LEG = 18.0                 # mm, length of the reference line and of the marked grid axis from the vertex
A_SCALE_GREY_ALPHA = 0.4     # fill opacity of the grey cell in the Scale tile
A_SCALE = 0.55               # second cell of the Scale tile, relative grid spacing
A_LABEL_DY = 4.3             # baseline of the labels below the tiles
MIN_VISIBLE = 1.0            # fields that would show less than this at a frame are left out
WHEEL_R = 12.0
H_A = TILE_TOP + TILE + A_LABEL_DY + 1.2

# C
R_POL = 11.0
C_HEAD = 5.31                # baseline of the column headings (their tops style.LETTER_ABOVE under the letter C)
C_ROW = (19.11, 46.11)       # row centres
C_X_CELL1, C_X_CELL2, C_X_SUM = 33.0, 65.0, 107.0
C_X_PLUS1, C_X_PLUS2, C_X_DOTS, C_X_EQ = 50.7, 80.0, 86.0, 92.0
C_SYMBOL_DY = 0.4            # moves "+", dots and "=" onto the centre line of the plots
H_C = C_ROW[1] + AX_DOWN + LABEL_DY + 1.3

# D
MAP_A = 8.5                  # grid spacing in the maps; also 1 unit of the time axis
MAP_RF = 0.25 * MAP_A        # field radius; the lettered fields have the same size
MAP_W, MAP_H = 42.0, 22.5
MAP_GAP_SHORT = 0.4          # gap between a short (aligned) arrow and the field it leaves or reaches
MAP_GAP_LONG = 1.0           # the same for a long (misaligned) arrow, so that no two arrows touch
D_TOP = (3.02, 39.02)        # top of the map and chart of each row (row 1: style.LETTER_ABOVE under the letter D)
X_CH = 61.0                  # vertical axis of the two charts of D
X_BRACE, BRACE_D = 109.5, 3.0
X_ARROW_END = 120.0
H_D = D_TOP[1] + AX_UP + AX_DOWN + LABEL_DY + 2.1

DIRECTIONS_DEG = np.arange(0.0, 360.0, 0.25)
TH = np.deg2rad(DIRECTIONS_DEG)


# ----------------------------------------------------------------------- model
def describe_von_mises(sd_deg):
    """Summed tuning of six von Mises cells of circular SD ``sd_deg``.

    See ``hexmod.sim.analytic.population_summary``.
    """
    s = analytic.population_summary(sd_deg, TH)
    return dict(tuning="von Mises", circular_sd_deg=sd_deg, kappa=s["kappa"], sixfold_formula=s["sixfold_formula"],
                sixfold_numeric=s["sixfold_numeric"], aligned_over_mean=s["aligned_over_mean"],
                misaligned_over_mean=s["misaligned_over_mean"])


def describe_cosine():
    """Summed tuning of six cosine cells ``1 + cos(theta - mu)``: exactly uniform, so the six-fold component is 0."""
    s = analytic.population_sum_cosine(TH)
    dev = np.rad2deg(np.abs(np.angle(np.exp(1j * 6 * TH)))) / 6.0       # distance to the nearest grid axis (deg)
    return dict(tuning="cosine", circular_sd_deg=float(circular.circular_sd_deg(0.5)), kappa=float("nan"),
                sixfold_formula=0.0, sixfold_numeric=2.0 * np.mean(s * np.cos(6 * TH)),
                aligned_over_mean=s[dev <= 15.0].mean(), misaligned_over_mean=s[dev > 15.0].mean())


def curves(d):
    """Cell preferring 0 deg, cell preferring 60 deg and the mean of the six cells, for one row of C."""
    if d["tuning"] == "cosine":
        return analytic.cosine_cell(TH, 0.0), analytic.cosine_cell(TH, 60.0), analytic.population_sum_cosine(TH)
    k = d["kappa"]
    return analytic.von_mises_cell(TH, 0.0, k), analytic.von_mises_cell(TH, 60.0, k), analytic.population_sum(TH, k)


def prediction_d():
    """Rate bars (fractions of the mean, averaging 1) and the beta of the prediction of D."""
    r = D_RATIO["fr_aligned_over_misaligned"]
    aligned, misaligned = 2.0 * r / (1.0 + r), 2.0 / (1.0 + r)
    return aligned, misaligned, analytic.beta_from_bars(aligned, misaligned)


# --------------------------------------------------------------------- canvas
class _Canvas:
    """The figure, its one drawing axes (mm, y downwards), the current panel and the fields drawn in the tiles."""

    def __init__(self, fig, ax):
        self.fig, self.ax = fig, ax
        self.panel = ""
        self.tile = None
        self.fields = []


def _new_canvas(w_mm, h_mm):
    fig = Figure(figsize=(w_mm / 25.4, h_mm / 25.4), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_mm)
    ax.set_ylim(h_mm, 0)          # y runs downwards, in mm
    ax.set_aspect("equal")
    ax.axis("off")
    common.tag(ax, panel="", name="drawing area")
    return _Canvas(fig, ax)


def _text(cv, x, y, s, **kw):
    kw.setdefault("fontsize", FS)
    kw.setdefault("color", "black")
    t = cv.ax.text(x, y, s, **kw)
    common.tag(t, panel=cv.panel)
    return t


def _text_size_mm(cv, s, **kw):
    t = cv.ax.text(0, 0, s, **kw)
    bb = t.get_window_extent(cv.fig.canvas.get_renderer())
    t.remove()
    return bb.width / cv.fig.dpi * 25.4, bb.height / cv.fig.dpi * 25.4


def _centre_to_baseline(yc, fontsize):
    """Baseline that puts capitals and digits of this size centred on yc (cap height 0.716 em)."""
    return yc + 0.358 * fontsize * PT


def _mixed_text(cv, x, y, parts, ha="center", fontsize=FS):
    """One line of (string, style, colour) pieces, baseline at y; phi is italic, words upright."""
    widths = [_text_size_mm(cv, s, fontsize=fontsize, style=st)[0] for s, st, _ in parts]
    total = sum(widths)
    xx = x - total / 2.0 if ha == "center" else (x if ha == "left" else x - total)
    left = xx
    for (s, st, col), w in zip(parts, widths):
        _text(cv, xx, y, s, fontsize=fontsize, style=st, color=col, ha="left", va="baseline")
        xx += w
    return left, total


def _vertical_label(cv, x, yc, upright, italic=None, fontsize=FS):
    """Upright words, optionally followed by an italic symbol, written upwards, centred on yc.
    x is the centre of a one-line label; a two-line label ends at the same right edge."""
    if italic is None:
        two = "\n" in upright
        _text(cv, x - (1.95 if two else 0.0), yc, upright, fontsize=fontsize, rotation=90, ha="center", va="center",
              multialignment="center", linespacing=1.1)
        return
    _, h1 = _text_size_mm(cv, upright, fontsize=fontsize, rotation=90)
    _, h2 = _text_size_mm(cv, italic, fontsize=fontsize, style="italic", rotation=90)
    _text(cv, x, yc + (h1 + h2) / 2.0, upright, fontsize=fontsize, rotation=90, ha="center", va="bottom")
    _text(cv, x, yc + (h1 + h2) / 2.0 - h1, italic, fontsize=fontsize, style="italic", rotation=90, ha="center",
          va="bottom")


def _arrow(ax, p0, p1, colour, shaft=1.0, head_l=2.3, head_w=HEAD_W, z=6):
    """Flat arrow from p0 to p1 (mm): a shaft and a triangular head, one polygon."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    length = np.hypot(*d)
    u = d / length
    n = np.array([-u[1], u[0]])
    hl = min(head_l, 0.75 * length)
    b = p1 - hl * u
    pts = [p0 + n * shaft / 2, b + n * shaft / 2, b + n * head_w / 2, p1,
           b - n * head_w / 2, b - n * shaft / 2, p0 - n * shaft / 2]
    ax.add_patch(Polygon(pts, closed=True, fc=colour, ec="none", zorder=z))


def _axis_arrow(ax, p0, p1):
    """Chart axis (or a plain thin arrow) ending in a small arrowhead."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    u = (p1 - p0) / np.hypot(*(p1 - p0))
    n = np.array([-u[1], u[0]])
    hl, hw = 1.7, 1.3
    b = p1 - hl * u
    ax.plot([p0[0], b[0]], [p0[1], b[1]], color="black", lw=LW, solid_capstyle="butt", zorder=5)
    ax.add_patch(Polygon([b + n * hw / 2, p1, b - n * hw / 2], closed=True, fc="black", ec="none", zorder=5))


def _solid_line(ax, p, q, clip=None):
    ln, = ax.plot([p[0], q[0]], [p[1], q[1]], color="black", lw=LW_BOLD, zorder=6, solid_capstyle="butt")
    if clip is not None:
        ln.set_clip_path(clip)


def _dashed_line(ax, p, q, clip=None, dash=1.05, gap=0.75):
    """Dashed line that starts and ends with a full dash (no stray short dash)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    length = np.hypot(*(q - p))
    u = (q - p) / length
    n = max(2, int(round((length + gap) / (dash + gap))))
    k = length / (n * dash + (n - 1) * gap)
    for i in range(n):
        a = p + u * i * (dash + gap) * k
        b = a + u * dash * k
        ln, = ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=LW_BOLD, zorder=6, solid_capstyle="butt")
        if clip is not None:
            ln.set_clip_path(clip)


def _panel_letter(cv, y0, letter, x_letter=X_LETTER):
    cv.panel = letter
    common.tag(_text(cv, x_letter, y0 + Y_TITLE, letter, fontsize=FS_LETTER, weight="bold", ha="left", va="baseline"),
               name="panel letter")


def _lattice(origin, spacing, rot_deg=0.0, n=6):
    """Field centres of a hexagonal grid; angles counterclockwise on the page (y runs down)."""
    r = np.deg2rad(rot_deg)
    a1 = np.array([np.cos(r), -np.sin(r)])
    a2 = np.array([np.cos(r + np.pi / 3), -np.sin(r + np.pi / 3)])
    return [np.asarray(origin, float) + spacing * (i * a1 + j * a2)
            for i in range(-n, n + 1) for j in range(-n, n + 1)]


def _fields_in_box(cv, box, origin, spacing, radius, colours, z, rot=0.0):
    """The fields of one cell inside a rectangular frame, cut at the frame."""
    x0, y0 = box.get_xy()
    x1, y1 = x0 + box.get_width(), y0 + box.get_height()
    for p in _lattice(origin, spacing, rot):
        show_x = min(p[0] + radius, x1) - max(p[0] - radius, x0)
        show_y = min(p[1] + radius, y1) - max(p[1] - radius, y0)
        if show_x >= MIN_VISIBLE and show_y >= MIN_VISIBLE:
            if cv.tile is not None:
                fc = colours[0]
                if isinstance(fc, str):
                    fill, alpha = fc, 1.0
                else:
                    fill, alpha = "#{:02X}{:02X}{:02X}".format(*(int(round(255 * v)) for v in fc[:3])), fc[3]
                cv.fields.append((cv.tile, fill, alpha, colours[1], p[0], p[1], radius, spacing, rot))
            c = Circle(p, radius, fc=colours[0], ec=colours[1], lw=LW, zorder=z)
            cv.ax.add_patch(c)
            c.set_clip_path(box)


def _grey_see_through(alpha):
    """Grey that, laid over white with this opacity, gives the grey fill FIELD[0] (as RGBA)."""
    g = int(FIELD[0][1:3], 16)
    v = int(round(255.0 - (255.0 - g) / alpha)) / 255.0
    return (v, v, v, alpha)


# ---- the two charts that close every row of C and the argument of D
def _rate_chart(cv, x_axis, y_top, h_aligned, h_misaligned, ylabel="Firing rate"):
    """Two bars, aligned and misaligned, on an axis without numbers."""
    ax = cv.ax
    y_base = y_top + AX_UP + AX_DOWN
    _axis_arrow(ax, (x_axis, y_base), (x_axis, y_top))
    x_end = x_axis + BAR_LEAD + 2 * BAR_W + BAR_GAP + BAR_LEAD
    ax.plot([x_axis, x_end], [y_base, y_base], color="black", lw=LW, solid_capstyle="butt", zorder=5)
    for j, (h, col, tcol, lab) in enumerate(((h_aligned, C_ALIGN, T_ALIGN, KEY_WORDS[0]),
                                              (h_misaligned, C_MISAL, T_MISAL, KEY_WORDS[1]))):
        xb = x_axis + BAR_LEAD + j * (BAR_W + BAR_GAP)
        ax.add_patch(Rectangle((xb, y_base - h), BAR_W, h, fc=col, ec="none", zorder=3))
        _text(cv, xb + BAR_W / 2.0, y_base + LABEL_DY, lab, color=tcol, ha="center", va="baseline")
    _vertical_label(cv, x_axis - YLAB_DX, (y_base + y_top) / 2.0 + 0.6, ylabel)
    return x_end


def _beta_chart(cv, x_axis, y_top, h_up):
    """One neutral bar from the zero line: up for a positive coefficient, down for a negative one."""
    ax = cv.ax
    y_base = y_top + AX_UP + AX_DOWN
    y_zero = y_top + AX_UP + ZERO_DY
    _axis_arrow(ax, (x_axis, y_base), (x_axis, y_top))
    x_end = x_axis + BAR_LEAD + BETA_W + BAR_LEAD
    ax.plot([x_axis, x_end], [y_zero, y_zero], color="black", lw=LW, solid_capstyle="butt", zorder=5)
    if abs(h_up) < 0.05:
        # a bar too small to see (less than 0.05 mm) is written as a value instead
        _text(cv, x_axis + BAR_LEAD + BETA_W / 2.0, y_zero - 1.4, "= 0" if h_up == 0 else "≈ 0", ha="center",
              va="baseline")
    else:
        ax.add_patch(Rectangle((x_axis + BAR_LEAD, min(y_zero, y_zero - h_up)), BETA_W, abs(h_up),
                               fc=C_BAR, ec="none", zorder=3))
    # the zero of the axis, then the axis label to its left
    w0 = _text_size_mm(cv, "0", fontsize=FS)[0]
    _text(cv, x_axis - 0.8, _centre_to_baseline(y_zero, FS), "0", ha="right", va="baseline")
    _vertical_label(cv, x_axis - 0.8 - w0 - 0.5 - 1.6, (y_base + y_top) / 2.0 + 0.6, NAME + " ", BETA)
    return x_end


# ------------------------------------------------------------------ panels A and B
def _tile(cv, x_left, y_top, kind):
    ax = cv.ax
    cv.tile = kind
    S, a, rf = TILE, TILE_A, TILE_RF
    box = Rectangle((x_left, y_top), S, S, fc="white", ec="none", zorder=1)
    ax.add_patch(box)
    c = np.array([x_left + S / 2.0, y_top + S / 2.0])
    up = np.array([a / 2.0, -a / (2.0 * np.sqrt(3.0))])     # to the centre of the triangle above right

    def axes_through(origin, rot=0.0):
        for ang in (rot, rot + 60.0, rot + 120.0):
            co, si = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
            ln, = ax.plot([origin[0] - 2 * S * co, origin[0] + 2 * S * co],
                          [origin[1] + 2 * S * si, origin[1] - 2 * S * si],
                          color=C_AXIS, lw=LW_THIN, zorder=4, solid_capstyle="butt")
            ln.set_clip_path(box)

    def at(origin, dist, ang_deg):
        return origin + dist * np.array([np.cos(np.deg2rad(ang_deg)), -np.sin(np.deg2rad(ang_deg))])

    if kind == "Orientation":
        # one cell, turned by A_ROT; the angle is marked at the field one spacing before the centre along the turned
        # axis, so that the arc and the label lie in the space between fields
        _fields_in_box(cv, box, c, a, rf, FIELD, 3, rot=A_ROT)
        axes_through(c, A_ROT)
        v = at(c, a, A_ROT + 180.0)                   # vertex of the angle, on the grid axis
        _dashed_line(ax, v, at(v, A_LEG, 0.0), box)   # horizontal reference
        _solid_line(ax, v, at(v, A_LEG, A_ROT), box)  # nearest grid axis
        t = np.deg2rad(np.linspace(0.0, A_ROT, 41))
        arc_, = ax.plot(v[0] + A_ARC_R * np.cos(t), v[1] - A_ARC_R * np.sin(t), color="black", lw=LW,
                        zorder=6, solid_capstyle="butt")
        arc_.set_clip_path(box)
        pl = at(v, A_ARC_R + A_PHI_GAP, A_ROT / 2.0)
        _text(cv, pl[0], _centre_to_baseline(pl[1], FS) - 0.35, PHI, style="italic", ha="center", va="baseline",
              zorder=7)
    elif kind == "Scale":
        # a second cell at A_SCALE of the spacing under semi-transparent grey fields (opaque outlines); each line joins
        # two neighbouring fields: solid for the grey cell, dashed for the second cell
        _fields_in_box(cv, box, c, A_SCALE * a, A_SCALE * rf, FIELD_B, 2)
        _fields_in_box(cv, box, c, a, rf, (_grey_see_through(A_SCALE_GREY_ALPHA), FIELD[1]), 3)
        axes_through(c)
        _solid_line(ax, c, at(c, a, 0.0), box)
        p1 = at(c, A_SCALE * a, 300.0)               # a field of the second cell, below right of the centre
        _dashed_line(ax, p1, p1 + np.array([A_SCALE * a, 0.0]), box)
    elif kind == "Offset":
        # a second cell shifted by a third of the unit cell; arrow from a grey field to the nearest field of the second
        _fields_in_box(cv, box, c, a, rf, FIELD, 2)
        _fields_in_box(cv, box, c + up, a, rf, FIELD_B, 2)
        axes_through(c)
        _arrow(ax, c, c + up, "black", shaft=LW_BOLD * PT, head_l=1.9, head_w=1.7, z=6)
    ax.add_patch(Rectangle((x_left, y_top), S, S, fc="none", ec="black", lw=LW, zorder=9))
    cv.tile = None
    yb = y_top + S + A_LABEL_DY
    if kind == "Orientation":
        _mixed_text(cv, c[0], yb, [("Orientation (", "normal", "black"), (PHI, "italic", "black"),
                                   (")", "normal", "black")])
    else:
        _text(cv, c[0], yb, kind, ha="center", va="baseline")


def _wheel(cv, y0):
    """The wheel of movement directions; returns the right end of its rightmost phi label (mm)."""
    ax = cv.ax
    cx, cy, R = X_WHEEL, y0 + TILE_TOP + TILE / 2.0, WHEEL_R
    right = cx + R
    for k in range(12):
        t = np.deg2rad(np.linspace(30.0 * k - 15.0, 30.0 * k + 15.0, 31))
        pts = [(cx, cy)] + [(cx + R * np.cos(v), cy - R * np.sin(v)) for v in t]
        col = C_ALIGN if k % 2 == 0 else C_MISAL
        ax.add_patch(Polygon(pts, closed=True, fc=col, ec=col, lw=0.2, zorder=1))
    for ang in (0, 60, 120):                      # thin grid axes, as in the polar plots of C
        co, si = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        ax.plot([cx - R * co, cx + R * co], [cy + R * si, cy - R * si], color=C_AXIS, lw=LW_THIN, zorder=2,
                solid_capstyle="butt")
    a = R / 2.33                                  # the circle holds the centre field and two rings
    rf = 0.24 * a
    for i in range(-2, 3):
        for j in range(-2, 3):
            if abs(i + j) <= 2:
                p = (cx + a * (i + 0.5 * j), cy - a * np.sqrt(3) / 2 * j)
                ax.add_patch(Circle(p, rf, fc=FIELD[0], ec=FIELD[1], lw=LW, zorder=3))
    ax.add_patch(Circle((cx, cy), R, fc="none", ec="black", lw=LW, zorder=4))
    # the grid orientation and one direction of each kind
    for ang, tail, col in ((0.0, "", "black"), (30.0, " + 30°", T_MISAL), (60.0, " + 60°", T_ALIGN)):
        sn = np.sin(np.deg2rad(ang))
        px = cx + (R + 0.9) * np.cos(np.deg2rad(ang))
        py = cy - (R + 0.9) * sn
        parts = [(PHI, "italic", col)] + ([(tail, "normal", col)] if tail else [])
        left, total = _mixed_text(cv, px, _centre_to_baseline(py, FS) - 0.5 - 0.9 * sn, parts, ha="left")
        right = max(right, left + total)
    _text(cv, cx, y0 + TILE_TOP + TILE + A_LABEL_DY, "Movement direction", ha="center", va="baseline")
    return right


def _key(cv, y0):
    """Colour key of B: Aligned above Misaligned, flush right to X1 and vertically centred on the wheel.

    Returns the left edge of the key (mm).
    """
    sw_w, sw_h, gap = 4.8, 2.6, 1.6
    widths = [_text_size_mm(cv, s, fontsize=FS)[0] for s in KEY_WORDS]
    x = X1 - (sw_w + gap + max(widths))
    cy = y0 + TILE_TOP + TILE / 2.0                  # centre of the wheel
    yb0 = cy - (KEY_PITCH - sw_h) / 2.0              # block (top of first swatch to last baseline) centred on cy
    for i, (lab, fill) in enumerate(zip(KEY_WORDS, (C_ALIGN, C_MISAL))):
        yb = yb0 + i * KEY_PITCH
        cv.ax.add_patch(Rectangle((x, yb - sw_h), sw_w, sw_h, fc=fill, ec="none", zorder=3))
        _text(cv, x + sw_w + gap, yb, lab, ha="left", va="baseline")
    return x


def _panels_a_b(cv, y0, font=hs.HOUSE_FONT):
    """Panels A and B. The colour key of B is placed for Arial: in Arial a key closer than ``KEY_GAP_MIN`` to the
    wheel labels is an error of the layout; in a fallback font (``font``) it is reported with a warning."""
    _panel_letter(cv, y0, "A")
    _panel_letter(cv, y0, "B", x_letter=X_B)
    x_key = _key(cv, y0)
    cv.panel = "A"
    for x, kind in zip(TILE_X, ("Orientation", "Scale", "Offset")):
        _tile(cv, x, y0 + TILE_TOP, kind)
    cv.panel = "B"
    x_wheel_right = _wheel(cv, y0)
    if x_key - x_wheel_right < KEY_GAP_MIN - 1e-9:
        message = (f"the colour key of B ({x_key:.2f} mm) is too close to the wheel labels "
                   f"({x_wheel_right:.2f} mm)")
        if font == hs.HOUSE_FONT:
            raise ValueError(message)
        warnings.warn(f"Figure 1: {message} in {font}; the layout is made for Arial (set HEXMOD_FONT_DIR to a "
                      "folder with the Arial TrueType files)")


# ------------------------------------------------------------------ panel C
def _polar(ax, cx, cy, rad, r_norm, colours, mean_circle=None):
    """Closed polar curve; r_norm is the radius as a fraction of the outer circle.
    Angles run counterclockwise from the right."""
    fill, edge = colours
    ax.add_patch(Circle((cx, cy), rad, fc="white", ec="none", zorder=1))
    xy = np.column_stack([cx + rad * r_norm * np.cos(TH), cy - rad * r_norm * np.sin(TH)])
    ax.add_patch(Polygon(xy, closed=True, fc=fill, ec="none", zorder=2))
    for a in (0, 60, 120):                       # the six grid axis directions
        c, s = np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))
        ax.plot([cx - rad * c, cx + rad * c], [cy + rad * s, cy - rad * s],
                color=C_AXIS, lw=LW_THIN, zorder=3, solid_capstyle="butt")
    if mean_circle is not None:
        ax.add_patch(Circle((cx, cy), rad * mean_circle, fc="none", ec="black", lw=LW_THIN,
                            ls=(0, (2.5, 2.0)), zorder=4))
    ax.add_patch(Polygon(xy, closed=True, fc="none", ec=edge, lw=LW_BOLD, zorder=5, joinstyle="round"))
    ax.add_patch(Circle((cx, cy), rad, fc="none", ec="black", lw=LW, zorder=6))


def _row_c(cv, cy, d, label):
    c1, c2, s = curves(d)
    _text(cv, X0, cy, label, ha="left", va="center", linespacing=1.15)
    _polar(cv.ax, C_X_CELL1, cy, R_POL, c1 / (CELL_RLIM * c1.max()), CELL1)
    _polar(cv.ax, C_X_CELL2, cy, R_POL, c2 / (CELL_RLIM * c2.max()), CELL2)
    _polar(cv.ax, C_X_SUM, cy, R_POL, s / SUM_RLIM, SUMC, mean_circle=1.0 / SUM_RLIM)
    _text(cv, C_X_CELL1 + R_POL + 0.7, _centre_to_baseline(cy, FS) - 0.5, PHI, style="italic", ha="left", va="baseline",
          zorder=7)
    for x, sym in ((C_X_PLUS1, "+"), (C_X_PLUS2, "+"), (C_X_DOTS, "···"), (C_X_EQ, "=")):
        _text(cv, x, cy + C_SYMBOL_DY, sym, fontsize=FS_SYMBOL, ha="center", va="center")
    _rate_chart(cv, X_FR, cy - AX_UP, d["aligned_over_mean"] * FR_MM_PER_MEAN,
                d["misaligned_over_mean"] * FR_MM_PER_MEAN)
    _beta_chart(cv, X_HM, cy - AX_UP, d["sixfold_formula"] * HM_MM_PER_UNIT)


def _panel_c(cv, y0, dn, db):
    _panel_letter(cv, y0, "C")
    k = "black"
    yh = y0 + C_HEAD
    _mixed_text(cv, C_X_CELL1, yh, [("Cell preferring ", "normal", k), (PHI, "italic", k)])
    _mixed_text(cv, C_X_CELL2, yh, [("Cell preferring ", "normal", k), (PHI, "italic", k), (" + 60°", "normal", k)])
    _text(cv, C_X_SUM, yh, "Sum of six cells", ha="center", va="baseline")
    _text(cv, X_PRED, yh, "Prediction", ha="center", va="baseline")
    _row_c(cv, y0 + C_ROW[0], dn, "Narrow\ntuning")
    _row_c(cv, y0 + C_ROW[1], db, "Broad\ntuning")


# ------------------------------------------------------------------ panel D
def _letter_field(cv, p, letter, z=7):
    cv.ax.add_patch(Circle(p, MAP_RF, fc="white", ec=FIELD[1], lw=LW, zorder=z))
    _text(cv, p[0], _centre_to_baseline(p[1], FS_SMALL), letter, fontsize=FS_SMALL, ha="center", va="baseline",
          zorder=z + 1)


def _field_map(cv, y_top, paths, letters):
    """Grid fields in a framed patch of the environment, drawn like the tiles of A.
    paths = (from, to, colour, gap) and letters = (field, letter), fields in grid units (right, up);
    gap is the distance between the arrow and the outline of the field it leaves or reaches."""
    ax = cv.ax
    box = Rectangle((X0, y_top), MAP_W, MAP_H, fc="white", ec="none", zorder=1)
    ax.add_patch(box)
    c = np.array([X0 + MAP_W / 2.0, y_top + MAP_H / 2.0])

    def xy(p):
        return c + MAP_A * np.array([p[0], -p[1]])

    _fields_in_box(cv, box, c, MAP_A, MAP_RF, FIELD, 3)
    for p, s in letters:
        _letter_field(cv, xy(p), s)
    for p, q, colour, g in paths:
        p0, q0 = xy(p), xy(q)
        u = (q0 - p0) / np.hypot(*(q0 - p0))
        _arrow(ax, p0 + u * (MAP_RF + g), q0 - u * (MAP_RF + g), colour, z=6)
    ax.add_patch(Rectangle((X0, y_top), MAP_W, MAP_H, fc="none", ec="black", lw=LW, zorder=9))


def _time_chart(cv, y_top):
    """Firing rate at three successive field visits; time between visits follows the distance."""
    ax = cv.ax
    x = X_CH
    base = y_top + AX_UP + AX_DOWN
    full = C_FULL * (AX_UP + AX_DOWN)
    bw, step, g = 7.0, MAP_A, 0.4
    tX = x + BAR_LEAD + bw / 2.0
    tY = tX + np.sqrt(3.0) * step              # X to Y: square root of 3 grid spacings
    tZ = tY + step                             # Y to Z: one grid spacing
    x_end = tZ + bw / 2.0 + 5.0
    _axis_arrow(ax, (x, base), (x, y_top))
    _axis_arrow(ax, (x, base), (x_end, base))
    _text(cv, x_end + 1.0, _centre_to_baseline(base, FS), "Time", ha="left", va="baseline")
    _vertical_label(cv, x - YLAB_DX, (base + y_top) / 2.0 + 0.6, "Firing rate")
    y_let = base + 1.0 + 0.45 + MAP_RF
    for t, f, letter in zip((tX, tY, tZ), D_RATIO["fr_fields_over_first"], "XYZ"):
        ax.add_patch(Rectangle((t - bw / 2.0, base - f * full), bw, f * full, fc=C_BAR, ec="none", zorder=3))
        ax.plot([t, t], [base, base + 1.0], color="black", lw=LW, solid_capstyle="butt", zorder=5)
        _letter_field(cv, (t, y_let), letter)
    _arrow(ax, (tX + MAP_RF + g, y_let), (tY - MAP_RF - g, y_let), T_MISAL)
    _arrow(ax, (tY + MAP_RF + g, y_let), (tZ - MAP_RF - g, y_let), T_ALIGN)
    yw = y_let + MAP_RF + 0.9 + 0.716 * FS_SMALL * PT
    _text(cv, (tX + tY) / 2.0, yw, "long", fontsize=FS_SMALL, ha="center", va="baseline")
    _text(cv, (tY + tZ) / 2.0, yw, "short", fontsize=FS_SMALL, ha="center", va="baseline")


def _brace(ax, x, y0, y1, d):
    """Curly brace that closes the two rows on the right; its point is at (x + d, middle)."""
    ym, q = (y0 + y1) / 2.0, d / 2.0
    r = 2.2
    verts = [(x, y0), (x + q, y0), (x + q, y0 + r), (x + q, ym - r), (x + q, ym), (x + d, ym),
             (x + q, ym), (x + q, ym + r), (x + q, y1 - r), (x + q, y1), (x, y1)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3]
    ax.add_patch(PathPatch(Path(verts, codes), fc="none", ec="black", lw=LW, capstyle="butt", zorder=5))
    return ym


def _panel_d(cv, y0):
    ax = cv.ax
    _panel_letter(cv, y0, "D")
    y1, y2 = y0 + D_TOP[0], y0 + D_TOP[1]
    h = np.sqrt(3.0) / 2.0
    # row 1: a long (X to Y, misaligned) and a short (Y to Z, aligned) inter-field interval
    pX, pY, pZ = (-1.5, h), (0.0, 0.0), (1.0, 0.0)
    _field_map(cv, y1, [(pX, pY, T_MISAL, MAP_GAP_LONG), (pY, pZ, T_ALIGN, MAP_GAP_SHORT)],
               [(pX, "X"), (pY, "Y"), (pZ, "Z")])
    _time_chart(cv, y1)
    # row 2: four short steps along a grid axis, two long steps 30 deg away from it
    paths = [((i, 0.0), (i + 1.0, 0.0), T_ALIGN, MAP_GAP_SHORT) for i in (-2.0, -1.0, 0.0, 1.0)]
    paths += [((-1.5, h), (0.0, 0.0), T_MISAL, MAP_GAP_LONG), ((0.0, 0.0), (1.5, -h), T_MISAL, MAP_GAP_LONG)]
    _field_map(cv, y2, paths, [])
    full = C_FULL * (AX_UP + AX_DOWN)
    _rate_chart(cv, X_CH, y2, full, full * D_RATIO["short_misaligned_over_aligned"], ylabel=LABEL_SHORT)
    # the two rows together give the prediction
    ym = _brace(ax, X_BRACE, y1, y2 + MAP_H, BRACE_D)
    _axis_arrow(ax, (X_BRACE + BRACE_D + 0.9, ym), (X_ARROW_END, ym))
    # prediction block (heading to bar labels) centred on the arrow; charts as in C
    head_dy = C_ROW[0] - AX_UP - C_HEAD          # heading above the arrow tip of the charts, as in C
    cap = 0.716 * FS * PT
    block = cap + head_dy + AX_UP + AX_DOWN + LABEL_DY + 0.6
    top = ym - block / 2.0 + cap + head_dy
    _text(cv, X_PRED, top - head_dy, "Prediction", ha="center", va="baseline")
    aligned, misaligned, beta = prediction_d()       # the two bars average to the mean, as in C
    _rate_chart(cv, X_FR, top, aligned * FR_MM_PER_MEAN, misaligned * FR_MM_PER_MEAN)
    _beta_chart(cv, X_HM, top, beta * HM_MM_PER_UNIT)


# ------------------------------------------------------------------------ build
def build(results_dir, out_dir):
    """Draw Figure 1 into ``out_dir`` (the figure reads no result files; ``results_dir`` is not used)."""
    font = hs.house_font()
    with mpl.rc_context():
        mpl.rcParams.update(mpl.rcParamsDefault)
        mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [font, "Liberation Sans"],
                             "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 600})
        dn, db = describe_von_mises(SD_NARROW), describe_cosine()
        yA = 0.0
        yC = yA + H_A + GAP
        yD = yC + H_C + GAP_CD
        cv = _new_canvas(W_MM, yD + H_D)
        _panels_a_b(cv, yA, font)
        _panel_c(cv, yC, dn, db)
        _panel_d(cv, yD)
        common.save_outputs(cv.fig, out_dir, FIGURE)
        _write_source_data(out_dir, cv, dn, db)


def _write_source_data(out_dir, cv, dn, db):
    common.write_source_data(out_dir, FIGURE, "A", "tile_fields", pd.DataFrame(
        cv.fields, columns=["tile", "fill", "fill_opacity", "outline", "centre_x_mm", "centre_y_mm_from_top",
                            "radius_mm", "grid_spacing_mm", "grid_rotation_deg_counterclockwise"]))
    table = {"direction_deg": DIRECTIONS_DEG}
    for row, d in (("narrow", dn), ("broad", db)):
        c1, c2, s = curves(d)
        table.update({f"{row}_cell_0deg": c1, f"{row}_cell_60deg": c2, f"{row}_sum_of_six": s})
    common.write_source_data(out_dir, FIGURE, "C", "tuning_curves", pd.DataFrame(table))
    common.write_source_data(out_dir, FIGURE, "C", "predictions", pd.DataFrame(
        [dict(row=row, tuning=d["tuning"], circular_sd_deg=d["circular_sd_deg"], kappa=d["kappa"],
              sixfold_component=d["sixfold_formula"], sixfold_component_numeric=d["sixfold_numeric"],
              rate_aligned_over_mean=d["aligned_over_mean"], rate_misaligned_over_mean=d["misaligned_over_mean"],
              beta_from_bars=analytic.beta_from_bars(d["aligned_over_mean"], d["misaligned_over_mean"]))
         for row, d in (("narrow", dn), ("broad", db))]))
    aligned, misaligned, beta = prediction_d()
    rows = [("row 1: firing rate at successive field visits", f"field {f}", v, "fraction of the first bar")
            for f, v in zip("XYZ", D_RATIO["fr_fields_over_first"])]
    rows += [("row 2: short intervals", "aligned", 1.0, "fraction of the aligned bar"),
             ("row 2: short intervals", "misaligned", D_RATIO["short_misaligned_over_aligned"],
              "fraction of the aligned bar"),
             ("prediction: firing rate", "aligned", aligned, "fraction of the mean rate"),
             ("prediction: firing rate", "misaligned", misaligned, "fraction of the mean rate"),
             ("prediction: HexMod beta", "beta", beta, "coefficient, (pi / 4) (aligned - misaligned)")]
    common.write_source_data(out_dir, FIGURE, "D", "schematic_bars",
                             pd.DataFrame(rows, columns=["chart", "bar", "value", "unit"]))
