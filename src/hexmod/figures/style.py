"""House style of the figures.

Every figure module uses this one style. It holds:

- ``use_style``: the matplotlib parameters (type, line weights, axes) and the typeface;
- ``PALETTE``, ``LOCAL``, ``tint``: one colour for every concept, and colours that belong to one panel only;
- ``VAR``, ``LABEL``, ``TICKS``: variable names as mathtext, and the axis labels and tick words built from them;
- ``new_figure``, ``axes_mm``, ``polar_mm``, ``text_mm``: figures and axes addressed in mm from the top left corner
  of the figure (fixed Agg canvas);
- ``panel_letter``, ``add_clearance``, ``panels_top``, ``settle_letters``, ``extent_mm``: panel letters (one line
  per row of panels, above the panels' topmost element) and the extents of drawn artists;
- ``colour_key``, ``line_key``, ``solid_key``, ``note_above``, ``words_above``: keys and notes;
- ``box_axes``, ``zero_line``, ``polar_style``, ``polar_bars``, ``even_polar_labels``, ``alignment_wedges``,
  ``DEG``, ``NUM``: axes frames and formatters;
- ``boxplot``, ``sig_stars``, ``sig_mark``, ``sig_bracket``, ``bracket_stack``, ``holm``, ``pt_to_data``,
  ``box_width_data``: box plots and significance marks;
- ``box_stats``, ``read_boxes``, ``read_bp``: the numbers of a box plot, computed from the values and read back from
  the drawing;
- ``histogram``, ``sem_band``, ``errorbars``, ``chance_band``, ``rate_map``, ``colour_bar``, ``field_outlines``,
  ``spikes_on_map``, ``stacked_traces``: other plot kinds;
- ``save_figure``, ``stable_svg_clip_ids``, ``type_sizes``, ``script_sizes``, ``svg_type_sizes``: PDF with editable
  text, SVG with text kept as text, PNG at 600 dpi, and checks of the type sizes;
- ``simulate_cvd``, ``colour_distance``, ``lightness``, ``palette_report``: colour-blindness and greyscale checks.

Panels carry no titles (the figure legends name them). A panel letter stands at the left of its panel, its capital
``LETTER_ABOVE`` (1.5 mm) above the topmost element of its row of panels, on one line with the other letters of the
row; ``settle_letters`` raises a row's line where a letter would come within ``LETTER_CLEAR`` (0.5 mm) of a label.

Sizes are final sizes: a figure is designed at the width it is printed at and is never scaled.

The typeface is Arial. Arial cannot be distributed with this repository: set ``HEXMOD_FONT_DIR`` to a folder that
holds the Arial TrueType files (``Arial.ttf``, ``Arial Bold.ttf``, ``Arial Italic.ttf``, ``Arial Bold Italic.ttf``)
or install Arial on the system. Without Arial the figures use Liberation Sans (same letter widths) or, failing that,
DejaVu Sans, and a warning is given once.
"""

import hashlib
import itertools
import re
import warnings
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from .. import config

MM = 1.0 / 25.4                     # inches per mm
PT = 25.4 / 72.0                    # mm per point

# ------------------------------------------------------------------------------------------------ sizes
WIDTH = {"single": 85.0, "onehalf": 114.0, "full": 174.0}      # mm; column widths of the journal
MAX_HEIGHT = 225.0                                             # mm; leaves room for the legend on the same page

FS_LETTER = 12.0      # pt, bold: panel letters
FS_TITLE = 10.0       # pt: axis labels, keys, headings inside a panel
FS_SMALL = 8.0        # pt: tick labels, notes, legends inside a panel
FS_MIN = 7.0          # pt: tick labels of grids of small maps only; nothing is smaller
# Subscripts and superscripts: matplotlib sets a script at 0.7 of the size of its text, so a subscript in 8 pt text
# would be 5.6 pt, below FS_MIN. Any text that carries a script (t_prev, tau_1, T_12, FR_2 ...) is therefore set at
# FS_SCRIPT_TEXT (10 pt: axis labels, colour keys, keys inside a panel, tick labels that carry a script), which puts
# the script at 7 pt. A script inside a script (e^{-t_prev/tau}) would be 4.9 pt and is not used in a figure. Check a
# saved figure with script_sizes() or svg_type_sizes().
SCRIPT_SCALE = 0.7    # matplotlib's ratio of script to text size (mathtext SHRINK_FACTOR)
FS_SCRIPT_TEXT = FS_TITLE

LW_FRAME = 0.8        # pt: axes frame, ticks, box outlines, error bars, significance brackets
LW_DATA = 1.5         # pt: data lines
LW_REF = 1.2          # pt: reference lines (identity diagonal, fitted line, median of a box)
LW_GUIDE = 0.5        # pt: guide lines inside a plot (grid axes, cross lines)
LW_MINOR = 0.6        # pt: minor ticks, edge of a chance band
TICK = 3.5            # pt: major tick length (minor 2.0), pointing outward

LETTER_X = 1.5        # mm: left edge of a panel letter, from the left edge of its column
LETTER_TOP = 1.5      # mm: top of the capital of the letters of the first row, below the top of the figure
LETTER_CLEAR = 0.5    # mm: least white between a panel letter and a label, tick label or key item it would touch
LETTER_ABOVE = 1.5    # mm: top of the capital above the topmost element of its row's panels
TITLE_PITCH = 1.1 * FS_TITLE * PT     # mm between two lines of 10 pt text (3.88)
CAP12 = FS_LETTER * 0.716 * PT        # mm: height of a 12 pt capital of the house typeface (3.03)
NOTE_UP = 1.0         # mm: lower edge of a note above the box of an axes
KEY = dict(swatch_w=4.8, swatch_h=2.6, gap_text=1.6, gap_item=4.5)    # mm: swatches and gaps of a colour key


# ------------------------------------------------------------------------------------------------ colours
def tint(colour, alpha=0.6):
    """Return the colour as it looks at this opacity on white (the fill tone of a line colour).

    Each channel is rounded down, as the renderer does (#F18D00 gives #F6BA66).
    """
    c = np.array(mpl.colors.to_rgb(colour)) * 255.0
    v = 255.0 - alpha * (255.0 - c)
    return "#%02X%02X%02X" % tuple(int(x + 1e-6) for x in v)


PALETTE = {
    # movement direction relative to the grid axes
    "aligned": "#009FB9",            # lines, text, arrows, the identity diagonal of Fig 2D
    "aligned_fill": "#66C5D5",       # areas: wedges, bars, boxes (the line colour at 60% on white)
    "misaligned": "#F18D00",
    "misaligned_fill": "#F6BA66",
    # grid module of the simulations, and small against large grid scale where a colour is needed
    "module_small": "#9159D6",
    "module_small_fill": "#BD9BE6",
    "module_large": "#402A8C",
    "module_large_fill": "#8C7FBA",
    # estimation method
    "method_location": "#1A6600",    # location-controlled estimate
    "method_location_fill": "#75A366",
    "method_simple": "#858585",      # simple estimate: the neutral mid grey (the baseline has no hue)
    "method_simple_fill": "#B5B5B5", # simple estimate as an area (overlaid polar bars of Fig 7C): the mid grey at 60%
    # adaptation
    "adaptation": "#BF2666",         # with adaptation; strong adaptation; a fitted adaptation model
    "adaptation_fill": "#D87CA3",
    "no_adaptation": "#858585",      # without adaptation; a second, simpler model: neutral mid grey
    # measured data and neutral marks (greys carry no category and are explained in the legend)
    "ink": "#000000",                # text, frames, medians, significance marks, measured curves and points
    "dot": "#3C3C3C",                # dots of a scatter plot, points on a box plot
    "neutral": "#858585",            # bars that are not split by a factor; a baseline condition
    "guide": "#929292",              # guide lines inside a plot, zero line
    "neutral_fill": "#D2D2D2",       # boxes and histograms that are not split by a factor; firing fields in schematics
    "chance": "#DCDCDC",             # band of a null or chance distribution
    "chance_edge": "#8C8C8C",
    "span": "#EBEBEB",               # shaded time spans (inside a firing field)
    "white": "#FFFFFF",              # trajectory and field outlines drawn on a colour map
    "significance": "#000000",       # asterisks and brackets
}
PALETTE = {k: v.upper() for k, v in PALETTE.items()}
assert PALETTE["aligned_fill"] == tint(PALETTE["aligned"]) and PALETTE["misaligned_fill"] == tint(PALETTE["misaligned"])
for _k in ("module_small", "module_large", "method_location", "method_simple", "adaptation"):
    assert PALETTE[_k + "_fill"] == tint(PALETTE[_k]), _k

# Colours that belong to one panel only; they name single cells or animals, not a concept, and are explained in the
# legend of that panel. They are not used anywhere else.
LOCAL = {
    "fig2_tuning_example": "#0000FF",      # Fig 2A, tuning curve and preferred direction of an example cell
    "fig2_tuning_mean": "#1F77B4",         # Fig 2C, mean tuning curve
    "fig2_polar_cross": "#B0B0B0",         # Fig 2B and 2C, cross lines (0.8 pt)
    "fig2_animals": ["#0F4BF7", "#C61E58", "#0FA626", "#581378", "#B775FE", "#354E03"],   # Fig 2D, six animals
    "fig2_animals_other": "#969696",       # Fig 2D, animals without a colour of their own
    "fig1_cell_reference": ("#D2D2D2", "#4D4D4D"),     # Fig 1A and 1C, fill and outline of the reference cell
    "fig1_cell_second": ("#B9A6E8", "#6F52C0"),        # Fig 1A, second cell
    "fig1_cell_third": ("#7A5FC9", "#3F2B8C"),         # Fig 1A, third cell (Module tile)
    "fig1_example_cell_1": ("#F3ABA2", "#D93A2B"),     # Fig 1C, cell preferring the grid orientation
    "fig1_example_cell_2": ("#B5DDAD", "#3E9E38"),     # Fig 1C, cell preferring 60 degrees further
    "fig1_sum": ("#DDDDDD", "#000000"),                # Fig 1C, sum of six cells
}

CMAP = {"rate": "viridis",          # firing rate maps, single cells and summed population (low = dark)
        "rate_under_colour": "gray",  # a rate map that lies under a coloured trajectory (low = black)
        "speed": "plasma"}          # running speed along a trajectory (use the range 0 to 0.9 of the map)

# line styles that repeat the colour, so that a figure also reads in greyscale
LINESTYLE = {"module_small": "-", "module_large": (0, (4, 2)), "adaptation": "-", "no_adaptation": (0, (4, 2)),
             "method_location": "-", "method_simple": "-", "model": "-", "model_alternative": (0, (4, 2))}

# ------------------------------------------------------------------------------------------------ variables
VAR = {
    "t_prev": r"$t_{\mathrm{prev}}$",                  # time since the previous field visit
    "T12": r"$T_{12}$", "T23": r"$T_{23}$",            # intervals between successive fields
    "FR": "FR", "FR1": r"$\mathrm{FR}_{1}$", "FR2": r"$\mathrm{FR}_{2}$", "FR3": r"$\mathrm{FR}_{3}$",
    "FRR": r"$(\mathrm{FR}_{2}-\mathrm{FR}_{3})/(\mathrm{FR}_{2}+\mathrm{FR}_{3})$",
    "alpha": r"$\alpha$", "alpha_odd": r"$\alpha_{\mathrm{odd}}$", "alpha_even": r"$\alpha_{\mathrm{even}}$",
    "tau": r"$\tau$", "tau1": r"$\tau_{1}$", "tau2": r"$\tau_{2}$",
    "beta": r"$\beta$", "beta_n": r"$\beta_{n}$", "hexmod_beta": r"HexMod $\beta$",
    "phi": r"$\varphi$", "theta": r"$\theta$",
    "alignment": r"$\cos[6(\theta-\varphi)]$",
    "proximity": r"$e^{-t_{\mathrm{prev}}/\tau}$",
    "R": r"$R$", "n": r"$n$", "sqrt3": "\u221A3",       # the root sign as a character of the house typeface
    "vm": "vm", "vm_s": "vm/s", "cm_s": "cm/s", "Hz": "Hz", "deg": "°",
}

LABEL = {
    "t_prev": r"$t_{\mathrm{prev}}$ (s)",
    "t_prev_long": r"Time since previous field, $t_{\mathrm{prev}}$ (s)",
    "fr_change": "Firing rate change (%)",
    "fr": "Firing rate (Hz)",
    "fr_ratio": "Firing rate ratio",
    "alpha": r"Adaptation coefficient $\alpha$",
    "alpha_odd": r"$\alpha$, odd minutes", "alpha_even": r"$\alpha$, even minutes",
    "location_input": "Location input", "adaptation_variable": "Adaptation variable",
    "proximity": "Proximity", "proximity_diff": "Proximity,\naligned − misaligned",
    "direction": "Movement direction", "alignment": "Grid alignment",
    "hexmod_beta": r"HexMod $\beta$", "beta": r"$\beta$", "symmetry": r"Symmetry ($n$-fold)",
    "detection": "Detection rate",
    "fr_diff": "Firing rate,\naligned − misaligned (%)",
    "fr_diff_mis": "Firing rate,\nmisaligned − aligned (%)",     # Fig 8D plots this sign
    "beta_pct": r"$\beta$ (% of mean rate)",       # Fig 7B, 7E: beta times 100 over the mean rate
    "best_shift": "Best time shift (ms)",           # Supp Fig 2
    "spatial_deviation": "Spatial deviation (grid scales)", "mean_fr": "Mean firing rate (Hz)",
    "fr_norm": "Firing rate (normalised)",          # Supp Fig 3B to D
    "summed_rate": "Summed rate (% of mean)",       # colour bar of the summed maps of Fig 6B, 6C
    "alignment_formula": "Grid alignment,\n" + r"$\cos[6(\theta-\varphi)]$",   # Fig 4B, label on two lines
    "movement_direction_deg": "Movement direction",  # ticks carry the degree sign (DEG); no unit in the label
    "grid_orientation": "Grid orientation", "grid_scale": "Grid scale", "adaptation_strength": "Adaptation strength",
    "speed_cm": "Speed (cm/s)", "speed_vm": "Speed (vm/s)",
    "x_cm": r"$x$ (cm)", "y_cm": r"$y$ (cm)", "x_vm": r"$x$ (vm)", "y_vm": r"$y$ (vm)",
    "time": "Time (s)", "time_shift": "Time shift (ms)", "spatial_information": "Spatial information",
    "cells": "Cells", "density": "Probability density",
}

TICKS = {
    "aligned": ("Aligned", "Misaligned"),
    "t_prev_bins": ("0 to 1", "1 to 2", "2 to 3", "> 3"),       # ranges are written with "to"
    "interval_order": (r"$T_{23} < T_{12}$", r"$T_{23} > T_{12}$"),
    "adaptation_groups": ("Strong", "Weak", "No"),             # Fig 5: thirds of the adaptation coefficient
    "scale_groups": ("Small", "Medium", "Large"),
    "proximity_groups": ("Top third", "Bottom third"),
    "modules": ("Small module", "Large module"),
    "methods": ("Simple", "Location-controlled"),
    "halves": ("Odd minutes", "Even minutes"),                  # Supp Fig 1A
    "shift": ("Original", "Shifted"),                           # Supp Fig 2
    "shift_titles": ("−120 ms", "0 ms (original)", "+120 ms"),  # Supp Fig 2A maps
    "all_cells": ("All cells",),                                # Fig 5A
    "grid_scale_rat": ("Small grid scale", "Large grid scale"), # Fig 4A headings (recorded cells)
    "scale_rows": ("Small scale", "Large scale"),               # Supp Fig 3A row labels (recorded cells)
    "participants": ("Participant 1", "Participant 2"),         # Fig 6B, 6C summed maps
}

DEG = FuncFormatter(lambda v, _: f"{v:g}°".replace("-", "−"))      # 30° ; the axis label then carries no unit
NUM = FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−"))       # shortest form, true minus sign

# ------------------------------------------------------------------------------------------------ typeface
HOUSE_FONT = "Arial"
FALLBACK_FONTS = ("Liberation Sans", "DejaVu Sans")
FONT_SUFFIXES = (".ttf", ".otf")

_REGISTERED = []           # font files registered from a font folder, in registration order
_FALLBACK_WARNED = False   # the warning about a missing Arial is given once per process


def register_fonts(font_dir=None):
    """Register the TrueType and OpenType files of a folder with matplotlib.

    Parameters
    ----------
    font_dir : str or Path, optional
        Folder with font files; default ``HEXMOD_FONT_DIR`` (nothing is registered when it is not set).

    Returns
    -------
    list of str
        Font files registered by this call (files registered before are skipped).
    """
    folder = config.font_dir() if font_dir is None else Path(font_dir).expanduser()
    if folder is None:
        return []
    if not folder.is_dir():
        warnings.warn(f"font folder {folder} (HEXMOD_FONT_DIR) does not exist")
        return []
    new = []
    for path in sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in FONT_SUFFIXES):
        name = str(path.resolve())
        if name in _REGISTERED:
            continue
        fm.fontManager.addfont(name)
        _REGISTERED.append(name)
        new.append(name)
    return new


def house_font():
    """Return the typeface of the figures, registering the fonts of ``HEXMOD_FONT_DIR`` first.

    Arial when it is in ``HEXMOD_FONT_DIR`` or installed; otherwise Liberation Sans (same letter widths) or DejaVu
    Sans, with one warning per process.
    """
    global _FALLBACK_WARNED
    register_fonts()
    have = {f.name for f in fm.fontManager.ttflist}
    if HOUSE_FONT in have:
        return HOUSE_FONT
    font = next((name for name in FALLBACK_FONTS if name in have), "DejaVu Sans")
    if not _FALLBACK_WARNED:
        warnings.warn(f"Arial is not available; the figures use {font}. Set HEXMOD_FONT_DIR to a folder with the "
                      "Arial TrueType files to match the published figures.")
        _FALLBACK_WARNED = True
    return font


def font_files(font):
    """Return the font files matplotlib uses for a family name (regular, bold, italic, as available)."""
    return sorted({f.fname for f in fm.fontManager.ttflist if f.name == font})


# ------------------------------------------------------------------------------------------------ style
def use_style():
    """Set the matplotlib parameters of the house style; call once before drawing a figure.

    Returns
    -------
    str
        The typeface in use (see ``house_font``).
    """
    font = house_font(); ink = PALETTE["ink"]
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [font, "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": font, "mathtext.it": font + ":italic",
        "mathtext.bf": font + ":bold",
        "mathtext.cal": font,     # unused; set so that matplotlib does not look for a cursive font
        "mathtext.default": "it",
        "font.size": FS_TITLE, "axes.labelsize": FS_TITLE, "axes.titlesize": FS_TITLE,
        "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL, "legend.fontsize": FS_SMALL,
        "lines.linewidth": LW_DATA, "patch.linewidth": 1.0, "axes.linewidth": LW_FRAME, "grid.linewidth": LW_FRAME,
        "xtick.major.width": LW_FRAME, "ytick.major.width": LW_FRAME, "xtick.minor.width": LW_MINOR,
        "ytick.minor.width": LW_MINOR,
        "xtick.major.size": TICK, "ytick.major.size": TICK, "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
        "axes.labelpad": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,     # left and bottom spines only
        "axes.edgecolor": ink, "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink, "text.color": ink,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": False, "legend.handlelength": 1.0, "legend.handleheight": 0.8, "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.4, "legend.borderpad": 0.0,
        "image.interpolation": "none", "image.cmap": CMAP["rate"],
        "figure.dpi": 100, "savefig.dpi": 600,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": True,
    })
    return font


# -------------------------------------------------------------------------------------------- figures and axes in mm
def new_figure(width="full", height_mm=100.0):
    """Return a figure of a journal width ('single', 'onehalf', 'full') or of a width in mm, on a fixed Agg canvas.

    The canvas is not interactive, so the size is not rounded to whole screen pixels.
    """
    w = WIDTH[width] if isinstance(width, str) else float(width)
    if height_mm > MAX_HEIGHT: warnings.warn(f"figure height {height_mm:.0f} mm is above {MAX_HEIGHT:.0f} mm")
    fig = Figure(figsize=(w * MM, height_mm * MM), dpi=600)
    FigureCanvasAgg(fig)
    return fig


def size_mm(fig):
    """Return the width and height of a figure in mm."""
    w, h = fig.get_size_inches(); return w / MM, h / MM


def fx(fig, x_mm):
    """Return a horizontal position in mm from the left edge as a figure fraction."""
    return x_mm / size_mm(fig)[0]


def fy(fig, y_mm):
    """Return a vertical position in mm from the top edge as a figure fraction."""
    return 1.0 - y_mm / size_mm(fig)[1]


def axes_mm(fig, left, top, width, height, full=False, sides=("left", "bottom"), **kw):
    """Add Cartesian axes; left and top of the box in mm from the top left corner of the figure.

    ``full`` and ``sides`` set the frame as in ``box_axes`` (left and bottom spines unless ``full=True``).
    """
    w, h = size_mm(fig)
    ax = fig.add_axes([left / w, 1.0 - (top + height) / h, width / w, height / h], **kw)
    if kw.get("projection") != "polar": box_axes(ax, full=full, sides=sides)
    return ax


def polar_mm(fig, cx, cy, diameter=19.558):
    """Add polar axes with the circular frame centred on (cx, cy) mm (19.558 mm: the frame of the Fig 2 polar plots)."""
    ax = axes_mm(fig, cx - diameter / 2, cy - diameter / 2, diameter, diameter, projection="polar")
    polar_style(ax)
    return ax


def text_mm(fig, x, y, s, **kw):
    """Write figure text at (x, y) mm from the top left corner (left aligned, on the baseline by default)."""
    kw.setdefault("ha", "left"); kw.setdefault("va", "baseline")
    return fig.text(fx(fig, x), fy(fig, y), s, **kw)


# ------------------------------------------------------------------------------------------------ letters, keys
def panel_letter(fig, letter, x=LETTER_X, row_top=LETTER_TOP, clear_of=None, not_above=None, row=None):
    """Write a panel letter: bold capital, 12 pt, no full stop; x: its left edge, row_top: top of the capital (mm).

    Parameters
    ----------
    clear_of : list, optional
        What the letter must not touch: an Axes stands for its y-axis label and its drawn y tick labels, any other
        artist (a text, a key swatch or line) for itself. When the figure is saved, ``settle_letters`` raises the
        letter only as far as needed to clear them by ``LETTER_CLEAR``.
    not_above : float, optional
        Top of the row (mm): a letter (or row line) that would have to rise above it is reported, so that the panel
        is lowered instead.
    row : hashable, optional
        Puts the letter on one line with every other letter of the same row: ``settle_letters`` moves the letters of
        a row together (if one of them has to rise, the whole line rises). ``row_top`` is then the row's letter
        line, ``LETTER_ABOVE`` above the topmost element of the row's panels (see ``panels_top``).

    Without these options the letter is placed at ``row_top`` and never moved.
    """
    t = text_mm(fig, x, row_top + CAP12, letter, fontsize=FS_LETTER, fontweight="bold")
    if clear_of is not None or not_above is not None or row is not None:
        t._hexmod_letter = dict(top=float(row_top), clear_of=list(clear_of or []), not_above=not_above, row=row)
    return t


def add_clearance(letter, *items):
    """Add artists that a panel letter placed with ``clear_of``, ``not_above`` or ``row`` must clear.

    For artists drawn after the letter (the letter is placed first where the order of the figure texts matters).
    """
    letter._hexmod_letter["clear_of"].extend(items)


def panels_top(letter_line, rise=0.0):
    """Return the top (mm) of the topmost element of a row's panels whose letters have their capital top at letter_line.

    It lies ``LETTER_ABOVE`` below the line, plus ``rise`` when the line had to rise to clear a label (the panels then
    move down by that amount and the letters stay on their line).
    """
    return float(letter_line) + LETTER_ABOVE + float(rise)


def extent_mm(fig, artist, renderer=None):
    """Return (left, top, right, bottom) of the window extent of an artist, in mm from the top left of the figure."""
    r = renderer or fig.canvas.get_renderer()
    b = artist.get_window_extent(r); s = 25.4 / fig.dpi; H = size_mm(fig)[1]
    return (b.x0 * s, H - b.y1 * s, b.x1 * s, H - b.y0 * s)


def _letter_obstacles(items):
    out = []
    for it in items:
        if isinstance(it, mpl.axes.Axes):
            if not (it.get_visible() and it.axison and it.yaxis.get_visible()):
                continue
            lab = it.yaxis.label
            if lab.get_visible() and lab.get_text().strip():
                out.append(lab)
            out += [t for t in it.yaxis.get_majorticklabels() + it.yaxis.get_minorticklabels()
                    if t.get_visible() and t.get_text().strip()]
        elif it.get_visible() and not (isinstance(it, mpl.text.Text) and not it.get_text().strip()):
            out.append(it)
    return out


def settle_letters(fig, clear=LETTER_CLEAR, strict=False):
    """Move the panel letters placed with ``clear_of``, ``not_above`` or ``row`` to their final height.

    Each such letter keeps its x and its top, unless it would touch one of its obstacles (horizontal and vertical
    gaps both under ``clear`` mm): it is then raised only as far as needed, so that its baseline stands ``clear`` mm
    above the top of that obstacle. A letter is never lowered. Letters placed with ``row`` are then settled
    together: every letter of a row takes the highest top found for any of them (one line per row). Called by
    ``save_figure``.

    Returns
    -------
    list of dict
        One record per letter (also kept as ``fig._hexmod_letter_report``): ``letter``, ``wanted_top_mm``,
        ``top_mm``, ``raised_mm``, ``raised_for``, ``ok``, ``problem`` and, for letters of a row, ``row`` and
        ``row_line_mm``. A letter (or row line) that would rise above ``not_above`` is not ok: a warning is given,
        or a ValueError raised with ``strict``.
    """
    letters = [t for t in fig.texts if getattr(t, "_hexmod_letter", None) is not None and t.get_visible()]
    if not letters:
        return []
    fig.canvas.draw(); r = fig.canvas.get_renderer(); rep = []; rows = {}
    for t in letters:
        info = t._hexmod_letter; want = top = info["top"]
        l, _, rr, _ = extent_mm(fig, t, r)
        obs = [(o, extent_mm(fig, o, r)) for o in _letter_obstacles(info["clear_of"])]
        by, ok, why = [], True, ""
        for _ in range(len(obs) + 1):       # rising past one obstacle can bring the letter to another one above it
            moved = False
            for o, (ol, ot, orr, ob) in obs:
                if max(ol - rr, l - orr) < clear and max(ot - (top + CAP12), top - ob) < clear:
                    # the letter goes above the obstacle (also when the obstacle starts above the letter's own top,
                    # as a long y label that runs past the top of its frame)
                    top = ot - clear - CAP12
                    by.append(o.get_text() if isinstance(o, mpl.text.Text) else type(o).__name__); moved = True
            if not moved:
                break
        if info.get("row") is not None:     # settled with its row below
            rows.setdefault(info["row"], []).append((t, want, top, by)); continue
        na = info.get("not_above")
        if na is not None and top < na - 1e-6:
            ok, why = False, f"would rise {na - top:.2f} mm above the row top {na:.2f} mm: lower the panel"
        if top != want:
            t.set_y(fy(fig, top + CAP12))
        rep.append(dict(letter=t.get_text(), wanted_top_mm=round(want, 3), top_mm=round(top, 3),
                        raised_mm=round(want - top, 3), raised_for=by, ok=ok, problem=why))
        if not ok:
            msg = f"panel letter {t.get_text()}: {why}"
            if strict:
                raise ValueError(msg)
            warnings.warn(msg)
    for key, members in rows.items():       # one line per row
        line = min(m[2] for m in members)
        lead = [m[0].get_text() for m in members if m[2] <= line + 1e-9 and m[2] < m[1] - 1e-9]
        for t, want, top, by in members:
            ok, why = True, ""
            na = t._hexmod_letter.get("not_above")
            if na is not None and line < na - 1e-6:
                ok, why = False, (f"row {key}: the letter line would rise {na - line:.2f} mm above {na:.2f} mm: "
                                  f"lower the row's panels by {na - line:.2f} mm")
            if line != want:
                t.set_y(fy(fig, line + CAP12))
            rfor = list(by) + [f"row line (letter {x})" for x in lead if x != t.get_text()]
            rep.append(dict(letter=t.get_text(), wanted_top_mm=round(want, 3), top_mm=round(line, 3),
                            raised_mm=round(want - line, 3), raised_for=rfor, ok=ok, problem=why, row=key,
                            row_line_mm=round(line, 3)))
            if not ok:
                msg = f"panel letter {t.get_text()}: {why}"
                if strict:
                    raise ValueError(msg)
                warnings.warn(msg)
    fig._hexmod_letter_report = rep
    return rep


def colour_key(fig, items, right, row_top=LETTER_TOP):
    """Draw swatches with words on one line (baseline one capital height under row_top), ending at x = right (mm).

    Parameters
    ----------
    items : list of (str, colour)
        Words and line colours, left to right. The swatch shows the fill tone (the line colour at 60%).
    """
    base = row_top + CAP12; r = fig.canvas.get_renderer(); x = right; arts = []
    for word, colour in reversed(list(items)):
        t = text_mm(fig, x, base, word, fontsize=FS_TITLE, ha="right")
        wtxt = t.get_window_extent(r).width * 25.4 / fig.dpi
        x0 = x - wtxt - KEY["gap_text"] - KEY["swatch_w"]
        p = Rectangle((fx(fig, x0), fy(fig, base)), KEY["swatch_w"] / size_mm(fig)[0],
                      KEY["swatch_h"] / size_mm(fig)[1],
                      transform=fig.transFigure, facecolor=colour, alpha=0.6, edgecolor="none", linewidth=0)
        fig.add_artist(p); arts += [t, p]; x = x0 - KEY["gap_item"]
    return arts


def line_key(fig, items, right, row_top=LETTER_TOP, fontsize=FS_TITLE, lw=LW_DATA):
    """Draw line samples with words on one line (baseline one capital height under row_top), ending at x = right
    (mm), for things drawn as lines.

    Parameters
    ----------
    items : list of (str, colour, linestyle)
        Left to right. The sample is as long as a swatch and sits at its middle.
    """
    base = row_top + CAP12; r = fig.canvas.get_renderer(); x = right; arts = []
    for word, colour, ls in reversed(list(items)):
        t = text_mm(fig, x, base, word, fontsize=fontsize, ha="right")
        wtxt = t.get_window_extent(r).width * 25.4 / fig.dpi
        x0 = x - wtxt - KEY["gap_text"] - KEY["swatch_w"]; ym = base - KEY["swatch_h"] / 2.0
        ln = Line2D([fx(fig, x0), fx(fig, x0 + KEY["swatch_w"])], [fy(fig, ym), fy(fig, ym)], transform=fig.transFigure,
                    color=colour, linestyle=ls, linewidth=lw, solid_capstyle="butt", dash_capstyle="butt")
        fig.add_artist(ln); arts += [t, ln]; x = x0 - KEY["gap_item"]
    return arts


def solid_key(fig, items, right, row_top=LETTER_TOP):
    """Draw solid swatches (no outline, no transparency) with words on one line, for bars drawn in line tones.

    Parameters
    ----------
    items : list of (str, colour)
        Left to right.
    """
    base = row_top + CAP12; r = fig.canvas.get_renderer(); x = right; arts = []
    for word, colour in reversed(list(items)):
        t = text_mm(fig, x, base, word, fontsize=FS_TITLE, ha="right")
        wtxt = t.get_window_extent(r).width * 25.4 / fig.dpi
        x0 = x - wtxt - KEY["gap_text"] - KEY["swatch_w"]
        p = Rectangle((fx(fig, x0), fy(fig, base)), KEY["swatch_w"] / size_mm(fig)[0],
                      KEY["swatch_h"] / size_mm(fig)[1],
                      transform=fig.transFigure, facecolor=colour, edgecolor="none", linewidth=0)
        fig.add_artist(p); arts += [t, p]; x = x0 - KEY["gap_item"]
    return arts


FS_SUBHEAD = FS_TITLE    # pt: sub-panel headings (column or condition headers inside a panel); numeric notes stay 8 pt
FS_CBAR = FS_SMALL       # pt: colour-bar labels (the bar sits right of a map or below a row of maps)
BOX_WIDTH_MM = 7.6       # mm: drawn width of a box in population box plots


def note_above(ax, s, ha="right"):
    """Write a short note above the box of an axes, 8 pt, its lower edge 1 mm above the box."""
    return ax.annotate(s, (1.0 if ha == "right" else 0.0, 1.0), xytext=(0, NOTE_UP / PT), textcoords="offset points",
                       xycoords="axes fraction", ha=ha, va="bottom", fontsize=FS_SMALL)


def words_above(fig, ax, items, gap_mm=2.0):
    """Write coloured words above the box of an axes, 8 pt, ending at its right edge (a key without swatches).

    Parameters
    ----------
    items : list of (str, colour)
        Left to right.
    """
    r = fig.canvas.get_renderer(); x_pt = 0.0; arts = []
    for word, colour in reversed(list(items)):
        t = ax.annotate(word, (1.0, 1.0), xytext=(x_pt, NOTE_UP / PT), textcoords="offset points",
                        xycoords="axes fraction", ha="right", va="bottom", fontsize=FS_SMALL, color=colour)
        x_pt -= t.get_window_extent(r).width * 72.0 / fig.dpi + gap_mm / PT; arts.append(t)
    return arts


# ------------------------------------------------------------------------------------------------ axes
def box_axes(ax, full=False, sides=("left", "bottom")):
    """Set the frame of a Cartesian axes.

    By default the left and bottom spines only, black, 0.8 pt; ``sides=("left",)`` or ``("bottom",)`` gives one
    spine; ``full=True`` keeps the four-sided box (maps, arena plots). Ticks point outward on the left and at the
    bottom; no grid.
    """
    show = ("left", "right", "bottom", "top") if full else tuple(sides)
    assert set(show) <= {"left", "right", "bottom", "top"}, sides
    for k, s in ax.spines.items(): s.set_visible(k in show); s.set_linewidth(LW_FRAME); s.set_color(PALETTE["ink"])
    ax.grid(False); ax.tick_params(which="both", top=False, right=False)


def zero_line(ax, y=0.0, axis="y"):
    """Draw the one reference line allowed inside a box: solid, guide grey, behind the data."""
    f = ax.axhline if axis == "y" else ax.axvline
    return f(y, color=PALETTE["guide"], linewidth=LW_FRAME, zorder=0.5)


def polar_style(ax, labels=True, cross=False, cross_colour=None, cross_lw=LW_GUIDE):
    """Set a circular frame of 0.8 pt, no radial ticks, angle labels at 0, 90, 180, 270 degrees close to the frame.

    ``cross=True`` keeps the two cross lines through the centre (guide grey, 0.5 pt, unless given).
    """
    ax.spines["polar"].set_linewidth(LW_FRAME); ax.spines["polar"].set_color(PALETTE["ink"])
    ax.set_yticks([])
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]) if (labels or cross) else [])
    ax.set_xticklabels(["0°", "90°", "180°", "270°"] if labels else [""] * 4 if cross else [])
    ax.grid(False)
    if cross: ax.grid(True, axis="x", color=PALETTE["guide"] if cross_colour is None else cross_colour,
                      linewidth=cross_lw)
    ax.tick_params(axis="x", pad=-2)


def even_polar_labels(fig, ax):
    """Right-align the left angle label of a polar plot ('180°') at the distance from the frame of the right label.

    matplotlib centres both labels on points equally far from the frame, so the wider left label would end closer to
    it. Call this after the tick labels are set.
    """
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    labs = ax.get_xticklabels(); locs = np.asarray(ax.get_xticks()); strings = [t.get_text() for t in labs]
    i0 = int(np.argmin(np.abs(locs))); i180 = int(np.argmin(np.abs(locs - np.pi)))
    if not strings[i0] or not strings[i180]: return None
    gap_px = labs[i0].get_window_extent(r).x0 - ax.spines["polar"].get_window_extent(r).x1
    size = labs[i180].get_fontsize(); new = list(strings); new[i180] = ""
    ax.set_xticks(ax.get_xticks()); ax.set_xticklabels(new)
    return ax.annotate(strings[i180], xy=(np.pi, ax.get_ylim()[1]), xycoords="data",
                       xytext=(-gap_px * 72.0 / fig.dpi, 0),
                       textcoords="offset points", ha="right", va="center", fontsize=size, annotation_clip=False)


def polar_bars(ax, centres_deg, counts, width_deg, colours, angle_factor=1.0):
    """Draw a histogram on a circle: bars at 60% opacity with white edges of 1 pt.

    ``angle_factor=6`` maps 60 degrees onto the full circle.
    """
    for c, n, col in zip(centres_deg, counts, colours):
        ax.bar(angle_factor * np.deg2rad(c), n, width=angle_factor * np.deg2rad(width_deg), align="center",
               facecolor=col, edgecolor="white", linewidth=1.0, alpha=0.6)
    ax.set_ylim(0, max(counts))


def polar_guides(ax, angles_deg=(0, 90), rmax=None):
    """Draw thin guide lines through the centre of a polar plot (grid axes or cross lines)."""
    rmax = ax.get_ylim()[1] if rmax is None else rmax
    for a in angles_deg:
        ax.plot(np.deg2rad([a, a + 180]), [rmax, rmax], color=PALETTE["guide"], linewidth=LW_GUIDE, zorder=1,
                solid_capstyle="butt")


def alignment_wedges(ax, orientation_deg, rmax):
    """Draw the background of a polar tuning plot: 30 degree wedges, aligned on the grid axes and misaligned between."""
    for i in range(6):
        ax.bar(np.deg2rad(orientation_deg + 60 * i), rmax, width=np.deg2rad(30), color=PALETTE["aligned"], alpha=0.6,
               linewidth=0)
        ax.bar(np.deg2rad(orientation_deg + 30 + 60 * i), rmax, width=np.deg2rad(30), color=PALETTE["misaligned"],
               alpha=0.6, linewidth=0)


# -------------------------------------------------------------------------------------------- box plots, significance
def boxplot(ax, groups, positions=None, colours=None, width=0.6, points=False, seed=0):
    """Draw a box plot in the house convention.

    Box: first to third quartile. Line: median. Whiskers: most extreme values within 1.5 interquartile ranges of the
    box. No caps, no notches, no outlier marks.

    Parameters
    ----------
    groups : list of array_like
        Values per box (non-finite values are dropped).
    positions : list of float, optional
        x positions (default 0, 1, ...).
    colours : list of colour, optional
        One fill per group (default neutral).
    width : float
        Box width in data units.
    points : bool
        Add every value as a dot (for groups of 50 values or fewer); the horizontal jitter uses
        ``numpy.random.default_rng(seed)``.

    Returns
    -------
    list of (float, float)
        Lower and upper whisker ends per group.
    """
    groups = [np.asarray(g, float)[np.isfinite(np.asarray(g, float))] for g in groups]
    positions = list(range(len(groups))) if positions is None else list(positions)
    colours = [PALETTE["neutral_fill"]] * len(groups) if colours is None else list(colours)
    ink = PALETTE["ink"]
    bp = ax.boxplot(groups, positions=positions, widths=width, whis=1.5, showfliers=False, showcaps=False, notch=False,
                    patch_artist=True, manage_ticks=False,
                    boxprops=dict(linewidth=LW_FRAME, edgecolor=ink), whiskerprops=dict(linewidth=LW_FRAME, color=ink),
                    medianprops=dict(linewidth=LW_REF, color=ink, solid_capstyle="butt"), zorder=3)
    for patch, c in zip(bp["boxes"], colours): patch.set_facecolor(c)
    if points:
        rng = np.random.default_rng(seed)
        for x, g in zip(positions, groups):
            ax.plot(x + rng.uniform(-0.3, 0.3, len(g)) * width, g, linestyle="", marker="o", ms=1.8, mfc=PALETTE["dot"],
                    mec="none", alpha=0.6, zorder=4)
    ends = []
    for i in range(len(groups)):
        lo = bp["whiskers"][2 * i].get_ydata(); hi = bp["whiskers"][2 * i + 1].get_ydata()
        ends.append((float(np.min(lo)), float(np.max(hi))))
    ax.__dict__.setdefault("_hexmod_boxes", []).append(dict(positions=positions, bp=bp, n=[len(g) for g in groups]))
    return ends


def box_stats(groups):
    """Return the numbers a house box plot draws, computed from the values (no drawing).

    Quartiles use linear interpolation (numpy's default, as matplotlib). Whisker ends are the most extreme values
    within 1.5 interquartile ranges of the box.

    Returns
    -------
    list of dict
        Per group: n, median, q1, q3, whisker_low, whisker_high.
    """
    out = []
    for g in groups:
        g = np.asarray(g, float); g = g[np.isfinite(g)]
        q1, med, q3 = np.percentile(g, [25, 50, 75]); iqr = q3 - q1
        lo = g[g >= q1 - 1.5 * iqr].min() if g.size else np.nan; hi = g[g <= q3 + 1.5 * iqr].max() if g.size else np.nan
        out.append(dict(n=int(g.size), median=float(med), q1=float(q1), q3=float(q3), whisker_low=float(min(lo, q1)),
                        whisker_high=float(max(hi, q3))))
    return out


def read_boxes(ax):
    """Read back what the house box plots of an axes drew (median, box edges, whisker ends), in data units.

    Reads the boxes drawn by ``boxplot``; for any other ``ax.boxplot`` call pass its return value to ``read_bp``.
    Used to check that a panel draws the numbers of its source data.
    """
    rows = []
    for rec in ax.__dict__.get("_hexmod_boxes", []):
        rows += read_bp(rec["bp"], rec["positions"])
    return rows


def read_bp(bp, positions=None):
    """Return the numbers drawn by one matplotlib boxplot call (the dict it returns); notched boxes read at corners."""
    rows = []
    nbox = len(bp["medians"]); positions = list(range(nbox)) if positions is None else list(positions)
    for i in range(nbox):
        med = bp["medians"][i].get_ydata(); box = bp["boxes"][i]
        yb = np.asarray(box.get_path().vertices)[:, 1] if hasattr(box, "get_path") else np.asarray(box.get_ydata())
        wl = bp["whiskers"][2 * i].get_ydata(); wh = bp["whiskers"][2 * i + 1].get_ydata()
        rows.append(dict(position=float(positions[i]), median=float(np.mean(med)), q1=float(np.min(yb)),
                         q3=float(np.max(yb)), whisker_low=float(np.min(wl)), whisker_high=float(np.max(wh))))
    return rows


def holm(p):
    """Return Holm step-down adjusted p values (the correction used within a panel)."""
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def sig_stars(p):
    """Return '*' for p < 0.05, '**' for p < 0.01, '***' for p < 0.001, and '' otherwise (no mark)."""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


_STAR = dict(fontsize=FS_TITLE, color=PALETTE["significance"], ha="center", va="baseline")
_STAR_LOW, _STAR_HIGH = 0.40, 0.74        # the asterisk of the house typeface sits between these fractions of its size


def sig_mark(ax, x, y, p, side="above", gap_pt=1.5):
    """Write the asterisks of a test of one group against zero just beyond the whisker end y (above or below).

    Put them on the side away from zero.
    """
    s = sig_stars(p)
    if not s: return None
    dy = gap_pt - _STAR_LOW * FS_TITLE if side == "above" else -gap_pt - _STAR_HIGH * FS_TITLE
    return ax.annotate(s, (x, y), xytext=(0, dy), textcoords="offset points", annotation_clip=False, **_STAR)


def sig_bracket(ax, x0, x1, y, p, gap_pt=1.0):
    """Draw a comparison between two groups: a flat black line of 0.8 pt at height y with the asterisks centred on it.

    A bracket is drawn only for a significant comparison and always carries asterisks.
    """
    s = sig_stars(p)
    if not s: return None
    ln, = ax.plot([x0, x1], [y, y], color=PALETTE["significance"], linewidth=LW_FRAME, solid_capstyle="butt",
                  clip_on=False, zorder=5)
    t = ax.annotate(s, ((x0 + x1) / 2.0, y), xytext=(0, gap_pt - _STAR_LOW * FS_TITLE), textcoords="offset points",
                    annotation_clip=False, **_STAR)
    return ln, t


def pt_to_data(ax, pt, axis="y"):
    """Return a length in points as a length in data units of an axes (linear scale).

    Call it after the limits and the axes size are set.
    """
    fig = ax.figure; bb = ax.get_position()
    if axis == "y":
        lo, hi = ax.get_ylim(); length_pt = bb.height * fig.get_size_inches()[1] * 72.0
    else:
        lo, hi = ax.get_xlim(); length_pt = bb.width * fig.get_size_inches()[0] * 72.0
    return pt * (hi - lo) / length_pt


def box_width_data(ax, width_mm=None):
    """Return the box width in x data units that gives a drawn width of width_mm (default ``BOX_WIDTH_MM``).

    Set the x limits and the axes size first.
    """
    return pt_to_data(ax, (BOX_WIDTH_MM if width_mm is None else width_mm) / PT, axis="x")


def bracket_stack(ax, pairs, pvals, y0, step_pt=9.5, order="span", gap_pt=1.0):
    """Draw several comparisons between groups above a box plot, stacked upwards from y0 (data units).

    One flat bracket per significant pair, one level every step_pt points (a bracket with its asterisks is about 8 pt
    high). Pairs that are not significant are left out and take no level. ``order='span'`` draws the shortest pairs
    lowest; ``'given'`` keeps the order of pairs. Set the y limits and the axes size first.

    Returns
    -------
    tuple
        (list of (pair, p, y) drawn, the y of the highest level or None).
    """
    pairs = [tuple(p) for p in pairs]; pvals = list(pvals)
    idx = list(range(len(pairs)))
    if order == "span": idx.sort(key=lambda i: (abs(pairs[i][1] - pairs[i][0]), min(pairs[i])))
    step = pt_to_data(ax, step_pt); drawn = []; level = 0
    for i in idx:
        if not sig_stars(pvals[i]): continue
        y = y0 + level * step
        sig_bracket(ax, pairs[i][0], pairs[i][1], y, pvals[i], gap_pt=gap_pt)
        drawn.append((pairs[i], float(pvals[i]), float(y))); level += 1
    return drawn, (y0 + (level - 1) * step if level else None)


# ------------------------------------------------------------------------------------------------ maps
def field_outlines(ax, field_loc, colour=None, lw=LW_FRAME, numbers=None, number_colour=None, number_size=FS_SMALL,
                   gap_mm=0.6, angles_deg=(45, 135, -45, -135, 90, -90, 0, 180), zorder=4):
    """Draw circular field outlines on a map, and optionally field numbers just outside chosen outlines.

    Parameters
    ----------
    field_loc : array_like, shape (k, 3)
        Rows (centre x, centre y, radius) in data units; the row index is the field number.
    colour : colour, optional
        Outline colour (default white, for a colour map; black on white).
    numbers : None, 'all' or list of int
        Fields to number. A number is set gap_mm outside its outline, at the first of angles_deg where it stays
        inside the axes, so that no number is cut by the frame.
    """
    from matplotlib.patches import Circle
    colour = PALETTE["white"] if colour is None else colour
    number_colour = colour if number_colour is None else number_colour
    field_loc = np.asarray(field_loc, float); arts = []
    for cx, cy, r in field_loc[:, :3]:
        c = Circle((cx, cy), r, fill=False, edgecolor=colour, linewidth=lw, zorder=zorder)
        ax.add_patch(c); arts.append(c)
    if numbers is None: return arts
    ids = range(len(field_loc)) if numbers == "all" else [int(k) for k in numbers]
    x0, x1 = sorted(ax.get_xlim()); y0, y1 = sorted(ax.get_ylim())
    fig = ax.figure; bb = ax.get_position(); W, H = fig.get_size_inches() * 25.4
    mm_x = (x1 - x0) / (bb.width * W); mm_y = (y1 - y0) / (bb.height * H)      # data units per mm
    half_w = 0.5 * 0.6 * number_size * PT * 2; half_h = 0.5 * 0.72 * number_size * PT    # rough half size of two digits
    for k in ids:
        cx, cy, r = field_loc[k, :3]
        for a in angles_deg:
            ux, uy = np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))
            px = cx + ux * (r + gap_mm * mm_x + half_w * mm_x * abs(ux))
            py = cy + uy * (r + gap_mm * mm_y + half_h * mm_y * abs(uy))
            if x0 + half_w * mm_x <= px <= x1 - half_w * mm_x and y0 + half_h * mm_y <= py <= y1 - half_h * mm_y: break
        arts.append(ax.text(px, py, f"{k:d}", ha="center", va="center", fontsize=number_size, color=number_colour,
                            zorder=zorder + 1, clip_on=False))
    return arts


def spikes_on_map(ax, x, y, size=1.6, rim=0.3, zorder=5):
    """Draw spike positions on a colour map: black dots with a thin white rim."""
    return ax.plot(x, y, linestyle="", marker="o", ms=size, mfc=PALETTE["ink"], mec=PALETTE["white"], mew=rim,
                   zorder=zorder)[0]


# ------------------------------------------------------------------------------------------------ stacked time courses
def stacked_traces(fig, left, top, width, heights, t, series, labels, window, visits=None, spikes_t=None, gap=1.5,
                   spike_row=2.2, number_row=True, xticks=None, xlabel=None, label_pad=None):
    """Draw time courses for one time window, stacked on one time axis (Fig 3B, 4B).

    Parameters
    ----------
    left, top, width : float
        Box of the stack in mm from the top left corner of the figure.
    heights : list of float
        Height of each trace box (mm), top to bottom; ``gap`` is the space between boxes (mm).
    series : list of array_like or None
        One 1-D array per box (drawn in black, 1.5 pt), or None (the caller draws into that box).
    labels : list of str
        y labels.
    visits : array_like, optional
        Rows (field number, t_enter, t_exit); each visit is shaded in every box and, with ``number_row``, its field
        number is written above the stack.
    spikes_t : array_like, optional
        Spike times, drawn once as a row of black ticks spike_row mm high just above the first box.

    Returns
    -------
    dict
        ``axes`` (list), ``spike_ax`` and ``numbers``; the caller sets y limits and ticks of each box.
    """
    t = np.asarray(t, float); w0, w1 = window; m = (t >= w0) & (t <= w1)
    y = top; out = dict(axes=[], spike_ax=None, numbers=[])
    if spikes_t is not None:
        sa = axes_mm(fig, left, y, width, spike_row); sa.set_xlim(w0, w1); sa.set_ylim(0, 1); sa.axis("off")
        st = np.asarray(spikes_t, float); st = st[(st >= w0) & (st <= w1)]
        sa.vlines(st, 0.1, 0.9, color=PALETTE["ink"], linewidth=LW_MINOR)
        out["spike_ax"] = sa; y += spike_row + 0.6
    for i, h in enumerate(heights):
        ax = axes_mm(fig, left, y, width, h); ax.set_xlim(w0, w1)
        if visits is not None:
            for fid, a, b in visits:
                ax.axvspan(max(a, w0), min(b, w1), color=PALETTE["span"], linewidth=0, zorder=0)
        if series[i] is not None:
            ax.plot(t[m], np.asarray(series[i])[m], color=PALETTE["ink"], linewidth=LW_DATA, zorder=3,
                    solid_joinstyle="round")
        if labels and labels[i]:
            ax.set_ylabel(labels[i], linespacing=1.1,
                          labelpad=label_pad if label_pad is not None else mpl.rcParams["axes.labelpad"])
        if i < len(heights) - 1:   # one time axis: ticks on the last box only
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        if xticks is not None: ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(NUM); out["axes"].append(ax); y += h + gap
    if xlabel: out["axes"][-1].set_xlabel(xlabel)
    if visits is not None and number_row:
        top_ax = out["spike_ax"] or out["axes"][0]
        for fid, a, b in visits:
            xm = 0.5 * (max(a, w0) + min(b, w1))
            out["numbers"].append(top_ax.annotate(f"{int(fid)}", (xm, 1.0), xycoords=("data", "axes fraction"),
                                                  xytext=(0, NOTE_UP / PT), textcoords="offset points", ha="center",
                                                  va="bottom", fontsize=FS_SMALL, color=PALETTE["ink"],
                                                  annotation_clip=False))
    return out


# ------------------------------------------------------------------------------------------------ other plot kinds
def histogram(ax, x, bins, colour=None, fill=None, **kw):
    """Draw a histogram as filled steps without bar edges, with the outline in the line colour."""
    colour = PALETTE["ink"] if colour is None else colour
    fill = PALETTE["neutral_fill"] if fill is None else fill
    n, edges, _ = ax.hist(x, bins=bins, histtype="stepfilled", facecolor=fill, edgecolor="none", linewidth=0, zorder=2,
                          **kw)
    ax.stairs(n, edges, color=colour, linewidth=LW_FRAME, zorder=3)
    return n, edges


def sem_band(ax, x, mean, sem, colour=None, **kw):
    """Draw a mean as a line of 1.5 pt with its s.e.m. as a band of the same colour at 30%, without an edge."""
    colour = PALETTE["ink"] if colour is None else colour
    ax.fill_between(x, np.asarray(mean) - np.asarray(sem), np.asarray(mean) + np.asarray(sem), color=colour, alpha=0.3,
                    linewidth=0, zorder=2)
    return ax.plot(x, mean, color=colour, linewidth=LW_DATA, zorder=3, **kw)[0]


def errorbars(ax, x, y, err, colour=None, **kw):
    """Draw points with error bars: no caps, 0.8 pt, marker of 3 pt."""
    colour = PALETTE["ink"] if colour is None else colour
    return ax.errorbar(x, y, yerr=err, linestyle="", marker="o", ms=3.0, mfc=colour, mec=colour, mew=0, ecolor=colour,
                       elinewidth=LW_FRAME, capsize=0, zorder=4, **kw)


SCATTER_MANY = dict(s=3.4, color=PALETTE["dot"], alpha=0.6, linewidths=0, zorder=3)                 # hundreds of points
SCATTER_FEW = dict(marker="o", linestyle="", ms=3.0, mfc=mpl.colors.to_rgba(PALETTE["dot"], 0.65), mec=PALETTE["dot"],
                   mew=0.5, zorder=4, clip_on=False)                         # up to about 30 points (with ax.plot)


def chance_band(ax, x, upper, lower=0.0):
    """Draw the band of a chance distribution from lower to upper, with its upper edge as a thin line."""
    ax.fill_between(x, lower, upper, color=PALETTE["chance"], linewidth=0, zorder=1)
    return ax.plot(x, upper, color=PALETTE["chance_edge"], linewidth=LW_MINOR, zorder=2, solid_joinstyle="round")[0]


def rate_map(ax, rate, extent, under_colour=False, **kw):
    """Draw a firing rate map: viridis, nearest neighbour, origin at the lower left, square pixels, full box.

    ``rate`` is indexed [x, y] and is shown transposed.
    """
    im = ax.imshow(np.asarray(rate).T, extent=extent, origin="lower", interpolation="nearest",
                   cmap=CMAP["rate_under_colour"] if under_colour else CMAP["rate"], **kw)
    ax.set_aspect("equal"); box_axes(ax, full=True)          # maps keep the full box
    return im


def colour_bar(fig, im, left, top, width, height, label=None, orientation="vertical", ticks=None):
    """Add a colour bar in its own axes (mm from the top left corner), with a 0.8 pt frame and its label in 8 pt."""
    w, h = size_mm(fig)
    cax = fig.add_axes([left / w, 1.0 - (top + height) / h, width / w, height / h])
    cb = fig.colorbar(im, cax=cax, orientation=orientation, ticks=ticks)
    cb.outline.set_linewidth(LW_FRAME); cb.ax.tick_params(labelsize=FS_SMALL, width=LW_FRAME, length=2.0)
    if label: cb.set_label(label, fontsize=FS_SMALL, labelpad=2.0)
    return cb


# ------------------------------------------------------------------------------------------------ saving
def type_sizes(fig):
    """Return the type sizes used in a figure (pt), smallest first."""
    return sorted({round(t.get_fontsize(), 2) for t in fig.findobj(mpl.text.Text)
                   if t.get_visible() and t.get_text().strip()})


def _script_depth(s):
    """Return the deepest level of sub- or superscript in a mathtext string (0: none)."""
    deepest = 0
    for seg in s.split("$")[1::2]:                      # the parts between dollar signs
        stack, level, pending = [], 0, False
        for c in seg:
            if c.isspace():
                continue
            if c in "_^":
                pending = True
            elif c == "{":
                stack.append(level)
                if pending: level += 1; deepest = max(deepest, level)
                pending = False
            elif c == "}":
                level = stack.pop() if stack else 0
            elif pending:                                # a script of one token (a letter, a digit or a command)
                deepest = max(deepest, level + 1); pending = False
    return deepest


def script_sizes(fig):
    """Return [(text, size of its smallest script in pt)] for every visible text that carries a sub- or superscript."""
    out = []
    for t in fig.findobj(mpl.text.Text):
        s = t.get_text()
        if t.get_visible() and s.count("$") >= 2:
            d = _script_depth(s)
            if d: out.append((s, round(t.get_fontsize() * SCRIPT_SCALE ** d, 2)))
    return out


def svg_type_sizes(svg_path):
    """Return every type size (pt) written in a saved SVG, scripts included, smallest first.

    The figure must be saved with ``svg.fonttype`` 'none' (text kept as text).
    """
    import re
    txt = open(svg_path, encoding="utf-8").read()
    return sorted({float(v) for v in re.findall(r"font:\s*(?:italic\s+|bold\s+|oblique\s+)*([0-9.]+)px", txt)})


def save_figure(fig, stem, check=True):
    """Save a figure at its designed size, without cropping, as PDF, SVG and PNG.

    The panel letters placed with ``clear_of``, ``not_above`` or ``row`` are settled first (``settle_letters``).
    PDF with editable text (TrueType, type 42), SVG with text kept as text, PNG at 600 dpi. No creation date is
    written and SVG element ids are derived from a fixed salt. matplotlib derives the id of a clip path that is not
    a rectangle (for example the circle of polar axes) from an object address instead; these ids are renamed after
    their outline once the SVG is written (``stable_svg_clip_ids``). Saving the same figure again therefore gives
    the same files.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    stem : str or Path
        Output path without extension.
    check : bool
        Warn when a text is smaller than ``FS_MIN``.

    Returns
    -------
    list of str
        The three written paths.
    """
    stem = str(stem)
    settle_letters(fig)
    if check:
        sizes = type_sizes(fig)
        if sizes and sizes[0] < FS_MIN:
            warnings.warn(f"{stem}: type of {sizes[0]} pt is below the smallest allowed size ({FS_MIN} pt)")
    with mpl.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "hexmod"}):
        fig.savefig(stem + ".pdf", metadata={"CreationDate": None})
        fig.savefig(stem + ".svg", metadata={"Date": None})
        fig.savefig(stem + ".png", dpi=600)
    stable_svg_clip_ids(stem + ".svg")
    return [stem + e for e in (".pdf", ".svg", ".png")]


_SVG_CLIP = re.compile(r'<clipPath id="([^"]+)">(.*?)</clipPath>', flags=re.S)


def stable_svg_clip_ids(path):
    """Rename the ids of the non-rectangular clip paths of an SVG file after their outline, in place.

    matplotlib names a rectangular clip path after its corners, but a clip path of another shape (for example the
    circle of polar axes) after the memory address of a Python object, so that name changes from run to run. The
    new id hashes the outline of the clip path and the number of identical outlines before it; it has the form of
    matplotlib's ids ('p' and 10 hexadecimal digits). Rectangular clip paths and all other elements are unchanged.

    Parameters
    ----------
    path : str or Path
        SVG file written by matplotlib.

    Returns
    -------
    int
        Number of renamed clip paths.
    """
    path = Path(path)
    text = path.read_bytes().decode("utf-8")
    clips = _SVG_CLIP.findall(text)
    rename, seen = {}, {}
    for cid, body in clips:
        if "<path" not in body:
            continue
        k = seen.get(body, 0)
        seen[body] = k + 1
        rename[cid] = "p" + hashlib.sha256(f"hexmod\n{k}\n{body}".encode("utf-8")).hexdigest()[:10]
    if not rename:
        return 0
    final = [rename.get(cid, cid) for cid, _ in clips]
    if len(set(final)) != len(final):
        raise RuntimeError(f"{path}: clip path ids are not unique after renaming")
    pattern = re.compile(r'(id="|url\(#)(' + "|".join(re.escape(c) for c in rename) + r')(["\)])')
    text = pattern.sub(lambda m: m.group(1) + rename[m.group(2)] + m.group(3), text)
    path.write_bytes(text.encode("utf-8"))
    return len(rename)


# ------------------------------------------------------------------------------------------------ colour vision
# Machado, Oliveira and Fernandes (2009), severity 1.0: red-blind (protan), green-blind (deutan), blue-blind (tritan)
_CVD = {"normal": np.eye(3),
        "protan": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
        "deutan": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                            [-0.011820, 0.042940, 0.968881]]),
        "tritan": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
                            [0.004733, 0.691367, 0.303900]])}
VISION = tuple(_CVD)


def _lin(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _unlin(c): c = np.clip(c, 0, 1); return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def simulate_cvd(colour, kind):
    """Return the colour as seen with a colour vision deficiency ('protan', 'deutan', 'tritan') or in grey ('grey')."""
    c = np.array(mpl.colors.to_rgb(colour))
    if kind == "grey":
        y = float(np.dot([0.2126729, 0.7151522, 0.0721750], _lin(c)))
        return mpl.colors.to_hex(_unlin(np.array([y, y, y]))).upper()
    return mpl.colors.to_hex(_unlin(_CVD[kind] @ _lin(c))).upper()


def _lab(colour):
    X = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]]) @ _lin(np.array(mpl.colors.to_rgb(colour)))
    X = X / np.array([0.95047, 1.0, 1.08883]); f = np.where(X > 216 / 24389, np.cbrt(X), (24389 / 27 * X + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def lightness(colour):
    """Return CIE L* (0 black, 100 white): what is left of a colour in a greyscale print."""
    return float(_lab(colour)[0])


def _de2000(l1, l2):
    L1, a1, b1 = l1; L2, a2, b2 = l2
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2); Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2; C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1 = np.degrees(np.arctan2(b1, a1p)) % 360; h2 = np.degrees(np.arctan2(b2, a2p)) % 360
    dL = L2 - L1; dC = C2p - C1p
    dh = 0.0 if C1p * C2p == 0 else ((h2 - h1 + 180) % 360) - 180
    dH = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2)
    Lb = (L1 + L2) / 2; Cbp = (C1p + C2p) / 2
    if C1p * C2p == 0: hb = h1 + h2
    else: hb = (h1 + h2) / 2 if abs(h1 - h2) <= 180 else ((h1 + h2 + 360) / 2 if h1 + h2 < 360 else (h1 + h2 - 360) / 2)
    T = (1 - 0.17 * np.cos(np.radians(hb - 30)) + 0.24 * np.cos(np.radians(2 * hb))
         + 0.32 * np.cos(np.radians(3 * hb + 6)) - 0.20 * np.cos(np.radians(4 * hb - 63)))
    SL = 1 + 0.015 * (Lb - 50) ** 2 / np.sqrt(20 + (Lb - 50) ** 2); SC = 1 + 0.045 * Cbp; SH = 1 + 0.015 * Cbp * T
    RT = -2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7)) * np.sin(np.radians(60 * np.exp(-((hb - 275) / 25) ** 2)))
    return float(np.sqrt((dL / SL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2 + RT * (dC / SC) * (dH / SH)))


def colour_distance(c1, c2, kind="normal"):
    """Return the CIEDE2000 distance between two colours, with normal vision or a colour vision deficiency.

    About 1 is a just noticeable difference for large patches side by side; thin lines and small marks need 10 or more.
    """
    if kind != "normal": c1, c2 = simulate_cvd(c1, kind), simulate_cvd(c2, kind)
    return _de2000(_lab(c1), _lab(c2))


# the colours a reader has to tell apart: the line tone of every concept and the four steps of grey
DISTINCT = ("aligned", "misaligned", "module_small", "module_large", "method_location", "adaptation", "ink", "dot",
            "neutral", "neutral_fill")
# fill tones stand for the same concept as their line tone, and the paired greys are one step of grey each
SAME_CONCEPT = [("aligned", "aligned_fill"), ("misaligned", "misaligned_fill"), ("module_small", "module_small_fill"),
                ("module_large", "module_large_fill"), ("method_location", "method_location_fill"),
                ("adaptation", "adaptation_fill"), ("method_simple", "method_simple_fill")]
# colours that share a figure (fills as drawn)
FIGURE_SETS = {
    "Fig 2": ["aligned", "aligned_fill", "misaligned", "misaligned_fill", "ink", "dot", "chance"],
    "Fig 3": ["neutral_fill", "ink", "span"],        # spikes and numbers black (white only on the colour map of A)
    "Fig 4": ["aligned", "aligned_fill", "misaligned", "misaligned_fill", "neutral_fill", "ink"],
    "Fig 5": ["neutral_fill", "ink"],
    "Fig 6": ["aligned_fill", "misaligned_fill", "ink"],
    "Fig 7": ["aligned_fill", "misaligned_fill", "neutral_fill", "dot", "ink", "method_simple", "method_simple_fill",
              "method_location", "method_location_fill"],     # no coloured tick labels or frames
    "Fig 8": ["aligned", "misaligned", "ink"],
    "Supp Fig 3": ["module_small", "module_small_fill", "module_large", "module_large_fill", "ink"],
    "Supp Fig 4": ["ink", "adaptation", "no_adaptation"],
}


def _pairs_min(names, colours=None):
    colours = PALETTE if colours is None else colours; skip = {frozenset(p) for p in SAME_CONCEPT}; rows = {}
    for kind in VISION:
        rows[kind] = min((colour_distance(colours[a], colours[b], kind), a, b)
                         for a, b in itertools.combinations(names, 2)
                         if frozenset((a, b)) not in skip and colours[a] != colours[b])
    return rows


def palette_report(verbose=True):
    """Return the smallest colour distances of the palette (normal vision and three deficiencies) and its lightness."""
    out = {"distinct": _pairs_min(DISTINCT)}
    fills = list(DISTINCT) + [b for _, b in SAME_CONCEPT]
    out["with_fills"] = _pairs_min(fills)
    out["figures"] = {k: _pairs_min(v) for k, v in FIGURE_SETS.items()}
    out["lightness"] = {k: round(lightness(PALETTE[k]), 1) for k in fills}
    out["overall_min_distinct"] = min(out["distinct"].values())
    out["overall_min_with_fills"] = min(out["with_fills"].values())
    if verbose:
        fmt = lambda r: "; ".join("%s %.1f (%s / %s)" % ((k,) + r[k]) for k in VISION)  # noqa: E731
        print("Line tones and greys that a reader must tell apart:\n   " + fmt(out["distinct"]))
        print("The same with the fill tones added:\n   " + fmt(out["with_fills"]))
        for k, r in out["figures"].items(): print("%-10s %s" % (k, fmt(r)))
        print("Lightness L* (greyscale):",
              ", ".join("%s %.0f" % kv for kv in sorted(out["lightness"].items(), key=lambda kv: kv[1])))
    return out
