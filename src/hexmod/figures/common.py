"""Helpers shared by the figure modules.

Inputs
    ``find_input`` locates a result file: first in the results folder, then (for the simulation results) in the
    shipped ``results_precomputed/``. A missing file raises ``MissingInput``, which ``step30_figures.py``
    reports as a skipped figure.

Outputs (in the figure folder ``figures/<Figure>/`` of the results folder)
    ``save_outputs`` writes ``<stem>.pdf`` (editable TrueType text), ``<stem>.svg`` (text kept as text),
    ``<stem>.png`` (600 dpi) and ``positions.csv`` (every axes box and every visible text, in mm from the bottom
    left corner of the figure). ``write_source_data`` writes the values drawn in a panel as
    ``source_data/<Figure><panel>_<description>.csv``.

Drawing helpers
    ``draw_triangular_lattice_lines`` (a grid lattice over a rate map), ``polar_hist`` (a histogram of angles on
    polar axes) and ``plot_trajectory_segment`` (a stretch of path over grid fields, optionally coloured by a
    variable and with time courses below).

Text drawn into a figure follows the repository text rules: no en or em dashes (the minus sign U+2212 is used for
negative numbers). ``save_outputs`` and ``write_source_data`` refuse text that breaks them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import patches
from matplotlib.collections import LineCollection
from matplotlib.text import Text

from .. import config
from .. import io as hio
from . import FIGURES, style

PathLike = Union[str, Path]

#: Characters that the text rules forbid in figure text and source data (dashes other than the hyphen).
FORBIDDEN_CHARACTERS = {"\u2012": "figure dash", "\u2013": "en dash", "\u2014": "em dash", "\u2015": "horizontal bar",
                        "\u2e3a": "two-em dash", "\u2e3b": "three-em dash", "\ufe58": "small em dash"}

POSITION_COLUMNS = ["element", "panel", "name", "text", "x0_mm", "y0_mm", "x1_mm", "y1_mm"]

_PRECOMPUTED: Dict[str, object] = {"dir": None, "prefer": False}
_USED_INPUTS: List[Path] = []
_PANEL_RE = re.compile(r"^[A-Z]{0,3}$")
_DESCRIPTION_RE = re.compile(r"^[A-Za-z0-9]+(_[A-Za-z0-9]+)*$")


class MissingInput(FileNotFoundError):
    """A result file needed by a figure does not exist (the step that writes it has not been run)."""


# ------------------------------------------------------------------------------------------------------------ inputs
def set_precomputed(path: Optional[PathLike] = None, prefer: bool = False) -> None:
    """Set where shipped simulation results are found and whether they are preferred.

    Parameters
    ----------
    path : str or Path, optional
        Folder laid out like the results folder (default ``<repo>/results_precomputed``).
    prefer : bool
        Use the shipped files for ``sim/`` inputs even when the results folder holds a file of the same name.
    """
    _PRECOMPUTED["dir"] = Path(path).expanduser() if path is not None else None
    _PRECOMPUTED["prefer"] = bool(prefer)


def precomputed_dir() -> Path:
    """Return the folder of shipped simulation results in use."""
    folder = _PRECOMPUTED["dir"]
    return Path(folder) if folder is not None else config.precomputed_dir()


def find_input(results_dir: PathLike, relpath: str, alternatives: Sequence[str] = ()) -> Path:
    """Return the path of a result file, given relative to the results folder.

    Files under ``sim/`` are taken from the shipped ``results_precomputed/`` when they are absent from the results
    folder (or always, after ``set_precomputed(prefer=True)``).

    Parameters
    ----------
    results_dir : str or Path
        The results folder.
    relpath : str
        Path of the file relative to ``results_dir``.
    alternatives : sequence of str
        Other names of the same file (for example its ``.gz`` copy), tried in order after ``relpath`` in each place.

    Raises
    ------
    MissingInput
        When the file exists in neither place under any of its names.
    """
    names = [Path(name).as_posix() for name in (relpath, *alternatives)]
    relpath = names[0]
    is_sim = relpath.startswith("sim/")
    local = [Path(results_dir) / name for name in names]
    shipped = [precomputed_dir() / name for name in names] if is_sim else []
    first = lambda paths: next((path for path in paths if path.exists()), None)  # noqa: E731
    found = None
    if is_sim and _PRECOMPUTED["prefer"]:
        found = first(shipped)
    if found is None:
        found = first(local)
    if found is None:
        found = first(shipped)
    if found is not None:
        if found not in _USED_INPUTS:
            _USED_INPUTS.append(found)
        return found
    raise MissingInput(f"{relpath} not found in {results_dir}" + (f" or {precomputed_dir()}" if is_sim else ""))


def used_inputs() -> List[Path]:
    """Return the files returned by ``find_input`` since the last ``reset_used_inputs`` (for the run record)."""
    return list(_USED_INPUTS)


def reset_used_inputs() -> None:
    """Forget the files returned by ``find_input`` so far."""
    _USED_INPUTS.clear()


def figure_dir(results_dir: PathLike, figure: str) -> Path:
    """Return ``<results_dir>/figures/<figure>``."""
    if figure not in FIGURES:
        raise ValueError(f"unknown figure {figure!r}; figures are {', '.join(FIGURES)}")
    return Path(results_dir) / "figures" / figure


# ------------------------------------------------------------------------------------------------------- text rules
def forbidden_characters(text: str) -> List[str]:
    """Return the names of forbidden dash characters found in a string."""
    return [label for ch, label in FORBIDDEN_CHARACTERS.items() if ch in text]


def visible_texts(fig) -> List[Tuple[Text, str, object]]:
    """Return the texts a figure draws, as (text, kind, owner axes or None), in drawing order of their owners.

    Only visible texts with a non-empty string are returned; for tick labels only those of ticks inside the view
    limits. ``kind`` is one of 'figure text', 'axes title', 'xlabel', 'ylabel', 'xtick', 'ytick', 'offset',
    'text' (axes texts and annotations) and 'legend'. Call after ``fig.canvas.draw()``.
    """
    out: List[Tuple[Text, str, object]] = []

    def add(t, kind, owner):
        if isinstance(t, Text) and t.get_visible() and t.get_text().strip():
            out.append((t, kind, owner))

    for t in fig.texts:
        add(t, "figure text", None)
    for a in fig.artists:
        add(a, "figure text", None)
    for leg in fig.legends:
        for t in leg.get_texts():
            add(t, "legend", None)
        if leg.get_title() is not None:
            add(leg.get_title(), "legend", None)
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        if ax.axison:
            for axis, name in ((ax.xaxis, "x"), (ax.yaxis, "y")):
                if not axis.get_visible():
                    continue
                add(axis.label, name + "label", ax)
                update = getattr(axis, "_update_ticks", None)
                ticks = update() if update is not None else axis.get_major_ticks() + axis.get_minor_ticks()
                for tick in ticks:
                    if tick.get_visible():
                        add(tick.label1, name + "tick", ax)
                        add(tick.label2, name + "tick", ax)
                add(axis.get_offset_text(), "offset", ax)
        for title in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)):
            if title is not None:
                add(title, "axes title", ax)
        for t in ax.texts:
            add(t, "text", ax)
        for a in ax.artists:
            add(a, "text", ax)
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                add(t, "legend", ax)
            add(leg.get_title(), "legend", ax)
    return out


def check_figure_text(fig) -> None:
    """Raise ValueError when a text of the figure contains a forbidden dash character."""
    fig.canvas.draw()
    bad = []
    for t, kind, _ in visible_texts(fig):
        found = forbidden_characters(t.get_text())
        if found:
            bad.append(f"{kind} {t.get_text()!r}: {', '.join(found)}")
    if bad:
        raise ValueError("figure text breaks the text rules (use a hyphen, 'to' or the minus sign U+2212):\n"
                         + "\n".join(bad))


# -------------------------------------------------------------------------------------------------------- positions
def tag(artist, panel: Optional[str] = None, name: Optional[str] = None):
    """Attach a panel letter and a name to an axes or a text, for ``positions.csv``; returns the artist."""
    if panel is not None:
        artist._hexmod_panel = str(panel)
    if name is not None:
        artist._hexmod_name = str(name)
    return artist


def _is_panel_letter(t: Text, kind: str) -> bool:
    s = t.get_text().strip()
    return (kind == "figure text" and len(s) == 1 and s.isupper() and abs(t.get_fontsize() - style.FS_LETTER) < 1e-6
            and str(t.get_fontweight()) in ("bold", "700"))


def positions(fig, decimals: int = 4) -> pd.DataFrame:
    """Return the boxes of all visible axes and texts of a figure, in mm from its bottom left corner.

    Columns ``element`` ('axes' or 'text'), ``panel``, ``name``, ``text``, ``x0_mm``, ``y0_mm`` (lower left
    corner), ``x1_mm``, ``y1_mm`` (upper right corner). Axes boxes are the axes rectangles (without tick labels);
    text boxes are the drawn extents of the strings. ``panel`` is the panel set with ``tag``, else the panel
    letter (12 pt bold figure text) that lies nearest above and to the left of the element's upper left corner.
    ``name`` is the tagged name, else the matplotlib label of an axes or the kind of a text. Line breaks in texts
    are written as ``\\n``.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width_mm, height_mm = style.size_mm(fig)
    px_mm = 25.4 / fig.dpi
    rows = []
    for i, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        bb = ax.get_position()
        if hasattr(ax, "_colorbar"):
            name = "colour bar"
        else:
            name = ax.get_label() if ax.get_label() and not ax.get_label().startswith("<") else f"axes {i}"
        rows.append(dict(element="axes", panel=getattr(ax, "_hexmod_panel", None),
                         name=getattr(ax, "_hexmod_name", name), text="",
                         x0_mm=bb.x0 * width_mm, y0_mm=bb.y0 * height_mm, x1_mm=bb.x1 * width_mm,
                         y1_mm=bb.y1 * height_mm))
    letters = []
    for t, kind, _ in visible_texts(fig):
        ext = t.get_window_extent(renderer)
        row = dict(element="text", panel=getattr(t, "_hexmod_panel", None), name=getattr(t, "_hexmod_name", kind),
                   text=t.get_text().replace("\n", "\\n"),
                   x0_mm=ext.x0 * px_mm, y0_mm=ext.y0 * px_mm, x1_mm=ext.x1 * px_mm, y1_mm=ext.y1 * px_mm)
        if _is_panel_letter(t, kind):
            if row["panel"] is None:
                row["panel"] = t.get_text().strip()
            if not hasattr(t, "_hexmod_name"):
                row["name"] = "panel letter"
            letters.append((row["panel"], row["x0_mm"], row["y1_mm"]))
        rows.append(row)
    tol = 1.0
    for row in rows:
        if row["panel"] is not None:
            continue
        cands = [(top, x0, letter) for letter, x0, top in letters
                 if x0 <= row["x0_mm"] + tol and top >= row["y1_mm"] - tol]
        row["panel"] = min(cands, key=lambda c: (round(c[0], 1), -c[1]))[2] if cands else ""
    df = pd.DataFrame(rows, columns=POSITION_COLUMNS)
    for col in ("x0_mm", "y0_mm", "x1_mm", "y1_mm"):
        df[col] = df[col].astype(float).round(decimals)
    return df


def write_positions(fig, path: PathLike) -> Path:
    """Write ``positions(fig)`` as CSV."""
    return hio.write_csv(positions(fig), path)


def save_outputs(fig, out_dir: PathLike, stem: str, check: bool = True) -> List[Path]:
    """Save a figure as PDF, SVG and PNG and write its ``positions.csv``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    out_dir : str or Path
        Figure folder (``figures/<Figure>`` in the results folder); created if needed.
    stem : str
        File stem, for example ``"Fig3"``.
    check : bool
        Check the text rules and the smallest type size before saving.

    Returns
    -------
    list of Path
        The PDF, SVG, PNG and positions files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if check:
        check_figure_text(fig)
    files = [Path(p) for p in style.save_figure(fig, out_dir / stem, check=check)]
    files.append(write_positions(fig, out_dir / "positions.csv"))
    return files


# ------------------------------------------------------------------------------------------------------ source data
def source_data_path(out_dir: PathLike, figure: str, panel: str, description: str) -> Path:
    """Return ``<out_dir>/source_data/<figure><panel>_<description>.csv`` after checking the name parts.

    ``panel`` is one to three capital letters, or '' for a file that belongs to the whole figure.
    """
    if figure not in FIGURES:
        raise ValueError(f"unknown figure {figure!r}; figures are {', '.join(FIGURES)}")
    if not _PANEL_RE.match(panel):
        raise ValueError(f"panel must be capital letters (or empty), got {panel!r}")
    if not _DESCRIPTION_RE.match(description):
        raise ValueError(f"description must be letters and digits joined by underscores, got {description!r}")
    return Path(out_dir) / "source_data" / f"{figure}{panel}_{description}.csv"


def as_table(table, pad: bool = False) -> pd.DataFrame:
    """Return a DataFrame from a DataFrame or a mapping of column name to 1-D values.

    With ``pad=True`` shorter columns are padded with missing values (written as empty fields); otherwise all
    columns must have the same length.
    """
    if isinstance(table, pd.DataFrame):
        return table
    if not isinstance(table, Mapping):
        raise TypeError("table must be a DataFrame or a mapping of column name to values")
    cols = {}
    for key, values in table.items():
        arr = np.asarray(values)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError(f"column {key!r} is not one-dimensional (shape {arr.shape}); use map_table or long_table")
        cols[str(key)] = arr
    lengths = {len(v) for v in cols.values()}
    if len(lengths) > 1:
        if not pad:
            raise ValueError(f"columns have different lengths {sorted(lengths)}; pass pad=True to pad them")
        n = max(lengths)
        cols = {k: pd.Series(v).reindex(range(n)) for k, v in cols.items()}
    return pd.DataFrame(cols)


def write_source_data(out_dir: PathLike, figure: str, panel: str, description: str, table,
                      pad: bool = False) -> Path:
    """Write the values drawn in a panel as ``source_data/<figure><panel>_<description>.csv``.

    Parameters
    ----------
    out_dir : str or Path
        Figure folder.
    figure, panel, description : str
        Name parts (see ``source_data_path``).
    table : DataFrame or mapping
        The values (see ``as_table``); floats are written at full precision (``%.17g``).
    pad : bool
        Allow columns of different lengths (padded with empty fields).

    Returns
    -------
    Path
    """
    path = source_data_path(out_dir, figure, panel, description)
    df = as_table(table, pad=pad)
    bad = [f"column name {c!r}" for c in df.columns if forbidden_characters(str(c))]
    for c in df.columns:
        if df[c].dtype == object:
            bad += [f"{c}: {v!r}" for v in df[c].dropna().unique() if isinstance(v, str) and forbidden_characters(v)]
    if bad:
        raise ValueError("source data break the text rules (no en or em dashes):\n" + "\n".join(bad[:20]))
    return hio.write_csv(df, path)


def long_table(groups: Mapping[str, Sequence[float]], group_name: str = "group",
               value_name: str = "value") -> pd.DataFrame:
    """Return values of several groups (for example the boxes of a box plot) as one long table.

    Rows keep the order of the groups and of the values within each group.
    """
    names, values = [], []
    for key, vals in groups.items():
        vals = np.asarray(vals).ravel()
        names.extend([key] * len(vals))
        values.append(vals)
    data = np.concatenate(values) if values else np.array([], float)
    return pd.DataFrame({group_name: names, value_name: data})


def map_table(array, x, y, value_name: str = "value", x_name: str = "x", y_name: str = "y") -> pd.DataFrame:
    """Return a 2-D map indexed [x, y] as a long table with one row per bin (x varies slowest).

    Parameters
    ----------
    array : array_like, shape (nx, ny)
    x, y : array_like
        Bin centres (length nx and ny) or bin edges (length nx + 1 and ny + 1, converted to centres).
    """
    a = np.asarray(array)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) == a.shape[0] + 1:
        x = 0.5 * (x[:-1] + x[1:])
    if len(y) == a.shape[1] + 1:
        y = 0.5 * (y[:-1] + y[1:])
    if len(x) != a.shape[0] or len(y) != a.shape[1]:
        raise ValueError(f"bins ({len(x)}, {len(y)}) do not match the map shape {a.shape}")
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return pd.DataFrame({x_name: xx.ravel(), y_name: yy.ravel(), value_name: a.ravel()})


# -------------------------------------------------------------------------------------------------- drawing helpers
CM = 1.0 / 2.54  # inches per cm (default figure sizes of the drawing helpers)


def draw_triangular_lattice_lines(dx, theta_deg, env_size_x, env_size_y, origin=(0, 0), ax=None, line_colour='k'):
    """Draw a triangular (hexagonal) lattice with lines of 0.5 pt joining neighbouring vertices.

    Parameters
    ----------
    dx : float
        Lattice spacing (grid scale).
    theta_deg : float
        Orientation of the lattice (degrees).
    env_size_x, env_size_y : float
        Plot domain; the axes limits are set to [0, env_size_x] and [0, env_size_y], the aspect to equal and the axis
        labels to 'x' and 'y'.
    origin : (float, float)
        Position of one lattice vertex.
    ax : matplotlib Axes, optional
        Axes to draw on (default a new 6 x 6 inch figure).
    line_colour : colour
        Line colour.

    Returns
    -------
    P : ndarray, shape (n, 2)
        Lattice vertices within one spacing of the domain.
    ax : matplotlib Axes
    """
    # lattice basis
    a1 = np.array([dx, 0])
    a2 = np.array([dx/2, dx*np.sqrt(3)/2])

    # rotation matrix
    th = np.deg2rad(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)],
                  [np.sin(th),  np.cos(th)]])
    b1 = R @ a1
    b2 = R @ a2

    # grid indices large enough to cover the domain
    diag = np.hypot(env_size_x, env_size_y)
    nmax = int(np.ceil((env_size_x + env_size_y + diag) / dx)) + 2
    I, J = np.meshgrid(np.arange(-nmax, nmax + 1), np.arange(-nmax, nmax + 1), indexing='ij')

    # lattice points
    P = origin + (I[..., None] * b1 + J[..., None] * b2)
    P = P.reshape(-1, 2)

    # keep only points within one spacing of the domain
    mask = (
        (P[:, 0] >= -dx) & (P[:, 0] <= env_size_x+dx) &
        (P[:, 1] >= -dx) & (P[:, 1] <= env_size_y+dx)
    )
    P = P[mask]

    # lines
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))

    for p in P:
        for v in (b1, b2, b2 - b1):  # connect to 3 nearest neighbours
            q = p + v
            if (-dx <= q[0] <= env_size_x+dx) and (-dx <= q[1] <= env_size_y+dx):
                ax.plot([p[0], q[0]], [p[1], q[1]], linewidth=0.5, color=line_colour)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, env_size_x)
    ax.set_ylim(0, env_size_y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(False)

    return P, ax


def polar_hist(ax, theta_list, theta_range, n_bins):
    """Draw a histogram of angles on polar axes, folded so that theta_range spans the full circle.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes to draw on; Cartesian axes are replaced by polar axes at the same position.
    theta_list : array_like
        Angles in degrees.
    theta_range : (float, float)
        (0, max angle) in degrees; a max angle of 60 maps 60 degrees onto the full circle.
    n_bins : int
        Number of bins.

    Returns
    -------
    bars : BarContainer
    """
    # replace Cartesian axes by polar axes
    fig = ax.figure
    if ax.name != 'polar':
        pos = ax.get_position()
        ax.remove()
        ax = fig.add_axes(pos, projection='polar')
    # bins
    if (theta_range[0] != 0) or (theta_range[1] < 0):
        raise ValueError("theta_range[0] must be 0 for polar plot.")
    fold = 360/theta_range[1]
    counts, bin_edges = np.histogram(theta_list, bins=n_bins, range=theta_range)
    bins = bin_edges * fold * np.pi/180  # adjust bins for folding
    # bin centres
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # bars on the polar axes
    bars = ax.bar(bin_centers, counts, width=(bins[1] - bins[0]), bottom=0.0, align='center', edgecolor='k', alpha=0.7)
    theta_label = []
    for bin in bin_edges[:-1]:
        theta_label.append(f'{bin:.0f}°')
    ax.set_xticks(bins[:-1])
    ax.set_xticklabels(theta_label)
    return bars


def plot_trajectory_segment(
    *,
    # trajectory
    x: np.ndarray = [0],                 # shape (T,), horizontal axis of the plot
    y: np.ndarray = [0],                 # shape (T,), vertical axis of the plot
    t: np.ndarray = [0],                 # shape (T,), increasing (seconds or samples)
    time_window: Tuple[float, float] = [1, 2],  # (t_start, t_end) in the units of t
    # grid fields
    field_loc: Optional[np.ndarray] = None,          # shape (K, 3): [peak_x, peak_y, radius], or a list of such
    show_field_id: bool = True,
    # optional colouring variable (for example running speed)
    colour_var: Optional[np.ndarray] = None,  # shape (T,)
    colour_label: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    # optional background image (for example a rate map or a dwell-time map)
    background: Optional[np.ndarray] = None,
    xbins: Optional[Sequence[float]] = None,        # bin edges or centres of the horizontal axis
    ybins: Optional[Sequence[float]] = None,        # bin edges or centres of the vertical axis
    background_is_row_x: bool = False,              # if True, the array is indexed [x, y]
    background_colourbar: bool = False,
    # optional spikes
    spikes: Optional[np.ndarray] = None,        # shape (T,), spike count or flag per sample
    # optional time courses drawn below the map in a second figure (for example firing rate, speed)
    data_1d: list = None,
    data_1d_label: list = None,
    data_1d_ylim: list = None,
    # optional field visits
    time_all_field: Optional[pd.DataFrame] = None,
    show_middle_visit: bool = False,  # if True, shade the two halves of each field visit in different colours
    # figure and axes
    ax=None,
    s: float = 8.0,                 # scatter size
    line_colour: Optional[str] = 'k',  # line colour of the path (ignored if colour_var is given)
    field_colour: str = ['k', 'r', 'b', 'c'],  # outline colour of each set of fields
    mark_start: bool = False,
    equal_aspect: bool = True,
    xticks: Optional[Sequence[float]] = None,
    yticks: Optional[Sequence[float]] = None,
    show_colorbar: bool = True,
    linewidth: float = 0.75,
    use_linecollection: bool = True,  # draw a colour-graded continuous line instead of coloured points
):
    """Plot a stretch of trajectory over grid fields, optionally coloured by a variable.

    The path within ``time_window`` is drawn as a line (or, with ``colour_var``, as a colour-graded line or coloured
    points). Field outlines are drawn as circles with their row index as number; spikes as red crosses. A background
    image is shown with ``origin='lower'`` and the extent of ``xbins`` and ``ybins`` (pass
    ``background_is_row_x=True`` when it is indexed [x, y]); its colour limits are the 1st and 99th percentiles.
    With ``data_1d``, time courses are drawn in a second figure, with field visits from ``time_all_field`` (columns
    fieldId, t_enter, t_exit) shaded grey.

    Returns
    -------
    list
        [ax] or, with ``data_1d``, [ax, axes of the time courses].
    """
    x = np.asarray(x)
    y = np.asarray(y)
    t = np.asarray(t)
    # time window
    if t.ndim != 1 or x.ndim != 1 or y.ndim != 1:
        raise ValueError("x, y, t must be 1-D arrays of equal length.")
    if not (len(x) == len(y) == len(t)):
        raise ValueError("x, y, t must have the same length.")

    t0, t1 = time_window

    idx = (t >= np.min(time_window)) & (t <= np.max(time_window))
    xw = x[idx]
    yw = y[idx]
    tw = t[idx]
    nseg = len(tw)
    # axes
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4*CM, 4*CM))

    # optional background image
    if background is not None:
        if xbins is None or ybins is None:
            raise ValueError("When providing a background image, also pass xbins and ybins to set its extent.")
        y_min, y_max = (np.min(ybins), np.max(ybins))
        x_min, x_max = (np.min(xbins), np.max(xbins))

        # an array indexed [x, y] is transposed, so that imshow gets rows = vertical axis
        bg = background.T if background_is_row_x else background
        if colour_var is not None:
            cmap = "gray"
        im = ax.imshow(
            bg,
            origin="lower",
            extent=(x_min, x_max, y_min, y_max),
            interpolation="nearest",
            aspect="auto",
            cmap=cmap,
            vmin=np.percentile(bg[~np.isnan(bg)], 1), vmax=np.percentile(bg[~np.isnan(bg)], 99)
        )
        if background_colourbar:
            ax.figure.colorbar(im, ax=ax)

    # colour by variable (line collection or scatter)
    mappable = None
    if colour_var is None:
        # path in one colour
        ax.plot(xw, yw, ms=max(1.0, s * 0.6), color=line_colour, linewidth=linewidth, zorder=3)
    else:
        cw = np.asarray(colour_var)[idx]
        # colour limits
        vmin_eff = np.nanmin(cw) if vmin is None else vmin
        vmax_eff = np.nanmax(cw) if vmax is None else vmax
        if not np.isfinite(vmin_eff) or not np.isfinite(vmax_eff) or vmin_eff == vmax_eff:
            # fall back to finite bounds
            finite = np.isfinite(cw)
            if finite.any():
                vmin_eff = np.nanmin(cw[finite])
                vmax_eff = np.nanmax(cw[finite]) if np.nanmax(cw[finite]) > vmin_eff else vmin_eff + 1.0
            else:
                vmin_eff, vmax_eff = 0.0, 1.0

        if use_linecollection and nseg >= 2:
            # coloured segments
            pts = np.column_stack((xw, yw)).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            cvals = 0.5 * (cw[:-1] + cw[1:])  # colour per segment
            lc = LineCollection(
                segs,
                cmap='viridis',
                norm=mpl.colors.Normalize(vmin=vmin_eff, vmax=vmax_eff),
                linewidths=linewidth,
            )
            lc.set_array(cvals)
            ax.add_collection(lc)
            ax.set_xlim(np.nanmin(x), np.nanmax(x))
            ax.set_ylim(np.nanmin(y), np.nanmax(y))
            mappable = lc
        else:
            sc = ax.scatter(
                xw, yw,
                c=cw,
                s=s,
                cmap=cmap,
                vmin=vmin_eff,
                vmax=vmax_eff,
                edgecolors="none",
            )
            mappable = sc
    # grid fields (circles and numbers)
    if field_loc is not None:
        if isinstance(field_loc, np.ndarray):
            field_loc_all = [field_loc]
        if isinstance(field_loc, list):
            field_loc_all = field_loc
        for field_set, field_loc in enumerate(field_loc_all):
            peaks_x = np.asarray(field_loc[:, 0])
            peaks_y = np.asarray(field_loc[:, 1])
            radii = np.asarray(field_loc[:, 2])
            theta = np.linspace(0.0, 2.0 * np.pi, 361)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            for k in range(len(radii)):
                cx, cy, r = peaks_x[k], peaks_y[k], radii[k]
                ax.plot(cx + r * cos_t, cy + r * sin_t, field_colour[field_set], linewidth=0.5)
                if show_field_id:
                    ax.text(cx, cy, f"{k:d}", ha="center", va="center", fontsize=9, color=field_colour[field_set],
                            zorder=5)
    # spikes within the window
    if spikes is not None:
        spikesw = np.where(spikes[idx])[0]
        if np.sum(spikesw) > 0:
            ax.plot(xw[spikesw], yw[spikesw], "x", ms=1, mew=1.0, zorder=4, color='r')

    # start marker
    if mark_start and len(xw) > 0:
        # the first position that is not missing
        yw_nonan = yw[~np.isnan(yw)]
        xw_nonan = xw[~np.isnan(xw)]
        ax.plot(xw_nonan[0], yw_nonan[0], "yx", ms=7, mew=1.5, zorder=6)

    # axes
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(np.min(xbins), np.max(xbins))
    ax.set_ylim(np.min(ybins), np.max(ybins))
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    if show_colorbar and mappable is not None:
        cbar = ax.figure.colorbar(mappable, ax=ax, pad=0.10, orientation='vertical')
        if colour_label:
            cbar.ax.set_title(colour_label)  # label above the vertical colour bar
    # title with the time window (in the units of t)
    if nseg > 0:  # if there is a path to plot
        ax.set_title(f"time=[{np.min(time_window):.0f}, {np.max(time_window):.0f}]s")
    ax_out = []
    ax_out.append(ax)
    # time courses
    if data_1d is not None:
        import matplotlib.pyplot as plt
        fig2, axes = plt.subplots(len(data_1d), 1, sharex=True, figsize=(6*CM, 2.0*CM*len(data_1d)))
        if len(data_1d) == 1:
            axes = [axes]
        for i, var_to_plot in enumerate(data_1d):
            axes[i].plot(t[idx], var_to_plot[idx], 'k')
            axes[i].set_ylabel(data_1d_label[i])
            if data_1d_ylim is not None:
                axes[i].set_ylim(data_1d_ylim[i])
            axes[i].set_xlim(time_window)

            if spikes is not None:
                spikesw = np.where((spikes > 0) & (t > time_window[0]) & (t < time_window[1]))[0]
                if np.sum(spikesw) > 0:
                    axes[i].plot(t[spikesw], var_to_plot[spikesw], "rx", ms=3, mew=1.0, zorder=5)
        axes[i].set_xlabel("time (s)")
        if time_all_field is not None:
            for ax_idx, ax in enumerate(axes):
                ax2 = ax.twinx()
                # shaded rectangles for the field visits
                time_all_field_subset = time_all_field[(time_all_field['t_exit'] > (time_window[0]))
                                                       & (time_all_field['t_enter'] < (time_window[1]))]
                for _, row in enumerate(time_all_field_subset.itertuples(index=False)):
                    if (not show_middle_visit):
                        t_enter = max(row.t_enter, time_window[0])
                        t_exit = min(row.t_exit, time_window[1])
                        color = (0.5, 0.5, 0.5, 1)
                        rect = patches.Rectangle(
                            (t_enter, 0),  # (x, y)
                            t_exit - t_enter,  # width
                            1,  # height
                            color=color,
                            alpha=0.2,
                        )
                        ax2.add_patch(rect)

                    if show_middle_visit:
                        t_enter = max(row.t_enter, time_window[0])
                        t_exit = min(row.t_exit, time_window[1])
                        t_mid = (t_enter+t_exit)/2
                        color_1st = (0.5, 0.5, 0.5, 1)
                        color_2nd = 'orange'
                        rect = patches.Rectangle(
                            (t_enter, 0),  # (x, y)
                            t_mid - t_enter,  # width
                            1,  # height
                            color=color_1st,
                            alpha=0.2,
                        )
                        ax2.add_patch(rect)

                        rect = patches.Rectangle(
                                (t_mid, 0),  # (x, y)
                                t_exit - t_mid,  # width
                                1,  # height
                                color=color_2nd,
                                alpha=0.2,
                            )
                        ax2.add_patch(rect)

                    if ax_idx == 0:
                        ax2.text(
                            (t_enter + t_exit) / 2,
                            1.0,
                            f"{int(row.fieldId)}",
                            color="r",
                            fontsize=8,
                            ha="center",
                            va="bottom",
                        )
                ax2.set_ylabel("")
                ax2.set_yticks([])  # no y ticks on the field visit axis
        fig2.tight_layout()
        ax_out.append(axes)
    return ax_out
