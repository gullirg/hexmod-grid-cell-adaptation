"""Helpers shared by the figure modules.

Inputs
    ``find_input`` locates a result file: first in the results folder, then (for the simulation results) in the
    shipped ``results_precomputed/``. A missing file raises ``MissingInput``, which ``step30_figures.py``
    reports as a skipped figure.

Outputs (in the figure folder ``figures/<Figure>/`` of the results folder)
    ``save_outputs`` writes ``<stem>.pdf`` (editable TrueType text), ``<stem>.svg`` (text kept as text) and
    ``<stem>.png`` (600 dpi). ``write_source_data`` writes the values drawn in a panel as
    ``source_data/<Figure><panel>_<description>.csv``.

Drawing helpers
    ``draw_triangular_lattice_lines`` (a grid lattice over a rate map).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Mapping, Sequence, Union

import numpy as np
import pandas as pd

from .. import config
from .. import io as hio
from . import FIGURES, style

PathLike = Union[str, Path]

_USED_INPUTS: List[Path] = []
_PANEL_RE = re.compile(r"^[A-Z]{0,3}$")
_DESCRIPTION_RE = re.compile(r"^[A-Za-z0-9]+(_[A-Za-z0-9]+)*$")


class MissingInput(FileNotFoundError):
    """A result file needed by a figure does not exist (the step that writes it has not been run)."""


# ------------------------------------------------------------------------------------------------------------ inputs
def find_input(results_dir: PathLike, relpath: str, alternatives: Sequence[str] = ()) -> Path:
    """Return the path of a result file, given relative to the results folder.

    Files under ``sim/`` are taken from the shipped ``results_precomputed/`` when they are absent from the results
    folder.

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
    shipped = [config.precomputed_dir() / name for name in names] if is_sim else []
    first = lambda paths: next((path for path in paths if path.exists()), None)  # noqa: E731
    found = first(local)
    if found is None:
        found = first(shipped)
    if found is not None:
        if found not in _USED_INPUTS:
            _USED_INPUTS.append(found)
        return found
    raise MissingInput(f"{relpath} not found in {results_dir}" + (f" or {config.precomputed_dir()}" if is_sim else ""))


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


# ---------------------------------------------------------------------------------------------------------- outputs
def save_outputs(fig, out_dir: PathLike, stem: str) -> List[Path]:
    """Save a figure as PDF, SVG and PNG (``style.save_figure``).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    out_dir : str or Path
        Figure folder (``figures/<Figure>`` in the results folder); created if needed.
    stem : str
        File stem, for example ``"Fig3"``.

    Returns
    -------
    list of Path
        The PDF, SVG and PNG files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [Path(p) for p in style.save_figure(fig, out_dir / stem)]


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
            raise ValueError(f"column {key!r} is not one-dimensional (shape {arr.shape}); use map_table")
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
    return hio.write_csv(as_table(table, pad=pad), path)


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
