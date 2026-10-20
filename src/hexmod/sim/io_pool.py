"""Simulated grid cell pools: conversion between the pool table and an npz file.

A pool is a pandas DataFrame with one row per simulated cell. Scalar columns hold one number per
cell (for example ``gridness``). Ragged columns hold one array per cell whose length differs
between cells: the field centres of the underlying lattice (``centers``, k x 2), the peak
coordinates, radii and peak rates of the fields (``peaks_x``, ``peaks_y``, ``radius``,
``peakFR_jitter``).

The npz layout keeps every value exactly and needs no pickle (it loads with
``allow_pickle=False``):

- ``columns``: the column names in table order (unicode array);
- ``<name>``: a scalar column, one value per cell, in its own dtype;
- ``<name>__data`` and ``<name>__shapes``: a ragged column, all arrays raveled in C order and
  concatenated, and one row of int64 dimensions per cell (see ``hexmod.io.pack_ragged``);
- ``index``: the row labels, stored only when they are not 0 to n - 1.

``load_pool(save_pool(table))`` returns a table equal to ``table`` in column order, dtypes, row
labels and every value (``pool_differences`` checks this), so results computed from a loaded pool
are identical to results computed from the table that was saved.

The pools of the two simulated modules are written by step 21 to ``results/sim/pools/``
(``POOL_FILES``); ``load_module_pool`` reads the pool of a module for steps 22 to 24, and
``pool_fingerprint`` identifies a pool by its contents (participant files of step 24 record it).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from hexmod.io import RAGGED_DATA, RAGGED_SHAPES, atomic_path, pack_ragged, sha256_file, unpack_ragged

PathLike = Union[str, Path]

COLUMNS_KEY = "columns"
INDEX_KEY = "index"
_RESERVED = {COLUMNS_KEY, INDEX_KEY}

__all__ = [
    "pool_to_arrays", "pool_from_arrays", "save_pool", "load_pool", "ragged_columns",
    "column_checksums", "pool_summary", "pool_differences", "COLUMNS_KEY", "INDEX_KEY",
    "MODULES", "POOL_FILES", "N_POOL_CELLS", "SHIPPED_LARGE_POOL", "load_module_pool", "pool_fingerprint",
    "fingerprint_from_summary",
]

#: The two simulated modules.
MODULES = ("small", "large")
#: Pool files of the modules written by step 21 under ``results/sim/pools/``.
POOL_FILES = {"small": "small_pool_40cm.npz", "large": "large_pool_70cm.npz"}
#: Cells of each pool of the paper.
N_POOL_CELLS = 100000
#: The shipped large-module pool, under ``inputs/simulation/``.
SHIPPED_LARGE_POOL = "large_module_pool_70cm.npz"


# ------------------------------------------------------------------------------------- helpers
def _is_ragged(values: np.ndarray) -> bool:
    """True when an object column holds one numpy array per row."""
    return values.dtype == object and all(isinstance(v, np.ndarray) for v in values)


def _object_array(items: List[np.ndarray]) -> np.ndarray:
    """One-dimensional object array holding ``items`` (never stacked into a numeric array)."""
    out = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        out[i] = item
    return out


def _has_default_index(table) -> bool:
    import pandas as pd

    idx = table.index
    return isinstance(idx, pd.RangeIndex) and idx.start == 0 and idx.step == 1 and idx.stop == len(table) \
        and idx.name is None


def ragged_columns(table) -> List[str]:
    """Names of the columns of ``table`` that hold one array per row."""
    return [c for c in table.columns if _is_ragged(table[c].to_numpy())]


# ------------------------------------------------------------------------------ table <-> arrays
def pool_to_arrays(table) -> Dict[str, np.ndarray]:
    """Convert a pool table into a dict of plain numpy arrays (the npz layout).

    Parameters
    ----------
    table : pandas.DataFrame
        One row per cell. Columns are numeric or boolean scalars, or object columns holding one
        numpy array per row (all of the same number of dimensions and dtype).

    Returns
    -------
    dict
        ``{"columns": names, <scalar name>: values, <ragged name>__data: ...,
        <ragged name>__shapes: ..., ["index": labels]}``.
    """
    out: Dict[str, np.ndarray] = {}
    names = [str(c) for c in table.columns]
    if len(set(names)) != len(names):
        raise ValueError("column names must be unique")
    for name in names:
        if name in _RESERVED or name.endswith(RAGGED_DATA) or name.endswith(RAGGED_SHAPES):
            raise ValueError(f"column name {name!r} is reserved in the pool layout")
    for col, name in zip(table.columns, names):
        values = table[col].to_numpy()
        if values.dtype != object:
            if values.dtype.kind not in "biuf":
                raise TypeError(f"column {name!r} has unsupported dtype {values.dtype}")
            out[name] = np.ascontiguousarray(values)
        elif _is_ragged(values):
            dtypes = {v.dtype for v in values}
            if len(dtypes) != 1:
                raise TypeError(f"ragged column {name!r} mixes dtypes {sorted(map(str, dtypes))}")
            data, shapes = pack_ragged(list(values), dtype=dtypes.pop())
            out[name + RAGGED_DATA] = data
            out[name + RAGGED_SHAPES] = shapes
        else:
            raise TypeError(f"column {name!r} holds objects that are not numpy arrays")
    out[COLUMNS_KEY] = np.array(names, dtype=str)
    if not _has_default_index(table):
        index = table.index.to_numpy()
        if index.dtype.kind not in "iu":
            raise TypeError("only integer row labels can be stored")
        out[INDEX_KEY] = np.ascontiguousarray(index)
    return out


def pool_from_arrays(arrays: Mapping[str, np.ndarray]):
    """Rebuild the pool table from the arrays written by ``pool_to_arrays``.

    Parameters
    ----------
    arrays : mapping
        Raw npz content (``hexmod.io.load_npz(path, unpack=False)`` or ``dict(np.load(path))``).

    Returns
    -------
    pandas.DataFrame
    """
    import pandas as pd

    names = [str(c) for c in np.asarray(arrays[COLUMNS_KEY]).tolist()]
    data = {}
    n = None
    for name in names:
        if name in arrays:
            col = np.asarray(arrays[name])
        elif name + RAGGED_DATA in arrays:
            col = _object_array(unpack_ragged(arrays[name + RAGGED_DATA], arrays[name + RAGGED_SHAPES]))
        else:
            raise KeyError(f"column {name!r} is missing from the pool arrays")
        if n is None:
            n = len(col)
        elif len(col) != n:
            raise ValueError(f"column {name!r} has {len(col)} rows, expected {n}")
        data[name] = col
    n = 0 if n is None else n
    if INDEX_KEY in arrays:
        index = pd.Index(np.asarray(arrays[INDEX_KEY]))
    else:
        index = pd.RangeIndex(n)
    return pd.DataFrame(data, index=index, columns=names)


# --------------------------------------------------------------------------------------- files
def save_pool(table, path: PathLike, compressed: bool = True) -> Path:
    """Save a pool table as an npz file (see the module docstring for the layout).

    Parameters
    ----------
    table : pandas.DataFrame
        Pool table.
    path : str or Path
        Output file (``.npz``).
    compressed : bool
        Write with ``np.savez_compressed`` (default) or ``np.savez``.

    Returns
    -------
    Path
    """
    arrays = pool_to_arrays(table)
    path = Path(path)
    with atomic_path(path, suffix=".npz") as tmp:
        with open(tmp, "wb") as f:
            (np.savez_compressed if compressed else np.savez)(f, **arrays)
    return path


def load_pool(path: PathLike):
    """Load a pool table written by ``save_pool``.

    Parameters
    ----------
    path : str or Path
        npz file.

    Returns
    -------
    pandas.DataFrame
    """
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return pool_from_arrays(arrays)


# ------------------------------------------------------------------------------------ checks
def _rounded(a: np.ndarray, digits: int) -> np.ndarray:
    """Floating-point values rounded to ``digits`` significant digits.

    The rounding goes through Python's decimal formatting, which is correctly rounded, so it gives
    the same result on every machine.
    """
    values = [float(f"{v:.{digits - 1}e}") for v in a.ravel().tolist()]
    return np.array(values, dtype=a.dtype).reshape(a.shape) + 0.0   # + 0.0 turns -0.0 into 0.0


def _update(h, array: np.ndarray, digits: Optional[int] = None) -> None:
    a = np.asarray(array)
    if digits is not None and a.dtype.kind == "f":
        a = _rounded(a, digits)
    if a.dtype.byteorder == ">" or (a.dtype.byteorder == "=" and np.little_endian is False):
        a = a.astype(a.dtype.newbyteorder("<"))
    a = np.ascontiguousarray(a)
    h.update(f"{a.dtype.kind}{a.dtype.itemsize}:{','.join(map(str, a.shape))};".encode("ascii"))
    h.update(a.tobytes())


def column_checksums(table, digits: Optional[int] = None) -> Dict[str, str]:
    """sha256 of every column of a pool table, independent of the file format.

    A scalar column is hashed as its values; a ragged column as its ``__shapes`` array followed by
    its ``__data`` array (``hexmod.io.pack_ragged``). Each array contributes its dtype kind and
    size, its shape and its little-endian bytes in C order, so equal checksums mean equal dtypes,
    shapes and values.

    Parameters
    ----------
    table : pandas.DataFrame
        Pool table.
    digits : int, optional
        Round floating-point values to this many significant digits before hashing. Such checksums
        do not change with differences in the last digits, as between builds of the numerical
        libraries. None (default) hashes the exact values.

    Returns
    -------
    dict
        ``{column name: sha256 hex digest}`` in column order.
    """
    arrays = pool_to_arrays(table)
    out: Dict[str, str] = {}
    for name in arrays[COLUMNS_KEY].tolist():
        h = hashlib.sha256()
        if name in arrays:
            _update(h, arrays[name], digits)
        else:
            _update(h, arrays[name + RAGGED_SHAPES])
            _update(h, arrays[name + RAGGED_DATA], digits)
        out[name] = h.hexdigest()
    return out


def pool_summary(table, path: PathLike = None, digits: Optional[int] = None) -> dict:
    """Describe a pool table: size, column order, columns with kind, dtype and sha256, and the file checksum.

    Parameters
    ----------
    table : pandas.DataFrame
        Pool table.
    path : str or Path, optional
        npz file of the pool; its size and sha256 are added when given.
    digits : int, optional
        Significant digits of the column checksums (``column_checksums``); when given, the summary
        records them as ``significant_digits``.

    Returns
    -------
    dict
    """
    sums = column_checksums(table, digits)
    ragged = set(ragged_columns(table))
    columns = {}
    for name in sums:
        values = table[name].to_numpy()
        if name in ragged:
            columns[name] = {"kind": "ragged", "dtype": str(values[0].dtype) if len(values) else "float64",
                             "ndim": int(values[0].ndim) if len(values) else 1,
                             "n_values": int(sum(v.size for v in values)), "sha256": sums[name]}
        else:
            columns[name] = {"kind": "scalar", "dtype": str(values.dtype), "sha256": sums[name]}
    out = {"n_cells": int(len(table)), "column_order": list(sums), "columns": columns}
    if digits is not None:
        out["significant_digits"] = int(digits)
    if path is not None:
        out["file_bytes"] = int(Path(path).stat().st_size)
        out["file_sha256"] = sha256_file(path)
    return out


def pool_differences(a, b) -> List[str]:
    """List every difference between two pool tables (an empty list means identical).

    Compares column names and order, row labels, scalar dtypes and values (exactly, NaN equal to
    NaN), and for ragged columns the type, dtype, shape and values of every array.

    Parameters
    ----------
    a, b : pandas.DataFrame
        Pool tables.

    Returns
    -------
    list of str
    """
    diffs: List[str] = []
    if list(a.columns) != list(b.columns):
        return [f"columns differ: {list(a.columns)} against {list(b.columns)}"]
    if len(a) != len(b):
        return [f"row counts differ: {len(a)} against {len(b)}"]
    if type(a.index) is not type(b.index) or not np.array_equal(a.index.to_numpy(), b.index.to_numpy()):
        diffs.append("row labels differ")
    for col in a.columns:
        x, y = a[col].to_numpy(), b[col].to_numpy()
        if x.dtype != y.dtype:
            diffs.append(f"{col}: dtype {x.dtype} against {y.dtype}")
            continue
        if x.dtype != object:
            if x.tobytes() != y.tobytes():
                diffs.append(f"{col}: {int(np.sum(x != y))} values differ")
            continue
        bad = 0
        for u, v in zip(x, y):
            if not (isinstance(u, np.ndarray) and isinstance(v, np.ndarray)) or u.dtype != v.dtype \
                    or u.shape != v.shape or np.ascontiguousarray(u).tobytes() != np.ascontiguousarray(v).tobytes():
                bad += 1
        if bad:
            diffs.append(f"{col}: {bad} arrays differ")
    return diffs


# ------------------------------------------------------------------------------ module pools
def load_module_pool(results_dir: PathLike, module: str) -> Tuple[object, Path]:
    """Pool of a module written by step 21, for steps 22 to 24.

    Parameters
    ----------
    results_dir : str or Path
        ``results/``.
    module : {"small", "large"}
        Module.

    Returns
    -------
    table : pandas.DataFrame
        The pool.
    path : Path
        The file it was read from: ``sim/pools/<POOL_FILES[module]>``, or for the large module the
        shipped ``inputs/simulation/large_module_pool_70cm.npz`` when step 21 has not been run.

    Raises
    ------
    SystemExit
        With a one-line message when the small pool is missing or a pool does not have
        ``N_POOL_CELLS`` cells.
    """
    from hexmod import config

    path = Path(results_dir) / "sim" / "pools" / POOL_FILES[module]
    if not path.is_file():
        if module != "large":
            raise SystemExit(f"error: {path} not found; run scripts/step21_simulation_pools.py first")
        path = config.inputs_dir() / "simulation" / SHIPPED_LARGE_POOL
    table = load_pool(path)
    if len(table) != N_POOL_CELLS:
        raise SystemExit(f"error: {path} holds {len(table)} cells, not {N_POOL_CELLS}; "
                         "run scripts/step21_simulation_pools.py again")
    return table, path


def _fingerprint(order, digests: Mapping[str, str], n_cells: int) -> str:
    h = hashlib.sha256(f"n_cells={int(n_cells)}\n".encode("ascii"))
    for name in order:
        h.update(f"{name}:{digests[name]}\n".encode("ascii"))
    return h.hexdigest()


def pool_fingerprint(table) -> str:
    """sha256 identifying a pool by its contents: its size and the checksum of every column, in order."""
    sums = column_checksums(table)
    return _fingerprint(list(sums), sums, len(table))


def fingerprint_from_summary(summary: Mapping) -> Optional[str]:
    """``pool_fingerprint`` of the pool described by an entry of a ``pool_checksums.json`` file.

    Returns None when the entry lacks the number of cells or a column checksum, or when its
    checksums are of rounded values (``significant_digits``).
    """
    if summary.get("significant_digits") is not None:
        return None
    try:
        order = list(summary["column_order"])
        digests = {name: summary["columns"][name]["sha256"] for name in order}
        return _fingerprint(order, digests, int(summary["n_cells"]))
    except (KeyError, TypeError, ValueError):
        return None
