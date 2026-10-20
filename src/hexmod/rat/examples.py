"""Arrays of the example cells drawn in Fig 2A, 3A, 3B, 4A, 4B, Supp Fig 2A and Supp Fig 3A.

The example cells are listed in ``inputs/rat/example_cells.csv`` (read with
``hexmod.preprocessing.labels.read_example_cells``). Their maps, fields and grid measures
come from the characterisation of step 02; this module recomputes only what the figures draw
on top of it. The ``*_outputs`` functions return the contents of the files of
``rat/examples/`` (docs/PIPELINE.md, step 11):

- ``fig2a_outputs``: direction tuning curves of the Fig 2A cells.
- ``trajectory_outputs`` (Fig 3A, 3B, 4A, 4B), built on ``trajectory_example``: the
  speed-filtered samples of a whole session with spikes, the field visits, the time since the
  last field visit (t_prev), the location input of the last visit and the adaptation variable
  exp(-t_prev / tau) times the location input, scaled to a maximum of 1 (tau = 2 s, as in the
  adaptation fits); the fields are numbered by their first visit in the drawn time window
  (``first_visit_order``). For Fig 4B it also returns the grid alignment of the running
  direction, cos(6 (direction - orientation)), and the direction in (-pi, pi].
- ``supp2a_outputs``, built on ``shifted_maps``: spike positions, fields and grid measures with
  all spikes shifted by the cell's best time shift, by 0 ms and by minus the best shift.
- ``supp3a_outputs``, built on ``lattice_origin``: rate maps, fields and the offset of the perfect
  triangular lattice fitted to the field centres.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ..circular import wrap60
from .characterise import characterise_gridcell
from .fields import extract_field_entry, field_visit_input
from .gridcell import GridCellAnalysis, make_cell_data
from .maps import quantify_field_deviation

__all__ = ["TAU_S", "VISIT_COLUMNS", "example_rows", "trajectory_example", "visits_in_window", "first_visit_order",
           "visits_table", "shifted_maps", "shift_prefix", "lattice_origin", "fig2a_outputs", "trajectory_outputs",
           "supp2a_outputs", "supp3a_outputs"]

#: Time constant (s) of the adaptation variable, as in the adaptation fits.
TAU_S = 2.0

#: Field-visit columns returned by ``fields.extract_field_entry`` and their names in the result files.
VISIT_COLUMNS = {"t_enter": "t_enter", "t_exit": "t_exit", "fieldId": "field", "t_prev": "t_prev",
                 "field_x": "field_x", "field_y": "field_y", "radius": "radius",
                 "direction_prev": "direction_prev_deg", "dur_eff": "duration_s", "FR": "rate_hz",
                 "FR_prevOutfield": "rate_prev_outfield_hz"}


# ----------------------------------------------------------------------------------- building blocks
def example_rows(cells, examples) -> List[int]:
    """Row of each example cell in the cohort table (``row`` column), in the order of ``examples``.

    Raises ``KeyError`` when a cell is not in the cohort.
    """
    index = {(str(s), int(c)): int(r) for s, c, r in zip(cells["session_id"], cells["cell_id"], cells["row"])}
    rows = []
    for s, c in zip(examples["session_id"], examples["cell_id"]):
        key = (str(s), int(c))
        if key not in index:
            raise KeyError(f"example cell {s} {c} is not in the cohort")
        rows.append(index[key])
    return rows


def trajectory_example(session_data, cell_id, animal_id, session_id, field_locations, field_rate_hz,
                       orientation_deg, tau_s=TAU_S, with_alignment=False):
    """Per-sample arrays of one cell over its whole session.

    Parameters
    ----------
    session_data : dict
        The session in the preprocessed trajectories (``results/preprocessed/trajectories.pkl``).
    cell_id : int
        Unit key within the session.
    animal_id, session_id : str
        Labels of the animal and the session.
    field_locations : (K, 3) ndarray
        Field centres x, y and radius (cm) from the characterisation.
    field_rate_hz : (K,) ndarray
        Mean rate inside each field (Hz) from the characterisation.
    orientation_deg : float
        Grid orientation (degrees); used only with ``with_alignment``.
    tau_s : float
        Time constant of the adaptation variable (s).
    with_alignment : bool
        Also return ``grid_alignment`` and ``movedir_rad_pm_pi`` (Fig 4B).

    Returns
    -------
    arrays : dict
        ``x``, ``y`` (cm), ``t`` (s), ``spikes`` (int64 counts), ``speed`` (cm/s),
        ``movedir_rad``, ``t_prev`` (s), ``in_field`` (int64: 0 outside fields, 1 and 2 first and
        second half of a visit), ``location_input``, ``adaptation_variable``; with
        ``with_alignment`` also ``grid_alignment`` and ``movedir_rad_pm_pi``.
    ev_df : pandas.DataFrame
        Field visits as returned by ``fields.extract_field_entry``.
    """
    cell_data = make_cell_data(session_data, cell_id, animal_id, session_id)
    gc_obj = GridCellAnalysis(cell_data)
    x, y, t = gc_obj.x, gc_obj.y, gc_obj.time
    spikes, speed, movedir_rad = gc_obj.spike_array, gc_obj.speed, gc_obj.move_dir
    field_locations = np.asarray(field_locations, float)
    peaks_x, peaks_y, radius = field_locations[:, 0], field_locations[:, 1], field_locations[:, 2]
    visits = extract_field_entry(x, y, t, peaks_x, peaks_y, radius, spikes, dt=0.02)
    ev_df, t_prev_vec, in_field = visits["ev_df"], visits["t_prev_vec"], visits["in_field"]
    inputs = field_visit_input(x, y, t, ev_df, np.asarray(field_rate_hz, float), speed)
    location_weight = inputs["adaptWeight_loc"]
    location_input = inputs["loc_instant"]
    tprev_factor = np.exp(-t_prev_vec / tau_s)
    adaptation_variable = tprev_factor * location_weight / np.max(location_weight)
    arrays = {
        "x": np.asarray(x, float), "y": np.asarray(y, float), "t": np.asarray(t, float),
        "spikes": np.asarray(spikes, dtype=np.int64), "speed": np.asarray(speed, float),
        "movedir_rad": np.asarray(movedir_rad, float).copy(), "t_prev": np.asarray(t_prev_vec, float),
        "in_field": np.asarray(in_field, dtype=np.int64), "location_input": np.asarray(location_input, float),
        "adaptation_variable": np.asarray(adaptation_variable, float),
    }
    if with_alignment:
        # computed on the direction samples as stored (float32 for some datasets), as for the figure
        grid_alignment = np.cos((movedir_rad - orientation_deg * np.pi / 180) * 6)
        direction = np.asarray(movedir_rad, float).copy()
        direction[direction > np.pi] = direction[direction > np.pi] - 2 * np.pi
        arrays["grid_alignment"] = np.asarray(grid_alignment, float)
        arrays["movedir_rad_pm_pi"] = direction
    return arrays, ev_df


def visits_in_window(ev_df, time_window) -> List[int]:
    """Field numbers of the visits that overlap a time window, in order of entry.

    A visit overlaps the window ``[t0, t1]`` when it ends after ``t0`` and starts before ``t1``.
    """
    t0, t1 = float(time_window[0]), float(time_window[1])
    sel = ev_df[(ev_df["t_exit"] > t0) & (ev_df["t_enter"] < t1)]
    return [int(f) for f in sel["fieldId"]]


def first_visit_order(ev_df, time_window, n_fields) -> np.ndarray:
    """Number each field by its first visit within the time window (1, 2, ...); 0 = not visited.

    Returns
    -------
    ndarray of int64, shape (n_fields,)
    """
    sequence = visits_in_window(ev_df, time_window)
    order = {f: i + 1 for i, f in enumerate(dict.fromkeys(sequence))}
    return np.array([order.get(k, 0) for k in range(int(n_fields))], dtype=np.int64)


def visits_table(ev_df):
    """Field visits with the column names of the result files (see ``VISIT_COLUMNS``)."""
    return ev_df.rename(columns=VISIT_COLUMNS)[list(VISIT_COLUMNS.values())]


def shift_prefix(shift_ms) -> str:
    """Key prefix of one spike shift in ``supp2a_shifted_maps.npz`` (for example ``shift_m120ms``)."""
    shift_ms = int(shift_ms)
    if shift_ms > 0:
        return f"shift_p{shift_ms}ms"
    if shift_ms < 0:
        return f"shift_m{-shift_ms}ms"
    return "shift_0ms"


def shifted_maps(session_data, cell_id, animal_id, session_id,
                 shifts_ms: Sequence[int]) -> Dict[str, Dict[str, np.ndarray]]:
    """Spike positions, fields and grid measures of one cell with its spikes shifted in time.

    For each shift the cell is characterised again with every spike time moved by the shift
    (minimum peak distance 2/3 of the scale, prominence 0.4, as for the cohort).

    Parameters
    ----------
    session_data : dict
        The session in the preprocessed trajectories.
    cell_id : int
        Unit key within the session.
    animal_id, session_id : str
        Labels.
    shifts_ms : sequence of int
        Shifts in ms (positive: spikes moved later).

    Returns
    -------
    dict
        ``{prefix: arrays}`` with prefix from ``shift_prefix`` and arrays ``spike_x``,
        ``spike_y`` (positions of the samples with spikes, cm), ``spike_count`` (int64),
        ``field_locations`` (K, 3), ``x_edges``, ``y_edges``, and 0-d ``gridness``, ``scale_cm``,
        ``orientation_deg``. Raises ``ValueError`` when the cell is no longer a grid cell with
        fields at one of the shifts.
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for shift_ms in shifts_ms:
        cell_data = make_cell_data(session_data, cell_id, animal_id, session_id,
                                   spike_shift_s=float(shift_ms) / 1000)
        char = characterise_gridcell({"cell_data": cell_data, "min_dist_btw_peaks_rel": 2 / 3, "prominence": 0.4})
        if char is None or "fieldLoc" not in char:
            raise ValueError(f"cell {session_id} {cell_id} is not a grid cell with fields at a shift of {shift_ms} ms")
        gc_obj = GridCellAnalysis(cell_data)
        spikes = np.asarray(gc_obj.spike_array)
        with_spikes = spikes > 0
        out[shift_prefix(shift_ms)] = {
            "spike_x": np.asarray(gc_obj.x)[with_spikes].astype(float),
            "spike_y": np.asarray(gc_obj.y)[with_spikes].astype(float),
            "spike_count": spikes[with_spikes].astype(np.int64),
            "field_locations": np.asarray(char["fieldLoc"], float),
            "x_edges": np.asarray(char["x_edges"], float),
            "y_edges": np.asarray(char["y_edges"], float),
            "gridness": np.array(float(char["gridness"])),
            "scale_cm": np.array(float(char["scale_cm"])),
            "orientation_deg": np.array(float(char["orientation_deg"])),
        }
    return out


def lattice_origin(field_locations, scale_cm, orientation_deg) -> np.ndarray:
    """Offset (cm) of the perfect triangular lattice fitted to the field centres of a cell.

    Uses ``maps.quantify_field_deviation`` with the cell's grid scale and orientation, as the
    characterisation does.

    Returns
    -------
    (2,) ndarray
    """
    field_locations = np.asarray(field_locations, float)
    origin, _, _, _, _ = quantify_field_deviation(field_locations[:, :-1], lattice_spacing=float(scale_cm),
                                                  lattice_orientation=np.deg2rad(float(orientation_deg)))
    return np.asarray(origin, float)


# ------------------------------------------------------------------------------- result files
def _labels(cells, row):
    """Animal, session and unit key of a cohort row."""
    return cells.loc[row, "animal_id"], cells.loc[row, "session_id"], int(cells.loc[row, "cell_id"])


def fig2a_outputs(cells, dir_maps, rows: Sequence[int]):
    """Fig 2A: table and tuning curves of the example conjunctive cells.

    Parameters
    ----------
    cells : pandas.DataFrame
        ``rat/cohort/cells.csv`` (index equal to ``row``).
    dir_maps : (n_cells, 72) ndarray
        ``rat/cohort/dir_maps.npy``.
    rows : sequence of int
        Cohort rows of the cells, in panel order.

    Returns
    -------
    table : pandas.DataFrame
        ``row``, ``session_id``, ``cell_id``, ``Rayleigh_mag``, ``Rayleigh_angle``,
        ``orientation_deg`` and ``deviation_deg`` (wrap60(Rayleigh_angle - orientation_deg)).
    arrays : dict
        ``dir_maps`` (n, 72) and ``rows`` (n,) int64.
    """
    import pandas as pd

    rows = [int(r) for r in rows]
    sub = cells.loc[rows]
    angle = sub["Rayleigh_angle"].to_numpy()
    orientation = sub["orientation_deg"].to_numpy()
    table = pd.DataFrame({"row": np.asarray(rows, dtype=np.int64), "session_id": sub["session_id"].to_numpy(),
                          "cell_id": sub["cell_id"].to_numpy(dtype=np.int64),
                          "Rayleigh_mag": sub["Rayleigh_mag"].to_numpy(),
                          "Rayleigh_angle": angle, "orientation_deg": orientation,
                          "deviation_deg": wrap60(angle - orientation)})
    return table, {"dir_maps": np.asarray(dir_maps)[rows], "rows": np.asarray(rows, dtype=np.int64)}


def trajectory_outputs(session_data, cells, row: int, field_locations, cells_full, time_window,
                       with_alignment: bool = False):
    """Fig 3A, 3B, 4A, 4B: the contents of ``<name>.npz`` and ``<name>_field_visits.csv``.

    Parameters
    ----------
    session_data : dict
        The cell's session in the preprocessed trajectories.
    cells : pandas.DataFrame
        ``rat/cohort/cells.csv`` (index equal to ``row``).
    row : int
        Cohort row of the cell.
    field_locations : list of (K, 3) arrays
        ``field_locations`` of ``rat/cohort/field_locations.npz``.
    cells_full : dict
        ``rat/cohort/cells_full.npz`` loaded with ``hexmod.io.load_npz``.
    time_window : (2,) array_like
        Drawn time window (s).
    with_alignment : bool
        Store the grid alignment of the running direction and the direction in (-pi, pi] (Fig 4B).

    Returns
    -------
    arrays : dict
        Per-sample arrays of ``trajectory_example``, ``rate_map``, ``x_edges``, ``y_edges``,
        ``field_locations``, ``field_rate_hz``, ``field_order``, ``time_window`` and 0-d
        ``gridness``, ``scale_cm``, ``orientation_deg``, ``row``.
    visits : pandas.DataFrame
        Field visits (``visits_table``).
    sequence : list of int
        Fields visited in the time window, in order of entry.
    """
    row = int(row)
    animal_id, session_id, cell_id = _labels(cells, row)
    fl = field_locations[row]
    arrays, ev_df = trajectory_example(session_data, cell_id, animal_id, session_id, fl,
                                       cells_full["field_rate_hz"][row], cells.loc[row, "orientation_deg"],
                                       with_alignment=with_alignment)
    window = np.asarray(time_window, float)
    arrays.update({
        "rate_map": cells_full["rate_map"][row], "x_edges": cells_full["x_edges"][row],
        "y_edges": cells_full["y_edges"][row],
        "field_locations": fl, "field_rate_hz": cells_full["field_rate_hz"][row],
        "field_order": first_visit_order(ev_df, window, len(fl)), "time_window": window,
        "gridness": np.array(float(cells.loc[row, "gridness"])),
        "scale_cm": np.array(float(cells.loc[row, "scale_cm"])),
        "orientation_deg": np.array(float(cells.loc[row, "orientation_deg"])), "row": np.array(row, dtype=np.int64),
    })
    return arrays, visits_table(ev_df), visits_in_window(ev_df, window)


def supp2a_outputs(session_data, cells, row: int, shift_ms, spatial_information):
    """Supp Fig 2A: the contents of ``supp2a_shifted_maps.npz``.

    The cell is mapped at its best shift (largest spatial information), at 0 and at minus the
    best shift.

    Parameters
    ----------
    session_data : dict
        The cell's session in the preprocessed trajectories.
    cells : pandas.DataFrame
        ``rat/cohort/cells.csv`` (index equal to ``row``).
    row : int
        Cohort row of the cell.
    shift_ms : (n_shifts,) array_like
        The time shifts of step 05 (``rat/prospective/shift_ms.npy``).
    spatial_information : (n_shifts,) array_like
        The cell's spatial information at each shift (its row of ``si_shift.npy``).

    Returns
    -------
    arrays : dict
        ``shift_ms``, ``spatial_information`` and, for each mapped shift, the arrays of
        ``shifted_maps`` under ``<prefix>__<name>``.
    best_shift_ms : int
    """
    row = int(row)
    si = np.asarray(spatial_information, float)
    best = int(np.asarray(shift_ms)[int(np.argmax(si))])
    shifts_ms = list(dict.fromkeys([best, 0, -best]))
    animal_id, session_id, cell_id = _labels(cells, row)
    maps = shifted_maps(session_data, cell_id, animal_id, session_id, shifts_ms)
    arrays = {"shift_ms": np.asarray(shift_ms, dtype=np.int64), "spatial_information": si}
    for prefix, values in maps.items():
        for key, value in values.items():
            arrays[f"{prefix}__{key}"] = value
    return arrays, best


def supp3a_outputs(cells, field_locations, cells_full, rows: Sequence[int], panels: Sequence[int]):
    """Supp Fig 3A: the contents of ``supp3a_cells.csv`` and ``supp3a_cells.npz``.

    Parameters
    ----------
    cells : pandas.DataFrame
        ``rat/cohort/cells.csv`` (index equal to ``row``).
    field_locations : list of (K, 3) arrays
        ``rat/cohort/field_locations.npz``.
    cells_full : dict
        ``rat/cohort/cells_full.npz`` loaded with ``hexmod.io.load_npz``.
    rows, panels : sequences
        Cohort row and panel number (1 to 12) of each cell.

    Returns
    -------
    table : pandas.DataFrame
        ``panel``, ``row``, ``session_id``, ``cell_id``, ``gridness``, ``scale_cm``,
        ``orientation_deg``, ``rate_map_max_hz``, ``env_size_cm`` (last x edge of the rate map),
        ``origin_x_cm``, ``origin_y_cm`` (lattice offset) and ``conjunctive`` (Rayleigh vector length
        above 0.3).
    arrays : dict
        ``pNN__rate_map``, ``pNN__x_edges``, ``pNN__y_edges``, ``pNN__field_locations``,
        ``pNN__origin`` for each panel NN.
    """
    import pandas as pd

    table, arrays = [], {}
    for row, panel in zip(rows, panels):
        row = int(row)
        fl = field_locations[row]
        origin = lattice_origin(fl, cells.loc[row, "scale_cm"], cells.loc[row, "orientation_deg"])
        rate_map = np.asarray(cells_full["rate_map"][row], float)
        p = f"p{int(panel):02d}"
        arrays.update({f"{p}__rate_map": rate_map, f"{p}__x_edges": cells_full["x_edges"][row],
                       f"{p}__y_edges": cells_full["y_edges"][row], f"{p}__field_locations": fl,
                       f"{p}__origin": origin})
        table.append({"panel": int(panel), "row": row, "session_id": cells.loc[row, "session_id"],
                      "cell_id": int(cells.loc[row, "cell_id"]), "gridness": cells.loc[row, "gridness"],
                      "scale_cm": cells.loc[row, "scale_cm"], "orientation_deg": cells.loc[row, "orientation_deg"],
                      "rate_map_max_hz": float(np.nanmax(rate_map)),
                      "env_size_cm": float(cells_full["x_edges"][row][-1]),
                      "origin_x_cm": origin[0], "origin_y_cm": origin[1],
                      "conjunctive": bool(cells.loc[row, "Rayleigh_mag"] > 0.3)})
    return pd.DataFrame(table), arrays
