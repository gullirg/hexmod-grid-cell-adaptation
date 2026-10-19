"""Tests of the figure style, the shared figure helpers and the figure step (no data needed)."""

from __future__ import annotations

import importlib.util
import re
import shutil
import struct
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

import matplotlib.font_manager as fm  # noqa: E402

from hexmod import io as hio  # noqa: E402
from hexmod.figures import FIGURES, MODULES, common, figure_name, module_name, style  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------------------- helpers
def demo_figure(text="Firing rate change (%)"):
    """A small figure in the house style with two panels, their letters on one line above the panels."""
    style.use_style()
    fig = style.new_figure("single", 47.0)
    ax = style.axes_mm(fig, 12.0, 11.0, 25.0, 26.0)
    style.panel_letter(fig, "A", row_top=style.LETTER_TOP, clear_of=[ax], not_above=style.LETTER_TOP, row=1)
    rng = np.random.default_rng(0)
    groups = [rng.normal(m, 1.0, 40) for m in (1.0, 0.0)]
    style.boxplot(ax, groups, colours=[style.PALETTE["aligned_fill"], style.PALETTE["misaligned_fill"]])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-4, 5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(style.TICKS["aligned"])
    ax.set_ylabel(text)
    style.sig_bracket(ax, 0, 1, 4.0, 1e-4)
    axm = style.axes_mm(fig, 52.0, 11.0, 22.0, 22.0, full=True)
    style.panel_letter(fig, "B", x=44.0, row_top=style.LETTER_TOP, clear_of=[axm], not_above=style.LETTER_TOP, row=1)
    im = style.rate_map(axm, rng.random((10, 10)), (0, 100, 0, 100))
    style.colour_bar(fig, im, 76.0, 11.0, 2.0, 22.0, label=style.LABEL["fr"])
    return fig, ax, axm


def png_size(path):
    data = Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def load_step30():
    spec = importlib.util.spec_from_file_location("step30_figures", REPO / "scripts" / "step30_figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------------------------------- constants
def test_style_constants():
    assert (style.FS_LETTER, style.FS_TITLE, style.FS_SMALL, style.FS_MIN) == (12.0, 10.0, 8.0, 7.0)
    assert style.FS_SCRIPT_TEXT * style.SCRIPT_SCALE == pytest.approx(style.FS_MIN)
    assert (style.LW_FRAME, style.LW_DATA, style.LW_REF, style.LW_GUIDE, style.LW_MINOR, style.TICK) == \
        (0.8, 1.5, 1.2, 0.5, 0.6, 3.5)
    assert style.WIDTH == {"single": 85.0, "onehalf": 114.0, "full": 174.0} and style.MAX_HEIGHT == 225.0
    assert (style.LETTER_X, style.LETTER_TOP, style.LETTER_ABOVE, style.LETTER_CLEAR, style.NOTE_UP) == \
        (1.5, 1.5, 1.5, 0.5, 1.0)
    assert style.CAP12 == pytest.approx(3.0311, abs=1e-4) and style.TITLE_PITCH == pytest.approx(3.8806, abs=1e-4)
    assert style.KEY == dict(swatch_w=4.8, swatch_h=2.6, gap_text=1.6, gap_item=4.5)
    assert style.BOX_WIDTH_MM == 7.6
    assert style.TICKS["adaptation_groups"] == ("Strong", "Weak", "No")
    assert style.TICKS["aligned"] == ("Aligned", "Misaligned")
    assert style.PALETTE["aligned"] == "#009FB9" and style.PALETTE["misaligned"] == "#F18D00"
    for line, fill in style.SAME_CONCEPT:
        assert style.PALETTE[fill] == style.tint(style.PALETTE[line])


def test_palette_is_readable():
    report = style.palette_report(verbose=False)
    assert report["overall_min_distinct"][0] > 5.0
    assert set(report["lightness"]) >= {"aligned", "misaligned"}


def test_use_style_parameters():
    font = style.use_style()
    rc = mpl.rcParams
    assert rc["font.sans-serif"][0] == font and rc["mathtext.rm"] == font
    assert rc["pdf.fonttype"] == 42 and rc["svg.fonttype"] == "none" and rc["savefig.dpi"] == 600
    assert not rc["axes.spines.top"] and not rc["axes.spines.right"]
    assert rc["xtick.major.size"] == style.TICK and rc["axes.linewidth"] == style.LW_FRAME


def test_box_axes_despine():
    style.use_style()
    fig = style.new_figure("single", 30.0)
    ax = style.axes_mm(fig, 5, 5, 20, 20)
    assert {k for k, s in ax.spines.items() if s.get_visible()} == {"left", "bottom"}
    ax2 = style.axes_mm(fig, 30, 5, 20, 20, full=True)
    assert all(s.get_visible() for s in ax2.spines.values())
    ax3 = style.axes_mm(fig, 55, 5, 20, 20, sides=("left",))
    assert {k for k, s in ax3.spines.items() if s.get_visible()} == {"left"}


def letter_top_mm(fig, letter):
    """Top of the capital of a panel letter, in mm from the top of the figure."""
    return (1 - letter.get_position()[1]) * style.size_mm(fig)[1] - style.CAP12


def test_panel_letter_geometry():
    style.use_style()
    fig = style.new_figure("full", 60.0)
    letter = style.panel_letter(fig, "C", x=40.0, row_top=20.0)
    w, h = style.size_mm(fig)
    x, y = letter.get_position()
    assert x * w == pytest.approx(40.0) and (1 - y) * h == pytest.approx(20.0 + style.CAP12)
    assert letter.get_fontsize() == 12.0 and letter.get_fontweight() == "bold"
    assert style.panels_top(20.0) == pytest.approx(20.0 + style.LETTER_ABOVE)
    assert style.panels_top(20.0, rise=2.0) == pytest.approx(22.0 + style.LETTER_ABOVE)
    assert style.settle_letters(fig) == []            # a letter placed without options is never moved


def test_letters_of_a_row_share_one_line():
    """A letter that would touch a label rises until its baseline clears it, and the other letters of its row follow."""
    style.use_style()
    fig = style.new_figure("full", 60.0)
    line = 10.0
    label = style.text_mm(fig, 11.0, 14.0, "Label under the letter", fontsize=style.FS_TITLE)
    ax = style.axes_mm(fig, 74.0, style.panels_top(line), 30.0, 30.0)
    a = style.panel_letter(fig, "A", x=10.0, row_top=line, clear_of=[label], row=1)
    b = style.panel_letter(fig, "B", x=60.0, row_top=line, clear_of=[ax], row=1)
    c = style.panel_letter(fig, "C", x=110.0, row_top=line, row=2)
    report = {r["letter"]: r for r in style.settle_letters(fig)}
    want = style.extent_mm(fig, label)[1] - style.LETTER_CLEAR - style.CAP12
    assert want < line
    assert letter_top_mm(fig, a) == pytest.approx(want, abs=1e-9)
    assert letter_top_mm(fig, b) == pytest.approx(want, abs=1e-9)      # B follows the line of its row
    assert letter_top_mm(fig, c) == pytest.approx(line)                # another row keeps its line
    assert report["A"]["raised_for"] == ["Label under the letter"]
    assert report["B"]["raised_for"] == ["row line (letter A)"] and report["C"]["raised_mm"] == 0
    assert all(r["ok"] for r in report.values()) and len(fig._hexmod_letter_report) == 3


def test_letter_that_must_rise_above_its_row_is_reported():
    style.use_style()
    fig = style.new_figure("full", 60.0)
    line = 10.0
    letter = style.panel_letter(fig, "A", x=10.0, row_top=line, not_above=line, row=1)
    label = style.text_mm(fig, 11.0, 14.0, "Label drawn after the letter", fontsize=style.FS_TITLE)
    style.add_clearance(letter, label)
    with pytest.raises(ValueError, match="lower the row's panels"):
        style.settle_letters(fig, strict=True)
    fig2 = style.new_figure("full", 60.0)
    ax = style.axes_mm(fig2, 14.0, style.panels_top(line), 30.0, 30.0)
    style.panel_letter(fig2, "A", x=1.5, row_top=line, clear_of=[ax], not_above=line)
    assert all(r["ok"] and r["raised_mm"] == 0 for r in style.settle_letters(fig2, strict=True))


# ---------------------------------------------------------------------------------------------------------- fonts
def test_fallback_font_warns_once(monkeypatch):
    monkeypatch.delenv("HEXMOD_FONT_DIR", raising=False)
    monkeypatch.setattr(fm.fontManager, "ttflist", [f for f in fm.fontManager.ttflist if f.name != "Arial"])
    monkeypatch.setattr(style, "_FALLBACK_WARNED", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = style.house_font()
        second = style.house_font()
    assert first == second and first in style.FALLBACK_FONTS
    assert sum("Arial is not available" in str(w.message) for w in caught) == 1


def test_register_fonts_from_folder(tmp_path, monkeypatch):
    source = Path(mpl.get_data_path()) / "fonts" / "ttf" / "DejaVuSerif-Bold.ttf"
    target = tmp_path / "fonts" / "Serif Bold.ttf"
    target.parent.mkdir()
    shutil.copyfile(source, target)
    monkeypatch.setenv("HEXMOD_FONT_DIR", str(target.parent))
    added = style.register_fonts()
    assert added == [str(target.resolve())]
    assert str(target.resolve()) in {f.fname for f in fm.fontManager.ttflist}
    assert style.register_fonts() == []


def _font_dir_with_arial():
    from hexmod import config

    folder = config.font_dir()
    if folder is None or not any(p.name.lower() == "arial.ttf" for p in folder.glob("*")):
        return None
    return folder


@pytest.mark.skipif(_font_dir_with_arial() is None, reason="HEXMOD_FONT_DIR with Arial.ttf not set")
def test_arial_from_font_dir_is_embedded(tmp_path):
    assert style.house_font() == "Arial"
    fig, *_ = demo_figure()
    files = common.save_outputs(fig, tmp_path, "Demo")
    pdf = files[0].read_bytes()
    assert b"ArialMT" in pdf and b"Arial-BoldMT" in pdf


# ----------------------------------------------------------------------------------------------- output files
def test_save_outputs_writes_all_files(tmp_path):
    fig, ax, axm = demo_figure()
    files = common.save_outputs(fig, tmp_path, "Demo")
    names = sorted(p.name for p in files)
    assert names == ["Demo.pdf", "Demo.png", "Demo.svg", "positions.csv"]
    pdf = (tmp_path / "Demo.pdf").read_bytes()
    assert b"FontFile2" in pdf and b"/Type3" not in pdf            # TrueType (type 42), editable text
    svg = (tmp_path / "Demo.svg").read_text(encoding="utf-8")
    assert "<text" in svg and "Aligned" in svg                       # text kept as text
    assert png_size(tmp_path / "Demo.png") == tuple(int(v * 600) for v in fig.get_size_inches())
    pos = hio.read_csv(tmp_path / "positions.csv", keep_default_na=False)
    assert list(pos.columns) == common.POSITION_COLUMNS
    box = pos[(pos.element == "axes")].iloc[0]
    assert box.x0_mm == pytest.approx(12.0, abs=1e-3) and box.y1_mm == pytest.approx(47.0 - 11.0, abs=1e-3)
    assert all(r["ok"] and r["raised_mm"] == 0 for r in fig._hexmod_letter_report)
    letters = pos[pos.name == "panel letter"]
    assert list(letters.text) == ["A", "B"] and list(letters.panel) == ["A", "B"]
    assert set(pos[pos.text == "Misaligned"].panel) == {"A"}
    assert set(pos[(pos.element == "axes")].panel) == {"A", "B"}
    assert "colour bar" in set(pos.name)


def test_outputs_are_reproducible(tmp_path):
    for k in (1, 2):
        fig, *_ = demo_figure()
        common.save_outputs(fig, tmp_path / str(k), "Demo")
    for name in ("Demo.pdf", "Demo.svg", "Demo.png", "positions.csv"):
        assert (tmp_path / "1" / name).read_bytes() == (tmp_path / "2" / name).read_bytes(), name


def test_polar_outputs_are_reproducible(tmp_path):
    figs = []                                # keep both figures alive, so that their clip path objects differ
    for k in (1, 2):
        style.use_style()
        fig = style.new_figure("single", 30.0)
        for j in range(2):
            ax = style.polar_mm(fig, 15.0 + 25.0 * j, 15.0)
            ax.plot(np.linspace(0, 2 * np.pi, 50), np.ones(50))
        figs.append(fig)
        style.save_figure(fig, tmp_path / f"Polar{k}")
    svg = [(tmp_path / f"Polar{k}.svg").read_bytes() for k in (1, 2)]
    assert svg[0] == svg[1]
    text = svg[0].decode("utf-8")
    ids = re.findall(r'<clipPath id="([^"]+)">', text)
    assert len(ids) == len(set(ids)) and set(re.findall(r"url\(#([^)]+)\)", text)) <= set(ids)
    assert style.stable_svg_clip_ids(tmp_path / "Polar1.svg") == 2      # renaming again keeps the same ids
    assert (tmp_path / "Polar1.svg").read_bytes() == svg[0]


def test_dashes_in_figure_text_are_refused(tmp_path):
    fig, *_ = demo_figure(text="Rate \u2013 change")
    with pytest.raises(ValueError, match="en dash"):
        common.save_outputs(fig, tmp_path, "Demo")


def test_tagged_panel_and_name(tmp_path):
    fig, ax, axm = demo_figure()
    common.tag(axm, panel="B", name="rate map")
    pos = common.positions(fig)
    row = pos[pos.name == "rate map"]
    assert len(row) == 1 and row.panel.iloc[0] == "B"


# ------------------------------------------------------------------------------------------------- source data
def test_source_data_round_trip(tmp_path):
    values = np.random.default_rng(1).standard_normal(7)
    path = common.write_source_data(tmp_path, "Fig3", "C", "firing_rate_change_percent",
                                    {"row": np.arange(7), "value": values})
    assert path == tmp_path / "source_data" / "Fig3C_firing_rate_change_percent.csv"
    back = hio.read_csv(path)
    assert np.array_equal(back["value"].to_numpy(), values)
    whole = common.write_source_data(tmp_path, "SuppFig2", "", "statistics", pd.DataFrame({"t": [1.5]}))
    assert whole.name == "SuppFig2_statistics.csv"


def test_source_data_checks(tmp_path):
    with pytest.raises(ValueError):
        common.write_source_data(tmp_path, "Fig9", "A", "x", {"a": [1]})
    with pytest.raises(ValueError):
        common.write_source_data(tmp_path, "Fig3", "a", "x", {"a": [1]})
    with pytest.raises(ValueError):
        common.write_source_data(tmp_path, "Fig3", "A", "bad-name", {"a": [1]})
    with pytest.raises(ValueError):
        common.write_source_data(tmp_path, "Fig3", "A", "x", {"a": [1, 2], "b": [1]})
    with pytest.raises(ValueError, match="dash"):
        common.write_source_data(tmp_path, "Fig3", "A", "x", {"label": ["0 \u2013 1 s"]})
    path = common.write_source_data(tmp_path, "Fig3", "A", "x", {"a": [1.0, 2.0], "b": [3.0]}, pad=True)
    back = hio.read_csv(path)
    assert np.isnan(back["b"].iloc[1])


def test_long_and_map_tables():
    t = common.long_table({"Aligned": [1.0, 2.0], "Misaligned": [3.0]})
    assert list(t.group) == ["Aligned", "Aligned", "Misaligned"] and list(t.value) == [1.0, 2.0, 3.0]
    m = common.map_table(np.arange(6.0).reshape(2, 3), [0.0, 1.0, 2.0], [0.0, 1.0, 2.0, 3.0])
    assert list(m.x) == [0.5, 0.5, 0.5, 1.5, 1.5, 1.5] and list(m.y) == [0.5, 1.5, 2.5] * 2
    assert list(m.value) == [0, 1, 2, 3, 4, 5]


def test_find_input(tmp_path):
    results = tmp_path / "results"
    shipped = tmp_path / "shipped"
    (results / "rat").mkdir(parents=True)
    (results / "rat" / "a.csv").write_text("x\n1\n")
    (shipped / "sim").mkdir(parents=True)
    (shipped / "sim" / "b.csv").write_text("x\n2\n")
    try:
        common.set_precomputed(shipped)
        common.reset_used_inputs()
        assert common.find_input(results, "rat/a.csv") == results / "rat" / "a.csv"
        assert common.find_input(results, "sim/b.csv") == shipped / "sim" / "b.csv"
        (results / "sim").mkdir()
        (results / "sim" / "b.csv").write_text("x\n3\n")
        assert common.find_input(results, "sim/b.csv") == results / "sim" / "b.csv"
        common.set_precomputed(shipped, prefer=True)
        assert common.find_input(results, "sim/b.csv") == shipped / "sim" / "b.csv"
        with pytest.raises(common.MissingInput):
            common.find_input(results, "rat/missing.csv")
        assert len(common.used_inputs()) == 3
        # alternative names: the preferred place is searched under every name before the other place
        (results / "sim" / "c.csv").write_text("x\n4\n")
        (shipped / "sim" / "c.csv.gz").write_bytes(b"")
        assert common.find_input(results, "sim/c.csv", ["sim/c.csv.gz"]) == shipped / "sim" / "c.csv.gz"
        common.set_precomputed(shipped)
        assert common.find_input(results, "sim/c.csv", ["sim/c.csv.gz"]) == results / "sim" / "c.csv"
        assert common.find_input(results, "sim/d.csv", ["sim/c.csv.gz"]) == shipped / "sim" / "c.csv.gz"
    finally:
        common.set_precomputed(None)


# ------------------------------------------------------------------------------------------------ drawing helpers
def test_triangular_lattice():
    style.use_style()
    fig = style.new_figure("single", 40.0)
    ax = style.axes_mm(fig, 5, 5, 30, 30, full=True)
    P, ax_out = common.draw_triangular_lattice_lines(dx=40.0, theta_deg=10.0, env_size_x=150, env_size_y=150,
                                                     origin=(20.0, 30.0), ax=ax, line_colour="w")
    assert ax_out is ax and ax.get_xlim() == (0, 150)
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    d[d == 0] = np.inf
    assert np.allclose(d.min(axis=1), 40.0)
    assert np.any(np.all(np.isclose(P, [20.0, 30.0]), axis=1))
    assert len(ax.lines) > 0 and all(line.get_linewidth() == 0.5 for line in ax.lines)


def test_polar_hist_replaces_cartesian_axes():
    fig = style.new_figure("single", 40.0)
    ax = fig.add_axes([0.1, 0.1, 0.4, 0.8])
    bars = common.polar_hist(ax, [5.0, 15.0, 15.5, 55.0], (0, 60), 6)
    assert len(fig.axes) == 1 and fig.axes[0].name == "polar"
    assert [b.get_height() for b in bars] == [1, 2, 0, 0, 0, 1]


def test_trajectory_segment_on_given_axes():
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    fig = style.new_figure("single", 60.0)
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])
    t = np.linspace(0, 20, 401)
    x = 50 + 30 * np.cos(t / 3)
    y = 50 + 30 * np.sin(t / 3)
    out = common.plot_trajectory_segment(x=x, y=y, t=t, time_window=(2, 12), colour_var=np.abs(np.gradient(x)),
                                         field_loc=np.array([[50.0, 50.0, 10.0], [20.0, 80.0, 8.0]]),
                                         xbins=[0, 100], ybins=[0, 100], ax=ax)
    assert out == [ax]
    assert len(fig.axes) == 2                         # the axes and its colour bar
    assert [t_.get_text() for t_ in ax.texts] == ["0", "1"]
    assert plt.get_fignums() == before                # no pyplot figure created


# ---------------------------------------------------------------------------------------------------- figure step
def test_figure_names():
    assert len(FIGURES) == 13 and FIGURES[0] == "Fig1" and FIGURES[-1] == "SuppFig5"
    assert figure_name("supp2") == "SuppFig2" == figure_name("SUPPFIG2")
    assert module_name("Fig3") == "hexmod.figures.fig3" and MODULES["SuppFig5"] == "supp5"
    with pytest.raises(ValueError):
        figure_name("Fig9")


def _fake_figure_module(build):
    module = types.ModuleType("hexmod.figures.fig1")
    module.build = build
    return module


def _found(root, pattern):
    """The paths under root that match a glob pattern (the step decides the layout of its results folder)."""
    return sorted(p for p in Path(root).rglob(pattern) if ".building" not in p.as_posix())


def _demo_build(results_dir, out_dir):
    fig, *_ = demo_figure()
    common.write_source_data(out_dir, "Fig1", "A", "values", {"value": [1.0, 2.0]})
    common.save_outputs(fig, out_dir, "Fig1")


def test_step30_builds_a_figure(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "hexmod.figures.fig1", _fake_figure_module(_demo_build))
    monkeypatch.setenv("HEXMOD_RESULTS", str(tmp_path))
    step30 = load_step30()
    assert step30.main(["--figure", "Fig1", "--results", str(tmp_path)]) == 0
    (out,) = _found(tmp_path, "figures/Fig1")
    assert sorted(p.name for p in out.iterdir()) == ["Fig1.pdf", "Fig1.png", "Fig1.svg", "positions.csv",
                                                     "source_data"]
    assert (out / "source_data" / "Fig1A_values.csv").is_file()
    (log,) = _found(tmp_path, "logs/step30_figures.json")
    record = hio.read_json(log)
    assert record["status"] == "ok" and record["notes"]["figures"]["Fig1"]["status"] == "ok"
    assert len(record["outputs"]) == 5


def test_step30_keeps_previous_output_on_failure(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "hexmod.figures.fig1", _fake_figure_module(_demo_build))
    monkeypatch.setenv("HEXMOD_RESULTS", str(tmp_path))
    step30 = load_step30()
    assert step30.main(["--figure", "Fig1", "--results", str(tmp_path)]) == 0
    (out,) = _found(tmp_path, "figures/Fig1")
    before = (out / "Fig1.pdf").read_bytes()

    def build(results_dir, out_dir):
        raise RuntimeError("broken")

    monkeypatch.setitem(sys.modules, "hexmod.figures.fig1", _fake_figure_module(build))
    assert step30.main(["--figure", "fig1", "--results", str(tmp_path)]) == 1
    assert (out / "Fig1.pdf").read_bytes() == before
    assert not (out.parent / ".Fig1.building").exists()


def test_step30_skips_missing_inputs(tmp_path, monkeypatch):
    def build(results_dir, out_dir):
        common.find_input(results_dir, "rat/cohort/cells.csv")

    monkeypatch.setitem(sys.modules, "hexmod.figures.fig1", _fake_figure_module(build))
    monkeypatch.setenv("HEXMOD_RESULTS", str(tmp_path))
    step30 = load_step30()
    args = ["--figure", "Fig1", "--results", str(tmp_path)]
    assert step30.main(args) == 0
    assert step30.main(args + ["--strict"]) == 1
    assert not _found(tmp_path, "figures/Fig1")


@pytest.mark.parametrize("module", sorted(set(MODULES.values())))
def test_figure_modules_take_results_and_output_folders(module):
    import inspect
    import importlib

    params = inspect.signature(importlib.import_module(f"hexmod.figures.{module}").build).parameters
    assert list(params) == ["results_dir", "out_dir"]


def test_fig1_key_gap_is_an_error_in_arial_and_a_warning_in_other_fonts(monkeypatch):
    """The colour key of Fig 1B is placed for Arial; in a fallback font a key too close to the wheel only warns."""
    from hexmod.figures import fig1

    monkeypatch.setattr(fig1, "KEY_GAP_MIN", 1000.0)      # every layout fails the check

    def panels(font):
        fig1._panels_a_b(fig1._new_canvas(fig1.W_MM, fig1.H_A), 0.0, font)

    with pytest.raises(ValueError, match="too close"):
        panels(style.HOUSE_FONT)
    with pytest.warns(UserWarning, match="made for Arial"):
        panels("Liberation Sans")
