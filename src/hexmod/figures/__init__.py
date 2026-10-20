"""Figures of the paper.

Each figure has one module that draws it from the result files only (never from raw data):
``fig2`` to ``fig8`` for Figures 2 to 8 and ``supp1`` to ``supp5`` for Supplementary Figures 1 to 5. Figure 1 is
a schematic and is not drawn by the code. A figure module exposes::

    build(results_dir, out_dir)

``results_dir`` is the results folder that the analysis steps wrote, ``out_dir`` the folder the figure writes to
(``figures/<Figure>/`` in the results folder).
``scripts/step30_figures.py`` calls ``build`` for each figure. The shared style is in ``hexmod.figures.style`` and
the shared helpers (inputs, output files, positions, source data, lattice and polar drawing) in
``hexmod.figures.common``.

This package is the only part of ``hexmod`` that imports matplotlib; importing ``hexmod.figures`` itself does not.
"""

from __future__ import annotations

from typing import Dict, Tuple

#: Figure names, in the order of the paper.
FIGURES: Tuple[str, ...] = tuple([f"Fig{i}" for i in range(2, 9)] + [f"SuppFig{i}" for i in range(1, 6)])

#: Module of each figure within ``hexmod.figures``.
MODULES: Dict[str, str] = {**{f"Fig{i}": f"fig{i}" for i in range(2, 9)},
                           **{f"SuppFig{i}": f"supp{i}" for i in range(1, 6)}}


def figure_name(name: str) -> str:
    """Return the canonical figure name for 'Fig3', 'fig3', 'SuppFig2', 'suppfig2' or 'supp2'."""
    key = name.strip().lower()
    for figure, module in MODULES.items():
        if key in (figure.lower(), module):
            return figure
    raise ValueError(f"unknown figure {name!r}; figures are {', '.join(FIGURES)}")


def module_name(figure: str) -> str:
    """Return the full module name that draws a figure, for example ``hexmod.figures.fig3``."""
    return f"{__name__}.{MODULES[figure_name(figure)]}"


__all__ = ["FIGURES", "MODULES", "figure_name", "module_name"]
