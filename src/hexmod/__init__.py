"""Analysis code for "Firing rate adaptation in grid cells produces macroscopic hexadirectional modulation".

The package holds the analysis library (rat recordings, simulations, statistics and figures).
The numbered scripts in ``scripts/`` run the analyses in order and write to ``results/``;
see ``docs/PIPELINE.md`` for every step, its inputs and its outputs.

Analysis modules never import matplotlib; only ``hexmod.figures`` does.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
