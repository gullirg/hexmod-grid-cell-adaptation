"""The simulation parameters stated in the Methods, as the simulation code uses them.

Each assertion names a quantity of the Methods sections on the simulations (population size, grid
scales, trajectories, adaptation, estimators, speed sweep) and checks the value in the code: the
defaults of ``hexmod.sim.pool.set_param``, the module parameters and the settings of the step
scripts 22 to 26. A change of any of these values changes the simulation results, so it must be
deliberate and must be reflected in the Methods.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

from hexmod.sim import pool, simulate, speed_sweep, trajectory_v1

REPO = Path(__file__).resolve().parents[1]


def load_script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def par():
    return pool.set_param()


def test_population_and_participants(par):
    step24 = load_script("step24_simulation_participants")
    assert step24.N_CELLS == 10000                    # grid cells summed per participant (one voxel)
    assert step24.N_PARTICIPANTS == 1500 and par["nSub"] == 30   # 50 experiments of 30 participants
    assert step24.ALPHA == 0.1


def test_grid_scales_and_environment(par):
    assert par["envSize_x"] == par["envSize_y"] == 100.0           # square environment of 100 vm
    assert pool.MODULE_PARAMS["small"]["gridScale"] == 40           # 40% of the environment
    assert pool.MODULE_PARAMS["large"]["gridScale"] == 70           # 70% of the environment
    assert par["minGridness"] == 0.3                                # cells kept with gridness of at least 0.3


def test_trajectories(par):
    source = inspect.getsource(trajectory_v1.make_trajectory)
    assert "dt = 0.1  # 10 Hz" in source                            # sampling rate 10 Hz
    assert par["nBlock"] == 4 and par["expDurBlock"] == 600         # four blocks of 10 min
    assert par["dur_ITI"] == 2.0 and par["dur_cue"] == 1.0          # intertrial interval 2 s, cue 1 s
    assert (par["minTurnDur"], par["maxTurnDur"]) == (0.8, 3.2)     # rotation 0.8 to 3.2 s
    assert par["moveSpeed"] == pytest.approx(100.0 / 6.0)           # fixed speed: 16.7 vm/s
    assert (par["minLen"], par["maxLen"]) == (40.0, 70.0)           # run distance 40 to 70 vm


def test_adaptation_and_analysis(par):
    assert par["adaptEff_mean"] == 0.1 and par["adaptEff_std"] == 0  # adaptation strength alpha
    assert par["adaptTau"] == 2.0                                    # time constant tau
    assert par["speedGamma_mean"] == 0                               # no speed modulation
    assert par["speedThres"] == 5.0                                  # samples at 5 vm/s or slower excluded
    assert simulate.FOLD_SYM_LIST == (1, 2, 3, 4, 5, 6, 7)           # control symmetries 1 to 7
    sig = inspect.signature(simulate.calc_hm_control_location)
    assert sig.parameters["spatialResolution"].default == 5          # 5 vm x 5 vm location bins
    assert sig.parameters["degResolution"].default == 5              # 5 deg direction bins
    assert "np.arange(-30, 361, 30) + 15" in inspect.getsource(simulate.calc_hm)   # 12 bins of 30 deg


def test_speed_sweep():
    step26 = load_script("step26_speed_sweep")
    assert step26.N_CELLS == 1024                                    # cells per ideal module
    assert step26.SCALES == {"small": 30, "medium": 50, "large": 70}  # 30 and 70 vm drawn, 50 vm also run
    assert step26.SPEEDS[0] == 5 and step26.SPEEDS[-1] == 149 and np.all(np.diff(step26.SPEEDS) == 4)
    assert len(step26.SPEEDS) * len(step26.SCALES) == 111
    assert step26.ALPHA == 0.1
    assert speed_sweep.RUN_LENGTH == 70                               # straight wall-to-wall runs
