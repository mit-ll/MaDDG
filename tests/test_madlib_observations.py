# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

"""
Test file: test_madlib_observations.py
Description: This file contains unit tests which test "edge cases" in
    the madlib._observations module.
"""

import pathlib
import sys

import astroforge as af
import numpy as np
from astropy.time import Time

# add parent directory of __file__ to sys.path, if isn't already included
if str(pathlib.Path(__file__).parents[1]) not in sys.path:
    sys.path.append(str(pathlib.Path(__file__).parents[1]))

import pytest

import madlib
from madlib._observation import Observation, combineObsCollections
from madlib._utils import MadlibException

# --- setup dummy sensor
SST_PARAMS = {
    "lat": -21.89567,
    "lon": 114.0898731,
    "alt": 0.067845,
    "dra": 0.3,
    "ddec": 0.3,
    "obs_per_collect": (3, 5),
    "obs_time_spacing": 1.5,
    "collect_gap_mean": 7200,
    "collect_gap_std": 800,
    "obs_limits": None,
    "obs_limits": None,
    "id": "SST",
}

seed = 4445


def test_Observation_madlib_exception_1():
    """Observations must be at the same time for computing a residual"""
    with pytest.raises(MadlibException):
        obs1 = Observation(mjd=54321.0)
        obs2 = Observation(mjd=43210.0)
        return obs1 - obs2


def test_Observation_madlib_exception_2():
    """Can only subtract two Observation objects"""
    with pytest.raises(MadlibException):
        obs1 = Observation(mjd=54321.0)
        obs2 = 54321.0
        return obs1 - obs2  # type: ignore


def test_Observation_special_handling_angles_that_wrap():
    obs1 = Observation(mjd=54321.0)
    obs1.ra = 359.0
    obs2 = Observation(mjd=54321.0)
    obs2.ra = 2.0
    residual = obs2 - obs1
    assert np.isclose(residual.ra, 3.0)  # type: ignore


def test_Observation_asarray():
    obs = Observation(mjd=54321.0)
    assert isinstance(obs, madlib._observation.Observation)
    assert isinstance(obs.asarray(), np.ndarray)


def test_ObservationResidual_asarray():
    obs1 = Observation(mjd=54321.0)
    obs2 = Observation(mjd=54321.0)
    residual = obs2 - obs1
    assert isinstance(residual.asarray(), np.ndarray)


def test_ObservationCollection_edge_cases(seed=seed):
    # --- generate an ObservationCollection
    # --- orbital state
    np.random.seed(seed)
    th = 2 * np.pi * np.random.rand()
    x0 = af.constants.Rgeo * np.array([np.cos(th), np.sin(th), 0.0])
    v0 = af.constants.Vgeo * np.array([-np.sin(th), np.cos(th), 0.0])
    a0 = np.zeros((3,))
    t_start = Time("2011-11-11T11:11:11", format="isot", scale="utc")
    epoch = t_start.mjd

    # --- define satellites
    sat0 = madlib.Satellite(epoch, x0, v0, a0)

    # --- setup sensor and generate ObservationCollection(s)
    sensor = madlib.GroundOpticalSensor(**SST_PARAMS)
    times = sensor.generate_obs_timing(epoch, epoch + 0.5)  # + 0.5 [days]

    ObsColl1 = sensor.observe(sat0, times)
    ObsColl2 = sensor.observe(sat0, times)

    ObsColl1_pos_observed_original_size = ObsColl1.pos_observed.size
    ObsColl1_pos_truth_original_size = ObsColl1.pos_truth.size
    ObsColl1 + ObsColl2  # type: ignore

    collectionList = [ObsColl1, ObsColl2]
    combinedObsCollections = combineObsCollections(collectionList)

    # --- check that ObsColl2 was concatenated with ObsColl1 during the addition line above
    # check if ObsColl1.pos_observed now contains ObsColl2.pos_observed
    assert all(
        ObsColl1.pos_observed[-ObsColl2.pos_observed.size :] == ObsColl2.pos_observed
    )
    assert all(ObsColl1.pos_truth[-ObsColl2.pos_truth.size :] == ObsColl2.pos_truth)
    # check if size is correct
    assert (
        ObsColl1.pos_observed.size
        == ObsColl1_pos_observed_original_size + ObsColl2.pos_observed.size
    )
    assert (
        ObsColl1.pos_truth.size
        == ObsColl1_pos_truth_original_size + ObsColl2.pos_truth.size
    )

    # --- check _obvervation.combineObsCollections()
    assert (
        ObsColl1.pos_observed.size + ObsColl2.pos_observed.size
        == combinedObsCollections.pos_observed.size
    )
    assert (
        ObsColl1.pos_truth.size + ObsColl2.pos_truth.size
        == combinedObsCollections.pos_truth.size
    )
