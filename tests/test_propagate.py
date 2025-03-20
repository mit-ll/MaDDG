# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

"""
Test file: test_propagate.py
Description: This file contains a unit tests which propagates a satellite,
    with and without a manuever, and compares the final lat&lon values to
    known values. This unit test in itself provides ~80% code coverage.
"""

import pathlib
import sys

import astroforge as af
import numpy as np
from astropy.time import Time

# add parent directory of __file__ to sys.path, if isn't already included
if str(pathlib.Path(__file__).parents[1]) not in sys.path:
    sys.path.append(str(pathlib.Path(__file__).parents[1]))
import madlib


def test_sat_impulse_maneuver(seed=4445):
    np.random.seed(seed)
    # --- orbital state
    th = 2 * np.pi * np.random.rand()
    x0 = af.constants.Rgeo * np.array([np.cos(th), np.sin(th), 0.0])
    v0 = af.constants.Vgeo * np.array([-np.sin(th), np.cos(th), 0.0])
    a0 = np.zeros((3,))
    t_start = Time("2011-11-11T11:11:11", format="isot", scale="utc")
    epoch = t_start.mjd

    # --- maneuver definition
    man_time = epoch + 5.0 / 24
    man_dv = np.array([0.0, 5.0, 0.0]) / 1000
    man = madlib.ImpulsiveManeuver(man_time, man_dv)

    # --- define satellites
    sat0 = madlib.Satellite(epoch, x0, v0, a0)
    sat1 = madlib.Satellite(epoch, x0, v0, a0, man)

    # --- setup sensor
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
        # "obs_limits": {"el": (20.0, 91.0)},
        "obs_limits": None,
        "id": "SST",
    }

    sensor = madlib.GroundOpticalSensor(**SST_PARAMS)
    times = sensor.generate_obs_timing(epoch, epoch + 2)  # + 2 [hours]

    manuv_f = sensor.observe(sat0, times).pos_truth
    manuv_t = sensor.observe(sat1, times).pos_truth
    manuv_f_dict = {
        "ra": np.array([x.ra for x in manuv_f]),
        "dec": np.array([x.dec for x in manuv_f]),
    }
    manuv_t_dict = {
        "ra": np.array([x.ra for x in manuv_t]),
        "dec": np.array([x.dec for x in manuv_t]),
    }

    assert all(
        np.isclose(
            np.array(
                [
                    manuv_f_dict["ra"][-1],
                    manuv_f_dict["dec"][-1],
                    manuv_t_dict["ra"][-1],
                    manuv_t_dict["dec"][-1],
                ]
            ),
            np.array(
                [
                    28.88567036042452,
                    3.295301473939045,
                    25.393658788650338,
                    3.3090154095997817,
                ]
            ),
        )
    )


def test_sat_continuous_maneuver(seed=4445):
    np.random.seed(seed)
    # --- orbital state
    th = 2 * np.pi * np.random.rand()
    x0 = af.constants.Rgeo * np.array([np.cos(th), np.sin(th), 0.0])
    v0 = af.constants.Vgeo * np.array([-np.sin(th), np.cos(th), 0.0])
    a0 = np.zeros((3,))
    t_start = Time("2011-11-11T11:11:11", format="isot", scale="utc")
    epoch = t_start.mjd

    # --- maneuver definition
    # ContinuousThrust Function Definition
    def acc_f(t):
        return np.array([0.0, 1.0e-7, 0.0])

    acc_t_range = (epoch, epoch + 3)
    # ContinuousManeuver Definition
    man = madlib.ContinuousManeuver(acc_f, acc_t_range)

    # --- define satellites
    sat0 = madlib.Satellite(epoch, x0, v0, a0)
    sat1 = madlib.ContinuousThrustSatellite(epoch, x0, v0, a0, man)

    # --- setup sensor
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
        # "obs_limits": {"el": (20.0, 91.0)},
        "obs_limits": None,
        "id": "SST",
    }

    sensor = madlib.GroundOpticalSensor(**SST_PARAMS)
    times = sensor.generate_obs_timing(epoch, epoch + 2)  # + 2 [hours]

    manuv_f = sensor.observe(sat0, times).pos_truth
    manuv_t = sensor.observe(sat1, times).pos_truth
    manuv_f_dict = {
        "ra": np.array([x.ra for x in manuv_f]),
        "dec": np.array([x.dec for x in manuv_f]),
    }
    manuv_t_dict = {
        "ra": np.array([x.ra for x in manuv_t]),
        "dec": np.array([x.dec for x in manuv_t]),
    }
    dt = np.array([Time(x.mjd, format="mjd").datetime for x in manuv_f])

    assert all(
        np.isclose(
            np.array(
                [
                    manuv_f_dict["ra"][-1],
                    manuv_f_dict["dec"][-1],
                    manuv_t_dict["ra"][-1],
                    manuv_t_dict["dec"][-1],
                ]
            ),
            np.array(
                [
                    28.88567036042452,
                    3.295301473939045,
                    30.024913619492622,
                    3.3128772755122093,
                ]
            ),
        )
    )
