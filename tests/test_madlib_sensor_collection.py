# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

"""
Test file: test_madlib_sensor.py
Description: This file contains unit tests which test "edge cases" in
    the madlib._sensor module.
"""

import astroforge as af
import numpy as np
from astropy.time import Time

import madlib
from madlib._utils import MadlibException
from madlib._sensor_collection import SensorCollection, SensorException

import pytest


seed = 4445

yaml = "configs/sample_sensor_network.yaml"


def test_SensorCollection_init_from_params():
    # Sensors
    sensor_params = SensorCollection.paramsFromYAML(yaml)
    sensor_objects = [
        madlib.GroundOpticalSensor(**params) for key, params in sensor_params.items()
    ]
    sensors = SensorCollection(sensor_objects)

    assert isinstance(sensors, madlib._sensor_collection.SensorCollection)
    assert sensors.sensorList == sensor_objects
    assert sensors.numSensors == len(sensor_objects)
    assert sensors.obsTimes == None


def test_SensorCollection_init_from_yaml():
    sensors = SensorCollection.fromYAML(yaml)

    assert isinstance(sensors, madlib._sensor_collection.SensorCollection)
    assert sensors.numSensors == 11
    assert sensors.obsTimes == None


def test_SensorCollection_generate_obs_timing():
    sensors = SensorCollection.fromYAML(yaml)

    ### SIMULATION TIMING
    t_start_mjd = Time("2011-11-11T11:11:11", format="isot", scale="utc").mjd
    t_end_mjd = t_start_mjd + 1

    assert sensors.obsTimes == None
    sensors.generate_obs_timing(t_start_mjd, t_end_mjd)
    assert sensors.obsTimes != None

    with pytest.raises(
        SensorException
    ):  # The observation schedule has already been generated.
        sensors.generate_obs_timing(t_start_mjd, t_end_mjd)


def test_add_sensor_post_timing():
    sensors = SensorCollection.fromYAML(yaml)

    ### SIMULATION TIMING
    t_start_mjd = Time("2011-11-11T11:11:11", format="isot", scale="utc").mjd
    t_end_mjd = t_start_mjd + 1

    SENSOR_PARAMS = {
        "lat": -21.89567,
        "lon": 114.0898731,
        "alt": 0.067845,
        "dra": 0.3,
        "ddec": 0.3,
        "obs_per_collect": 3,
        "obs_time_spacing": 1.5,
        "collect_gap_mean": 7200,
        "collect_gap_std": 800,
        "obs_limits": None,
        "id": "SENSOR",
    }

    assert sensors.numSensors == 11

    new_sensor_1 = madlib.GroundOpticalSensor(**SENSOR_PARAMS)
    sensors.add_sensor(new_sensor_1)
    assert sensors.numSensors == 12

    sensors.generate_obs_timing(t_start_mjd, t_end_mjd)

    new_sensor_2 = madlib.GroundOpticalSensor(**SENSOR_PARAMS)

    with pytest.raises(SensorException):
        sensors.add_sensor(new_sensor_2)

    assert sensors.numSensors == 12


def test_SensorCollection_observe(seed=seed):
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

    # --- sensors
    sensors = SensorCollection.fromYAML(yaml)

    # --- setup sensor and generate ObservationCollection(s)
    sensors.generate_obs_timing(epoch, epoch + 1)  # + 1 [days]

    observations = sensors.observe(sat0)
    assert isinstance(observations, madlib._observation.ObservationCollection)
    assert isinstance(observations.pos_observed[0], madlib._observation.Observation)
    assert isinstance(observations.pos_truth[0], madlib._observation.Observation)


def test_SensorCollection_observe_student_dist(seed=seed):
    # --- generate an ObservationCollection with different noise distributions
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

    # --- sensors
    sensors = SensorCollection.fromYAML("tests/inputs/students_t_sensors.yaml")

    # --- setup sensor and generate ObservationCollection(s)
    sensors.generate_obs_timing(epoch, epoch + 1)  # + 1 [days]

    observations = sensors.observe(sat0)
    assert isinstance(observations, madlib._observation.ObservationCollection)
    assert isinstance(observations.pos_observed[0], madlib._observation.Observation)
    assert isinstance(observations.pos_truth[0], madlib._observation.Observation)


def test_SensorCollection_error(seed=seed):
    """Try and fail to make observations without generating obs timing"""
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

    # --- sensors
    sensors = SensorCollection.fromYAML(yaml)

    failed = False
    try:
        observations = sensors.observe(sat0)
    except SensorException:
        failed = True

    assert failed


class TestInvalidYAML:
    """Group of tests for handling invalid YAML structures"""

    def test_missing_sensor_list(self):
        """Every sensor network YAML needs a top-level attribute called sensor_list."""
        failed = False
        try:
            params = SensorCollection.paramsFromYAML(
                "tests/inputs/invalid_sensor_1.yaml"
            )
        except MadlibException:
            failed = True

        assert failed

    def test_missing_required(self):
        """Make sure a YAML is invalid if one of its sensors is missing a required attribute."""
        failed = False
        try:
            params = SensorCollection.paramsFromYAML(
                "tests/inputs/invalid_sensor_2.yaml"
            )
        except MadlibException:
            failed = True

        assert failed

    def test_unknown_property(self):
        """Make sure a YAML is invalid if one of its sensors has an unexpected
        (in this case, misspelled) property."""
        failed = False
        try:
            params = SensorCollection.paramsFromYAML(
                "tests/inputs/invalid_sensor_3.yaml"
            )
        except MadlibException:
            failed = True

        assert failed
