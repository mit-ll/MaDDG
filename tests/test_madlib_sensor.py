# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

"""
Test file: test_madlib_sensor.py
Description: This file contains unit tests which test "edge cases" in
    the madlib._sensor module.
"""

import pathlib
import sys

import astroforge as af
import numpy as np
from astropy.time import Time
from astropy.utils import iers

# add parent directory of __file__ to sys.path, if isn't already included
if str(pathlib.Path(__file__).parents[1]) not in sys.path:
    sys.path.append(str(pathlib.Path(__file__).parents[1]))

import madlib
from madlib._utils import MadlibException
from madlib._satellite import Satellite
from madlib._sensor import _Sensor, pos_to_lat_lon
from madlib._observation import Observation

import pytest


seed = 4445


def test_Sensor_NotImplementedErrors():
    with pytest.raises(NotImplementedError):
        _Sensor.generate_obs_timing(None, 54321.0, 54322.0)  # type: ignore

    with pytest.raises(NotImplementedError):
        _epoch = 54321.0
        _pos = np.zeros((3,))
        _vel = np.zeros((3,))
        _accel = np.zeros((3,))
        _manuever_info = None
        sat = Satellite(
            _epoch,
            _pos,
            _vel,
            _accel,
            _manuever_info,
        )
        _Sensor.observe(None, sat, 54321.0)  # type: ignore


def test_invalid_pos_to_lat_lon():
    pos = np.array(
        [
            [8000, 8000, 8000],
            [8000, 8000, 8000],
            [8000, 8000, 8000],
        ]
    )

    times = np.array([1, 2, 3, 4])

    with pytest.raises(ValueError):
        pos_to_lat_lon(pos, times)


class TestGroundOpticalSensor:
    """Test behavior of GroundOpticalSensor objects."""

    def test_GroundOpticalSensor_generate_obs_timing(self, seed=seed):
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

        # --- setup sensor
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

        sensor = madlib.GroundOpticalSensor(**SENSOR_PARAMS)

        assert not isinstance(sensor.obs_per_collect, (tuple, list))

        with pytest.raises(ValueError):
            sensor.generate_obs_timing(
                epoch, epoch - 2
            )  # - 2 [days], raises ValueError

        assert isinstance(sensor.generate_obs_timing(epoch, epoch + 2), np.ndarray)

    def test_obs_timing_edge(self, seed=seed):
        """If an observation collection contains obs after the simulation end time,
        the entire collection must be discarded."""
        np.random.seed(seed)
        SENSOR_PARAMS = {
            "lat": -21.89567,
            "lon": 114.0898731,
            "alt": 0.067845,
            "dra": 0.3,
            "ddec": 0.3,
            "obs_per_collect": 5,
            "obs_time_spacing": 3 * 3600,
            "collect_gap_mean": 10 * 3600,
            "collect_gap_std": 0,
            "obs_limits": {"el": (20.0, 91.0)},
            "id": "SENSOR",
        }
        sensor = madlib.GroundOpticalSensor(**SENSOR_PARAMS)

        t_start = Time("2011-11-11T11:11:11", format="isot", scale="utc")
        epoch = t_start.mjd
        times = sensor.generate_obs_timing(epoch, epoch + 1)

        # Only 2 of the 3 collects should have succeeded
        assert len(times) == 10

    def test_invalid_limits(self, seed=seed):
        """Test a sensor that contains some invalid observation limits."""
        SENSOR_PARAMS = {
            "lat": -21.89567,
            "lon": 114.0898731,
            "alt": 0.067845,
            "dra": 0.3,
            "ddec": 0.3,
            "obs_per_collect": 1,
            "obs_time_spacing": 1.5,
            "collect_gap_mean": 3600,
            "collect_gap_std": 800,
            "obs_limits": {"el": (20.0, 91.0), "el_radians": (0, np.pi)},
            "id": "Sensor",
        }

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
        sensor = madlib.GroundOpticalSensor(**SENSOR_PARAMS)
        times = sensor.generate_obs_timing(epoch, epoch + 1)

        with pytest.raises(MadlibException):
            obs = sensor.observe(sat0, times)

    def test_invalid_observation(self, seed=seed):
        """Test that limit validation fails on obs without values."""
        SENSOR_PARAMS = {
            "lat": -21.89567,
            "lon": 114.0898731,
            "alt": 0.067845,
            "dra": 0.3,
            "ddec": 0.3,
            "obs_per_collect": 1,
            "obs_time_spacing": 1.5,
            "collect_gap_mean": 3600,
            "collect_gap_std": 800,
            "obs_limits": {"el": (20.0, 91.0)},
            "id": "Sensor",
        }
        sensor = madlib.GroundOpticalSensor(**SENSOR_PARAMS)

        obs = Observation(mjd=1000)

        failed = False
        try:
            sensor.validate_limits(obs)
        except MadlibException:
            failed = True

        assert failed

    def test_obs_time_inputs(self):
        """The observe function should still work properly if the
        time input is a single float or a (start, end) tuple."""

        SENSOR_PARAMS = {
            "lat": 0.0,
            "lon": 0.0,
            "alt": 0.0,
            "dra": 0.3,
            "ddec": 0.3,
            "obs_per_collect": 1,
            "obs_time_spacing": 1.5,
            "collect_gap_mean": 3600,
            "collect_gap_std": 800,
            "obs_limits": {"el": (20.0, 91.0)},
            "id": "Sensor",
        }
        sensor = madlib.GroundOpticalSensor(**SENSOR_PARAMS)

        t_start = Time("2011-11-11T11:11:11", format="isot", scale="utc")
        epoch = t_start.mjd
        sat = Satellite.from_GEO_longitude(lon=0.0, epoch=epoch)

        obs_1 = sensor.observe(sat, times=epoch + 1)
        obs_2 = sensor.observe(sat, times=(epoch, epoch + 3))

        assert len(obs_1.pos_observed) == 1
        assert len(obs_2.pos_observed) > 1

    def test_truth_position(self):
        """Test the functionality of truth lat/lon/alt positions."""

        SENSOR_1_PARAMS = {
            "lat": -10,
            "lon": 10,
            "alt": 100.0,
            "dra": 0.01,
            "ddec": 0.01,
            "obs_per_collect": 1,
            "obs_time_spacing": 1.5,
            "collect_gap_mean": 3600,
            "collect_gap_std": 800,
            "obs_limits": {"el": (20.0, 91.0)},
            "id": "Sensor",
            "lat_truth": 0.0,
            "lon_truth": 0,
            "alt_truth": 0,
        }

        SENSOR_2_PARAMS = {
            "lat": 0,
            "lon": 0,
            "alt": 0.0,
            "dra": 0.01,
            "ddec": 0.01,
            "obs_per_collect": 1,
            "obs_time_spacing": 1.5,
            "collect_gap_mean": 3600,
            "collect_gap_std": 800,
            "obs_limits": {"el": (20.0, 91.0)},
            "id": "Sensor",
        }

        sensor_1 = madlib.GroundOpticalSensor(**SENSOR_1_PARAMS)
        sensor_2 = madlib.GroundOpticalSensor(**SENSOR_2_PARAMS)

        assert ~np.isclose(sensor_1.lon, sensor_1.lon_truth, rtol=0, atol=0.1)
        assert ~np.isclose(sensor_1.lat, sensor_1.lat_truth, rtol=0, atol=0.1)
        assert ~np.isclose(sensor_1.alt, sensor_1.alt_truth, rtol=0, atol=0.1)

        assert np.isclose(sensor_2.lon, sensor_2.lon_truth)
        assert np.isclose(sensor_2.lat, sensor_2.lat_truth)
        assert np.isclose(sensor_2.alt, sensor_2.alt_truth)

        t_start = Time("2011-11-11T11:11:11", format="isot", scale="utc")
        epoch = t_start.mjd

        sat = madlib.Satellite.from_GEO_longitude(lon=0.0, epoch=epoch)
        times_1 = sensor_1.generate_obs_timing(epoch, epoch + 1)
        times_2 = sensor_2.generate_obs_timing(epoch, epoch + 1)

        obs_1 = sensor_1.observe(sat, times_1)
        obs_2 = sensor_2.observe(sat, times_2)

        ra_obs_1 = [p.ra for p in obs_1.pos_observed]
        ra_exp_1 = [p.ra for p in obs_1.pos_expected]

        ra_obs_2 = [p.ra for p in obs_2.pos_observed]
        ra_exp_2 = [p.ra for p in obs_2.pos_expected]

        assert all(~np.isclose(ra_obs_1, ra_exp_1, rtol=0, atol=0.1))
        assert all(np.isclose(ra_obs_2, ra_exp_2, rtol=0, atol=0.1))


class TestSpaceOpticalSensor:
    """Test the behavior of SpaceOpticalSensor objects"""

    def test_SpaceOpticalSensor_generate_obs_timing(self, seed=seed):
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

        # --- setup sensor
        sensor_satellite = madlib.Satellite.from_keplerian(
            epoch=t_start.mjd,
            inclination_rad=0.0,
            raan_rad=0.0,
            argp_rad=0.0,
            ecc=0.0,
            semi_major_axis_km=7000,
            mean_anomaly_rad=0.0,
        )
        SENSOR_PARAMS = {
            "sensor_satellite": sensor_satellite,
            "dra": 0.3,
            "ddec": 0.3,
            "obs_per_collect": 3,
            "obs_time_spacing": 1.5,
            "collect_gap_mean": 7200,
            "collect_gap_std": 800,
            "obs_limits": None,
            "id": "SENSOR",
        }

        sensor = madlib.SpaceOpticalSensor(**SENSOR_PARAMS)

        assert not isinstance(sensor.obs_per_collect, (tuple, list))

        with pytest.raises(ValueError):
            sensor.generate_obs_timing(
                epoch, epoch - 2
            )  # - 2 [days], raises ValueError

        assert isinstance(sensor.generate_obs_timing(epoch, epoch + 2), np.ndarray)

    def test_ObservationCollection_no_obs(self, seed=seed):
        # --- setup sensor
        t_start = Time("2011-11-11T11:11:11", format="isot", scale="utc")
        epoch = t_start.mjd

        sensor_satellite = madlib.Satellite.from_GEO_longitude(lon=0.0, epoch=epoch)
        SENSOR_PARAMS = {
            "sensor_satellite": sensor_satellite,
            "dra": 0.3,
            "ddec": 0.3,
            "obs_per_collect": 3,
            "obs_time_spacing": 1.5,
            "collect_gap_mean": 7200,
            "collect_gap_std": 800,
            "obs_limits": None,
            "id": "SENSOR",
        }

        sensor = madlib.SpaceOpticalSensor(**SENSOR_PARAMS)

        # --- generate an ObservationCollection
        # --- orbital state
        np.random.seed(seed)

        # --- define satellites
        sat0 = madlib.Satellite.from_GEO_longitude(lon=180.0, epoch=epoch)

        # --- setup sensor and generate ObservationCollection(s)
        times = sensor.generate_obs_timing(epoch, epoch + 2)  # + 2 [days]

        ObsColl1 = sensor.observe(sat0, times)
        assert isinstance(ObsColl1, madlib._observation.ObservationCollection)
        assert ObsColl1.count_valid_observations() == 0

    def test_ObservationCollection_no_times(self, seed=seed):
        # --- setup sensor
        t_start = Time("2011-11-11T11:11:11", format="isot", scale="utc")
        epoch = t_start.mjd

        sensor_satellite = madlib.Satellite.from_GEO_longitude(lon=0.0, epoch=epoch)
        SENSOR_PARAMS = {
            "sensor_satellite": sensor_satellite,
            "dra": 0.3,
            "ddec": 0.3,
            "obs_per_collect": 3,
            "obs_time_spacing": 1.5,
            "collect_gap_mean": 7200,
            "collect_gap_std": 800,
            "obs_limits": None,
            "id": "SENSOR",
        }

        sensor = madlib.SpaceOpticalSensor(**SENSOR_PARAMS)

        # --- generate an ObservationCollection
        # --- orbital state
        np.random.seed(seed)

        # --- define satellites
        sat0 = madlib.Satellite.from_GEO_longitude(lon=1.0, epoch=epoch)

        # --- setup sensor and generate ObservationCollection(s)
        times = (epoch, epoch + 1e-6)

        ObsColl1 = sensor.observe(sat0, times)
        assert isinstance(ObsColl1, madlib._observation.ObservationCollection)
        assert len(ObsColl1.pos_observed) == 0

    def test_ObservationCollection_mistaken_orbit(self, seed=seed):
        # --- setup sensor
        t_start = Time("2011-11-11T11:11:11", format="isot", scale="utc")
        epoch = t_start.mjd

        sensor_reported = madlib.Satellite.from_GEO_longitude(lon=0.0, epoch=epoch)
        sensor_truth = madlib.Satellite.from_GEO_longitude(lon=5.0, epoch=epoch)
        SENSOR_PARAMS = {
            "sensor_satellite": sensor_reported,
            "sensor_satellite_truth": sensor_truth,
            "dra": 0.0,
            "ddec": 0.0,
            "obs_per_collect": 3,
            "obs_time_spacing": 1.5,
            "collect_gap_mean": 7200,
            "collect_gap_std": 800,
            "obs_limits": None,
            "id": "SENSOR",
        }

        sensor = madlib.SpaceOpticalSensor(**SENSOR_PARAMS)

        # --- generate an ObservationCollection
        # --- orbital state
        np.random.seed(seed)

        # --- define satellites
        sat0 = madlib.Satellite.from_GEO_longitude(lon=1.0, epoch=epoch)

        # --- setup sensor and generate ObservationCollection(s)
        times = (epoch, epoch + 1)

        ObsColl1 = sensor.observe(sat0, times)
        assert isinstance(ObsColl1, madlib._observation.ObservationCollection)

        ra_diff = np.abs(ObsColl1.pos_observed[0].ra - ObsColl1.pos_expected[0].ra)
        assert ra_diff > 100

    def test_ObservationCollection_good_orbit(self, seed=seed):
        # --- setup sensor
        t_start = Time("2011-11-11T11:11:11", format="isot", scale="utc")
        epoch = t_start.mjd

        sensor_reported = madlib.Satellite.from_GEO_longitude(lon=0.0, epoch=epoch)
        sensor_truth = madlib.Satellite.from_GEO_longitude(lon=0.0, epoch=epoch)
        SENSOR_PARAMS = {
            "sensor_satellite": sensor_reported,
            "sensor_satellite_truth": sensor_truth,
            "dra": 0.0,
            "ddec": 0.0,
            "obs_per_collect": 3,
            "obs_time_spacing": 1.5,
            "collect_gap_mean": 7200,
            "collect_gap_std": 800,
            "obs_limits": None,
            "id": "SENSOR",
        }

        sensor = madlib.SpaceOpticalSensor(**SENSOR_PARAMS)

        # --- generate an ObservationCollection
        # --- orbital state
        np.random.seed(seed)

        # --- define satellites
        sat0 = madlib.Satellite.from_GEO_longitude(lon=1.0, epoch=epoch)

        # --- setup sensor and generate ObservationCollection(s)
        times = (epoch, epoch + 1)

        ObsColl1 = sensor.observe(sat0, times)
        assert isinstance(ObsColl1, madlib._observation.ObservationCollection)

        ra_diff = np.abs(ObsColl1.pos_observed[-1].ra - ObsColl1.pos_truth[-1].ra)
        assert ra_diff < 0.01

    def test_observation_single_time(self):
        t_start = Time("2011-11-11T11:11:11", format="isot", scale="utc")
        epoch = t_start.mjd

        sensor_reported = madlib.Satellite.from_GEO_longitude(lon=0.0, epoch=epoch)
        sensor_truth = madlib.Satellite.from_GEO_longitude(lon=0.0, epoch=epoch)
        SENSOR_PARAMS = {
            "sensor_satellite": sensor_reported,
            "sensor_satellite_truth": sensor_truth,
            "dra": 0.0,
            "ddec": 0.0,
            "obs_per_collect": 3,
            "obs_time_spacing": 1.5,
            "collect_gap_mean": 7200,
            "collect_gap_std": 800,
            "obs_limits": None,
            "id": "SENSOR",
        }

        sensor = madlib.SpaceOpticalSensor(**SENSOR_PARAMS)

        sat = madlib.Satellite.from_GEO_longitude(lon=1.0, epoch=epoch)

        obs = sensor.observe(sat, times=epoch + 1)

        assert len(obs.pos_observed) == 1


class TestCrossTags:
    def test_ground_cross_tags(self):
        """Test cross tags on ground-based sensors."""

        t_start = Time("2011-11-11T11:11:11", format="isot", scale="utc")
        epoch = t_start.mjd

        sat = madlib.Satellite.from_GEO_longitude(lon=0.0, epoch=epoch)
        sat_crs = sat.create_cross_tag(
            cross_mjd=epoch + 0.5,
            delta_pos_km=np.array([0, 0, 0]),
            delta_vel_kms=np.array([0, 0, -10]),
        )

        SENSOR_PARAMS = {
            "lat": 0,
            "lon": 0,
            "alt": 0.0,
            "dra": 0.01,
            "ddec": 0.01,
            "obs_per_collect": 1,
            "obs_time_spacing": 1.5,
            "collect_gap_mean": 3600,
            "collect_gap_std": 800,
            "obs_limits": {"el": (20.0, 91.0)},
            "id": "Sensor",
            "cross_tag": sat_crs,
            "cross_tag_limit_arcsec": 1000,
        }

        sensor = madlib.GroundOpticalSensor(**SENSOR_PARAMS)
        times = sensor.generate_obs_timing(epoch, epoch + 1)
        obs = sensor.observe(sat, times)

        ra_obs = np.array([p.ra for p in obs.pos_observed])
        ra_exp = np.array([p.ra for p in obs.pos_expected])
        ra_tru = np.array([p.ra for p in obs.pos_truth])

        assert np.sqrt(np.mean((ra_exp - ra_obs) ** 2)) > 10
        assert np.sqrt(np.mean((ra_exp - ra_tru) ** 2)) < 1e-4

    def test_space_cross_tags(self):
        """Test cross-tags on space-based sensors."""

        t_start = Time("2011-11-11T11:11:11", format="isot", scale="utc")
        epoch = t_start.mjd

        sat = madlib.Satellite.from_GEO_longitude(lon=0.0, epoch=epoch)
        sat_crs = sat.create_cross_tag(
            cross_mjd=epoch + 0.5,
            delta_pos_km=np.array([0, 0, 0]),
            delta_vel_kms=np.array([0, 0, -10]),
        )

        sensor = madlib.Satellite.from_keplerian(
            epoch=epoch,
            inclination_rad=0,
            raan_rad=0,
            argp_rad=0,
            ecc=0,
            semi_major_axis_km=7000,
            mean_anomaly_rad=0,
        )
        SENSOR_PARAMS = {
            "sensor_satellite": sensor,
            "dra": 0.0,
            "ddec": 0.0,
            "obs_per_collect": 3,
            "obs_time_spacing": 1.5,
            "collect_gap_mean": 7200,
            "collect_gap_std": 800,
            "obs_limits": None,
            "id": "SENSOR",
            "cross_tag": sat_crs,
            "cross_tag_limit_arcsec": 1000,
        }

        sensor = madlib.SpaceOpticalSensor(**SENSOR_PARAMS)

        times = sensor.generate_obs_timing(epoch, epoch + 1)
        obs = sensor.observe(sat, times)

        ra_obs = np.array([p.ra for p in obs.pos_observed])
        ra_exp = np.array([p.ra for p in obs.pos_expected])
        ra_tru = np.array([p.ra for p in obs.pos_truth])

        assert np.sqrt(np.mean((ra_exp - ra_obs) ** 2)) > 10
        assert np.sqrt(np.mean((ra_exp - ra_tru) ** 2)) < 1e-4


def test_pos_to_lat_lon():
    """Test the TETED position to geo lat/lon converter. All angles are in degrees.
    Data used for this test are taken from the JPL Horizons ephemeris of Mars."""

    # If this test is run offline, the dates used in this test are likely too old
    # for the auto IERS file. Disable the max age to prevent an error.
    iers.conf.auto_max_age = None

    times = [
        Time("2024-01-01T00:00:00", format="isot", scale="utc"),
        Time("2024-04-01T00:00:00", format="isot", scale="utc"),
        Time("2024-07-01T00:00:00", format="isot", scale="utc"),
        Time("2024-10-01T00:00:00", format="isot", scale="utc"),
    ]

    # ICRF coordinates (roughly equal to TETED) of Mars computed via JPL Horizons
    x_sun = np.array(
        [
            [-1.907478131413594e07, -3.308234951571553e08, -1.472059527740356e08],
            [2.871238765262319e08, -1.108970588982779e08, -5.534484559072558e07],
            [1.818940970143346e08, 1.722968856880029e08, 6.998787648240359e07],
            [-4.648912080798431e07, 1.641523211579444e08, 7.302598368902390e07],
        ]
    )

    # Precise Mars latitude (declination) and right ascension computed via Horizons
    # (using ICRF coordinates)
    truth_lat = np.array([-23.95204, -10.19497, 15.60632, 23.17272])
    ra = np.array([266.69554, 338.87794, 43.44463, 105.80949])

    # Sub-Mars longitude can be calculated from sidereal time and right ascension
    sidereal_times = np.array(
        [_t.sidereal_time("apparent", "greenwich").value for _t in times]
    )
    hour_angles = (360.0 / 24.0) * sidereal_times - ra
    truth_lon = -hour_angles
    # Wrap angles
    truth_lon[truth_lon < -180] += 360.0
    truth_lon[truth_lon > 180] += 360

    times_mjd = np.array([_t.mjd for _t in times])
    lat, lon = pos_to_lat_lon(x_sun, times_mjd)

    assert all(np.isclose(lat, truth_lat, rtol=0.0, atol=0.01))
    assert all(np.isclose(lon, truth_lon, rtol=0.0, atol=0.01))
