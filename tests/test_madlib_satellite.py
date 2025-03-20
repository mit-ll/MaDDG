# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

"""
Test file: test_madlib_satellite.py
Description: This file contains unit tests which test "edge cases" in
    the madlib._satellite module.
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
from madlib._utils import MadlibException
from madlib._satellite import Satellite, ContinuousThrustSatellite
from madlib._maneuver import ImpulsiveManeuver

import pytest


seed = 4445


def test_Satellite_kwargs_NotImplementedError():
    _epoch = 54321.0
    _pos = np.zeros((3,))
    _vel = np.zeros((3,))
    _accel = np.zeros((3,))
    _manuever_info = None
    with pytest.raises(NotImplementedError):
        Satellite(
            _epoch,
            _pos,
            _vel,
            _accel,
            _manuever_info,
            today="Tuesday",  # **kwargs, should raise NotImplementedError
        )


def test_Satellite_property_and_setters():
    _epoch = 54321.0
    _pos = np.zeros((3,))
    _vel = np.zeros((3,))
    _accel = np.zeros((3,))

    # --- Satellite w/ ImpulseManeuver
    _manuever_info = None
    sat = Satellite(
        _epoch,
        _pos,
        _vel,
        _accel,
        _manuever_info,
    )

    maneuver = ImpulsiveManeuver(
        54321.0 + 1.0,  # _time (mjd)
        np.ones((3,)),  # _dv
    )

    # --- @maneuver.setter
    assert sat.maneuver == None
    sat.maneuver = maneuver
    assert sat.maneuver == maneuver

    # --- @property : epoch
    assert sat.epoch == _epoch

    # --- @epoch.setter & @property
    assert sat.epoch == _epoch
    sat.epoch = _epoch + 1
    assert sat.epoch == _epoch + 1

    # --- @x.setter & @property
    assert all(sat.x == _pos)
    sat.x = _pos + 1
    assert all(sat.x == _pos + 1)

    # --- @v.setter & @property
    assert all(sat.v == _vel)
    sat.v = _vel + 1
    assert all(sat.v == _vel + 1)

    # --- @a.setter & @property
    assert all(sat.acc == _accel)
    sat.acc = _accel + 1
    assert all(sat.acc == _accel + 1)

    # --- Satellite.copy() test
    sat_copy = sat.copy()
    assert all(
        [
            sat_copy.does_maneuver == sat.does_maneuver,
            sat_copy.maneuver.time == sat.maneuver.time,  # type: ignore
            all(sat_copy.maneuver.dv == sat.maneuver.dv),  # type: ignore
            sat_copy.epoch == sat.epoch,
            all(sat_copy.x == sat.x),
            all(sat_copy.v == sat.v),
            all(sat_copy.acc == sat.acc),
        ]
    )

    # --- __str__() and __repr__() methods test
    sat_str = str(sat)
    sat_repr = repr(sat)
    assert isinstance(sat_str, str)
    assert isinstance(sat_repr, str)


def test_Satellite_invalid_instantiations():
    _epoch = 54321.0
    _pos = [1, 2, 3]
    _vel = np.zeros((3,))
    _accel = np.zeros((3,))

    with pytest.raises(TypeError):
        Satellite(
            _epoch,
            _pos,  # type: ignore
            _vel,
            _accel,
        )

    _pos = np.zeros((1, 3))

    with pytest.raises(ValueError):
        Satellite(
            _epoch,
            _pos,
            _vel,
            _accel,
        )


def test_Satellite_propagate_method(seed=seed):
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
    sat0 = madlib.Satellite(epoch, x0, v0, a0)  # maneuver = False
    sat1 = madlib.Satellite(epoch, x0, v0, a0, man)  # manuever = True

    # --- propagate, times is a scalar
    _times = epoch + 1
    _X, _V = sat0.propagate(_times)
    assert np.allclose(_X, [2.83737953e04, 3.11888127e04, -3.64526039e-01])
    assert np.allclose(_V, [-2.27434029e00, 2.06902511e00, 1.99407148e-04])

    # --- propagate, raise ValueError: One of the propagation times is the exact same time as the maneuver.
    with pytest.raises(ValueError):
        sat1.propagate(sat1.maneuver.time)  # type: ignore


def test_Satellite_from_GEO_longitude(seed=seed):
    np.random.seed(seed)
    # Satellite
    t_start = Time("2011-11-11T11:11:11", format="isot", scale="utc")
    epoch = t_start.mjd
    sat_longitude = 360 * np.random.random()
    sat_observed = madlib.Satellite.from_GEO_longitude(sat_longitude, epoch)

    assert sat_observed.epoch == epoch
    assert isinstance(sat_observed, madlib._satellite.Satellite)


def test_Satellite_from_keplerian(seed=seed, rtol=1e-3, atol=1e-3):
    np.random.seed(seed)
    # First, let's create a satellite from Cartesian coordinates we can use for comparison
    th = 2 * np.pi * np.random.rand()
    x0 = af.constants.Rgeo * np.array([np.cos(th), np.sin(th), 0.0])
    v0 = af.constants.Vgeo * np.array([-np.sin(th), np.cos(th), 0.0])
    a0 = np.zeros((3,))
    t_start = Time("2011-11-11T11:11:11", format="isot", scale="utc")
    epoch = t_start.mjd
    sat0 = madlib.Satellite(epoch, x0, v0, a0)
    assert all(sat0.x == x0)
    assert all(sat0.v == v0)
    assert all(sat0.acc == a0)

    # Convert Cartesian -> Keplerian elements
    keplerian_elements = af.coordinates.cartesian_to_keplerian(sat0.x, sat0.v)

    # Create Satellite from Keplerian elements
    sat1 = madlib.Satellite.from_keplerian(
        epoch=epoch,
        inclination_rad=keplerian_elements["inclination_rad"],
        raan_rad=keplerian_elements["raan_rad"],
        argp_rad=keplerian_elements["argp_rad"],
        ecc=keplerian_elements["eccentricity"],
        semi_major_axis_km=keplerian_elements["semi_major_axis_km"],
        mean_anomaly_rad=keplerian_elements["mean_anomaly_rad"],
    )

    # Check: sat1 should be the same as sat0
    assert np.allclose(sat0.x, sat1.x, rtol=rtol, atol=atol)
    assert np.allclose(sat0.v, sat1.v, rtol=rtol, atol=atol)
    assert np.allclose(sat0.acc, sat1.acc, rtol=rtol, atol=atol)

    # Let's do it again but with a bunch of randomly created satellite orbits
    for i in range(1000):
        sat_longitude = 360 * np.random.random()
        sat_rand = madlib.Satellite.from_GEO_longitude(sat_longitude, epoch)
        keplerian_elements = af.coordinates.cartesian_to_keplerian(
            sat_rand.x, sat_rand.v
        )
        sat_from_keplerian = madlib.Satellite.from_keplerian(
            epoch=epoch,
            inclination_rad=keplerian_elements["inclination_rad"],
            raan_rad=keplerian_elements["raan_rad"],
            argp_rad=keplerian_elements["argp_rad"],
            ecc=keplerian_elements["eccentricity"],
            semi_major_axis_km=keplerian_elements["semi_major_axis_km"],
            mean_anomaly_rad=keplerian_elements["mean_anomaly_rad"],
        )
        assert np.allclose(sat_rand.x, sat_from_keplerian.x, rtol=rtol, atol=atol)
        assert np.allclose(sat_rand.v, sat_from_keplerian.v, rtol=rtol, atol=atol)
        assert np.allclose(sat_rand.acc, sat_from_keplerian.acc, rtol=rtol, atol=atol)


def test_ContinuousThrustSatellite_edge_cases(seed=seed):
    np.random.seed(seed)
    # --- orbital state
    th = 2 * np.pi * np.random.rand()
    x0 = af.constants.Rgeo * np.array([np.cos(th), np.sin(th), 0.0])
    v0 = af.constants.Vgeo * np.array([-np.sin(th), np.cos(th), 0.0])
    a0 = np.zeros((3,))
    t_start = Time("2011-11-11T11:11:11", format="isot", scale="utc")
    epoch = t_start.mjd

    # --- ContinuousThrustSatellite w/ ContinuousManeuver
    # ContinuousThrust Function Definition
    def acc_f(t):
        return np.array([0.0, 1.0e-7, 0.0])

    acc_t_range = (epoch, epoch + 3)
    # ContinuousManeuver Definition
    man = madlib.ContinuousManeuver(acc_f, acc_t_range)
    sat = madlib.ContinuousThrustSatellite(
        epoch,
        x0,
        v0,
        a0,
        man,
    )

    # --- check case: _time_range input to ContinuousManeuver is reversed
    acc_t_range_rev = (acc_t_range[1], acc_t_range[0])
    man_rev_t = madlib.ContinuousManeuver(acc_f, acc_t_range_rev)
    assert acc_t_range == man_rev_t._time_range

    # --- propagate, times is a scalar
    _times = epoch + 1
    _X, _V = sat.propagate(_times)
    assert all(
        np.isclose(_X, np.array([2.79082377e04, 3.14487181e04, -3.31180976e-01]))
    )
    assert all(
        np.isclose(_V, np.array([-2.29974384e00, 2.05373079e00, 2.00473790e-04]))
    )


def test_ContinuousThrustSatellite_invalid_propagate(seed=seed):
    """Trying to apply a continuous thrust on an ordinary satellite should fail."""
    np.random.seed(seed)
    # --- orbital state
    th = 2 * np.pi * np.random.rand()
    x0 = af.constants.Rgeo * np.array([np.cos(th), np.sin(th), 0.0])
    v0 = af.constants.Vgeo * np.array([-np.sin(th), np.cos(th), 0.0])
    a0 = np.zeros((3,))
    t_start = Time("2011-11-11T11:11:11", format="isot", scale="utc")
    epoch = t_start.mjd

    # --- ContinuousThrustSatellite w/ ContinuousManeuver
    # ContinuousThrust Function Definition
    def acc_f(t):
        return np.array([0.0, 1.0e-7, 0.0])

    acc_t_range = (epoch, epoch + 3)
    # ContinuousManeuver Definition
    man = madlib.ContinuousManeuver(acc_f, acc_t_range)
    sat = madlib.Satellite(
        epoch,
        x0,
        v0,
        a0,
        man,
    )

    failed = False
    try:
        sat.propagate(epoch + 1)
    except MadlibException:
        failed = True

    assert failed


def test_ContinuousThrustSatellite_no_maneuver(seed=seed):
    """A ContinuousThrustSatellite with no maneuver should behave like
    a regular Satellite with no maneuver."""
    np.random.seed(seed)
    # --- orbital state
    th = 2 * np.pi * np.random.rand()
    x0 = af.constants.Rgeo * np.array([np.cos(th), np.sin(th), 0.0])
    v0 = af.constants.Vgeo * np.array([-np.sin(th), np.cos(th), 0.0])
    a0 = np.zeros((3,))
    t_start = Time("2011-11-11T11:11:11", format="isot", scale="utc")
    epoch = t_start.mjd

    # ContinuousManeuver Definition
    sat_c = madlib.ContinuousThrustSatellite(epoch, x0, v0, a0, None)

    # Ordinary satellite with no maneuver
    sat_n = madlib.Satellite(epoch, x0, v0)

    _times = epoch + 1
    Xc, Vc = sat_c.propagate(_times)
    Xn, Vn = sat_n.propagate(_times)
    assert np.allclose(Xc, Xn)
    assert np.allclose(Vc, Vn)


class TestTrueOrbits:

    th = 2 * np.pi * np.random.rand()
    x0 = af.constants.Rgeo * np.array([np.cos(th), np.sin(th), 0.0])
    v0 = af.constants.Vgeo * np.array([-np.sin(th), np.cos(th), 0.0])
    a0 = np.zeros((3,))
    t_start = Time("2011-11-11T11:11:11", format="isot", scale="utc")
    epoch: float = t_start.mjd

    th_true = th + 0.1
    x0_true = af.constants.Rgeo * np.array([np.cos(th_true), np.sin(th_true), 0.001])
    v0_true = af.constants.Vgeo * np.array([-np.sin(th_true), np.cos(th_true), 0.001])
    a0_true = a0 + 1e-5
    epoch_true = epoch + 0.1

    sat_default = madlib.Satellite(epoch, x0, v0)

    sat_modified = madlib.Satellite(
        epoch,
        x0,
        v0,
        a0,
        epoch_true=epoch_true,
        pos_true=x0_true,
        vel_true=v0_true,
        acc_true=a0_true,
    )

    def test_define_true_orbit(self):

        assert self.sat_default.epoch == self.sat_default.epoch_true
        assert self.sat_default.x_true is not None
        assert all(np.isclose(self.sat_default.x, self.sat_default.x_true))
        assert all(np.isclose(self.sat_default.v, self.sat_default.v_true))
        assert all(np.isclose(self.sat_default.acc, self.sat_default.acc_true))

        assert self.sat_modified.epoch != self.sat_modified.epoch_true
        assert self.sat_modified.x_true is not None
        assert all(~np.isclose(self.sat_modified.x, self.sat_modified.x_true))
        assert all(~np.isclose(self.sat_modified.v, self.sat_modified.v_true))
        assert all(~np.isclose(self.sat_modified.acc, self.sat_modified.acc_true))

    def test_propagate_true_orbit(self):
        x_default, v_default = self.sat_default.propagate(self.epoch + 0.5)
        x_default_true, v_default_true = self.sat_default.propagate(
            self.epoch + 0.5, use_true_orbit=True
        )

        x_modified, v_modified = self.sat_modified.propagate(self.epoch + 0.5)
        x_modified_true, v_modified_true = self.sat_modified.propagate(
            self.epoch + 0.5, use_true_orbit=True
        )

        assert np.allclose(x_default, x_default_true)
        assert np.allclose(v_default, v_default_true)
        assert ~np.allclose(x_modified, x_modified_true)
        assert ~np.allclose(v_modified, v_modified_true)

    def test_propagate_true_orbit_continuous(self):
        np.random.seed(0)
        # --- orbital state
        th = 2 * np.pi * np.random.rand()
        x0 = af.constants.Rgeo * np.array([np.cos(th), np.sin(th), 0.0])
        v0 = af.constants.Vgeo * np.array([-np.sin(th), np.cos(th), 0.0])
        a0 = np.zeros((3,))
        t_start = Time("2011-11-11T11:11:11", format="isot", scale="utc")
        epoch = t_start.mjd

        # --- ContinuousThrustSatellite w/ ContinuousManeuver
        # ContinuousThrust Function Definition
        def acc_f(t):
            return np.array([0.0, 1.0e-7, 0.0])

        acc_t_range = (epoch, epoch + 3)
        # ContinuousManeuver Definition
        man = madlib.ContinuousManeuver(acc_f, acc_t_range)

        th_true = th + 0.1
        x0_true = af.constants.Rgeo * np.array(
            [np.cos(th_true), np.sin(th_true), 0.001]
        )
        v0_true = af.constants.Vgeo * np.array(
            [-np.sin(th_true), np.cos(th_true), 0.001]
        )
        a0_true = a0 + 1e-5
        epoch_true = epoch + 0.1

        sat = madlib.ContinuousThrustSatellite(
            epoch,
            x0,
            v0,
            a0,
            man,
            epoch_true=epoch_true,
            pos_true=x0_true,
            vel_true=v0_true,
            acc_true=a0_true,
        )

        assert sat.epoch != sat.epoch_true
        assert sat.x_true is not None
        assert ~np.allclose(sat.x, sat.x_true)
        assert ~np.allclose(sat.v, sat.v_true)
        assert ~np.allclose(sat.acc, sat.acc_true)

        x, v = sat.propagate(epoch + 1.0)
        x_true, v_true = sat.propagate(epoch + 1.0, use_true_orbit=True)

        assert ~np.allclose(x, x_true)
        assert ~np.allclose(v, v_true)

    def test_setters(self):
        self.sat_default.epoch_true = self.epoch_true
        self.sat_default.x_true = self.x0_true
        self.sat_default.v_true = self.v0_true
        self.sat_default.acc_true = self.a0_true

        assert self.sat_default.epoch != self.sat_default.epoch_true
        assert self.sat_default.x_true is not None
        assert ~np.allclose(self.sat_default.x, self.sat_default.x_true)
        assert ~np.allclose(self.sat_default.v, self.sat_default.v_true)
        assert ~np.allclose(self.sat_default.acc, self.sat_default.acc_true)


class TestCrossTags:
    t_start = Time("2011-11-11T11:11:11", format="isot", scale="utc")
    epoch = t_start.mjd
    sat = madlib.Satellite.from_GEO_longitude(lon=0.0, epoch=epoch)

    def test_zero_cross_tag(self):
        cross_sat = self.sat.create_cross_tag(
            cross_mjd=self.epoch,
            delta_pos_km=np.zeros(3),
            delta_vel_kms=np.zeros(3),
        )

        assert np.isclose(cross_sat.epoch, self.sat.epoch, atol=0.1, rtol=0)
        assert np.allclose(cross_sat.x, self.sat.x)
        assert np.allclose(cross_sat.v, self.sat.v)
        assert np.allclose(cross_sat.acc, self.sat.acc)

        x, v = self.sat.propagate(self.epoch + 1)
        x_cross, v_cross = cross_sat.propagate(self.epoch + 1)

        assert np.allclose(x, x_cross)
        assert np.allclose(v, v_cross)

    def test_nonzero_cross_tag(self):
        cross_sat = self.sat.create_cross_tag(
            cross_mjd=self.epoch + 0.5,
            delta_pos_km=np.array([10, 10, 10]),
            delta_vel_kms=np.array([-5, -5, -5]),
        )

        assert ~np.isclose(cross_sat.epoch, self.sat.epoch, atol=0.1, rtol=0)
        assert ~np.allclose(cross_sat.x, self.sat.x)
        assert ~np.allclose(cross_sat.v, self.sat.v)

        x, v = self.sat.propagate(np.array([self.epoch + 0.5, self.epoch + 1.0]))
        x_cross, v_cross = cross_sat.propagate(
            np.array([self.epoch + 0.5, self.epoch + 1.0])
        )

        assert np.allclose(x[0] + 10, x_cross[0], rtol=0, atol=0.1)
        assert np.allclose(v[0] - 5, v_cross[0], rtol=0, atol=0.1)

        assert ~np.allclose(x[1], x_cross[1])
        assert ~np.allclose(v[1], v_cross[1])
