# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

"""
Implementation of the Satellite class, which models the propagation of a
satellite (with or without a maneuver).
"""

import copy
from typing import Optional, Self

import astroforge as af
import numpy as np
from astropy import time
from numpy.typing import NDArray

from astroforge.coordinates import keplerian_to_cartesian
from astroforge.constants import GM as GM_Earth

from ._maneuver import ContinuousManeuver, ImpulsiveManeuver
from ._utils import MadlibException


class Satellite:
    """
    Satellite class for propagating a satellite (either with or without a maneuver).
    """

    _maneuver: Optional[ImpulsiveManeuver] | Optional[ContinuousManeuver] | None = (
        None  # contains maneuver information, if it exists
    )
    _epoch: float  # epoch of the satellite orbit (MJD, UTC)
    _xx: NDArray[np.float64]  # satellite position at epoch, shape = (3,)
    _vv: NDArray[np.float64]  # satellite velocity at epoch, shape = (3,)
    _aa: NDArray[np.float64]  # satellite anomalous acceleration at epoch, shape = (3,)

    # Optional TRUE orbit
    _epoch_true: float
    _xx_true: NDArray[np.float64]
    _vv_true: NDArray[np.float64]
    _aa_true: NDArray[np.float64]

    def __init__(
        self,
        epoch: float,
        pos: NDArray[np.float64],
        vel: NDArray[np.float64],
        acc: NDArray[np.float64] = np.zeros((3,)),
        maneuver_info: (
            Optional[ImpulsiveManeuver] | Optional[ContinuousManeuver] | None
        ) = None,
        epoch_true: float | None = None,
        pos_true: NDArray[np.float64] | None = None,
        vel_true: NDArray[np.float64] | None = None,
        acc_true: NDArray[np.float64] | None = None,
        **kwargs,
    ):
        """Instantiate a Satellite object.

        Parameters
        ----------
        epoch : float
            Timestamp (MJD, UTC) indicating the time at which this orbital state is defined
        pos : NDArray[np.float64]
            Vector describing the satellite position at epoch.
            Coordinate system is TETED. Units are km. Shape: (3,)
        vel : NDArray[np.float64]
            Vector describing the satellite velocity at epoch.
            Coordinate system is TETED. Units are km/s. Shape: (3,)
        acc : NDArray[np.float64], optional
            Vector describing the satellite anomalous acceleration at epoch.
            Coordinate system is TETED. Units are km/s/s. Shape: (3,), by default np.zeros((3,))
        maneuver_info : Optional[ImpulsiveManeuver], optional
            Maneuver object describing when and how the satellite performs an
            impulsive maneuver, by default None
        epoch_true : float | None, optional
            If not None, then this is the ACTUAL epoch of the orbit, and <epoch> is
            a reported value with some error. Units are MJD in UTC system. Default is None.
        pos_true : NDArray[np.float64] | None, optional
            If not None, then this vector is the satellite's ACTUAL position at epoch,
            and <pos> is a reported value with some error. Coordinate system is TETED. Units
            are km. Shape: (3,). Default is None.
        vel_true : NDArray[np.float64] | None, optional
            If not None, then this vector is the satellite's ACTUAL velocity at epoch,
            and <vel> is a reported value with some error. Coordinate system is TETED. Units
            are km/s. Shape: (3,). Default is None.
        acc_true : NDArray[np.float64] | None, optional
            If not None, then this vector is the satellite's ACTUAL anomalous acceleration at epoch,
            and <acc> is a reported value with some error. Coordinate system is TETED. Units
            are km/s/s. Shape: (3,). Default is None.

        Raises
        ------
        NotImplementedError
            kwargs are not yet supported
        """

        self.validate_input_vector(pos, "Position")
        self.validate_input_vector(vel, "Velocity")
        self.validate_input_vector(acc, "Anomalous Acceleration")

        self._epoch = epoch
        self._xx = pos
        self._vv = vel
        self._aa = acc
        self._maneuver = maneuver_info

        self.true_orbit_exists = False

        if epoch_true is not None:
            self.true_orbit_exists = True
            self.epoch_true = epoch_true
        else:
            self._epoch_true = epoch

        if pos_true is not None:
            self.true_orbit_exists = True
            self._xx_true = pos_true
        else:
            self._xx_true = pos

        if vel_true is not None:
            self.true_orbit_exists = True
            self._vv_true = vel_true
        else:
            self._vv_true = vel

        if acc_true is not None:
            self.true_orbit_exists = True
            self._aa_true = acc_true
        else:
            self._aa_true = acc

        if len(kwargs.items()) != 0:
            raise NotImplementedError("kwargs are not yet supported")

    @property
    def does_maneuver(self) -> bool:
        return self._maneuver is not None

    @property
    def maneuver(self) -> ImpulsiveManeuver | ContinuousManeuver | None:
        return self._maneuver

    @maneuver.setter
    def maneuver(self, val: ImpulsiveManeuver | ContinuousManeuver | None):
        self._maneuver = val

    @property
    def epoch(self) -> float:
        return self._epoch

    @epoch.setter
    def epoch(self, val: float):
        self._epoch = val

    @property
    def epoch_true(self) -> float:
        return self._epoch_true

    @epoch_true.setter
    def epoch_true(self, val: float):
        self.true_orbit_exists = True
        self._epoch_true = val

    @property
    def x(self) -> NDArray[np.float64]:
        return self._xx

    @x.setter
    def x(self, val: NDArray[np.float64]):
        self.validate_input_vector(val, "Position")
        self._xx = val

    @property
    def x_true(self) -> NDArray[np.float64] | None:
        return self._xx_true

    @x_true.setter
    def x_true(self, val: NDArray[np.float64]):
        self.validate_input_vector(val, "Position")
        self.true_orbit_exists = True
        self._xx_true = val

    @property
    def v(self) -> NDArray[np.float64]:
        return self._vv

    @v.setter
    def v(self, val: NDArray[np.float64]):
        self.validate_input_vector(val, "Velocity")
        self._vv = val

    @property
    def v_true(self) -> NDArray[np.float64]:
        return self._vv_true

    @v_true.setter
    def v_true(self, val: NDArray[np.float64]):
        self.validate_input_vector(val, "Velocity")
        self.true_orbit_exists = True
        self._vv_true = val

    @property
    def acc(self) -> NDArray[np.float64]:
        return self._aa

    @acc.setter
    def acc(self, val: NDArray[np.float64]):
        self.validate_input_vector(val, "Anomalous Acceleration")
        self._aa = val

    @property
    def acc_true(self) -> NDArray[np.float64]:
        return self._aa_true

    @acc_true.setter
    def acc_true(self, val: NDArray[np.float64]):
        self.validate_input_vector(val, "Anomalous Acceleration")
        self.true_orbit_exists = True
        self._aa_true = val

    def copy(self) -> Self:
        return copy.deepcopy(self)

    def validate_input_vector(self, input_vector, input_name):
        if not isinstance(input_vector, np.ndarray):
            raise TypeError(f"{input_name} must be a NumPy array")

        if input_vector.shape != (3,):
            raise ValueError(f"{input_name} must have shape (3,)")

    def propagate(
        self,
        times: float | NDArray[np.float64],
        ignore_maneuvers: bool = False,
        use_true_orbit: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Propagates the satellite to the time(s) given.

        Parameters
        ----------
        times : float | NDArray[np.float64]
            Timestamp(s) for evaluating the satellite propagation (MJD, UTC)
        ignore_maneuvers : bool
            If True, propagate without applying any maneuvers, by default False.
        use_true_orbit : bool
            If True, and if input values were given for <epoch_true>, <pos_true>,
            <vel_true>, or <aa_true>, use these values for the true orbit to propagate
            instead of the reported values (<epoch>, <pos>, <vel>, <aa>)

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            First element of tuple is X : (N,3), position vector(s) at the time(s)
            requested. Coordinate system is TETED. Units are km.
            Second element of tuple is V : (N,3), velocity vector(s) at the time(s)
            requested, Coordinate system is TETED. Units are km/s.

        Raises
        ------
        ValueError
            One of the propagation times is the exact same time as the maneuver.
        """

        if isinstance(self.maneuver, ContinuousManeuver) and not ignore_maneuvers:
            raise MadlibException(
                "You cannot call Satellite.propagate() when Satellite._manuever is not type ImpulseManeuver. "
                "If you are trying to propagate a satellite with a ContinuousManeuver, use Satellite "
                "subclass ContinuousThrustSatellite."
            )

        if use_true_orbit and self.true_orbit_exists:
            epoch = self._epoch_true
            xx = self._xx_true
            vv = self._vv_true
            aa = self._aa_true
        else:
            epoch = self._epoch
            xx = self._xx
            vv = self._vv
            aa = self._aa

        scalar_input = False
        if isinstance(times, (float, int)):
            scalar_input = True
            times = np.asarray([times])

        # If the epoch is included in the propagation times, it could cause an
        # error in the numerical solver. Slightly perturb any such cases.
        times[times == epoch] += 1e-8

        if (
            (not self.does_maneuver)
            or (ignore_maneuvers)
            or (self.maneuver is not None and self.maneuver.time > times.max())
        ):
            # first propagate from the epoch to the first requested time
            T = np.array([epoch, times[0]])
            X, V = af.propagators.mp_srp(xx, vv, aa, T)
            X0, V0 = X[1], V[1]

            # no need to propagate further if only one time was provided
            if scalar_input or len(times) == 1:
                return (
                    X0[np.newaxis, :],
                    V0[np.newaxis, :],
                )  # Reshape needed for consistency

            # because there isn't a maneuver we can just propagate to all of the times all at once
            X, V = af.propagators.mp_srp(X0, V0, aa, times)
            return X, V
        else:
            # satellite does a maneuver sometime within the requested times

            # save the maneuver info, check for requested times at the same time as the maneuver
            mantime = self.maneuver.time  # type: ignore
            mandv = self.maneuver.dv  # type: ignore
            if mantime in times:
                raise ValueError(
                    "One of the propagation times is the exact same time as the maneuver."
                )

            # list of pos/vels from the various time segments
            Xkeep, Vkeep = [], []

            # propagate from the epoch through all of the requested pre-maneuver
            # times (if exist), and finally to the maneuver time
            T = np.hstack([epoch, times[times < mantime], mantime])
            X, V = af.propagators.mp_srp(xx, vv, aa, T)
            Xkeep.append(X[1:-1])
            Vkeep.append(V[1:-1])

            # apply delta-v at the time of the maneuver
            # first rotate the maneuver into ECI space
            x, v = X[-1], V[-1]
            R = self.rv2rsw(x, v)
            dv_eci = R.T @ mandv
            Xpm = x
            Vpm = v + dv_eci

            # propagate from the maneuver time to the rest of the requsted times
            # Note: there should be times after the maneuver time, otherwise we would have
            # entered the if above (self.maneuver.time > times.max())
            T = np.hstack([mantime, times[times > mantime]])
            X, V = af.propagators.mp_srp(Xpm, Vpm, aa, T)
            Xkeep.append(X[1:])
            Vkeep.append(V[1:])

            # stack the pos/vels to be a single array
            X = np.vstack(Xkeep)
            V = np.vstack(Vkeep)

            return X, V

    def create_cross_tag(
        self,
        cross_mjd: float,
        delta_pos_km: NDArray[np.float64],
        delta_vel_kms: NDArray[np.float64],
    ):
        """Create a satellite that will cross nearby this satellite,
        potentially creating a series of misattribution events.

        Args:
            cross_mjd : float
                A time (in MJD) at which the two objects will be nearby
            delta_pos_km : NDArray[np.float64]
                A vector representing the distance between the two satellites
                at time <cross_mjd> (ECI, km)
            delta_vel_kms : NDArray[np.float64]
                A vector representing the difference in the two satellites'
                velocities at time <cross_mjd> (ECI, km/s)

        Returns:
            Satellite
                The Satellite object for the crossing satellite, with epoch
                set at <cross_mjd>
        """

        # If the cross time is close to the epoch time, just use the epoch pos/vel
        if np.isclose(cross_mjd, self.epoch, rtol=0, atol=1e-6):
            x = self.x
            v = self.v
        else:
            # Propagate the satellite's initial orbit to the cross_mjd
            x, v = self.propagate(cross_mjd, ignore_maneuvers=True)
            # Satellite expects position as shape (3,) but x and v will be (1,3)
            x = x[0]
            v = v[0]

        # Add offsets.
        x_cross = x + delta_pos_km
        v_cross = v + delta_vel_kms

        cross_sat = Satellite(epoch=cross_mjd, pos=x_cross, vel=v_cross)

        return cross_sat

    @staticmethod
    def rv2rsw(r: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute rotation matrix from ECI frame to RSW frame.

        Parameters
        ----------
        r : NDArray[np.float64]
            Position vector (ECI, km)
        v : NDArray[np.float64]
            Velocity vector (ECI, km/s)

        Returns
        -------
        NDArray[np.float64]
            Rotation matrix from ECI frame to RSW frame
        """

        unit = lambda z: z / np.sqrt(z @ z)

        # get unit vector in each dimension
        # radial
        rvec = unit(r)

        # cross-track
        wvec = np.cross(r, v)
        wvec = unit(wvec)

        # along-track
        svec = np.cross(wvec, rvec)
        svec = unit(svec)

        # create rotation matrix
        T = np.zeros((3, 3))
        T[0, :] = rvec
        T[1, :] = svec
        T[2, :] = wvec

        return T

    @classmethod
    def from_GEO_longitude(cls, lon: float, epoch: float) -> Self:
        """Create a Satellite in GEO at the given longitude

        Parameters
        ----------
        lon : float
            Longitude of the GEO satellite (deg)
        epoch : float
            Timestamp (MJD, UTC) indicating the time at which this orbital state is defined

        Returns
        -------
        Self
            Instance of the Satellite class
        """

        lon = np.deg2rad(lon)

        # note: this only works for GEO because GEO has zero velocity in ITRS
        x_itrs = af.constants.Rgeo * np.array([np.cos(lon), np.sin(lon), 0.0])
        x, v = af.coordinates.PosVelConversion(
            af.coordinates.ITRSToTETED, epoch, x_itrs, np.zeros(3)
        )
        return cls(epoch, x, v)

    @classmethod
    def from_keplerian(
        cls,
        epoch: float,
        inclination_rad: float,
        raan_rad: float,
        argp_rad: float,
        ecc: float,
        semi_major_axis_km: float,
        mean_anomaly_rad: float,
        GM: float = GM_Earth,
    ) -> Self:
        """Create a Satellite from Keplerian elements

        Parameters
        ----------
        epoch : float
            Timestamp (MJD, UTC) indicating the time at which this orbital state is defined
        inclination_rad : float
            Orbital inclination in radians
        raan_rad : float
            Right ascension of the ascending node in radians
        argp_rad : float
            Argument of pericenter in radians
        ecc : float
            Eccentricity, a number between 0 and 1
        semi_major_axis_km : float
            Semi-major axis in km
        mean_anomaly_rad : float
            Mean anomaly in radians
        GM : float, optional
            Standard gravitational parameter (the product of the gravitational constant and the mass of a given astronomical body such as the Sun or Earth) in km^3/s^2. Earth's standard gravitational parameter by default (398600.4418 km^3/s^2)

        Returns
        -------
        Self
            Instance of the Satellite class
        """

        x, v = keplerian_to_cartesian(
            inclination_rad,
            raan_rad,
            argp_rad,
            ecc,
            semi_major_axis_km,
            mean_anomaly_rad,
            GM,
        )

        # Reshape from (1,3) to (3,)
        x = x.ravel()
        v = v.ravel()

        return cls(epoch, x, v)

    def __str__(self):
        epochstr = time.Time(self.epoch, format="mjd").isot
        s = ""
        s += "Satellite:\n"
        s += f"  epoch = {epochstr}\n"
        s += f"  x = {self.x.tolist()}\n"
        s += f"  v = {self.v.tolist()}\n"
        s += f"  acc = {self.acc.tolist()}"
        return s

    def __repr__(self):
        return self.__str__()


class ContinuousThrustSatellite(Satellite):
    """Satellite subclass for propagating an orbit that uses continuous thrust (i.e. ContinuousManeuver)"""

    def __init__(
        self,
        epoch: float,
        pos: NDArray[np.float64],
        vel: NDArray[np.float64],
        acc: NDArray[np.float64] = np.zeros((3,)),
        maneuver_info: Optional[ContinuousManeuver] | None = None,
        epoch_true: float | None = None,
        pos_true: NDArray[np.float64] | None = None,
        vel_true: NDArray[np.float64] | None = None,
        acc_true: NDArray[np.float64] | None = None,
        **kwargs,
    ):
        super().__init__(
            epoch,
            pos,
            vel,
            acc,
            maneuver_info,
            epoch_true=epoch_true,
            pos_true=pos_true,
            vel_true=vel_true,
            acc_true=acc_true,
        )

        self._propfun = self._setup_propfun()

    def propagate(
        self,
        times: float | NDArray[np.float64],
        ignore_maneuvers: bool = False,
        use_true_orbit: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Top level continuous thrust propagate method. Propagates the satellite to the time(s) given.

        Parameters
        ----------
        times : float | NDArray[np.float64]
            Timestamp(s) for evaluating the satellite propagation (MJD, UTC)
        ignore_maneuvers : bool
            If True, propagate without applying any maneuvers, by default False.
        use_true_orbit : bool
            If True, and if input values were given for <epoch_true>, <pos_true>,
            <vel_true>, or <aa_true>, use these values for the true orbit to propagate
            instead of the reported values (<epoch>, <pos>, <vel>, <aa>)

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            First element of tuple is X : (N,3), position vector(s) at the time(s)
            requested. Coordinate system is TETED. Units are km.
            Second element of tuple is V : (N,3), velocity vector(s) at the time(s)
            requested, Coordinate system is TETED. Units are km/s.
        """

        scalar_input = False
        if isinstance(times, (float, int)):
            scalar_input = True
            times = np.asarray([times])

        if ignore_maneuvers:
            return super().propagate(
                times,
                ignore_maneuvers=True,
                use_true_orbit=use_true_orbit,
            )

        if use_true_orbit and self.true_orbit_exists:
            epoch = self._epoch_true
            xx = self._xx_true
            vv = self._vv_true
            aa = self._aa_true
        else:
            epoch = self._epoch
            xx = self._xx
            vv = self._vv
            aa = self._aa

        # do an initial propagation to the first requested time, that way
        # we can ensure the propagation need only go one direction
        T = np.array([epoch, times[0]])
        X, V = self._propfun(xx, vv, aa, T)
        X0, V0 = X[1], V[1]

        # no need to propagate further if only one time was provided
        if scalar_input or len(times) == 1:
            return X0, V0

        X, V = self._propfun(X0, V0, aa, times)
        return X, V

    def _setup_propfun(self):
        # the static force model used in numerical integration; this does not include the continuous thrust
        self._fm = af.force_models.F_mp_srp

        def full_eom(t: float, xxdot: NDArray[np.float64]) -> NDArray[np.float64]:
            """Full equations of motion (EOM) to be used by the propagator's ODE solver

            Parameters
            ----------
            t : float
                Timestamp for evaluating the satellite propagation
            xxdot : NDArray[np.float64]
                Position, velocity, & acceleration state vectors concatenated
                into a single 1D array, shape: (9,)

            Returns
            -------
            NDArray[np.float64]
                The derivative, velocity, acceleration and jerk concatenated into a single
                array of shape: (9,). Acceleration due to continuous thrust acceleration
                added, if applicable.
            """
            dx = self._fm(t, xxdot)
            if self.maneuver is not None and callable(self.maneuver):
                dx[3:6] += self.maneuver(t / 86400.0)

            return dx

        def propfun(
            x0: NDArray[np.float64],
            v0: NDArray[np.float64],
            alpha: NDArray[np.float64],
            T: NDArray[np.float64],
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            """ContinuousThrustSatellite object propagation function

            Parameters
            ----------
            x0 : NDArray[np.float64]
                Position vector, shape: (3,)
            v0 : NDArray[np.float64]
                Velocity vector, shape: (3,)
            alpha : NDArray[np.float64]
                Acceleration vector, shape: (3,)
            T : NDArray[np.float64]
                Array of timestamps (MJD, UTC) to propagate to, shape: (N,)

            Returns
            -------
            tuple[NDArray[np.float64], NDArray[np.float64]]
                First element of tuple is X : (N,3), position vector(s) at the time(s)
                requested.
                Second element of tuple is V : (N,3), velocity vector(s) at the time(s)
                requested.
            """
            y0 = np.hstack([x0, v0, alpha])
            T = T.copy()
            T += af.coordinates.dut1utc(T) / 86400.0
            T *= 86400.0

            out = af.propagators.propagator(full_eom, y0, T, atol=1e-9, rtol=1e-9)

            X = out[:, :3]
            V = out[:, 3:6]
            return (X, V)

        return propfun
