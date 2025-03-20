# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from abc import abstractmethod
from typing import Protocol, Tuple

import astroforge.coordinates as afc
import numpy as np
from astroforge import R_sun, R_earth
from numpy.typing import NDArray

from madlib._utils import MadlibException
from ._observation import Observation, ObservationCollection
from ._satellite import Satellite
from ._utils import calc_separation_angle


class _Sensor(Protocol):
    """Class is not yet implemented"""

    @abstractmethod
    def generate_obs_timing(self, start: float, end: float) -> NDArray[np.float64]:
        """Given a start time and an end time (in MJD) as well as the sensor's
        defined parameters, generate an array of observation times (also in MJD).

        Parameters
        ----------
        start : float
            Earliest possible observation timestamp (in MJD)
        end : float
            Latest possible observation timestamp (in MJD)

        Returns
        -------
        NDArray[np.float64]
            An array of times (in MJD) at which an observation will be made.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @abstractmethod
    def observe(self, sat: Satellite, times: float | NDArray[np.float64]):
        """Observe a given satellite with this sensor at the times given.

        Parameters
        ----------
        sat : Satellite
            Satellite object to observe
        times : float | NDArray[np.float64]
            Time(s) to observe the satellite. Specified as MJD in the UTC system.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError


class _OpticalSensor(_Sensor):
    """Generic class for optical sensors"""

    id: str | None

    # site metric accuracy
    dra: float  # arcsec
    ddec: float  # arcsec

    # timing
    obs_per_collect: int | tuple[int, int]
    obs_time_spacing: float
    collect_gap_mean: float
    collect_gap_std: float

    # observation limits
    obs_limits: dict | None

    def __init__(
        self,
        dra: float,
        ddec: float,
        collect_gap_mean: float,
        obs_limits: dict | None = None,
        collect_gap_std: float = 0.0,
        obs_per_collect: int | tuple[int, int] = 1,
        obs_time_spacing: float = 1.0,
        students_dof: int | None = None,
        id: str | None = None,
        cross_tag: Satellite | None = None,
        cross_tag_limit_arcsec: float = 100.0,
    ):
        """Construct an _OpticalSensor object

        Parameters
        ----------
        dra : float
            Metric accuracy in the right ascension direction
        ddec : float
            Metric accuracy in the declination direction
        collect_gap_mean : float
            Average time (seconds) between collects on a satellite
        obs_limits : dict | None, optional
            Dictionary of the limits on what this sensor can observe. This can be
            obvious (e.g. can't observe through the Earth), or based on sensor
            sensitivity (e.g. range limit for radar), by default None
        collect_gap_std : float, optional
            Standard deviation of times (seconds) between collects on a satellite, by default 0.0
        obs_per_collect : int | tuple[int, int], optional
            Typical number of observations per collect. This can be a constant, or
            a tuple of (min, max) for randomly sampling, by default 1
        obs_time_spacing : float, optional
            Time between observations within a collect, by default 1.0
        id : str | None, optional
            Unique Sensor ID, by default None
        cross_tag : madlib.Satellite | None, optional
            An optional nearby satellite that the sensor has misattributed
            as the target, by default None (no misattribution)
        cross_tag_limit_arcsec : float, optional
            The maximum allowed separation (in arcseconds) between the target's
            expected position and the cross_tag satellite. At further separations,
            observations of the misattributed satellite will be ignored.
            By default 100.0
        students_dof : int | None, optional
            Degrees of freedom to use for T-distribution when generating noise, by default None (Gaussian)
        """

        self.dra = dra
        self.ddec = ddec
        self.obs_per_collect = obs_per_collect
        self.obs_time_spacing = obs_time_spacing
        self.collect_gap_mean = collect_gap_mean
        self.collect_gap_std = collect_gap_std
        self.obs_limits = obs_limits
        self.id = id
        self.cross_tag = cross_tag
        self.cross_tag_limit_arcsec = cross_tag_limit_arcsec
        self.students_dof = students_dof

    def generate_obs_timing(self, start: float, end: float) -> NDArray[np.float64]:
        """Randomly generate a realistic time sampling of observations.

        Parameters
        ----------
        start : float
            Timestamp (MJD) corresponding to the beginning of the observing period
        end : float
            Timestamp (MJD) corresponding to the end of the observing period

        Returns
        -------
        NDArray[np.float64]
            Array of MJD timestamps corresponding to typical observing times on a satellite

        Raises
        ------
        ValueError
            Start timestamp must be before end timestamp
        """

        if start >= end:
            raise ValueError(f"start ({start}) must be before end ({end})")

        def gap_gen() -> float:
            return self.collect_gap_mean + self.collect_gap_std * np.random.randn()

        istuple = isinstance(self.obs_per_collect, (tuple, list))

        def nobs_gen() -> int:
            if istuple:
                return np.random.randint(
                    self.obs_per_collect[0], self.obs_per_collect[1] + 1  # type: ignore
                )
            else:
                return self.obs_per_collect  # type: ignore

        # --- generate collect times
        collect_times = []
        latest = start
        first = True
        while latest < end:
            # generate a gap time at random
            gap = gap_gen() / 86400

            # if this is the first gap, randomly scale it between 0-1
            # (otherwise the first observation will be biased late)
            if first:
                gap *= np.random.rand()
                first = False

            # add the gap time to the latest collect time, and append to the
            # list if it isn't too late
            latest += gap
            if latest < end:
                collect_times.append(latest)

        # --- generate observation times within each collect
        obs_times = []
        for coll in collect_times:
            # generate a (possibly) random number of observations in this collect
            nobs = nobs_gen()

            # compute the time of each observation given the collect start time
            times = coll + (np.arange(nobs) * self.obs_time_spacing) / 86400

            # handle edge effect at the end of the observing period
            # all observations in this collect must be before the end, otherwise
            # none are saved
            if (times < end).all():
                obs_times.append(times)

        if len(obs_times) > 0:
            obs_times = np.hstack(obs_times)
        else:
            obs_times = np.array([])
        return obs_times

    def validate_limits(self, obs: Observation) -> bool:
        """Determine whether Observation is possible based on the sensor limits.

        Parameters
        ----------
        obs : Observation
            madlib.Observation class for holding observables

        Returns
        -------
        bool
            Whether or not the Observation is possible
        """

        if not self.obs_limits:
            return True

        flag = True
        obsdict = obs.__dict__
        for key, val in self.obs_limits.items():
            if key in obsdict:
                if obsdict[key] is not None:
                    flag = flag and obsdict[key] > val[0] and obsdict[key] < val[1]
                else:
                    msg = f"ERROR: Invalid observation, value of {key} is None"
                    raise MadlibException(msg)
            else:
                msg = (
                    f"ERROR: {key} is an invalid variable for observation limits. "
                    "Check the YAML configuration for sensor {self.id}"
                )
                raise MadlibException(msg)

        return flag


class GroundOpticalSensor(_OpticalSensor):
    """Class for modeling the observing of a satellite with an optical sensor"""

    # site geography
    _site_reported_itrs: NDArray[np.float64]
    lat: float
    lon: float
    alt: float

    # site metric accuracy
    dra: float  # arcsec
    ddec: float  # arcsec

    # timing
    obs_per_collect: int | tuple[int, int]
    collect_gap_mean: float
    collect_gap_std: float

    # observation limits
    obs_limits: dict | None

    def __init__(
        self,
        lat: float,
        lon: float,
        alt: float,
        dra: float,
        ddec: float,
        collect_gap_mean: float,
        obs_limits: dict | None = None,
        collect_gap_std: float = 0.0,
        obs_per_collect: int | tuple[int, int] = 1,
        obs_time_spacing: float = 1.0,
        id: str | None = None,
        lat_truth: float | None = None,
        lon_truth: float | None = None,
        alt_truth: float | None = None,
        cross_tag: Satellite | None = None,
        cross_tag_limit_arcsec: float = 100.0,
        **extras,
    ):
        """Construct a GroundOpticalSensor object

        Parameters
        ----------
        lat : float
            Site geodetic latitude (deg)
        lon : float
            Site geodetic longitude (deg)
        alt : float
            Site altitude above WGS-84 reference ellipsoid
        dra : float
            Metric accuracy in the right ascension direction
        ddec : float
            Metric accuracy in the declination direction
        collect_gap_mean : float
            Average time (seconds) between collects on a satellite
        obs_limits : dict | None, optional
            Dictionary of the limits on what this sensor can observe. This can be
            obvious (e.g. can't observe through the Earth), or based on sensor
            sensitivity (e.g. range limit for radar), by default None
        collect_gap_std : float, optional
            Standard deviation of times (seconds) between collects on a satellite, by default 0.0
        obs_per_collect : int | tuple[int, int], optional
            Typical number of observations per collect. This can be a constant, or
            a tuple of (min, max) for randomly sampling, by default 1
        obs_time_spacing : float, optional
            Time between observations within a collect, by default 1.0
        id : str | None, optional
            Unique Sensor ID, by default None
        lat_truth : float | None, optional
            If not None, the latitude specified by <lat> is reported by
            the sensor as its position, but it is incorrect. The actual
            position is specified by <lat_truth>. By default None.
        lon_truth : float | None, optional
            If not None, the longitude specified by <lon> is reported by
            the sensor as its position, but it is incorrect. The actual
            position is specified by <lon_truth>. By default None.
        alt_truth : float | None, optional
            If not None, the altitude specified by <alt> is reported by
            the sensor as its position, but it is incorrect. The actual
            position is specified by <alt_truth>. By default None.
        cross_tag : madlib.Satellite | None, optional
            An optional nearby satellite that the sensor has misattributed
            as the target, by default None (no misattribution)
        cross_tag_limit_arcsec : float, optional
            The maximum allowed separation (in arcseconds) between the target's
            expected position and the cross_tag satellite. At further separations,
            observations of the misattributed satellite will be ignored.
            By default 100.0
        """

        super().__init__(
            dra=dra,
            ddec=ddec,
            collect_gap_mean=collect_gap_mean,
            obs_limits=obs_limits,
            collect_gap_std=collect_gap_std,
            obs_per_collect=obs_per_collect,
            obs_time_spacing=obs_time_spacing,
            id=id,
            cross_tag=cross_tag,
            cross_tag_limit_arcsec=cross_tag_limit_arcsec,
        )

        self.lat = lat
        self.lon = lon
        self.alt = alt
        self._site_reported_itrs = afc.LatLonAltToITRS(lat, lon, alt)

        self.lat_truth = lat
        self.lon_truth = lon
        self.alt_truth = alt

        if lat_truth is not None:
            self.lat_truth = lat_truth
        if lon_truth is not None:
            self.lon_truth = lon_truth
        if alt_truth is not None:
            self.alt_truth = alt_truth

        self._site_truth_itrs = afc.LatLonAltToITRS(
            self.lat_truth, self.lon_truth, self.alt_truth
        )

    def observe(
        self,
        target_satellite: Satellite,
        times: float | NDArray[np.float64] | Tuple[float, float],
    ) -> ObservationCollection:
        """Observe a satellite with this sensor model at the times given. Observations
        are computed in three forms:
            - truth: Measurements that would be returned if there were no random or
                systematic errors and knowledge of the orbit and maneuvers was perfect.
                These are the observations you would see in a perfect world.
            - expected: Measurements without random noise, but also without maneuver knowledge
                and with any systematic errors on sensor position and/or satellite orbit.
                These are the observations that you THINK you should get, given your actual knowledge
                of the system.
            - measured: The actual output of the simulated system, including all random and
                systematic sources of error and following the target satellite's full trajectory.
                This can also contain cross-tag events.

        Parameters
        ----------
        target_satellite : Satellite
            madlib.Satellite object to observe
        times : float | NDArray[np.float64] | Tuple[float, float]
            Time(s) to observe the satellite. Specified as MJD in the UTC system.
                - If a single float is given, observation will be made at just that time.
                - If a numpy array is given, observations will be made at each time in the array.
                - If a tuple of two floats is given, they will represent the start and end
                  of observations, and observation times will be generated accordingly

        Returns
        -------
        ObservationCollection
            A container object holding a realistic observation of the satellite
            given the sensor noise parameters and the "true" observation that
            excludes all noise sources.
        """

        # Handle scalar and tuple time inputs
        if isinstance(times, (float, int)):
            times = np.asarray([times])

        if isinstance(times, tuple):
            start_mjd = min(times)
            end_mjd = max(times)
            times = self.generate_obs_timing(start_mjd, end_mjd)

        num_obs = len(times)

        # Return empty ObservationCollection if there are no observation times
        if num_obs == 0:
            pos_observed = np.array([])
            pos_truth = np.array([])
            pos_expected = np.array([])

            return ObservationCollection(
                pos_observed=pos_observed,
                pos_truth=pos_truth,
                pos_expected=pos_expected,
            )

        # Propagate satellite to the desired times, with and without maneuvers
        x_target, v_target = target_satellite.propagate(times, use_true_orbit=True)
        x_target_expected, v_target_expected = target_satellite.propagate(
            times,
            ignore_maneuvers=True,
        )

        # Coordinate conversion to observables
        # (observables for an optical sensor are RA/Dec)
        # First rotate the site's reported and true locations into TETED
        x_site_reported, v_site_reported = self._site_loc_TETED(times)
        x_site_truth, v_site_truth = self._site_loc_TETED(times, use_true_location=True)

        # Now compute the true and expected RA/Dec, range, and range_rate
        (fp_state_truth, r_truth, r_rate_truth) = afc.PosVelToFPState(
            x_target, v_target, x_site_truth, v_site_truth
        )
        ra_truth = fp_state_truth[:, 0] * 180 / np.pi
        dec_truth = fp_state_truth[:, 1] * 180 / np.pi
        az_truth, el_truth = self._eci_to_az_el(x_target, times)

        (fp_state_expected, r_expected, r_rate_expected) = afc.PosVelToFPState(
            x_target_expected,
            v_target_expected,
            x_site_reported,
            v_site_reported,
        )
        ra_expected = fp_state_expected[:, 0] * 180 / np.pi
        dec_expected = fp_state_expected[:, 1] * 180 / np.pi
        az_expected, el_expected = self._eci_to_az_el(x_target_expected, times)

        # Compute solar elevation at observation times (assumed invariant to expectation (small errors))
        x_sun = np.array([R_sun(_t) for _t in times])
        az_sun, el_sun = self._eci_to_az_el(x_sun, times)

        # Collect into true (what you SHOULD see) and expected (what you THINK you'll see) observations
        obs_truth: np.ndarray[Observation, np.dtype[np.float64]] = np.array(
            [
                Observation(
                    mjd=times[n],
                    ra=ra_truth[n],
                    dec=dec_truth[n],
                    az=az_truth[n],
                    el=el_truth[n],
                    sun_el=el_sun[n],
                    sensor_id=self.id,
                )
                for n in range(num_obs)
            ]
        )

        obs_expected: np.ndarray[Observation, np.dtype[np.float64]] = np.array(
            [
                Observation(
                    mjd=times[n],
                    ra=ra_expected[n],
                    dec=dec_expected[n],
                    az=az_expected[n],
                    el=el_expected[n],
                    sun_el=el_sun[n],
                    sensor_id=self.id,
                )
                for n in range(num_obs)
            ]
        )

        ### Actual/Observed measurements

        # Apply misattributed observations (cross-tags), if requested
        if self.cross_tag is not None:
            x_target, v_target = self.cross_tag.propagate(times, use_true_orbit=True)

            # Coordinate conversion to observables using target's and sensor's true positions
            (fp_state_measured, r_measured, r_rate_measured) = afc.PosVelToFPState(
                x_target, v_target, x_site_truth, v_site_truth
            )
            ra_measured = fp_state_measured[:, 0] * 180 / np.pi
            dec_measured = fp_state_measured[:, 1] * 180 / np.pi

        # If this isn't a cross-tag, then the measurement is just the noise-addled truth
        else:
            r_measured = r_truth
            ra_measured = ra_truth
            dec_measured = dec_truth

        # Apply noise sources
        ra_delta = self.dra / 3600 * np.random.randn(num_obs)
        dec_delta = self.ddec / 3600 * np.random.randn(num_obs)
        ra_noisy = ra_measured + ra_delta
        dec_noisy = dec_measured + dec_delta

        x_noisy = np.zeros(x_target.shape)
        for n in range(num_obs):
            x_noisy[n] = x_site_truth[n] + r_measured[n] * spherical_to_cartesian(
                ra_noisy[n], dec_noisy[n]
            )

        az_noisy, el_noisy = self._eci_to_az_el(x_noisy, times)

        # Collect actual observables (what you DO see)
        obs_measured: np.ndarray[Observation, np.dtype[np.float64]] = np.array(
            [
                Observation(
                    mjd=times[n],
                    ra=ra_noisy[n],
                    dec=dec_noisy[n],
                    az=az_noisy[n],
                    el=el_noisy[n],
                    range_=None,  # N/A for optical sensors
                    range_rate=None,  # N/A for optical sensors
                    sun_el=el_sun[n],
                    sensor_id=self.id,
                )
                for n in range(num_obs)
            ]
        )

        # Remove observations that are not within the sensor limits
        valid = np.array([self.validate_limits(x) for x in obs_truth])
        obs_measured = obs_measured[valid]
        obs_truth = obs_truth[valid]
        obs_expected = obs_expected[valid]

        observations = ObservationCollection(
            pos_observed=obs_measured,
            pos_truth=obs_truth,
            pos_expected=obs_expected,
        )

        return observations

    def _eci_to_az_el(
        self, x: NDArray[np.float64], mjd: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Convert TETED positions to Az/El from the site.

        Parameters
        ----------
        x : NDArray[np.float64]
            TETED positions
        mjd : NDArray[np.float64]
            timestamps (mjd)

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            Az/El results
        """

        # rotate TETED -> ITRS -> SEZ -> Az/El
        out = []
        for n in range(len(x)):
            # breakpoint()
            xitrs = afc.TETEDToITRS(mjd[n], x[n])
            xsez = afc.ITRSToSEZ(xitrs, self._site_reported_itrs, self.lat, self.lon)
            az, el, _ = afc.SEZToAzElRange(xsez)
            out.append([az, el])

        out = np.vstack(out)
        return out[:, 0], out[:, 1]

    def _site_loc_TETED(
        self,
        times: NDArray[np.float64],
        use_true_location: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Rotate the site location to TETED for a given set of times.

        Parameters
        ----------
        times : NDArray[np.float64]
            timestamps at which to compute the site location
        use_true_location : bool, optional
            If True, use the sensor's true position (as opposed to reported)
            in the calculation. By default False.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            site locations for timestamps
        """

        if use_true_location:
            site_itrs = self._site_truth_itrs
        else:
            site_itrs = self._site_reported_itrs

        temp = np.vstack(
            [
                afc.PosVelConversion(
                    afc.ITRSToTETED, times[n], site_itrs, np.zeros((3,))
                )
                for n in range(len(times))
            ]
        )
        xsite = temp[0::2]
        vsite = temp[1::2]

        return xsite, vsite


class SpaceOpticalSensor(_OpticalSensor):
    """Class for modeling observations with an optical sensor in Earth orbit."""

    # site metric accuracy
    dra: float  # arcsec
    ddec: float  # arcsec

    # timing
    obs_per_collect: int | tuple[int, int]
    collect_gap_mean: float
    collect_gap_std: float

    # observation limits
    obs_limits: dict | None

    def __init__(
        self,
        sensor_satellite: Satellite,
        dra: float,
        ddec: float,
        collect_gap_mean: float,
        sensor_satellite_truth: Satellite | None = None,
        obs_limits: dict | None = None,
        collect_gap_std: float = 0.0,
        obs_per_collect: int | tuple[int, int] = 1,
        obs_time_spacing: float = 1.0,
        id: str | None = None,
        cross_tag: Satellite | None = None,
        cross_tag_limit_arcsec: float = 100.0,
        **extras,
    ):
        """Construct a SpaceOpticalSensor object

        Parameters
        ----------
        sensor_satellite : madlib.Satellite
            Satellite object governing sensor motion
        dra : float
            Metric accuracy in the right ascension direction
        ddec : float
            Metric accuracy in the declination direction
        collect_gap_mean : float
            Average time (seconds) between collects on a satellite
        sensor_satellite_truth: madlib.Satellite | None, optional
            If not None, this defines the *actual* orbit of the satellite, turning
            the <satellite> argument into the *expected* orbit. This will affect
            residual calculations. By default, None.
        obs_limits : dict | None, optional
            Dictionary of the limits on what this sensor can observe. This can be
            obvious (e.g. can't observe through the Earth), or based on sensor
            sensitivity (e.g. range limit for radar), by default None
        collect_gap_std : float, optional
            Standard deviation of times (seconds) between collects on a satellite, by default 0.0
        obs_per_collect : int | tuple[int, int], optional
            Typical number of observations per collect. This can be a constant, or
            a tuple of (min, max) for randomly sampling, by default 1
        obs_time_spacing : float, optional
            Time between observations within a collect, by default 1.0
        id : str | None, optional
            Unique Sensor ID, by default None
        cross_tag : madlib.Satellite | None, optional
            An optional nearby satellite that the sensor has misattributed
            as the target, by default None (no misattribution)
        cross_tag_limit_arcsec : float, optional
            The maximum allowed separation (in arcseconds) between the target's
            expected position and the cross_tag satellite. At further separations,
            observations of the misattributed satellite will be ignored.
            By default 100.0
        """

        super().__init__(
            dra=dra,
            ddec=ddec,
            collect_gap_mean=collect_gap_mean,
            obs_limits=obs_limits,
            collect_gap_std=collect_gap_std,
            obs_per_collect=obs_per_collect,
            obs_time_spacing=obs_time_spacing,
            id=id,
            cross_tag=cross_tag,
            cross_tag_limit_arcsec=cross_tag_limit_arcsec,
        )

        self.sensor_satellite = sensor_satellite
        self.sensor_satellite_truth = sensor_satellite_truth

    def observe(
        self,
        target_satellite: Satellite,
        times: float | NDArray[np.float64] | Tuple[float, float],
    ):
        """Observe a satellite with this sensor model at the times given. Observations
        are computed in three forms:
            - truth: Measurements that would be returned if there were no random or
                systematic errors and knowledge of the target's orbit and maneuvers was perfect.
                These are the observations you would see in a perfect world.
            - expected: Measurements without random noise, but also without maneuver knowledge
                and with any systematic errors on sensor position and/or satellite orbit.
                These are the observations that you THINK you should get, given your actual knowledge
                of the system.
            - measured: The actual output of the simulated system, including all random and
                systematic sources of error and following the target satellite's full trajectory.
                This can also contain cross-tag events.

        Parameters
        ----------
        target_satellite : Satellite
            madlib.Satellite object to observe
        times : float | NDArray[np.float64] | Tuple[float, float]
            Time(s) to observe the satellite. Specified as MJD in the UTC system.
                - If a single float is given, observation will be made at just that time.
                - If a numpy array is given, observations will be made at each time in the array.
                - If a tuple of two floats is given, they will represent the start and end
                  of observations, and observation times will be generated accordingly

        Returns
        -------
        ObservationCollection
            A container object holding a realistic observation of the satellite
            given the sensor noise parameters and the "true" observation that
            excludes all noise sources.
        """

        # Handle scalar and tuple time inputs
        if isinstance(times, (float, int)):
            times = np.asarray([times])

        if isinstance(times, tuple):
            start_mjd = min(times)
            end_mjd = max(times)
            times = self.generate_obs_timing(start_mjd, end_mjd)

        num_obs = len(times)

        # Return empty ObservationCollection if there are no observation times
        if num_obs == 0:
            pos_observed = np.array([])
            pos_truth = np.array([])
            pos_expected = np.array([])

            return ObservationCollection(
                pos_observed=pos_observed,
                pos_truth=pos_truth,
                pos_expected=pos_expected,
            )

        # Propagate sensor expected position to the desired times
        x_sensor_reported, v_sensor_reported = self.sensor_satellite.propagate(times)

        # If a true orbit has been given, propagate it as well
        if self.sensor_satellite_truth is not None:
            x_sensor_truth, v_sensor_truth = self.sensor_satellite_truth.propagate(
                times
            )
        else:
            x_sensor_truth = x_sensor_reported
            v_sensor_truth = v_sensor_reported

        # Propagate satellite's true and expected orbits to the desired times
        x_target, v_target = target_satellite.propagate(times, use_true_orbit=True)
        x_target_expected, v_target_expected = target_satellite.propagate(
            times, ignore_maneuvers=True
        )

        # Compute true and expected RA/Dec, range, and range_rate
        (fp_state_truth, r_truth, r_rate_truth) = afc.PosVelToFPState(
            x_target,
            v_target,
            x_sensor_truth,
            v_sensor_truth,
        )
        ra_truth = fp_state_truth[:, 0] * 180 / np.pi
        dec_truth = fp_state_truth[:, 1] * 180 / np.pi

        (fp_state_expected, r_expected, r_rate_expected) = afc.PosVelToFPState(
            x_target_expected,
            v_target_expected,
            x_sensor_reported,
            v_sensor_reported,
        )
        ra_expected = fp_state_expected[:, 0] * 180 / np.pi
        dec_expected = fp_state_expected[:, 1] * 180 / np.pi

        # Compute solar separation angle at observation times
        # (using the sensor's true orbit)
        x_sun = np.array([R_sun(_t) for _t in times])
        sun_angle_deg = calc_separation_angle(
            x_sun - x_sensor_truth,
            x_target - x_sensor_truth,
            in_deg=True,
        )

        # Collect into true (what you SHOULD see) and expected (what you THINK you'll see) observations
        obs_truth: np.ndarray[Observation, np.dtype[np.float64]] = np.array(
            [
                Observation(
                    mjd=times[n],
                    ra=ra_truth[n],
                    dec=dec_truth[n],
                    sun_separation=sun_angle_deg[n],
                    sensor_id=self.id,
                )
                for n in range(num_obs)
            ]
        )

        obs_expected: np.ndarray[Observation, np.dtype[np.float64]] = np.array(
            [
                Observation(
                    mjd=times[n],
                    ra=ra_expected[n],
                    dec=dec_expected[n],
                    sun_separation=sun_angle_deg[n],
                    sensor_id=self.id,
                )
                for n in range(num_obs)
            ]
        )

        ### Actual/Observed Measurements

        # Apply misattributed observations, if requested
        if self.cross_tag is not None:
            x_target, v_target = self.cross_tag.propagate(times, use_true_orbit=True)

            # Compute (observed) RA/Dec, range, and range_rate
            (fp_state_obs, r_obs, r_rate_obs) = afc.PosVelToFPState(
                x_target, v_target, x_sensor_truth, v_sensor_truth
            )
            ra_measured = fp_state_obs[:, 0] * 180 / np.pi
            dec_measured = fp_state_obs[:, 1] * 180 / np.pi

        # If this isn't a cross-tag, then the measurement is just the noise-addled truth
        else:
            ra_measured = ra_truth
            dec_measured = dec_truth

        # Apply noise sources
        ra_delta = self.dra / 3600 * np.random.randn(num_obs)
        dec_delta = self.ddec / 3600 * np.random.randn(num_obs)
        ra_noisy = ra_measured + ra_delta
        dec_noisy = dec_measured + dec_delta

        # Sun angle stays the same (we want to know if it's *actually*
        # too bright to observe)

        # Collect actual observables (what you DO see)
        obs_measured: np.ndarray[Observation, np.dtype[np.float64]] = np.array(
            [
                Observation(
                    mjd=times[n],
                    ra=ra_noisy[n],
                    dec=dec_noisy[n],
                    sun_separation=sun_angle_deg[n],
                    sensor_id=self.id,
                )
                for n in range(num_obs)
            ]
        )

        # Remove observations that are not within the sensor limits

        ## Any viewing vector that is obscured by Earth is not valid
        sensor_dist = np.linalg.norm(x_sensor_truth, axis=1)
        viewing_vector = x_target - x_sensor_truth
        ### Calculate the angle from center of Earth, to sensor, to target.
        ### Note: If this angle is obtuse, it is automatically valid.
        viewing_angle = np.pi - np.abs(
            calc_separation_angle(x_sensor_truth, viewing_vector)
        )
        ### Calculate minimum height above Earth for viewing vector
        min_view_height = sensor_dist * np.sin(viewing_angle)
        view_valid = (min_view_height > R_earth) | (viewing_angle > np.pi / 2)

        ## Check the other sensor limits, then combine
        valid = view_valid * np.array([self.validate_limits(x) for x in obs_truth])

        obs_measured = obs_measured[valid]
        obs_truth = obs_truth[valid]
        obs_expected = obs_expected[valid]

        observations = ObservationCollection(
            pos_observed=obs_measured,
            pos_truth=obs_truth,
            pos_expected=obs_expected,
        )

        return observations


def pos_to_lat_lon(
    pos: NDArray[np.float64], times: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute latitude & longitude for given position(s) and time(s).

    Parameters
    ----------
    pos : NDArray[np.float64]
        position vector
    times : NDArray[np.float64]
        array of timestamps

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        computed latitudes and longitudes

    Raises
    ------
    ValueError
        The number of positions and times must be equal (for now)
    """

    # TODO(jake): refactor this code so as not to make the assumption that pos is a 2D array
    if len(pos) != len(times):
        raise ValueError("The number of positions and times must be equal (for now)")
    N = len(pos)

    # first rotate from TETED to ITRS
    pos_itrs = np.vstack([afc.TETEDToITRS(times[n], pos[n]) for n in range(N)])

    # compute lat/lon/alt from ITRS position
    lla = np.vstack(
        [afc.ITRSToLatLonAlt(pos_itrs[n]) for n in range(N)], dtype=np.float64
    )

    return lla[:, 0], lla[:, 1]


def spherical_to_cartesian(ra: float, dec: float) -> NDArray[np.float64]:
    """Convert spherical coordinates (ra, dec) to cartesian coordinates.

    Parameters
    ----------
    ra : float
        Right Ascension
    dec : float
        Declination

    Returns
    -------
    NDArray[np.float64]
        Cartesian coordinates
    """
    ra *= np.pi / 180
    dec *= np.pi / 180
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
