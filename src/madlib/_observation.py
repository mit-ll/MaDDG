# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import List, Self

import numpy as np
from numpy.typing import NDArray

from ._utils import MadlibException


@dataclass(kw_only=True)
class Observation:
    """
    Class for holding observables. All angles are in degrees.

    Parameters
    ----------
    mjd : float
        Timestamp of the observation, described as a MJD in UTC

    ra : float | None
        Topocentric right ascension angle, by default None

    dec : float | None
        Topocentric declination angle, by default None

    az : float | None
        Azimuth angle, by default None

    el : float | None
        Elevation angle, by default None

    range_ : float | None
        Distance between sensor and target, by default None

    range_rate : float | None
        Time rate of change of the distance between the sensor and
        target, by default None

    lat : float | None
        Geodetic latitude, by default None

    lon : float | None
        Geodetic longitude, by default None

    sun_el : float | None
        Elevation angle of sun, by default None

    sun_separation : float | None
        Separation angle between target and sun, by default None

    sensor_id : str | None
        Unique Sensor ID, by default None
    """

    # time (MJD, UTC)
    mjd: float

    # standard observables for optics
    ra: float | None = None
    dec: float | None = None

    # standard observables for radar
    az: float | None = None
    el: float | None = None
    range_: float | None = None
    range_rate: float | None = None

    # non-standard observables
    lat: float | None = None
    lon: float | None = None

    # Position of sun
    sun_el: float | None = None
    sun_separation: float | None = None

    # Sensor ID for bookkeeping
    sensor_id: str | None = None

    _keys = [
        "ra",
        "dec",
        "az",
        "el",
        "range_",
        "range_rate",
        "lat",
        "lon",
        "sun_el",
        "sun_separation",
    ]

    def __sub__(self, other: Self) -> "ObservationResidual":
        """Subtracts another instance of Observation class.

        Parameters
        ----------
        other : Self
            The other instance of Observation class to subract

        Returns
        -------
        Self
            The subtracted result

        Raises
        ------
        MadlibException
            Can only subtract two Observation objects
        MadlibException
            Observations must be at the same time for computing a residual
        """

        if not isinstance(other, Observation):
            raise MadlibException("Can only subtract two Observation objects")

        d1 = self.__dict__
        d2 = other.__dict__
        diff = {
            key: d1[key] - d2[key]
            for key in self._keys
            if d1[key] is not None and d2[key] is not None
        }

        # special handling of angles that wrap at 0/360
        for key in ("ra", "az", "lon"):
            if d1[key] is not None and d2[key] is not None:
                temp = np.unwrap(np.array([d1[key], d2[key]]), period=360)
                diff[key] = temp[0] - temp[1]

        # add the timestamp to the dict
        if abs(d1["mjd"] - d2["mjd"]) > 1e-9:
            raise MadlibException(
                "Observations must be at the same time for computing a residual"
            )

        diff["mjd"] = d1["mjd"]

        # Remove the solar elevation
        _ = diff.pop("sun_el", None)

        return ObservationResidual(**diff)

    def asarray(self) -> NDArray[np.float64]:
        """Convert this observation to a flat 1-D array"""
        return np.array(
            [val if val is not None else np.NaN for val in self.__dict__.values()]
        )


@dataclass(kw_only=True)
class ObservationResidual:
    """
    Class for holding the difference between two observables.

    Parameters
    ----------
    mjd : float
        Timestamp of the observation, described as a MJD in UTC

    ra : float | None
        Topocentric right ascension angle difference, by default None

    dec : float | None
        Topocentric declination angle difference, by default None

    az : float | None
        Azimuth angle difference, by default None

    el : float | None
        Elevation angle difference, by default None

    range_ : float | None
        Distance between sensor and target difference, by default None

    range_rate : float | None
        Time rate of change of the distance between the sensor and target
        difference, by default None

    lat : float | None
        Geodetic latitude difference, by default None

    lon : float | None
        Geodetic longitude difference, by default None
    """

    # time (MJD, UTC)
    mjd: float

    # standard observables for optics
    ra: float | None = None
    dec: float | None = None

    # standard observables for radar
    az: float | None = None
    el: float | None = None
    range_: float | None = None
    range_rate: float | None = None

    # non-standard observables
    lat: float | None = None
    lon: float | None = None

    def asarray(self) -> NDArray[np.float64]:
        """Convert this observation to a flat 1-D array"""
        return np.array(
            [val if val is not None else np.NaN for val in self.__dict__.values()]
        )


@dataclass(kw_only=True)
class ObservationCollection:
    """Class for holding observed and true positions of satellites.

    Parameters
    ----------
    pos_observed: np.ndarray[Observation, np.dtype[np.float64]]
        Realistic observations of a satellite given sensor noise parameters

    pos_truth: np.ndarray[Observation, np.dtype[np.float64]]
        True observations of a satellite ignoring all noise sources

    pos_expected: np.ndarray[Observation, np.dtype[np.float64]]
        Observations expected if no noise or maneuvers occur

    Raises
    ------
    MadlibException
        Can only add two ObservationCollection objects
    """

    pos_observed: np.ndarray[Observation, np.dtype[np.float64]]
    pos_truth: np.ndarray[Observation, np.dtype[np.float64]]
    pos_expected: np.ndarray[Observation, np.dtype[np.float64]]

    def __add__(self, other: "ObservationCollection"):
        self.pos_observed = np.concatenate((self.pos_observed, other.pos_observed))
        self.pos_truth = np.concatenate((self.pos_truth, other.pos_truth))
        self.pos_expected = np.concatenate((self.pos_expected, other.pos_expected))

    def count_valid_observations(self):
        return len(self.pos_observed)


def combineObsCollections(
    collectionList: List[ObservationCollection],
) -> ObservationCollection:
    """Given observations of a satellite from multiple sensors, combine them into a single object.

    Parameters
    ----------
    collectionList : List[ObservationCollection]
        List of observations from multiple sensors of a single satellite.

    Returns
    -------
    ObservationCollection
        Combined collection of observations from all sensors.
    """

    pos_observed = np.concatenate([col.pos_observed for col in collectionList])
    pos_truth = np.concatenate([col.pos_truth for col in collectionList])
    pos_expected = np.concatenate([col.pos_expected for col in collectionList])

    combinedCollection = ObservationCollection(
        pos_observed=pos_observed,
        pos_truth=pos_truth,
        pos_expected=pos_expected,
    )

    return combinedCollection
