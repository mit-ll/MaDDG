# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray


class ImpulsiveManeuver:
    """
    ImpulseManeuver class holds an impulse maneuver definition, which includes the time of the maneuver, and the impulsive delta-v.

    Properties
    ----------
    time : float
        Timestamp of the maneuver (MJD, UTC)
    dv : NDArray[np.float64]
        3D Array of the impulsive delta-v (RSW frame)
    """

    _time: float
    _dv: NDArray[np.float64]

    def __init__(self, time: float, dv: NDArray[np.float64]):
        """Initialize the ImpulseManeuver class.

        Parameters
        ----------
        time : float
            Timestamp of the maneuver (MJD, UTC)
        dv : NDArray[np.float64]
            3D Array of the impulsive delta-v (RSW frame)
        """
        self._time = time
        self._dv = dv

    @property
    def time(self) -> float:
        return self._time

    @property
    def dv(self) -> NDArray[np.float64]:
        return self._dv


class ContinuousManeuver:
    """
    ContinuousManeuver class holds a continuous maneuver definition, which includes the acceleration
    function defining the maneuver, and the time range over which the maneuver occurs.
    """

    _accel_func: Callable
    _time_range: Tuple[float, float]

    def __init__(self, f: Callable, time_range: Tuple[float, float]):
        """Create a ContinuousManeuver object

        Parameters
        ----------
        f : Callable
            Function which evaluates the acceleration at a given time
        time_range : Tuple[float, float]
            Time range within which this maneuver is valid
        """
        self._accel_func = f
        if time_range[0] >= time_range[1]:
            a, b = time_range
            time_range = (b, a)

        self._time_range = time_range

    def __call__(self, t: float) -> NDArray[np.float64]:
        """Compute the acceleration for this maneuver at the time given

        Parameters
        ----------
        t : float
            Timestamp (MJD, UT1) to evaluate the acceleration

        Returns
        -------
        NDArray[np.float64]
            Acceleration vector, shape: (3,). (TETED, km/s^2)
        """
        if self._time_range[0] <= t <= self._time_range[1]:
            return self._accel_func(t)

        return np.zeros(3)
