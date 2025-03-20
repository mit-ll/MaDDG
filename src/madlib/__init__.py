# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

"""
Maneuver Detection Library (MaDLib)
"""

from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

from ._maneuver import ContinuousManeuver, ImpulsiveManeuver
from ._observation import Observation, ObservationResidual
from ._satellite import ContinuousThrustSatellite, Satellite
from ._sensor import GroundOpticalSensor, SpaceOpticalSensor
from ._sensor_collection import SensorCollection

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "MaDDG"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
