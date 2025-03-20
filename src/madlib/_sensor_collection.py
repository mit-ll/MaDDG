# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import List, Sequence

import numpy as np
import yaml
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from numpy.typing import NDArray

from madlib._utils import MadlibException, sensor_yaml_schema

from ._observation import ObservationCollection, combineObsCollections
from ._satellite import Satellite
from ._sensor import GroundOpticalSensor, _Sensor


class SensorException(Exception):
    """SensorException class"""

    def __init__(self, message=None):
        super().__init__(message)


class SensorCollection:
    """Class containing multiple sensor objects that can generate a
    comprehensive observing schedule and collate observations."""

    def __init__(self, sensorList: Sequence[_Sensor]):
        """Initialize SensorCollection class

        Parameters
        ----------
        sensorList : List[Sensor]
            List of Sensors to include in observation network
        """
        self.sensorList = list(sensorList)
        self.numSensors = len(sensorList)

        self.obsTimes: List[NDArray[np.float64]] | None = None

    @staticmethod
    def paramsFromYAML(
        yaml_file: str | Path,
    ):
        """Parse a YAML into sensor parameters, returned as a list of dicts"""
        try:
            with open(yaml_file, "r") as f:
                sensor_data = yaml.safe_load(f)
                validate(sensor_data, sensor_yaml_schema)
        except ValidationError as e:
            if e.message == "'sensor_list' is a required property":
                msg = f"The sensor YAML file {yaml_file} must contain the top-level property 'sensor_list'."
            else:
                e_path = str(e.path)
                section = "/".join(list(e_path))
                msg = (
                    f"Error in the sensor YAML file [{yaml_file}] in the entry [{section}]: "
                    f"{e.message}"
                )
            raise MadlibException(msg) from None

        sensor_data = sensor_data["sensor_list"]

        return sensor_data

    @classmethod
    def fromYAML(cls, yaml_file: str):
        """Instantiate a SensorCollection object from a YAML file

        Parameters
        ----------
        yaml_file : str
            Path to YAML file defining the sensors in the collection
        """
        sensor_data = cls.paramsFromYAML(yaml_file)
        sensors = [GroundOpticalSensor(**params) for key, params in sensor_data.items()]
        sensor_network = cls(sensors)

        return sensor_network

    def generate_obs_timing(self, start: float, end: float):
        """Given a start time and an end time (in MJD), generate an
        array of observation times (also in MJD) based on the sensors'
        defined parameters.

        Parameters
        ----------
        start : float
            Earliest possible observation timestamp (MJD)
        end : float
            Latest possible observation (MJD)

        Raises
        ------
        SensorException
            The observation schedule has already been generated.
        """
        if self.obsTimes is not None:
            message = "The observation schedule has already been generated."
            raise SensorException(message=message)

        self.obsTimes = [
            sensor.generate_obs_timing(start, end) for sensor in self.sensorList
        ]

    def add_sensor(self, sensor: _Sensor):
        """Add a new sensor to the existing collection, provided the
        sensor timing has not already been generated.

        Parameters
        ----------
        sensor : _Sensor
            The sensor object to add to the collection
        """
        if self.obsTimes is not None:
            message = (
                "Cannot add new sensors to a SensorCollection "
                "if observation timing has already been generated."
            )
            raise (SensorException(message=message))

        self.sensorList.append(sensor)
        self.numSensors = len(self.sensorList)

    def observe(self, target_satellite: Satellite) -> ObservationCollection:
        """Given a madlib.Satellite, generate an ObservationCollection

        Parameters
        ----------
        target_satellite : Satellite
            madlib.Satellite is a class for propagating a satellite

        Returns
        -------
        ObservationCollection
            Observations of a satellite from multiple sensors, combined into a single object.

        Raises
        ------
        SensorException
            The observation schedule has already been generated.
        """
        if self.obsTimes is None:
            message = "obsTimes is None. Did you forget to generate obs timing first?"
            raise SensorException(message=message)
        else:
            obsCollections: List[ObservationCollection] = [
                sensor.observe(target_satellite, obstimes)
                for sensor, obstimes in zip(self.sensorList, self.obsTimes)
            ]
            observations: ObservationCollection = combineObsCollections(obsCollections)

        return observations
