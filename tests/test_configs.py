# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

"""
Test file: test_configs.py
Description: This file contains unit tests for the `configs` module
"""

from madlib._sensor_collection import SensorCollection


def test_sensor_YAML():
    """Test that the sample sensor network YAML file has a valid schema."""
    yaml = "configs/sample_sensor_network.yaml"
    network = SensorCollection.fromYAML(yaml)

    # If the test makes it this far, then the validation check succeeded
    assert True
