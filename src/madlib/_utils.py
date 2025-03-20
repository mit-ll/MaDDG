# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import numpy as np
from numpy.typing import NDArray


class MadlibException(Exception):
    """MadlibException Class"""

    pass


sensor_yaml_schema = {
    "type": "object",
    "properties": {
        "sensor_list": {
            "type": "object",
            "patternProperties": {
                ".": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "lat": {"type": "number", "minimum": -90, "maximum": 90},
                        "lon": {"type": "number", "minimum": -180, "maximum": 360},
                        "alt": {"type": "number"},
                        "dra": {"type": "number", "minimum": 0},
                        "ddec": {"type": "number", "minimum": 0},
                        "students_dof": {"type": "number", "exclusiveMinimum": 0},
                        "obs_per_collect": {"type": "integer", "exclusiveMinimum": 0},
                        "obs_time_spacing": {"type": "number", "minimum": 0},
                        "collect_gap_mean": {"type": "number", "minimum": 0},
                        "collect_gap_std": {"type": "number", "minimum": 0},
                        "obs_limits": {
                            "type": ["object", "null"],
                            "properties": {
                                "el": {
                                    "type": "array",
                                    "items": {
                                        "type": "number",
                                        "minimum": -90,
                                        "maximum": 90,
                                    },
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                                "az": {
                                    "type": "array",
                                    "items": {
                                        "type": "number",
                                        "minimum": -180,
                                        "maximum": 180,
                                    },
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                                "sun_el": {
                                    "type": "array",
                                    "items": {
                                        "type": "number",
                                        "minimum": -90,
                                        "maximum": 90,
                                    },
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                                "dec": {
                                    "type": "array",
                                    "items": {
                                        "type": "number",
                                        "minimum": -90,
                                        "maximum": 90,
                                    },
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                                "range_": {
                                    "type": "array",
                                    "items": {
                                        "type": "number",
                                        "minimum": 0,
                                    },
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                            },
                        },
                        "weather": {
                            "type": "object",
                            "properties": {
                                "cloud_prob": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "cloud_duration_mean": {"type": "number", "minimum": 0},
                                "cloud_duration_std": {"type": "number", "minimum": 0},
                            },
                        },
                    },
                    "required": [
                        "lat",
                        "lon",
                        "alt",
                        "dra",
                        "ddec",
                        "collect_gap_mean",
                    ],
                    "additionalProperties": False,
                }
            },
        }
    },
    "required": ["sensor_list"],
}


def calc_separation_angle(
    v1: NDArray[np.float64],
    v2: NDArray[np.float64],
    in_deg: bool = False,
):
    """Returns the angle between vectors v1 and v2, both with
    shapes (N, 3). Output is in radians by default, in degrees if
    <in_deg> is True."""

    shape_1 = v1.shape
    shape_2 = v2.shape

    if (len(shape_1) != 2) or (len(shape_2) != 2):
        raise MadlibException("Input arrays must have shape (N,3).")

    if (shape_1[1] != 3) or (shape_2[1] != 3):
        raise MadlibException("Input arrays must have shape (N,3).")

    if shape_1[0] != shape_2[0]:
        raise MadlibException(
            "Input arrays must have the same number of rows. "
            f"The first input has {shape_1[0]} rows, and "
            f"the second input has {shape_2[0]}"
        )

    norm_1 = np.linalg.norm(v1, axis=1, keepdims=True)
    norm_2 = np.linalg.norm(v2, axis=1, keepdims=True)

    v1_u = v1 / norm_1
    v2_u = v2 / norm_2

    dot = np.clip(np.sum(v1_u * v2_u, axis=1), -1.0, 1.0)
    angles = np.arccos(dot)

    if in_deg:
        angles *= 180.0 / np.pi

    return angles
