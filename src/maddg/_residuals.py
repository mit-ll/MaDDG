# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
from astropy.time import Time

import madlib


def calculate_residuals(
    sensors: madlib.SensorCollection,
    satellite: madlib.Satellite,
    sim_duration_days: float,
    t_start_mjd: float = Time.now().mjd,
) -> pd.DataFrame | None:
    """Calculates the residuals.

    Parameters
    ----------
    sensors : madlib.SensorCollection
        Collection of Sensors
    satellite : madlib.Satellite
        The Satellite object observed in the simulation
    sim_duration_days : float
        Duration of the simulation (days)
    t_start_mjd : float, optional
        Time (MJD) at the start of the simulation, by default astropy.time.Time.now().mjd

    Returns
    -------
    pd.DataFrame | None
        pandas DataFrame of results (or None, if nobs == 0)

    Raises
    ------
    MadlibException
        Raised if nobs != nexpected...
        number of actual and expected observations were not
        the same, so residuals cannot be calculated. This
        is likely a random occurrence.
    """
    ### SIMULATION TIMING

    t_end_mjd = t_start_mjd + sim_duration_days
    sensors.generate_obs_timing(t_start_mjd, t_end_mjd)

    ### GENERATE OBSERVATIONS
    obs = sensors.observe(satellite)

    actual_obs = obs.pos_observed
    predicted_obs = obs.pos_expected

    nobs = obs.count_valid_observations()

    ### CALCULATE AND RETURN RESIDUALS
    output = None
    if nobs > 0:
        obs_res = np.array([actual_obs[n] - predicted_obs[n] for n in range(nobs)])
        ra_res = np.array([obs_res[n].ra for n in range(nobs)]) * 3600
        dec_res = np.array([obs_res[n].dec for n in range(nobs)]) * 3600
        t = np.array([x.mjd for x in actual_obs])
        sensor_ids = np.array([x.sensor_id for x in predicted_obs])

        output = {
            "MJD": t,
            "RA Arcsec": ra_res,
            "DEC Arcsec": dec_res,
            "SensorID": sensor_ids,
        }
        output = pd.DataFrame(output)

    return output
