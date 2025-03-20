# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import argparse
import logging
import pathlib
import shutil
import sys
import time
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from astropy.time import Time
from hydra.conf import HydraConf, JobConf
from hydra_zen import store

import madlib
from maddg._residuals import calculate_residuals
from maddg._sim_launcher import launcher
from madlib._utils import MadlibException

log_task = logging.getLogger("simulator_task")
log_task.setLevel(logging.WARNING)


def parseArgs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "num_pairs",
        type=int,
        help="Number of Maneuver/Non-Maneuver pairs per simulation launch",
    )

    parser.add_argument(
        "sensor_yaml",
        type=str,
        help="Path to a YAML file defining the sensor network for the simulation",
    )

    parser.add_argument(
        "--start_mjd",
        type=float,
        default=-1.0,
        help="MJD at start of simulation. If <0, start at current system time. (default: -1.0)",
    )

    parser.add_argument(
        "--sim_duration_days",
        type=float,
        default=3.0,
        help=("Duration of simulation in whole days\n" "  (default: %(default)f)"),
    )

    parser.add_argument(
        "--mtype",
        type=str,
        choices=["impulse", "continuous", "all"],
        default="impulse",
        help=("Maneuver type for orbit simulations\n" "  (default: %(default)s)"),
    )

    parser.add_argument(
        "--dv_ric_mean_kms",
        type=float,
        nargs=3,
        default=[0, 0, 0],
        help=(
            "Mean values for impulsive RIC thrust vector normal distributions, in km/s\n"
            "  (default: %(default)s)"
        ),
    )

    parser.add_argument(
        "--dv_ric_std_kms",
        type=float,
        nargs=3,
        default=[0, 0.1, 1],
        help=(
            "Standard deviations for impulsive RIC thrust vector normal distributions, in km/s\n"
            "  (default: %(default)s)"
        ),
    )

    parser.add_argument(
        "--sensor_dra",
        type=float,
        default=None,
        help=(
            "Sensor metric accuracy in the right ascension direction (arcsec).\n"
            "  If not set, value is None, and `dra` value in sensor_yaml file\n"
            "  will be used. (default: None)"
        ),
    )

    parser.add_argument(
        "--sensor_ddec",
        type=float,
        default=None,
        help=(
            "Sensor metric accuracy in the declination direction (arcsec).\n"
            "  If not set, value is None, and `dra` value in sensor_yaml file\n"
            "  will be used. (default: None)"
        ),
    )

    parser.add_argument(
        "--cont_thrust_duration_days",
        type=float,
        help=(
            "Duration in days of continuous thrust, which begins at start of simulation\n"
            "  (default: %(default)s)"
        ),
    )

    parser.add_argument(
        "--cont_thrust_model",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Continous thrust model:\n"
            "  0 = continuous thrust in [0,1,0] direction\n"
            "  1 = continuous thrust in random direction\n"
            "  (relevant only for --mtype continuous)\n"
            "  (default: %(default)s)"
        ),
    )

    parser.add_argument(
        "--cont_thrust_mag",
        type=float,
        default=1.0e-7,
        help=(
            "Magnitude of the constant acceleration (km/s^2) for any continuous constant trust models\n"
            "  (relevant only for --mtype continuous, --cont_thrust_mode {0, 1})\n"
            "  (default: %(default)s)"
        ),
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs",
        help="Path to output directory",
    )

    parser.add_argument(
        "--submitit",
        type=str,
        default="",
        help="Optional JSON config file defining how to launch jobs across multiple GPUs using submitit",
    )

    parser.add_argument(
        "--multirun_root",
        type=str,
        default="",
        help="Optional path to directory where multirun results should be saved (./multirun by default)",
    )

    parser.add_argument(
        "--pred_err",
        type=float,
        default=0.0,
        help="Fractional error on predicted initial orbital state",
    )

    parser.add_argument(
        "--rm_multirun_root",
        action="store_true",
        help="Raise this flag to remove the multirun root directory after merging data",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Raise this flag to proceed with overwriting any data in outdir if outdir exists and is not empty",
    )

    parser.add_argument(
        "--sims_per_task",
        type=int,
        default=1,
        help=(
            "Number of simulations per task function. Must evenly divide into `num_pairs`.\n"
            "  (default: %(default)i)"
        ),
    )

    return parser


def simulator_task(
    seq_id: int,
    sensor_params: dict,
    maneuver_type: int,
    sim_duration_days: float,
    num_sim_pairs: int | None = None,
    start_mjd: float | None = None,
    dv_ric_mean_kms: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    dv_ric_std_kms: Tuple[float, float, float] = (0.0, 0.1, 1.0),
    cont_thrust_duration_days: float | None = None,
    cont_thrust_mag: float = 1e-7,
    cont_thrust_model: int = 0,
    pred_err: float = 0.0,
    random_seed: int | None = None,
    sims_per_task: int = 1,
) -> pd.DataFrame | None:
    """Generates a satellite, propagates it with and without a maneuver.

    Parameters
    ----------
    seq_id : int
        Unique ID of satellite
    sensor_params: dict
        Dictionary defining the parameters for each active sensor
    maneuver_type : int
        Specifies whether ot not the satellite is performing a maneuver:
        0 = no maneuver,
        1 = impulse maneuver,
        2 = continuous maneuver,
    sim_duration_days : float
        Duration of the simulation (days)
    num_sim_pairs : int, optional
        Number of simulation to perform per maneuver type.
    start_mjd : float, optional
        MJD at which the simulation should begin, by default None (current MJD)
    dv_ric_mean_kms : Tuple[float, float, float], optional
        Mean values of normal distributions to use when sampling
        the radial, in-track, and cross-track delta-V values, respectively,
        of impulsive maneuvers. In units of km/s, by default (0.0, 0.0, 0.0)
    dv_ric_std_kms : Tuple[float, float, float], optional
        Standard deviations of normal distributions to use when sampling
        the radial, in-track, and cross-track delta-V values, respectively,
        of impulsive maneuvers. In units of km/s, by default (0.0, 0.1, 1.0)
    cont_thrust_duration_days : float | None, optional
        Duration in days of the continuous maneuver that begins at simulation start, by default None
        (if None, maneuver duration is equal to simulation duration)
        Magnitude of the continuous thrust (km/s/s) (default 1e-7)
    cont_thrust_model : int, optional
        Which continuous thrust model to use:
        0 = applies a continuous thrust in the [0,1,0] direction,
        1 = applies a continuous thrust in a random direction,
        (default 0)
    pred_err : float, optional
        Fractional error on predicted initial orbital state, by default 0.0
    random_seed : int, optional
        Random seed to use for numpy, by default None
    sims_per_task : int, optional
        Number of simulations to perform per task function, by default 1

    Returns
    -------
    pd.core.frame.DataFrame
        Results (residuals)
    or, None (if no result)

    Raises
    ------
    ValueError
        Occurs if supplied `cont_thrust_model` is not a supported option
    """
    # Set the random seed for numpy
    np.random.seed(seed=random_seed)

    # Offset `seq_id` based off of value of `maneuver_type` so that each sequence ID is unique
    if num_sim_pairs is not None:
        seq_id = seq_id + num_sim_pairs * maneuver_type

    residual_dfs = []
    for seq_id in np.arange(seq_id, seq_id + sims_per_task):
        if cont_thrust_duration_days is None:
            cont_thrust_duration_days = sim_duration_days

        # Declarations
        sat_observed = None

        sensors = [
            madlib.GroundOpticalSensor(**params)
            for key, params in sensor_params.items()
        ]
        sensor_network = madlib.SensorCollection(sensors)

        # Timing
        if start_mjd is None:
            epoch = Time.now().mjd + np.random.random()
        else:
            epoch = start_mjd

        # Satellite Parameters
        sat_longitude = 360 * np.random.random()

        # Define Maneuver and setup appropriate Satellite Class
        maneuver = None
        maneuver_start_mjd = None
        maneuver_end_mjd = None
        maneuver_r_kms = None
        maneuver_i_kms = None
        maneuver_c_kms = None
        if maneuver_type == 0:
            """no maneuver"""
            # Setup Satellite
            sat_observed = madlib.Satellite.from_GEO_longitude(sat_longitude, epoch)
        elif maneuver_type == 1:
            """impulse maneuver"""
            # Maneuver Definition
            man_time = epoch + sim_duration_days * np.random.random()
            maneuver_start_mjd = man_time

            mean_rad, mean_in, mean_crs = dv_ric_mean_kms
            std_rad, std_in, std_crs = dv_ric_std_kms
            dv_rad = mean_rad + std_rad * np.random.randn()
            dv_in = mean_in + std_in * np.random.randn()
            dv_crs = mean_crs + std_crs * np.random.randn()

            man_dv = np.array([dv_rad, dv_in, dv_crs]) / 1000
            maneuver = madlib.ImpulsiveManeuver(man_time, man_dv)

            maneuver_r_kms = dv_rad
            maneuver_i_kms = dv_in
            maneuver_c_kms = dv_crs
            # Setup Satellite
            sat_observed = madlib.Satellite.from_GEO_longitude(sat_longitude, epoch)
        elif maneuver_type == 2:
            """continuous maneuver"""

            if cont_thrust_model == 0:
                accel_vec = np.array([0.0, 1.0, 0.0]) * cont_thrust_mag

                def acc_f(t):
                    return accel_vec

            elif cont_thrust_model == 1:
                rand_unit_vec = np.random.randn(3)
                rand_unit_vec /= np.linalg.norm(rand_unit_vec)
                accel_vec = rand_unit_vec * cont_thrust_mag

                def acc_f(t):
                    return accel_vec

            else:
                raise ValueError(f"{cont_thrust_model = } is not a supported option")

            man_time = (epoch, epoch + cont_thrust_duration_days)
            maneuver = madlib.ContinuousManeuver(acc_f, man_time)
            # Setup Satellite
            sat_observed = madlib.ContinuousThrustSatellite.from_GEO_longitude(
                sat_longitude, epoch
            )

            maneuver_start_mjd = man_time[0]
            maneuver_end_mjd = man_time[1]
            maneuver_r_kms, maneuver_i_kms, maneuver_c_kms = accel_vec

        if isinstance(
            sat_observed, (madlib.Satellite, madlib.ContinuousThrustSatellite)
        ):
            # If prediction error has been specified, use it to define a "true" orbit
            if np.abs(pred_err) > 1e-8:
                r_err = 1 + pred_err * np.random.randn(3)
                v_err = 1 + pred_err * np.random.randn(3)

                true_r = sat_observed.x * r_err
                true_v = sat_observed.v * v_err

                sat_observed.x_true = true_r
                sat_observed.v_true = true_v

            sat_observed.maneuver = maneuver

            # Observe and calculate residuals
            residual_df = calculate_residuals(
                sensors=sensor_network,
                satellite=sat_observed,
                sim_duration_days=sim_duration_days,
                t_start_mjd=epoch,
            )

            if residual_df is not None:
                residual_df["Maneuver"] = maneuver_type
                residual_df["Sequence"] = int(seq_id)
                residual_df["Maneuver_Start_MJD"] = maneuver_start_mjd
                residual_df["Maneuver_End_MJD"] = maneuver_end_mjd
                residual_df["Maneuver_DV_Radial_KmS"] = maneuver_r_kms
                residual_df["Maneuver_DV_InTrack_KmS"] = maneuver_i_kms
                residual_df["Maneuver_DV_CrossTrack_KmS"] = maneuver_c_kms
                residual_dfs.append(residual_df)

    if len(residual_dfs) == 0:
        return None
    else:
        return pd.concat(residual_dfs, ignore_index=True)


if __name__ == "__main__":
    time_start = time.time()

    # parse arguments
    parser = parseArgs()
    args = parser.parse_args()

    # check if num_pairs is evenly divisible by sims_per_task
    if args.num_pairs % args.sims_per_task != 0:
        warnings.warn(
            "`num_pairs` is not evenly divisible by `sims_per_task`. Number of sequences generated may be different than expected."
        )

    # check outdir for any existing files
    try:
        outdir_has_files = any(pathlib.Path(args.outdir).iterdir())
    except FileNotFoundError:
        outdir_has_files = False

    # exit (abort simulation) if outdir is not empty, unless --overwrite was specified
    # otherwise, delete existing folder and proceed
    if outdir_has_files and (not args.overwrite):
        # abort simulation (do nothing)
        sys.exit(
            "Outdir exists and is not empty. --overwrite argument was not passed. Aborting simulation."
        )
    elif outdir_has_files and args.overwrite:
        # delete all files in outdir and proceed
        shutil.rmtree(args.outdir)
        print(
            "INFO :: outdir exists and --overwrite argument was passed. Existing data was deleted."
        )

    # customize Hydra's configuration
    store(HydraConf(job=JobConf(chdir=True)))
    store.add_to_hydra_store(overwrite_ok=True)

    simulator_method = simulator_task
    num_sim_pairs = args.num_pairs
    sensor_yaml = args.sensor_yaml
    outdir = args.outdir
    dv_ric_mean_kms = (
        args.dv_ric_mean_kms[0],
        args.dv_ric_mean_kms[1],
        args.dv_ric_mean_kms[2],
    )
    dv_ric_std_kms = (
        args.dv_ric_std_kms[0],
        args.dv_ric_std_kms[1],
        args.dv_ric_std_kms[2],
    )

    sim_duration_days = args.sim_duration_days

    if not args.cont_thrust_duration_days:
        cont_thrust_duration_days = sim_duration_days
    else:
        cont_thrust_duration_days = args.cont_thrust_duration_days

    cont_thrust_mag = args.cont_thrust_mag
    cont_thrust_model = args.cont_thrust_model
    pred_err = args.pred_err
    submitit = args.submitit
    multirun_root = args.multirun_root
    rm_multirun_root = args.rm_multirun_root
    mtype = args.mtype
    sensor_dra = args.sensor_dra
    sensor_ddec = args.sensor_ddec

    start_mjd = args.start_mjd
    if start_mjd < 0:
        start_mjd = None

    print("Setting up job launcher...")

    launcher(
        simulator_method,
        mtype,
        num_sim_pairs,
        sensor_yaml,
        outdir,
        dv_ric_mean_kms=dv_ric_mean_kms,
        dv_ric_std_kms=dv_ric_std_kms,
        cont_thrust_duration_days=cont_thrust_duration_days,
        cont_thrust_mag=cont_thrust_mag,
        cont_thrust_model=cont_thrust_model,
        submitit=submitit,
        multirun_root=multirun_root,
        rm_multirun_root=rm_multirun_root,
        start_mjd=start_mjd,
        sim_duration_days=sim_duration_days,
        pred_err=pred_err,
        sensor_dra=sensor_dra,
        sensor_ddec=sensor_ddec,
        sims_per_task=args.sims_per_task,
    )
    time_stop = time.time()
    print(f"INFO :: Elapsed time (sec) = {time_stop - time_start}")
