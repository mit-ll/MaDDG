# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import json
import shutil
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from hydra_zen import launch, make_config

from madlib import SensorCollection
from madlib._utils import MadlibException


class NotImplementedError(Exception):
    """NotImplementedError Exception"""

    pass


def create_task_fn(method: Callable) -> Callable:
    """Create a task function for hydra. Config (cfg) will be passed in via hydra.

    Parameters
    ----------
    method : Callable
        _description_

    Returns
    -------
    Callable
        _description_
    """

    def task_fn(cfg):
        try:
            output = method(**cfg)

            if output is not None:
                output.to_csv("output.csv", index=False)

        except MadlibException as e:
            with open("error.txt", "w") as f:
                f.write(str(e))

    return task_fn


def launcher(
    simulator_method: Callable,
    mtype: str,
    num_sim_pairs: int,
    sensor_yaml: str | Path,
    outdir: str | Path,
    dv_ric_mean_kms: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    dv_ric_std_kms: Tuple[float, float, float] = (0.0, 0.1, 1.0),
    cont_thrust_duration_days: float | None = None,
    cont_thrust_mag: float = 1e-7,
    cont_thrust_model: int = 0,
    submitit: str = "",
    multirun_root: str = "",
    rm_multirun_root: bool = False,
    start_mjd: float | None = None,
    sim_duration_days: float = 3.0,
    random_seed: int | None = None,
    pred_err: float = 0.0,
    sensor_dra: float | None = None,
    sensor_ddec: float | None = None,
    sims_per_task: int = 1,
) -> None:
    """Hydra job launcher wrapper.

    Parameters
    ----------
    simulator_method : Callable
        Task function
    mtype : str
        Maneuver type:
        "impulse" = ImpulseManeuver,
        "continuous" = ContinuousManeuver,
    num_sim_pairs : int
        Number of simulations to perform per maneuver type.
    sensor_yaml : str | Path
        Path to YAML file defining the sensor network for the simulation
    outdir : str | Path
        Path to output directory where the concatenated results will be saved in a .csv file (complete.csv)
    dv_ric_mean_kms : Tuple[float, float, float], optional
        Mean values of normal distributions to use when sampling
        the radial, in-track, and cross-track delta-V values, respectively,
        of impulsive maneuvers. In units of km/s, by default [0.0, 0.0, 0.0]
    dv_ric_std_kms : Tuple[float, float, float], optional
        Standard deviations of normal distributions to use when sampling
        the radial, in-track, and cross-track delta-V values, respectively,
        of impulsive maneuvers. In units of km/s, by default [0.0, 0.1, 1.0]
    cont_thrust_duration_days : float | None, optional
        Duration in days of the continuous maneuver that begins at simulation start, by default None
        (if None, maneuver duration is equal to simulation duration)
    cont_thrust_mag : float, optional
        Magnitude of the continuous thrust (km/s/s), by default 1e-7
    cont_thrust_model : int, optional
        Which continuous thrust model to use:
        0 = applies a continuous thrust in the [0,1,0] direction,
        1 = applies a continuous thrust in a random direction,
        by default 0
    submitit : string, optional
        If specified, the path to a config JSON file defining how to launch jobs across multiple GPUs using submitit, by default None (serial launch only)
    multirun_root : string, optional
        If specified, the path to a directory where multirun output will be stored, by default None (./multirun will be used)
    rm_multirun_root : bool, optional
        Whether or not to delete the hydra multirun directory after finishing the simulation, by default False
    start_mjd : float, optional
        MJD at which the simulation should begin, by default None (current MJD)
    sim_duration_days : float, optional
        Duration of the simulation (days), by default 3.0
    random_seed : int, optional
        Random seed to use for numpy, by default None
    pred_err : float
        Fractional error on predicted initial orbital state
    sensor_dra : float, optional
        Sensor metric accuracy in the right ascension direction (arcsec).
        If not set, value is None, and `dra` value in sensor_yaml file
        will be used, by default: None
    sensor_ddec : float, optional
        Sensor metric accuracy in the declination direction (arcsec).
        If not set, value is None, and `dra` value in sensor_yaml file
        will be used, by default: None
    sims_per_task : int, optional
        Number of simulations to perform per task function, by default 1

    Raises
    ------
    NotImplementedError
        If an mtype was requested that is not yet implemented.
    """

    error_runs = []
    log_runs = []

    # Parse the sensor YAML file
    sensor_data = SensorCollection.paramsFromYAML(sensor_yaml)

    # Update sensor `dra` if was given as an input argument
    if sensor_dra is not None:
        for key in sensor_data.keys():
            sensor_data[key]["dra"] = sensor_dra

    # Update sensor `ddec` if was given as an input argument
    if sensor_ddec is not None:
        for key in sensor_data.keys():
            sensor_data[key]["ddec"] = sensor_ddec

    if cont_thrust_duration_days is None:
        cont_thrust_duration_days = sim_duration_days

    Conf = make_config(
        seq_id=0,
        sensor_params=sensor_data,
        maneuver_type=0,
        num_sim_pairs=num_sim_pairs,
        dv_ric_mean_kms=dv_ric_mean_kms,
        dv_ric_std_kms=dv_ric_std_kms,
        cont_thrust_duration_days=0,
        cont_thrust_mag=1e-7,
        cont_thrust_model=0,
        start_mjd=start_mjd,
        sim_duration_days=sim_duration_days,
        random_seed=random_seed,
        pred_err=pred_err,
        sims_per_task=sims_per_task,
    )

    # useful into to stout
    print(f"INFO :: {mtype = }")
    print(f"INFO :: {sim_duration_days = }")
    print(f"INFO :: {sims_per_task = }")

    # hydra_zen overrides: initialize
    overrides = []

    # hydra_zen overrides: mtype specific cases
    if mtype == "impulse":
        overrides += [
            f"maneuver_type=0,1",
        ]
    elif mtype == "continuous":
        print(f"INFO :: {cont_thrust_duration_days = }")
        print(f"INFO :: {cont_thrust_mag = }")
        print(f"INFO :: {cont_thrust_model = }")
        overrides += [
            f"maneuver_type=0,2",
            f"cont_thrust_duration_days={cont_thrust_duration_days}",
            f"cont_thrust_mag={cont_thrust_mag}",
            f"cont_thrust_model={cont_thrust_model}",
        ]
    else:
        # all mtypes case?
        # maneuver_type=0,1,2
        raise NotImplementedError("An mtype was requested that is not yet implemented.")

    # create task function
    task_fn = create_task_fn(simulator_method)

    # define seq_id string
    # seq_id = ",".join([str(n) for n in range(num_sim_pairs)])
    seq_id = ",".join([str(n) for n in np.arange(0, num_sim_pairs, sims_per_task)])

    # setup hydra_zen overrides
    overrides += [
        f"seq_id={seq_id}",
    ]

    # configure submitit if using
    if submitit != "":
        with open(submitit, "r") as f:
            submitit_overrides = json.load(f)

        # Make sure the JSON contents are a list of strings
        str_list_check = all(isinstance(n, str) for n in submitit_overrides)
        if (type(submitit_overrides) != list) or (not str_list_check):
            raise MadlibException(
                f"The specified submitit configuration file {submitit} is not formatted properly. "
                "The JSON must be a list of strings."
            )

        overrides += submitit_overrides

    else:
        overrides += [
            "hydra.job.chdir=True",
        ]

    # configure multirun directory if non-default
    if multirun_root != "":
        # Append datetime structure to root
        multirun_dir = Path(multirun_root) / "${now:%Y-%m-%d}/${now:%H-%M-%S}"
        multirun_dir = str(multirun_dir)
        overrides += [
            f"hydra.sweep.dir={multirun_dir}",
        ]

    # setup outdir
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # launch jobs
    (jobs,) = launch(
        Conf,
        task_fn,
        multirun=True,
        to_dictconfig=True,
        overrides=overrides,
        version_base="1.3",
    )

    # initialize df_merged dataframe that will contain all output.csv files concatenated
    df_merged = pd.DataFrame()

    # get multirun root directory
    rundir = Path(jobs[0].working_dir).parent

    # concatenate all output.csv files
    jobfiles = sorted(rundir.rglob("output.csv"))
    for csv_file in jobfiles:
        df_temp = pd.read_csv(csv_file)
        df_merged = pd.concat((df_merged, df_temp), ignore_index=True)

    # export merged to disk
    df_merged.to_csv(outdir / "complete.csv", index=False)

    # concatenate all error.txt files
    errfiles = sorted(rundir.rglob("error.txt"))
    for err in errfiles:
        with open(err, "r") as f:
            text = f.read()
        error_runs.append((err, text))

    # concatenate all zen_launch.log files
    logfiles = sorted(rundir.rglob("zen_launch.log"))
    for logfile in logfiles:
        with open(logfile, "r") as f:
            text = f.read()
        log_runs.append((logfile, text))

    # keep track of errors checkpoint-style
    with open(outdir / "errors.txt", "w") as f:
        for errfile, errtext in error_runs:
            f.write(str(errfile) + "\n")
            f.write(errtext + "\n\n")

    # keep track of zen_launch logs checkpoint-style
    with open(outdir / "logs.txt", "w") as f:
        for logfile, logtext in log_runs:
            if logtext != "":
                f.write(str(logfile) + "\n")
                f.write(logtext + "\n\n")

    # export merged to disk
    df_merged.to_csv(outdir / "complete.csv", index=False)

    # copy multirun.yaml file to outdir if exists
    try:
        multirun_yaml = next(rundir.glob("multirun.yaml"))
        shutil.copy2(multirun_yaml, outdir / "multirun.yaml")
    except (StopIteration, OSError):
        pass

    # remove multirun root directory?
    if rm_multirun_root:
        # DIRECTORIES WILL ONLY BE REMOVED IF THEIR CONTENTS ARE
        # EXCLUSIVELY HYDRA-ZEN MULTIRUN OUTPUTS
        for job in jobs:
            working_dir = Path(job.working_dir)
            hydra_dir = working_dir / ".hydra"
            if hydra_dir.exists() and hydra_dir.is_dir():
                shutil.rmtree(hydra_dir)

            (working_dir / "zen_launch.log").unlink(missing_ok=True)
            (working_dir / "output.csv").unlink(missing_ok=True)
            (working_dir / "error.txt").unlink(missing_ok=True)

            contents = list(working_dir.glob("*"))
            if len(contents) == 0:
                shutil.rmtree(working_dir)

        if (rundir / ".submitit").exists():
            shutil.rmtree(rundir / ".submitit")

        parent = rundir.parent
        grandparent = parent.parent

        (rundir / "multirun.yaml").unlink(missing_ok=True)
        contents = list(rundir.glob("*"))
        if len(contents) == 0:
            shutil.rmtree(rundir)

        parent_contents = list(parent.glob("*"))
        if len(parent_contents) == 0:
            shutil.rmtree(parent)

        grandparent_contents = list(grandparent.glob("*"))
        if len(grandparent_contents) == 0:
            shutil.rmtree(grandparent)
