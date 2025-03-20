# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from hydra.conf import HydraConf, JobConf
from hydra_zen import launch, make_config, store, zen
from hydra_zen.typing._implementations import DataClass

from madlib._utils import MadlibException

from madlib import SensorCollection

plt.rcParams.update({"figure.max_open_warning": 0})


def parseArgs():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Script description",
    )
    # Add the required arguments
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="The path to the input data file (.csv)",
    )
    parser.add_argument(
        "--sensor_yaml",
        type=str,
        required=True,
        help="Path to a YAML file defining the sensor network for the simulation",
    )
    # Optional arguments
    parser.add_argument(
        "--cloud_prob",
        type=str,
        default="0.5",
        help="Probability of a cloud event blocking the sky during any nighttime observable window for each sensor",
    )
    parser.add_argument(
        "--cloud_duration_mean",
        type=str,
        default="10800.0",
        help="Mean duration of a cloud event (seconds)",
    )
    parser.add_argument(
        "--cloud_duration_std",
        type=str,
        default="3600.0",
        help="Standard deviation of a cloud event (seconds)",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of dropout datasets to generate",
    )
    parser.add_argument(
        "--save_copy_of_original",
        action="store_true",
        help="Raise this flag to save copy of the original input data along side of the modified dataset with dropouts",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Raise this flag to generate and save plots",
    )
    parser.add_argument(
        "--submitit",
        type=str,
        default="",
        help="Optional JSON config file defining how to launch jobs across multiple GPUs using submitit",
    )

    return parser


def gen_random_duration(
    loc: float,
    scale: float,
) -> float:
    """Generate a random weather duration, drawing from the random normal distribution.

    Parameters
    ----------
    loc : float
        Mean ("center") of the distrubtion.
    scale : float
        Standard deviation (spread or "width") of the distribution. Must be non-negative.

    Returns
    -------
    float
        Random weather duration (note: always positive)
    """
    duration = -1.0
    # generate a duration sample until it is non-negative:
    while duration < 0.0:
        duration = np.random.normal(loc=loc, scale=scale)
    return duration


def get_sensor(
    sensor_list: List[Dict[str, Any]],
    sensor_ID: str,
) -> Dict[str, Any] | None:
    """Get sensor dictionary that corresponds to sensor_ID.

    Parameters
    ----------
    sensor_list : List[Dict[str, Any]]
        List of sensors
    sensor_ID : str
        Dictionary within sensor_list with a sensor_ID that matches this parameter will be returned.

    Returns
    -------
    Dict[str, Any] | None
    """
    sensor_match = None
    for sensor in sensor_list:
        if sensor["id"] == sensor_ID:
            sensor_match = sensor
            break
    return sensor_match


def update_sensor_weather_stats(
    sensor_list: List[Dict[str, Any]],
    idx: int | None = None,
    cloud_prob: float = 0.5,
    cloud_duration_mean: float = 10800.0,
    cloud_duration_std: float = 7200.0,
) -> List[Dict[str, Any]]:
    """Updates sensor weather stats for the sensors in supplied sensor_list.

    Parameters
    ----------
    sensor_list : List[Dict[str, Any]]
        List of sensors
    idx : int | None, optional
        Sensor index, by default None. If None, all sensors in the sensor list will be updated.
    cloud_prob : float, optional
       Probability of a cloud blocking a measurement: 0 - 1, by default 0.5
    cloud_duration_mean : float, optional
        Average duration of each cloud event (seconds), by default 10800.0
    cloud_duration_std : float, optional
        Standard deviation of each cloud event (seconds), by default 7200.0

    Returns
    -------
    List[Dict[str, Any]]
        _description_
    """
    s_l = sensor_list.copy()
    if isinstance(idx, int):
        s_l[idx]["weather"]["cloud_prob"] = cloud_prob
        s_l[idx]["weather"]["cloud_duration_mean"] = cloud_duration_mean
        s_l[idx]["weather"]["cloud_duration_std"] = cloud_duration_std
    else:
        for sensor in s_l:
            sensor["weather"]["cloud_prob"] = cloud_prob
            sensor["weather"]["cloud_duration_mean"] = cloud_duration_mean
            sensor["weather"]["cloud_duration_std"] = cloud_duration_std
    return s_l


def plot_sensor_ID(
    sensor_ID: str,
    df: pd.DataFrame,
    df_drop: pd.DataFrame,
    sensor_list: List[Dict[str, Any]],
    save_without_displaying: bool = False,
    save_path: str = "./plots/sensor_ID/",
) -> None:
    """This function plots all measurements for a given sensor_ID, before and after a synthetic "weather event".
    The data dropout regions are highlighted in the after plot.

    Parameters
    ----------
    sensor_ID : str
        The unique sensor ID
    df : pd.DataFrame
        The pandas DataFrame of a MaDDG output file, with an additional ``Datetime`` column
    df_drop : pd.DataFrame
        The pandas DataFrame of a MaDDG output file, with mising data, and an additional ``Datetime`` column
    sensor_list : List[Dict[str, Any]]
        List of sensors
    save_without_displaying : bool, optional
        Whether or not to save plots to the hydra multirun directory, without displaying the plots during runtime, by default False
    save_path : str, optional
        Path to save the plots within the hydra multirun directory, by default "./plots/sensor_ID/"
    """
    sensor = get_sensor(sensor_list, sensor_ID)
    if sensor is not None:
        try:
            hours = mdates.HourLocator(interval=8)
            timeFmt = mdates.DateFormatter("%H:")
            m_size = 4
            _ax_idx = [0, 2]
            _plt_types = ["original", "with dropout"]
            fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(8, 14))
            for _idx, _df in enumerate([df, df_drop]):
                # group by 'SensorID' and 'Maneuver'
                grouped_df = _df.groupby(["SensorID", "Maneuver"])
                try:
                    data_manuv = grouped_df.get_group((sensor_ID, 1)).sort_values("MJD")
                except KeyError:
                    data_manuv = grouped_df.get_group((sensor_ID, 2)).sort_values("MJD")
                data_no_manuv = grouped_df.get_group((sensor_ID, 0)).sort_values("MJD")

                axs[_ax_idx[_idx]].scatter(
                    data_manuv["Datetime"],
                    data_manuv["LON Arcsec"],
                    label="manuv=True",
                    s=m_size,
                )
                axs[_ax_idx[_idx]].scatter(
                    data_no_manuv["Datetime"],
                    data_no_manuv["LON Arcsec"],
                    label="manuv=False",
                    s=m_size,
                )
                axs[_ax_idx[_idx]].xaxis.set_major_locator(hours)
                axs[_ax_idx[_idx]].xaxis.set_major_formatter(timeFmt)
                # highlight dropouts
                if _idx == 1:
                    for drop in sensor["weather"]["drops"]:
                        axs[_ax_idx[_idx]].axvspan(
                            drop[0], drop[1], color="yellow", alpha=0.3
                        )
                axs[_ax_idx[_idx]].grid()
                axs[_ax_idx[_idx]].set_ylabel("LON Arcsec")
                axs[_ax_idx[_idx]].set_title(
                    f"Sensor_ID = {sensor_ID}; {_plt_types[_idx]}"
                )
                # -
                axs[_ax_idx[_idx] + 1].scatter(
                    data_manuv["Datetime"],
                    data_manuv["LAT Arcsec"],
                    label="manuv=True",
                    s=m_size,
                )
                axs[_ax_idx[_idx] + 1].scatter(
                    data_no_manuv["Datetime"],
                    data_no_manuv["LAT Arcsec"],
                    label="manuv=False",
                    s=m_size,
                )
                axs[_ax_idx[_idx] + 1].xaxis.set_major_locator(hours)
                axs[_ax_idx[_idx] + 1].xaxis.set_major_formatter(timeFmt)
                # highlight dropouts
                if _idx == 1:
                    for drop in sensor["weather"]["drops"]:
                        axs[_ax_idx[_idx] + 1].axvspan(
                            drop[0], drop[1], color="yellow", alpha=0.3
                        )
                        axs[_ax_idx[_idx] + 1].set_xlabel("Time (%H:)")
                axs[_ax_idx[_idx] + 1].grid()
                axs[_ax_idx[_idx] + 1].set_ylabel("LAT Arcsec")
                axs[_ax_idx[_idx] + 1].set_title(
                    f"Sensor_ID = {sensor_ID}; {_plt_types[_idx]}"
                )

            # create legend for all plots and put it outside of plot area
            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(1, 0.9),
                borderaxespad=0.0,
            )
            plt.tight_layout()

            # save plot?
            if save_without_displaying:
                outpath = Path(save_path)
                outpath.mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    outpath / f"plot__sensor_id_{sensor_ID}.png", bbox_inches="tight"
                )
                plt.close()
        except KeyError:
            pass


def plot_seq_id(
    seq_id: int,
    df: pd.DataFrame,
    df_drop: pd.DataFrame,
    save_without_displaying: bool = False,
    save_path: str = "./plots/seq_id/",
) -> None:
    """This function plots the manuver=[{1,2}, 0] pair for a given seq_id.
    The plot also displays the measurements that are dropped due to a synthetic weather event.

    Parameters
    ----------
    seq_id : int
        The seq_id to plot
    df : pd.DataFrame
        The pandas dataframe of a MaDDg output file, with an additional 'Datetime' column
    df_drop : pd.DataFrame
        The pandas dataframe of a MaDDg output file, with missing data, and an additonal
        'Datetime' column
    save_without_displaying : bool, optional
        Whether or not to save the plots without displaying them, by default False
    save_path : str, optional
        Path to save the plots in the hydra multirun folder, by default "./plots/seq_id/"
    """
    idx = seq_id

    # --- make missing data df
    df_merged = pd.merge(df_drop, df, how="outer", indicator=True)
    # select the rows that are in df but not in df_drop
    df_missing = df_merged[df_merged["_merge"] == "right_only"]

    try:
        # --- group by 'Sequence' and 'Maneuver'
        grouped_df = df.groupby(["Sequence", "Maneuver"])
        grouped_df_missing = df_missing.groupby(["Sequence"])
        try:
            data_manuv = grouped_df.get_group((idx, 1)).sort_values("MJD")
        except KeyError:
            data_manuv = grouped_df.get_group((idx, 2)).sort_values("MJD")
        data_no_manuv = grouped_df.get_group((idx, 0)).sort_values("MJD")
        data_missing = grouped_df_missing.get_group(idx).sort_values("MJD")

        # --- make & display multiplot
        hours = mdates.HourLocator(interval=8)
        timeFmt = mdates.DateFormatter("%H:")
        m_size = 4
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
        axs[0].scatter(
            data_manuv["Datetime"],
            data_manuv["LON Arcsec"],
            label="manuv=True",
            s=m_size,
        )
        axs[0].scatter(
            data_no_manuv["Datetime"],
            data_no_manuv["LON Arcsec"],
            label="manuv=False",
            s=m_size,
        )
        axs[0].scatter(
            data_missing["Datetime"],
            data_missing["LON Arcsec"],
            label="dropped",
            marker="x",
            color="red",
        )
        axs[0].xaxis.set_major_locator(hours)
        axs[0].xaxis.set_major_formatter(timeFmt)
        axs[0].grid()
        axs[0].set_title(f"LON Arcsec; seq_ID = {idx}")
        # -
        axs[1].scatter(
            data_manuv["Datetime"],
            data_manuv["LAT Arcsec"],
            label="manuv=True",
            s=m_size,
        )
        axs[1].scatter(
            data_no_manuv["Datetime"],
            data_no_manuv["LAT Arcsec"],
            label="manuv=False",
            s=m_size,
        )
        axs[1].scatter(
            data_missing["Datetime"],
            data_missing["LAT Arcsec"],
            label="dropped",
            marker="x",
            color="red",
        )
        axs[1].xaxis.set_major_locator(hours)
        axs[1].xaxis.set_major_formatter(timeFmt)
        axs[1].set_xlabel("Time (%H:)")
        axs[1].grid()
        axs[1].set_title(f"LAT Arcsec; seq_ID = {idx}")

        # create legent for all plots and put it outside of plot area
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1, 0.9),
            borderaxespad=0.0,
        )
        plt.tight_layout()

        # save plot?
        if save_without_displaying:
            outpath = Path(save_path)
            outpath.mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath / f"plot__seq_id_{seq_id}.png", bbox_inches="tight")
            plt.close()
    except KeyError:
        pass


def dropout(
    path: str,
    sensor_list: List[Dict[str, Any]],
    sensor_idx: int | None = None,
    cloud_prob: float = 0.5,
    cloud_duration_mean: float = 10800.0,
    cloud_duration_std: float = 7200.0,
    num_runs: int = 1,
    save_copy_of_original: bool = True,
    save_plots: bool = False,
) -> None:
    """Applies weather related data dropout to a MaDDG data output file (.csv).

    Parameters
    ----------
    path : str
        Path to the MaDDG data output file (.csv) to apply weather related data dropout
    sensor_list : List[Dict[str, Any]]
        List of sensors
    sensor_idx : int | None, optional
        Sensor index, by default None
    cloud_prob : float, optional
        Probability of a cloud blocking a measurement: 0 - 1, by default 0.5
    cloud_duration_mean : float, optional
        Average duration of each cloud event (seconds), by default 10800.0
    cloud_duration_std : float, optional
        Standard deviation of each cloud event (seconds), by default 7200.0
    num_runs : int, optional
        Number of dropout simulations to run, by default 1
    save_copy_of_original : bool, optional
        Whether or not to include a copy of the original data in the dropout simulation
        results directory, by default True
    save_plots : bool, optional
        Whether or not to generate and save dropout plots, by default False
    """
    for num_run in range(num_runs):
        try:
            # update sensor_list params
            sensor_list = update_sensor_weather_stats(
                sensor_list=sensor_list,
                idx=sensor_idx,
                cloud_prob=cloud_prob,
                cloud_duration_mean=cloud_duration_mean,
                cloud_duration_std=cloud_duration_std,
            )

            # read data from csv
            df = pd.read_csv(path)
            ts_mjd = Time(df["MJD"], format="mjd")
            ts_datetime = ts_mjd.to_value("datetime64")
            df["Datetime"] = ts_datetime
            df.sort_values("MJD", inplace=True)
            sensor_IDs = df["SensorID"].unique()

            # group by 'SensorID'
            grouped_df = df.groupby(["SensorID"])

            seq_IDs = []
            dfs = []
            # for idx in tqdm(range(sensor_IDs.size)):
            for sensor_ID in sensor_IDs:
                # get sensor dict that corresponds to sensor_ID
                sensor = get_sensor(sensor_list, sensor_ID)
                assert sensor is not None

                # add blockout range key to sensor
                sensor["weather"]["drops"] = []

                # get subset of data associated with sensor_ID
                data = grouped_df.get_group(sensor_ID).sort_values("MJD").copy()

                # get seq_IDs in data subset
                seq_IDs = df["Sequence"].unique()

                # --- group data by day by looking for a large gap between measurements
                # set threshold for maximum gap between the elapsed times
                gap_threshold = 10000  # seconds

                # compute the gaps between the elapsed times
                gaps = (
                    (data["Datetime"] - data["Datetime"].shift(1))
                    .dt.total_seconds()
                    .values
                )

                # create labels for each contiguous group of times separated by gaps smaller than gap_threshold
                labels = np.cumsum(gaps > gap_threshold)

                # add labels to data as 'Cluster' column
                data["Cluster"] = labels

                # get unique cluster IDs
                clusters = data["Cluster"].unique()

                # group by 'Cluster'
                grouped_data = data.groupby(["Cluster"])

                # generate random dropout & duration for each "Cluster"
                rands = [np.random.rand() for _ in range(clusters.size)]
                durations = [
                    gen_random_duration(
                        loc=sensor["weather"]["cloud_duration_mean"],
                        scale=sensor["weather"]["cloud_duration_std"],
                    )
                    for _ in range(clusters.size)
                ]

                # --- weather dropout
                dfs_filtered = []
                for rand, duration, cluster in zip(rands, durations, clusters):
                    data_cluster = grouped_data.get_group(cluster).copy()
                    if (rand <= sensor["weather"]["cloud_prob"]) and (
                        cloud_duration_mean != 0
                    ):
                        # drop = True, i.e. cloud coverage event occurs
                        # log useful numbers
                        cluster_duration = (
                            data_cluster["Datetime"].max()
                            - data_cluster["Datetime"].min()
                        )
                        cloud_duration = np.timedelta64(int(duration), "s")
                        if cloud_duration > cluster_duration:
                            # mark all measurements as cloudy
                            datetime_start = data_cluster["Datetime"].min()
                            datetime_stop = data_cluster["Datetime"].max()
                        else:
                            # mark subset of measurements as cloudy for specified duration
                            datetime_start = data_cluster.sample(n=1)[
                                "Datetime"
                            ].values[0]
                            datetime_stop = datetime_start + cloud_duration
                            # make sure the end time does not exceed the latest datetime in the cluster
                            if datetime_stop > data_cluster["Datetime"].max():
                                datetime_stop = data_cluster["Datetime"].max()
                                datetime_start = datetime_stop - cloud_duration

                        mask = (data_cluster["Datetime"] >= datetime_start) & (
                            data_cluster["Datetime"] <= datetime_stop
                        )
                        data_cluster["Clouds"] = mask
                        sensor["weather"]["drops"].append(
                            [datetime_start, datetime_stop]
                        )
                    else:
                        # drop = False, i.e. no clouds tonight
                        data_cluster["Clouds"] = False

                    dfs_filtered.append(data_cluster)
                dfs.append(pd.concat(dfs_filtered, axis=0))

            # After Loop Post-process ------------------------------------------|
            dfc = pd.concat(dfs, axis=0).sort_values("MJD")
            df_drop = df.drop(dfc[dfc["Clouds"] == True].index)

            # --- save plots
            if save_plots:
                # seq ID plots:
                for seq_ID in sorted(seq_IDs):
                    plot_seq_id(
                        seq_ID,
                        df,
                        df_drop,
                        save_without_displaying=True,
                        save_path=f"./plots/num_run_{num_run}/seq_id/",
                    )  # type: ignore
                # sensor ID plots:
                for sensor_ID in sensor_IDs:
                    plot_sensor_ID(
                        sensor_ID,
                        df,
                        df_drop,
                        sensor_list,
                        save_without_displaying=True,
                        save_path=f"./plots/num_run_{num_run}/sensor_ID/",
                    )

            # save dataset csv
            df_drop.drop(columns=["Datetime"]).sort_index().to_csv(
                f"output_dropped__{num_run}.csv", index=False
            )
            # sensor_list with dropout information pickle
            pd.to_pickle(sensor_list, f"sensor_list__{num_run}.pkl")
            # copy of original, unaltered dataset
            if save_copy_of_original and (num_run == 0):
                df.drop(columns=["Datetime"]).sort_index().to_csv(
                    "output.csv", index=False
                )
        except (TypeError, KeyError):
            pass


def launcher(
    config: type[DataClass],
    task_function: Callable[[Any], Any],
    overrides: list[str],
    submitit: str = "",
    return_output_dir: bool = False,
) -> None | Path:
    """Launches a multirun hydra job via hydra-zen

    Parameters
    ----------
    config : type[DataClass]
        Hydra-zen config with user-defined field names and, optionally,
        associated default values and/or type annotations.
    task_function : Callable[[Any], Any]
        The function that Hydra will execute. Its input will be config,
        which has been modified via the specified overrides
    overrides : list[str]
        Sets/overrides values in hydra-zen ``config``
    submitit : str, optional
        If specified, the path to a config JSON file defining how to launch jobs across multiple GPUs using submitit, by default None (serial launch only)
    return_output_dir : bool, optional
        If True, the data output path will be returned (otherwise None is returned), by default False

    Returns
    -------
    None | Path
        Returns None by default, unless input argument ``return_output_dir`` is True, then the data output path will be returned

    Raises
    ------
    MadlibException
        The specified submitit configuration file (JSON) must be a list of strings
    """
    # add submitit Config overrides if submitit bool argument is True
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

    # launch jobs (in parallel if using submitit)
    (jobs,) = launch(
        config=config,
        task_function=task_function,
        to_dictconfig=True,
        overrides=overrides,
        version_base="1.3",
        multirun=True,
    )

    if return_output_dir:
        # get multirun root directory
        return Path(jobs[0].working_dir).parent


# %%

if __name__ == "__main__":
    # parse CLI arguments
    parser = parseArgs()
    args = parser.parse_args()

    # customize Hydra's configuration
    store(HydraConf(job=JobConf(chdir=True)))
    store.add_to_hydra_store(overwrite_ok=True)

    # Parse the sensor YAML file
    sensor_data = SensorCollection.paramsFromYAML(args.sensor_yaml)
    sensor_list = [
        sensor_data[key] for key in sensor_data.keys()
    ]  # TODO: Legacy sensor_list format. Refactor to use sensor_data dict

    # zen wrapper that will auto-extract, resolve, and instantiate fields from
    # an input config based on dropout()'s signature
    task_function = zen(dropout)

    DropoutConfig = make_config(
        path=args.path,
        sensor_list=sensor_list,
        sensor_idx=None,
        cloud_prob=0.5,
        cloud_duration_mean=10800.0,
        cloud_duration_std=3600.0,
        num_runs=args.num_runs,
        save_copy_of_original=args.save_copy_of_original,
        save_plots=args.save_plots,
    )

    overrides = [
        f"cloud_prob={args.cloud_prob}",
        f"cloud_duration_mean={args.cloud_duration_mean}",
        f"cloud_duration_std={args.cloud_duration_std}",
    ]

    launcher(
        config=DropoutConfig,
        task_function=task_function,
        overrides=overrides,
        submitit=args.submitit,
    )
