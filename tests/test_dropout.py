# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time
from hydra.conf import HydraConf, JobConf
from hydra_zen import make_config, zen, ZenStore

# add parent directory of __file__ to sys.path, if isn't already included
if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.append(str(Path(__file__).parents[1]))

from madlib import SensorCollection
from madlib._utils import MadlibException
from scripts import dropout
import pytest


PROJECT_ROOT = str(Path(__file__).parents[1])


def get_default_overrides() -> list:
    return [
        f"hydra.sweep.dir={PROJECT_ROOT}/tests/outputs/multirun/"
        + "${now:%Y-%m-%d}/${now:%H-%M-%S}",
        "hydra.sweep.subdir=${hydra.job.num}",
        "cloud_prob=1.0",
        "cloud_duration_mean=3600,7200",
        "cloud_duration_std=0.0",
    ]


def add_datetime(df: pd.DataFrame) -> None:
    """Adds a datetime64 column (inplace) to a DataFrame with an existing "MJD" column

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame with an existing "MJD" column
    """
    ts_mjd = Time(df["MJD"], format="mjd")
    ts_datetime = ts_mjd.to_value("datetime64")
    df["Datetime"] = ts_datetime
    df.sort_values("MJD", inplace=True)


class TestArgs:
    """Test behavior of argument parsing"""

    def test_argument_defaults(self):
        """Test default argument values"""
        inputs = [
            "--path",
            "inputs/complete.impulse.csv",
            "--sensor_yaml",
            "/path/to/sensor.yaml",
        ]

        parser = dropout.parseArgs()
        args = parser.parse_args(inputs)

        # Count number of input arguments
        assert len(args.__dict__) == 9

        assert isinstance(args.path, str)
        assert args.path == "inputs/complete.impulse.csv"
        assert isinstance(args.sensor_yaml, str)
        assert args.sensor_yaml == "/path/to/sensor.yaml"
        assert isinstance(args.cloud_prob, str)
        assert args.cloud_prob == "0.5"
        assert isinstance(args.cloud_duration_mean, str)
        assert args.cloud_duration_mean == "10800.0"
        assert isinstance(args.cloud_duration_std, str)
        assert args.cloud_duration_std == "3600.0"
        assert isinstance(args.num_runs, int)
        assert args.num_runs == 1
        assert isinstance(args.save_copy_of_original, bool)
        assert args.save_copy_of_original == False
        assert isinstance(args.save_plots, bool)
        assert args.save_plots == False
        assert isinstance(args.submitit, str)
        assert args.submitit == ""

    def test_argument_inputs(self):
        """Test that input values work as expected"""
        inputs = [
            "--path",
            "inputs/complete.impulse.csv",
            "--sensor_yaml",
            "/path/to/sensor.yaml",
            "--cloud_prob",
            "0.1",
            "--cloud_duration_mean",
            "60.0",
            "--cloud_duration_std",
            "10.0",
            "--num_runs",
            "10",
            "--save_copy_of_original",
            "--save_plots",
            "--submitit",
            PROJECT_ROOT + "/configs/submitit_slurm_template.json",
        ]

        parser = dropout.parseArgs()
        args = parser.parse_args(inputs)

        # Count number of input arguments
        assert len(args.__dict__) == 9

        assert isinstance(args.path, str)
        assert args.path == "inputs/complete.impulse.csv"
        assert isinstance(args.sensor_yaml, str)
        assert args.sensor_yaml == "/path/to/sensor.yaml"
        assert isinstance(args.cloud_prob, str)
        assert args.cloud_prob == "0.1"
        assert isinstance(args.cloud_duration_mean, str)
        assert args.cloud_duration_mean == "60.0"
        assert isinstance(args.cloud_duration_std, str)
        assert args.cloud_duration_std == "10.0"
        assert isinstance(args.num_runs, int)
        assert args.num_runs == 10
        assert isinstance(args.save_copy_of_original, bool)
        assert args.save_copy_of_original == True
        assert isinstance(args.save_plots, bool)
        assert args.save_plots == True
        assert isinstance(args.submitit, str)
        assert args.submitit == PROJECT_ROOT + "/configs/submitit_slurm_template.json"


class TestDropout:
    """Test dropout core behavior"""

    def test_dropout_general_functionality_impulse(self):
        """Test the core functionality of the dropout script with impulse maneuver datasets"""

        # customize Hydra's configuration
        store1 = ZenStore(overwrite_ok=True)
        store1(HydraConf(job=JobConf(chdir=True)))
        store1.add_to_hydra_store(overwrite_ok=True)

        # Parse the sensor YAML file
        sensor_data = SensorCollection.paramsFromYAML(
            PROJECT_ROOT + "/configs/sample_sensor_network.yaml"
        )
        sensor_list = [sensor_data[key] for key in sensor_data.keys()]

        # zen wrapper that will auto-extract, resolve, and instantiate fields from
        # an input config based on dropout()'s signature
        task_function = zen(dropout.dropout)

        DropoutConfig = make_config(
            path=PROJECT_ROOT + "/tests/inputs/complete.impulse.csv",
            sensor_list=sensor_list,
            sensor_idx=None,
            cloud_prob=0.5,
            cloud_duration_mean=10800.0,
            cloud_duration_std=3600.0,
            num_runs=1,
            save_copy_of_original=True,
            save_plots=True,
        )

        overrides = get_default_overrides()

        output_dir = dropout.launcher(
            config=DropoutConfig,
            task_function=task_function,
            overrides=overrides,
            submitit="",
            return_output_dir=True,
        )

        assert output_dir is not None
        # Check if data files exist and that data was "dropped"
        # Top level
        assert len(list(output_dir.glob("multirun.yaml"))) > 0
        assert len(list(output_dir.glob("*/"))) == 2
        # Sweep Param 1
        assert (output_dir / "0/output.csv").exists()
        df = pd.read_csv(output_dir / "0/output.csv")
        assert (output_dir / "0/output_dropped__0.csv").exists()
        df_drop = pd.read_csv(output_dir / "0/output_dropped__0.csv")
        assert df_drop.shape[0] < df.shape[0]
        assert len(list((output_dir / "0").glob("*.pkl"))) > 0
        sensor_list = pd.read_pickle(list((output_dir / "0").glob("*.pkl"))[0])
        assert isinstance(sensor_list, list)
        assert (output_dir / "0/plots").exists()
        assert len(list((output_dir / "0/plots").rglob("*.png"))) > 0
        # Sweep Param 2
        assert (output_dir / "1/output.csv").exists()
        assert (output_dir / "1/output_dropped__0.csv").exists()
        df_drop = pd.read_csv(output_dir / "1/output_dropped__0.csv")
        assert df_drop.shape[0] < df.shape[0]
        assert len(list((output_dir / "1").glob("*.pkl"))) > 0
        sensor_list = pd.read_pickle(list((output_dir / "1").glob("*.pkl"))[0])
        assert isinstance(sensor_list, list)
        assert (output_dir / "0/plots").exists()
        assert len(list((output_dir / "0/plots").rglob("*.png"))) > 0

        # Test plotting edge cases ---------------------------------------------|
        # Add Datetime column to DataFrames (required for plotting routines)
        add_datetime(df)
        add_datetime(df_drop)

        # Plot edge cases: sensor_ID -------------------------------------------|
        # save_without_displaying is False
        sensor_ID = "A1"
        assert (
            dropout.plot_sensor_ID(
                sensor_ID,
                df=df,
                df_drop=df_drop,
                sensor_list=sensor_list,
                save_without_displaying=False,
            )
            is None
        )

        # Plot sensor_ID does not exist
        sensor_ID = "INVALID_SENSOR_ID"
        assert (
            dropout.plot_sensor_ID(
                sensor_ID,
                df=df,
                df_drop=df_drop,
                sensor_list=sensor_list,
                save_without_displaying=False,
            )
            is None
        )

        # Plot edge cases: seq_ID ----------------------------------------------|
        # KeyError Exception Case
        seq_ID = 99999999
        assert (
            dropout.plot_seq_id(
                seq_ID,
                df=df,
                df_drop=df_drop,
                save_without_displaying=False,
            )
            is None
        )

        # save_without_displaying is False
        seq_ID = 0
        assert (
            dropout.plot_seq_id(
                seq_ID,
                df=df,
                df_drop=df_drop,
                save_without_displaying=False,
            )
            is None
        )

        # Remove multirun folder in entirety
        shutil.rmtree(PROJECT_ROOT + "/tests/outputs/multirun")

    def test_dropout_none_type_return(self):
        """Test dropout function with None type return"""

        # customize Hydra's configuration
        store1 = ZenStore(overwrite_ok=True)
        store1(HydraConf(job=JobConf(chdir=True)))
        store1.add_to_hydra_store(overwrite_ok=True)

        # Parse the sensor YAML file
        sensor_data = SensorCollection.paramsFromYAML(
            PROJECT_ROOT + "/configs/sample_sensor_network.yaml"
        )
        sensor_list = [sensor_data[key] for key in sensor_data.keys()]

        # zen wrapper that will auto-extract, resolve, and instantiate fields from
        # an input config based on dropout()'s signature
        task_function = zen(dropout.dropout)

        DropoutConfig = make_config(
            path=PROJECT_ROOT + "/tests/inputs/complete.impulse.csv",
            sensor_list=sensor_list,
            sensor_idx=None,
            cloud_prob=0.5,
            cloud_duration_mean=10800.0,
            cloud_duration_std=3600.0,
            num_runs=1,
            save_copy_of_original=True,
            save_plots=False,
        )

        overrides = get_default_overrides()

        output = dropout.launcher(
            config=DropoutConfig,
            task_function=task_function,
            overrides=overrides,
            submitit="",
            return_output_dir=False,
        )
        assert output is None

        # Remove multirun folder in entirety
        shutil.rmtree(PROJECT_ROOT + "/tests/outputs/multirun")

    def test_dropout_with_submitit_json_inputs(self):
        """Tests the dropout script while using JSON file paths for the submitit argument"""

        # Parse the sensor YAML file
        sensor_data = SensorCollection.paramsFromYAML(
            PROJECT_ROOT + "/configs/sample_sensor_network.yaml"
        )
        sensor_list = [sensor_data[key] for key in sensor_data.keys()]

        # zen wrapper that will auto-extract, resolve, and instantiate fields from
        # an input config based on dropout()'s signature
        task_function = zen(dropout.dropout)

        DropoutConfig = make_config(
            path=PROJECT_ROOT + "/tests/inputs/complete.impulse.csv",
            sensor_list=sensor_list,
            sensor_idx=None,
            cloud_prob=0.5,
            cloud_duration_mean=10800.0,
            cloud_duration_std=3600.0,
            num_runs=1,
            save_copy_of_original=True,
            save_plots=False,
        )

        # Test with submitit config
        try:
            # Try to use `srun` if on a system with SLURM
            overrides = get_default_overrides()

            output_dir = dropout.launcher(
                config=DropoutConfig,
                task_function=task_function,
                overrides=overrides,
                submitit=PROJECT_ROOT + "/configs/submitit_slurm_template.json",
                return_output_dir=True,
            )
            assert output_dir is not None
        except RuntimeError:
            # RuntimeError occurs if running these tests on a machine without SLURM,
            # use `submitit_test.json` which uses `hydra/launcher=submitit_local` in
            # this case.
            overrides = get_default_overrides()

            output_dir = dropout.launcher(
                config=DropoutConfig,
                task_function=task_function,
                overrides=overrides,
                submitit=PROJECT_ROOT + "/tests/inputs/submitit_test.json",
                return_output_dir=True,
            )
            assert output_dir is not None

        # Test with submitit invalid config
        with pytest.raises(MadlibException):
            overrides = get_default_overrides()

            dropout.launcher(
                config=DropoutConfig,
                task_function=task_function,
                overrides=overrides,
                submitit=PROJECT_ROOT + "/tests/inputs/submitit_invalid.json",
            )

        # Remove multirun folder in entirety
        shutil.rmtree(PROJECT_ROOT + "/tests/outputs/multirun")

    def test_dropout_general_functionality_continuous(self):
        """Test the core functionality of the dropout script with continuous maneuver datasets"""
        # customize Hydra's configuration
        store2 = ZenStore()
        store2(HydraConf(job=JobConf(chdir=True)))
        store2.add_to_hydra_store(overwrite_ok=True)

        # Parse the sensor YAML file
        sensor_data = SensorCollection.paramsFromYAML(
            PROJECT_ROOT + "/configs/sample_sensor_network.yaml"
        )
        sensor_list = [sensor_data[key] for key in sensor_data.keys()]

        # zen wrapper that will auto-extract, resolve, and instantiate fields from
        # an input config based on dropout()'s signature
        task_function = zen(dropout.dropout)

        DropoutConfig = make_config(
            path=PROJECT_ROOT + "/tests/inputs/complete.continuous.csv",
            sensor_list=sensor_list,
            sensor_idx=None,
            cloud_prob=0.5,
            cloud_duration_mean=10800.0,
            cloud_duration_std=3600.0,
            num_runs=1,
            save_copy_of_original=True,
            save_plots=True,
        )

        overrides = get_default_overrides()

        output_dir = dropout.launcher(
            config=DropoutConfig,
            task_function=task_function,
            overrides=overrides,
            submitit="",
            return_output_dir=True,
        )

        assert output_dir is not None
        # Check if data files exist and that data was "dropped"
        # Top level
        assert len(list(output_dir.glob("multirun.yaml"))) > 0
        assert len(list(output_dir.glob("*/"))) == 2
        # Sweep Param 1
        assert (output_dir / "0/output.csv").exists()
        df = pd.read_csv(output_dir / "0/output.csv")
        assert (output_dir / "0/output_dropped__0.csv").exists()
        df_dropped = pd.read_csv(output_dir / "0/output_dropped__0.csv")
        assert df_dropped.shape[0] < df.shape[0]
        assert len(list((output_dir / "0").glob("*.pkl"))) > 0
        sensor_list = pd.read_pickle(list((output_dir / "0").glob("*.pkl"))[0])
        assert isinstance(sensor_list, list)
        # Sweep Param 2
        assert (output_dir / "1/output.csv").exists()
        assert (output_dir / "1/output_dropped__0.csv").exists()
        df_dropped = pd.read_csv(output_dir / "1/output_dropped__0.csv")
        assert df_dropped.shape[0] < df.shape[0]
        assert len(list((output_dir / "1").glob("*.pkl"))) > 0
        sensor_list = pd.read_pickle(list((output_dir / "1").glob("*.pkl"))[0])
        assert isinstance(sensor_list, list)

        # Remove multirun folder in entirety
        shutil.rmtree(PROJECT_ROOT + "/tests/outputs/multirun")

    def test_dropout_fn_edge_cases(self):
        """Test dropout core functionality edge cases"""

        # Bad sensor list
        bad_sensor_list = [
            "this",
            "is",
            "a",
            "bad",
            "sensor_list",
        ]

        # Test bad sensor list (i.e. elements are not dicts)
        assert (
            dropout.dropout(
                path=PROJECT_ROOT + "/tests/inputs/complete.impulse.csv",
                sensor_list=bad_sensor_list,  # type: ignore
                sensor_idx=None,
                cloud_prob=0.5,
                cloud_duration_mean=10800.0,
                cloud_duration_std=3600.0,
                num_runs=1,
            )
            is None
        )

        # More than 1 num_run and save_plots = False
        # customize Hydra's configuration
        store4 = ZenStore(overwrite_ok=True)
        store4(HydraConf(job=JobConf(chdir=True)))
        store4.add_to_hydra_store(overwrite_ok=True)

        # zen wrapper that will auto-extract, resolve, and instantiate fields from
        # an input config based on dropout()'s signature
        task_function = zen(dropout.dropout)

        # Parse the sensor YAML file
        sensor_data = SensorCollection.paramsFromYAML(
            PROJECT_ROOT + "/configs/sample_sensor_network.yaml"
        )
        sensor_list = [sensor_data[key] for key in sensor_data.keys()]

        # zen wrapper that will auto-extract, resolve, and instantiate fields from
        # an input config based on dropout()'s signature
        task_function = zen(dropout.dropout)

        DropoutConfig = make_config(
            path=PROJECT_ROOT + "/tests/inputs/complete.impulse.csv",
            sensor_list=sensor_list,
            sensor_idx=None,
            cloud_prob=0.5,
            cloud_duration_mean=10800.0,
            cloud_duration_std=3600.0,
            num_runs=2,
            save_copy_of_original=True,
            save_plots=False,
        )

        overrides = get_default_overrides()

        output_dir = dropout.launcher(
            config=DropoutConfig,
            task_function=task_function,
            overrides=overrides,
            submitit="",
            return_output_dir=True,
        )

        assert output_dir is not None
        # Check if data files exist and that data was "dropped"
        # Top level
        assert len(list(output_dir.glob("multirun.yaml"))) > 0
        assert len(list(output_dir.glob("*/"))) == 2
        # Sweep Param 1
        assert (output_dir / "0/output.csv").exists()
        df = pd.read_csv(output_dir / "0/output.csv")
        assert (output_dir / "0/output_dropped__0.csv").exists()
        df_drop = pd.read_csv(output_dir / "0/output_dropped__0.csv")
        assert df_drop.shape[0] < df.shape[0]
        assert len(list((output_dir / "0").glob("*.pkl"))) > 0
        sensor_list = pd.read_pickle(list((output_dir / "0").glob("*.pkl"))[0])
        assert isinstance(sensor_list, list)

        # Remove multirun folder in entirety
        shutil.rmtree(PROJECT_ROOT + "/tests/outputs/multirun")

    def test_dropout_low_probability(self):
        """Test the core functionality of the dropout script when probability of a weather event is low (zero)"""
        # customize Hydra's configuration
        store4 = ZenStore(overwrite_ok=True)
        store4(HydraConf(job=JobConf(chdir=True)))
        store4.add_to_hydra_store(overwrite_ok=True)

        # zen wrapper that will auto-extract, resolve, and instantiate fields from
        # an input config based on dropout()'s signature
        task_function = zen(dropout.dropout)

        # Parse the sensor YAML file
        sensor_data = SensorCollection.paramsFromYAML(
            PROJECT_ROOT + "/configs/sample_sensor_network.yaml"
        )
        sensor_list = [sensor_data[key] for key in sensor_data.keys()]

        # zen wrapper that will auto-extract, resolve, and instantiate fields from
        # an input config based on dropout()'s signature
        task_function = zen(dropout.dropout)

        DropoutConfig = make_config(
            path=PROJECT_ROOT + "/tests/inputs/complete.impulse.csv",
            sensor_list=sensor_list,
            sensor_idx=None,
            cloud_prob=0.5,
            cloud_duration_mean=10800.0,
            cloud_duration_std=3600.0,
            num_runs=1,
            save_copy_of_original=True,
            save_plots=True,
        )

        overrides = [
            f"hydra.sweep.dir={PROJECT_ROOT}/tests/outputs/multirun/"
            + "${now:%Y-%m-%d}/${now:%H-%M-%S}",
            "hydra.sweep.subdir=${hydra.job.num}",
            "cloud_prob=0.0",
            "cloud_duration_mean=3600,7200",
            "cloud_duration_std=0.0",
        ]

        output_dir = dropout.launcher(
            config=DropoutConfig,
            task_function=task_function,
            overrides=overrides,
            submitit="",
            return_output_dir=True,
        )

        assert output_dir is not None
        # Check if data files exist and that data was "dropped"
        # Top level
        assert len(list(output_dir.glob("multirun.yaml"))) > 0
        assert len(list(output_dir.glob("*/"))) == 2
        # Sweep Param 1
        assert (output_dir / "0/output.csv").exists()
        df = pd.read_csv(output_dir / "0/output.csv")
        assert (output_dir / "0/output_dropped__0.csv").exists()
        df_drop = pd.read_csv(output_dir / "0/output_dropped__0.csv")
        assert len(list((output_dir / "0").glob("*.pkl"))) > 0
        sensor_list = pd.read_pickle(list((output_dir / "0").glob("*.pkl"))[0])
        assert isinstance(sensor_list, list)
        assert (output_dir / "0/plots").exists()
        assert len(list((output_dir / "0/plots").rglob("*.png"))) > 0
        # Sweep Param 2
        assert (output_dir / "1/output.csv").exists()
        assert (output_dir / "1/output_dropped__0.csv").exists()
        df_drop = pd.read_csv(output_dir / "1/output_dropped__0.csv")
        assert len(list((output_dir / "1").glob("*.pkl"))) > 0
        sensor_list = pd.read_pickle(list((output_dir / "1").glob("*.pkl"))[0])
        assert isinstance(sensor_list, list)
        assert (output_dir / "0/plots").exists()
        assert len(list((output_dir / "0/plots").rglob("*.png"))) > 0

        # Remove multirun folder in entirety
        shutil.rmtree(PROJECT_ROOT + "/tests/outputs/multirun")

    def test_dropout_get_sensor(self):
        """Test dropout.get_sensor()"""
        # Parse the sensor YAML file
        sensor_data = SensorCollection.paramsFromYAML(
            PROJECT_ROOT + "/configs/sample_sensor_network.yaml"
        )
        sensor_list = [sensor_data[key] for key in sensor_data.keys()]

        sensor = dropout.get_sensor(sensor_list, "INVALID_SENSOR_ID")
        assert sensor is None

    def test_dropout_update_sensor_weather_stats_with_idx(self):
        """Test dropout.update_sensor_weather_stats() index parameter"""
        # Parse the sensor YAML file
        sensor_data = SensorCollection.paramsFromYAML(
            PROJECT_ROOT + "/configs/sample_sensor_network.yaml"
        )
        sensor_list = [sensor_data[key] for key in sensor_data.keys()]

        idx = 1

        # check initial values
        sensor = sensor_list[idx]
        sensor_id_original = sensor["id"]
        assert np.isclose(sensor["weather"]["cloud_prob"], 0.5)
        assert np.isclose(sensor["weather"]["cloud_duration_mean"], 10800.0)
        assert np.isclose(sensor["weather"]["cloud_duration_std"], 7200.0)

        # update sensor using idx
        sensor_list = dropout.update_sensor_weather_stats(
            sensor_list,
            idx=idx,
            cloud_prob=0.555,
            cloud_duration_mean=12345.0,
            cloud_duration_std=321.0,
        )

        # check that the values were changed
        sensor = sensor_list[idx]
        assert np.isclose(sensor["weather"]["cloud_prob"], 0.555)
        assert np.isclose(sensor["weather"]["cloud_duration_mean"], 12345.0)
        assert np.isclose(sensor["weather"]["cloud_duration_std"], 321.0)
        assert sensor["id"] == sensor_id_original
