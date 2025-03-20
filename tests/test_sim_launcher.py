# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

"""
Test file: test_hz_launcher.py
Description: This file contains unit tests for the `_sim_launcher` module
"""

from maddg._sim_launcher import launcher
from scripts.hz_launcher import simulator_task
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
from madlib._utils import MadlibException
import yaml
from unittest import mock

sensor_yaml_11 = "configs/sample_sensor_network.yaml"
sensor_yaml_blind = "tests/inputs/blind_sensor.yaml"


def clear_outputs():
    """Delete contents of tests/outputs"""
    outputs_dir = Path("tests/outputs")
    if outputs_dir.is_dir():
        for f in outputs_dir.glob("*"):
            f.unlink()


def count_experiments(multirun_dir: str):
    """Given a directory containing hydra-zen multirun output, count the number
    of experiments that the directory currently contains."""
    multirun_path = Path(multirun_dir)
    experiment_count = 0
    experiment_dirs = []
    if multirun_path.is_dir():
        experiment_dirs = list(multirun_path.glob("*/*"))
        experiment_count = len(experiment_dirs)

    return experiment_count, experiment_dirs


def clear_experiment(experiment_dir: Path):
    """Given an experiment directory, delete it and its contents. If its
    parent is then empty, delete the parent. Same for grandparent."""
    parent = experiment_dir.parent
    grandparent = parent.parent
    shutil.rmtree(experiment_dir)

    siblings = list(parent.glob("*"))
    if len(siblings) == 0:
        shutil.rmtree(parent)

        aunts = list(grandparent.glob("*"))
        if len(aunts) == 0:
            shutil.rmtree(grandparent)


class TestLauncher:
    """Test the behavior of the launcher function"""

    def test_impulsive(self):
        """Test launching experiments with impulsive maneuvers"""

        # This test should generate files and directories under ./multirun,
        # as well as summary outputs under ./tests/outputs. We'll start by
        # clearing out the ./tests/outputs directory and measuring how many
        # experiments already exist under ./multirun
        clear_outputs()
        experiment_count, experiment_dirs = count_experiments("multirun")

        num_sim_pairs = 1
        outdir = "tests/outputs"
        mtype = "impulse"
        sim_duration_days = 3
        start_mjd = 60196.5

        launcher(
            simulator_method=simulator_task,
            mtype=mtype,
            num_sim_pairs=num_sim_pairs,
            sensor_yaml=sensor_yaml_11,
            outdir=outdir,
            start_mjd=start_mjd,
            sim_duration_days=sim_duration_days,
            random_seed=0,
        )

        # The number of multirun experiments should have increased by 1
        new_experiment_count, new_experiment_dirs = count_experiments("multirun")
        assert new_experiment_count == experiment_count + 1

        output_csv = Path("tests/outputs/complete.csv")
        errors_txt = Path("tests/outputs/errors.txt")

        assert output_csv.exists()
        assert errors_txt.exists()

        results = pd.read_csv(output_csv)

        assert results["Sequence"].isin([0, 1]).all()
        assert len(results) > 0
        assert sum(results["Maneuver"] == 0) > 0
        assert sum(results["Maneuver"] == 1) > 0

        no_mnvr = results.loc[results["Maneuver"] == 0]
        mnvr = results.loc[results["Maneuver"] == 1]

        assert all(no_mnvr["Maneuver_Start_MJD"].isna())
        assert all(no_mnvr["Maneuver_End_MJD"].isna())
        assert all(no_mnvr["Maneuver_DV_Radial_KmS"].isna())
        assert all(no_mnvr["Maneuver_DV_InTrack_KmS"].isna())
        assert all(no_mnvr["Maneuver_DV_CrossTrack_KmS"].isna())

        assert all(mnvr["Maneuver_Start_MJD"] > start_mjd)
        assert all(mnvr["Maneuver_Start_MJD"] < start_mjd + sim_duration_days)
        assert all(mnvr["Maneuver_End_MJD"].isna())
        np.testing.assert_allclose(mnvr["Maneuver_DV_Radial_KmS"], 0, atol=1e-6)
        assert all(np.abs(mnvr["Maneuver_DV_InTrack_KmS"]) > 1e-6)
        assert all(np.abs(mnvr["Maneuver_DV_CrossTrack_KmS"]) > 1e-6)

        with open(errors_txt, "r") as f:
            errors = f.read()

        assert errors == ""

        # Cleanup
        clear_outputs()
        experiment_dir = (set(new_experiment_dirs) - set(experiment_dirs)).pop()
        clear_experiment(experiment_dir)

    def test_impulsive_with_modified_sensor_dra_ddec(self):
        """Test launching experiments with impulsive maneuvers"""

        # This test should generate files and directories under ./multirun,
        # as well as summary outputs under ./tests/outputs. We'll start by
        # clearing out the ./tests/outputs directory and measuring how many
        # experiments already exist under ./multirun
        clear_outputs()
        experiment_count, experiment_dirs = count_experiments("multirun")

        num_sim_pairs = 1
        outdir = "tests/outputs"
        mtype = "impulse"
        sim_duration_days = 3
        start_mjd = 60196.5

        launcher(
            simulator_method=simulator_task,
            mtype=mtype,
            num_sim_pairs=num_sim_pairs,
            sensor_yaml=sensor_yaml_11,
            outdir=outdir,
            start_mjd=start_mjd,
            sim_duration_days=sim_duration_days,
            random_seed=0,
            sensor_ddec=12.0,
            sensor_dra=7.7,
        )

        # The number of multirun experiments should have increased by 1
        new_experiment_count, new_experiment_dirs = count_experiments("multirun")
        assert new_experiment_count == experiment_count + 1

        output_csv = Path("tests/outputs/complete.csv")
        errors_txt = Path("tests/outputs/errors.txt")
        multirun_yaml_path = Path("tests/outputs/multirun.yaml")

        assert output_csv.exists()
        assert errors_txt.exists()

        results = pd.read_csv(output_csv)

        assert results["Sequence"].isin([0, 1]).all()
        assert len(results) > 0
        assert sum(results["Maneuver"] == 0) > 0
        assert sum(results["Maneuver"] == 1) > 0

        no_mnvr = results.loc[results["Maneuver"] == 0]
        mnvr = results.loc[results["Maneuver"] == 1]

        assert all(no_mnvr["Maneuver_Start_MJD"].isna())
        assert all(no_mnvr["Maneuver_End_MJD"].isna())
        assert all(no_mnvr["Maneuver_DV_Radial_KmS"].isna())
        assert all(no_mnvr["Maneuver_DV_InTrack_KmS"].isna())
        assert all(no_mnvr["Maneuver_DV_CrossTrack_KmS"].isna())

        assert all(no_mnvr["Maneuver_Start_MJD"].isna())
        assert all(no_mnvr["Maneuver_End_MJD"].isna())
        assert all(no_mnvr["Maneuver_DV_Radial_KmS"].isna())
        assert all(no_mnvr["Maneuver_DV_InTrack_KmS"].isna())
        assert all(no_mnvr["Maneuver_DV_CrossTrack_KmS"].isna())

        with open(errors_txt, "r") as f:
            errors = f.read()

        assert errors == ""

        # check `dra` and `ddec` were changed
        with open(multirun_yaml_path, "r") as file:
            multirun_yaml = yaml.safe_load(file)

        multirun_yaml_sensor_dras = [
            multirun_yaml["sensor_params"][sensor]["dra"]
            for sensor in multirun_yaml["sensor_params"].keys()
        ]
        multirun_yaml_sensor_ddecs = [
            multirun_yaml["sensor_params"][sensor]["ddec"]
            for sensor in multirun_yaml["sensor_params"].keys()
        ]
        np.testing.assert_allclose(multirun_yaml_sensor_dras, 7.7, atol=1e-6)
        np.testing.assert_allclose(multirun_yaml_sensor_ddecs, 12.0, atol=1e-6)

        # Cleanup
        clear_outputs()
        experiment_dir = (set(new_experiment_dirs) - set(experiment_dirs)).pop()
        clear_experiment(experiment_dir)

    def test_continuous(self):
        """Test launching experiments with continuous maneuvers"""

        # This test should generate files and directories under ./multirun,
        # as well as summary outputs under ./tests/outputs. We'll start by
        # clearing out the ./tests/outputs directory and measuring how many
        # experiments already exist under ./multirun
        clear_outputs()
        experiment_count, experiment_dirs = count_experiments("multirun")

        num_sim_pairs = 1
        outdir = "tests/outputs"
        mtype = "continuous"
        sim_duration_days = 0.5
        start_mjd = 60196.5
        cont_thrust_duration_days = 2.0
        cont_thrust_mag = 1e-5
        cont_thrust_model = 0

        launcher(
            simulator_method=simulator_task,
            mtype=mtype,
            num_sim_pairs=num_sim_pairs,
            sensor_yaml=sensor_yaml_11,
            outdir=outdir,
            cont_thrust_duration_days=cont_thrust_duration_days,
            cont_thrust_mag=cont_thrust_mag,
            cont_thrust_model=cont_thrust_model,
            start_mjd=start_mjd,
            sim_duration_days=sim_duration_days,
            random_seed=0,
        )

        # The number of multirun experiments should have increased by 1
        new_experiment_count, new_experiment_dirs = count_experiments("multirun")
        assert new_experiment_count == experiment_count + 1

        output_csv = Path("tests/outputs/complete.csv")
        errors_txt = Path("tests/outputs/errors.txt")

        assert output_csv.exists()
        assert errors_txt.exists()

        results = pd.read_csv(output_csv)

        assert results["Sequence"].isin([0, 2]).all()
        assert len(results) > 0
        assert sum(results["Maneuver"] == 0) > 0
        assert sum(results["Maneuver"] == 2) > 0

        no_mnvr = results.loc[results["Maneuver"] == 0]
        mnvr = results.loc[results["Maneuver"] == 2]

        assert all(no_mnvr["Maneuver_Start_MJD"].isna())
        assert all(no_mnvr["Maneuver_End_MJD"].isna())
        assert all(no_mnvr["Maneuver_DV_Radial_KmS"].isna())
        assert all(no_mnvr["Maneuver_DV_InTrack_KmS"].isna())
        assert all(no_mnvr["Maneuver_DV_CrossTrack_KmS"].isna())

        np.testing.assert_allclose(mnvr["Maneuver_Start_MJD"], start_mjd, atol=1e-6)
        np.testing.assert_allclose(
            mnvr["Maneuver_End_MJD"], start_mjd + cont_thrust_duration_days, atol=1e-6
        )
        np.testing.assert_allclose(mnvr["Maneuver_DV_Radial_KmS"], 0, atol=1e-6)
        np.testing.assert_allclose(mnvr["Maneuver_DV_InTrack_KmS"], 1e-5, atol=1e-6)
        np.testing.assert_allclose(mnvr["Maneuver_DV_CrossTrack_KmS"], 0, atol=1e-6)

        with open(errors_txt, "r") as f:
            errors = f.read()

        assert errors == ""

        # Cleanup
        clear_outputs()
        experiment_dir = (set(new_experiment_dirs) - set(experiment_dirs)).pop()
        clear_experiment(experiment_dir)

    def test_no_obs(self):
        """Test that a simulation with no observations creates an empty csv"""
        clear_outputs()
        experiment_count, experiment_dirs = count_experiments("multirun")

        num_sim_pairs = 1
        outdir = "tests/outputs_blank"
        mtype = "impulse"
        sim_duration_days = 0.5
        start_mjd = 60196.5

        launcher(
            simulator_method=simulator_task,
            mtype=mtype,
            num_sim_pairs=num_sim_pairs,
            sensor_yaml=sensor_yaml_blind,
            outdir=outdir,
            start_mjd=start_mjd,
            sim_duration_days=sim_duration_days,
            random_seed=0,
        )

        # The number of multirun experiments should have increased by 1
        new_experiment_count, new_experiment_dirs = count_experiments("multirun")
        assert new_experiment_count == experiment_count + 1

        output_csv = Path(outdir) / "complete.csv"
        errors_txt = Path(outdir) / "errors.txt"

        assert output_csv.exists()
        assert errors_txt.exists()

        with open(output_csv, "r") as f:
            output = f.read()

        assert output.strip() == ""

        with open(errors_txt, "r") as f:
            errors = f.read()

        assert errors == ""

        # Cleanup
        clear_outputs()
        shutil.rmtree(outdir)
        experiment_dir = (set(new_experiment_dirs) - set(experiment_dirs)).pop()
        clear_experiment(experiment_dir)

    def test_auto_cleanup(self):
        """Test that the rm_multirun_root option will clean up multirun directories."""

        # This test should generate files and directories under ./multirun,
        # as well as summary outputs under ./tests/outputs. We'll start by
        # clearing out the ./tests/outputs directory and measuring how many
        # experiments already exist under ./multirun
        clear_outputs()
        experiment_count, experiment_dirs = count_experiments("multirun")

        num_sim_pairs = 1
        outdir = "tests/outputs"
        mtype = "impulse"
        sim_duration_days = 0.5
        start_mjd = 60196.5

        launcher(
            simulator_method=simulator_task,
            mtype=mtype,
            num_sim_pairs=num_sim_pairs,
            sensor_yaml=sensor_yaml_11,
            outdir=outdir,
            start_mjd=start_mjd,
            sim_duration_days=sim_duration_days,
            rm_multirun_root=True,
            random_seed=0,
        )

        # The number of multirun experiments should not have changed
        new_experiment_count, new_experiment_dirs = count_experiments("multirun")
        assert new_experiment_count == experiment_count

        output_csv = Path("tests/outputs/complete.csv")
        errors_txt = Path("tests/outputs/errors.txt")

        assert output_csv.exists()
        assert errors_txt.exists()

        results = pd.read_csv(output_csv)

        assert results["Sequence"].isin([0, 1]).all()
        assert len(results) > 0
        assert sum(results["Maneuver"] == 0) > 0
        assert sum(results["Maneuver"] == 1) > 0

        with open(errors_txt, "r") as f:
            errors = f.read()

        assert errors == ""

        # Cleanup
        clear_outputs()

    def test_prediction_error(self):
        """Test launching experiments with orbit-misestimation"""

        # This test should generate files and directories under ./multirun,
        # as well as summary outputs under ./tests/outputs. We'll start by
        # clearing out the ./tests/outputs directory and measuring how many
        # experiments already exist under ./multirun
        clear_outputs()
        experiment_count, experiment_dirs = count_experiments("multirun")

        num_sim_pairs = 1
        outdir = "tests/outputs"
        mtype = "impulse"
        sim_duration_days = 3
        start_mjd = 60196.5

        launcher(
            simulator_method=simulator_task,
            mtype=mtype,
            num_sim_pairs=num_sim_pairs,
            sensor_yaml=sensor_yaml_11,
            outdir=outdir,
            start_mjd=start_mjd,
            sim_duration_days=sim_duration_days,
            random_seed=0,
            pred_err=0.1,
        )

        # The number of multirun experiments should have increased by 1
        new_experiment_count, new_experiment_dirs = count_experiments("multirun")
        assert new_experiment_count == experiment_count + 1

        output_csv = Path("tests/outputs/complete.csv")
        errors_txt = Path("tests/outputs/errors.txt")

        assert output_csv.exists()
        assert errors_txt.exists()

        results = pd.read_csv(output_csv)

        assert results["Sequence"].isin([0, 1]).all()
        assert len(results) > 0
        assert sum(results["Maneuver"] == 0) > 0
        assert sum(results["Maneuver"] == 1) > 0

        no_mnvr = results.loc[results["Maneuver"] == 0]
        mnvr = results.loc[results["Maneuver"] == 1]

        assert all(no_mnvr["Maneuver_Start_MJD"].isna())
        assert all(no_mnvr["Maneuver_End_MJD"].isna())
        assert all(no_mnvr["Maneuver_DV_Radial_KmS"].isna())
        assert all(no_mnvr["Maneuver_DV_InTrack_KmS"].isna())
        assert all(no_mnvr["Maneuver_DV_CrossTrack_KmS"].isna())

        assert all(mnvr["Maneuver_Start_MJD"] > start_mjd)
        assert all(mnvr["Maneuver_Start_MJD"] < start_mjd + sim_duration_days)
        assert all(mnvr["Maneuver_End_MJD"].isna())
        np.testing.assert_allclose(mnvr["Maneuver_DV_Radial_KmS"], 0, atol=1e-6)
        assert all(np.abs(mnvr["Maneuver_DV_InTrack_KmS"]) > 1e-6)
        assert all(np.abs(mnvr["Maneuver_DV_CrossTrack_KmS"]) > 1e-6)

        # Make sure that even the non-maneuver case has large residuals
        assert np.abs(no_mnvr["RA Arcsec"].iloc[0]) > 1000.0
        assert np.abs(no_mnvr["DEC Arcsec"].iloc[0]) > 1000.0

        with open(errors_txt, "r") as f:
            errors = f.read()

        assert errors == ""

        # Cleanup
        clear_outputs()
        experiment_dir = (set(new_experiment_dirs) - set(experiment_dirs)).pop()
        clear_experiment(experiment_dir)

    def test_auto_cleanup_with_error(self):
        """Test that the rm_multirun_root option will clean up multirun directories
        even if an error.txt file is produced."""

        clear_outputs()
        experiment_count, experiment_dirs = count_experiments("multirun")

        num_sim_pairs = 1
        outdir = "tests/outputs"
        mtype = "impulse"
        sim_duration_days = 3
        start_mjd = 60196.5

        invalid_sensor_yaml = "tests/inputs/invalid_sensor_4.yaml"

        launcher(
            simulator_method=simulator_task,
            mtype=mtype,
            num_sim_pairs=num_sim_pairs,
            sensor_yaml=invalid_sensor_yaml,
            outdir=outdir,
            dv_ric_mean_kms=(1000, 1000, 1000),
            start_mjd=start_mjd,
            sim_duration_days=sim_duration_days,
            rm_multirun_root=True,
            random_seed=0,
        )

        # The number of multirun experiments should not have changed
        new_experiment_count, new_experiment_dirs = count_experiments("multirun")
        assert new_experiment_count == experiment_count

        output_csv = Path("tests/outputs/complete.csv")
        errors_txt = Path("tests/outputs/errors.txt")

        assert output_csv.exists()
        assert errors_txt.exists()

        with open(errors_txt, "r") as f:
            errors = f.read()

        assert errors != ""

        # Cleanup
        clear_outputs()

    def test_cleanup_without_hydra_dir(self):
        """Make sure nothing breaks if the .hydra folder does not exist
        when it's time to delete it."""

        original_exists = Path.exists

        # We'll need to mock Path.exists() to make this test work
        def mock_exists(self):
            if self.stem == ".hydra":
                return False
            else:
                return original_exists(self)

        with mock.patch.object(Path, "exists", mock_exists):
            clear_outputs()
            experiment_count, experiment_dirs = count_experiments("multirun")

            num_sim_pairs = 1
            outdir = "tests/outputs"
            mtype = "impulse"
            sim_duration_days = 3
            start_mjd = 60196.5

            invalid_sensor_yaml = "tests/inputs/invalid_sensor_4.yaml"

            launcher(
                simulator_method=simulator_task,
                mtype=mtype,
                num_sim_pairs=num_sim_pairs,
                sensor_yaml=invalid_sensor_yaml,
                outdir=outdir,
                dv_ric_mean_kms=(1000, 1000, 1000),
                start_mjd=start_mjd,
                sim_duration_days=sim_duration_days,
                rm_multirun_root=True,
                random_seed=0,
            )

            # The number of multirun experiments should have increased
            new_experiment_count, new_experiment_dirs = count_experiments("multirun")
            assert new_experiment_count == experiment_count + 1

            output_csv = Path("tests/outputs/complete.csv")
            errors_txt = Path("tests/outputs/errors.txt")

            assert output_csv.exists()
            assert errors_txt.exists()

            with open(errors_txt, "r") as f:
                errors = f.read()

            assert errors != ""

            # Cleanup
            clear_outputs()

    def test_auto_cleanup_custom_root(self):
        """Test that the rm_multirun_root option will clean up multirun directories."""
        clear_outputs()
        multirun_root = "tests/outputs/multirun"
        experiment_count, experiment_dirs = count_experiments(multirun_root)

        num_sim_pairs = 1
        outdir = "tests/outputs"
        mtype = "impulse"
        sim_duration_days = 0.5
        start_mjd = 60196.5

        launcher(
            simulator_method=simulator_task,
            mtype=mtype,
            num_sim_pairs=num_sim_pairs,
            sensor_yaml=sensor_yaml_11,
            outdir=outdir,
            start_mjd=start_mjd,
            sim_duration_days=sim_duration_days,
            multirun_root=multirun_root,
            rm_multirun_root=True,
            random_seed=0,
        )

        # The number of multirun experiments should not have changed
        new_experiment_count, new_experiment_dirs = count_experiments(multirun_root)
        assert new_experiment_count == experiment_count

        output_csv = Path("tests/outputs/complete.csv")
        errors_txt = Path("tests/outputs/errors.txt")

        assert output_csv.exists()
        assert errors_txt.exists()

        results = pd.read_csv(output_csv)

        assert results["Sequence"].isin([0, 1]).all()
        assert len(results) > 0
        assert sum(results["Maneuver"] == 0) > 0
        assert sum(results["Maneuver"] == 1) > 0

        with open(errors_txt, "r") as f:
            errors = f.read()

        assert errors == ""

        # Cleanup
        clear_outputs()

    def test_auto_cleanup_partial(self):
        """Check that directories are not deleted if they contain a non-hydra file."""
        clear_outputs()
        multirun_root = "tests/outputs/multirun_partial"
        experiment_count, experiment_dirs = count_experiments(multirun_root)

        num_sim_pairs = 1
        outdir = "tests/outputs"
        mtype = "impulse"
        sim_duration_days = 0.5
        start_mjd = 60196.5

        original_exists = Path.exists

        # We'll need to mock Path.exists() to make this test work
        def mock_exists(self):
            if self.stem == ".hydra":
                # Create new files to prevent complete erasure
                files = list(self.glob("../*"))
                for f in files:
                    newfile = Path(f"{f}.blank")
                    with open(newfile, "w") as f:
                        f.write("")

            return original_exists(self)

        with mock.patch.object(Path, "exists", mock_exists):

            launcher(
                simulator_method=simulator_task,
                mtype=mtype,
                num_sim_pairs=num_sim_pairs,
                sensor_yaml=sensor_yaml_11,
                outdir=outdir,
                start_mjd=start_mjd,
                sim_duration_days=sim_duration_days,
                multirun_root=multirun_root,
                rm_multirun_root=True,
                random_seed=0,
            )

            # The number of multirun experiments should have increased
            new_experiment_count, new_experiment_dirs = count_experiments(multirun_root)
            assert new_experiment_count == experiment_count + 1

            output_csv = Path("tests/outputs/complete.csv")
            errors_txt = Path("tests/outputs/errors.txt")

            assert output_csv.exists()
            assert errors_txt.exists()

            with open(errors_txt, "r") as f:
                errors = f.read()

            assert errors == ""

            # Cleanup
            experiment_dir = (set(new_experiment_dirs) - set(experiment_dirs)).pop()
            clear_experiment(experiment_dir)
            clear_outputs()

    def test_yaml_copy_exception(self):
        """Check that code still runs as expected if multirun.yaml is not found at the end."""
        clear_outputs()
        multirun_root = "tests/outputs_noyaml/multirun_noyaml"
        experiment_count, experiment_dirs = count_experiments(multirun_root)

        num_sim_pairs = 1
        outdir = "tests/outputs_noyaml"
        mtype = "impulse"
        sim_duration_days = 0.5
        start_mjd = 60196.5

        if Path(outdir).exists() and Path(outdir).is_dir():
            shutil.rmtree(Path(outdir))

        original_glob = Path.glob

        # We'll need to mock Path.exists() to make this test work
        def mock_glob(self, args):
            if args == "multirun.yaml":
                (self / "multirun.yaml").unlink(missing_ok=True)
            return original_glob(self, args)

        with mock.patch.object(Path, "glob", mock_glob):
            launcher(
                simulator_method=simulator_task,
                mtype=mtype,
                num_sim_pairs=num_sim_pairs,
                sensor_yaml=sensor_yaml_11,
                outdir=outdir,
                start_mjd=start_mjd,
                sim_duration_days=sim_duration_days,
                multirun_root=multirun_root,
                rm_multirun_root=True,
                random_seed=0,
            )

            # The number of multirun experiments should not have increased
            new_experiment_count, new_experiment_dirs = count_experiments(multirun_root)
            assert new_experiment_count == experiment_count

            # The copied multirun.yaml should not exist
            assert not (Path(outdir) / "multirun.yaml").exists()

            output_csv = Path(outdir) / "complete.csv"
            errors_txt = Path(outdir) / "errors.txt"

            assert output_csv.exists()
            assert errors_txt.exists()

            with open(errors_txt, "r") as f:
                errors = f.read()

            assert errors == ""

            # Cleanup
            clear_outputs()
            shutil.rmtree(outdir)


class TestSubmitit:
    """These tests will check the behavior of the hydra submitit launcher"""

    def test_submitit(self):
        """Test that the submitit overrides work (NOTE: This will not actually run submitit)"""
        clear_outputs()
        experiment_count, experiment_dirs = count_experiments("multirun")

        num_sim_pairs = 1
        outdir = "tests/outputs"
        mtype = "impulse"
        sim_duration_days = 0.5
        start_mjd = 60196.5

        launcher(
            simulator_method=simulator_task,
            mtype=mtype,
            num_sim_pairs=num_sim_pairs,
            sensor_yaml=sensor_yaml_11,
            outdir=outdir,
            start_mjd=start_mjd,
            sim_duration_days=sim_duration_days,
            rm_multirun_root=True,
            random_seed=0,
            submitit="tests/inputs/submitit_test.json",
        )

        # The number of multirun experiments should not have changed
        new_experiment_count, new_experiment_dirs = count_experiments("multirun")
        assert new_experiment_count == experiment_count

        output_csv = Path("tests/outputs/complete.csv")
        errors_txt = Path("tests/outputs/errors.txt")

        assert output_csv.exists()
        assert errors_txt.exists()

        results = pd.read_csv(output_csv)

        assert results["Sequence"].isin([0, 1]).all()
        assert len(results) > 0
        assert sum(results["Maneuver"] == 0) > 0
        assert sum(results["Maneuver"] == 1) > 0

        with open(errors_txt, "r") as f:
            errors = f.read()

        assert errors == ""

        # Cleanup
        clear_outputs()

    def test_invalid(self):
        """Test that a submitit JSON with invalid format will be rejected"""
        clear_outputs()
        experiment_count, experiment_dirs = count_experiments("multirun")

        num_sim_pairs = 1
        outdir = "tests/outputs"
        mtype = "impulse"
        sim_duration_days = 0.5
        start_mjd = 60196.5

        failed = False
        try:
            launcher(
                simulator_method=simulator_task,
                mtype=mtype,
                num_sim_pairs=num_sim_pairs,
                sensor_yaml=sensor_yaml_11,
                outdir=outdir,
                start_mjd=start_mjd,
                sim_duration_days=sim_duration_days,
                rm_multirun_root=True,
                random_seed=0,
                submitit="tests/inputs/submitit_invalid.json",
            )
        except MadlibException:
            failed = True

        assert failed

        # The number of multirun experiments should not have changed
        new_experiment_count, new_experiment_dirs = count_experiments("multirun")
        assert new_experiment_count == experiment_count

        output_csv = Path("tests/outputs/complete.csv")
        errors_txt = Path("tests/outputs/errors.txt")

        assert not output_csv.exists()
        assert not errors_txt.exists()

    def test_auto_cleanup(self):
        """Test that the rm_multirun_root option will clean up multirun directories."""

        # This test should generate files and directories under ./multirun,
        # as well as summary outputs under ./tests/outputs. We'll start by
        # clearing out the ./tests/outputs directory and measuring how many
        # experiments already exist under ./multirun
        clear_outputs()
        experiment_count, experiment_dirs = count_experiments("multirun")

        num_sim_pairs = 1
        outdir = "tests/outputs"
        mtype = "impulse"
        sim_duration_days = 0.5
        start_mjd = 60196.5

        launcher(
            simulator_method=simulator_task,
            mtype=mtype,
            num_sim_pairs=num_sim_pairs,
            sensor_yaml=sensor_yaml_11,
            outdir=outdir,
            start_mjd=start_mjd,
            sim_duration_days=sim_duration_days,
            rm_multirun_root=True,
            random_seed=0,
            submitit="tests/inputs/submitit_test.json",
        )

        # The number of multirun experiments should not have changed
        new_experiment_count, new_experiment_dirs = count_experiments("multirun")
        assert new_experiment_count == experiment_count

        output_csv = Path("tests/outputs/complete.csv")
        errors_txt = Path("tests/outputs/errors.txt")

        assert output_csv.exists()
        assert errors_txt.exists()

        results = pd.read_csv(output_csv)

        assert results["Sequence"].isin([0, 1]).all()
        assert len(results) > 0
        assert sum(results["Maneuver"] == 0) > 0
        assert sum(results["Maneuver"] == 1) > 0

        with open(errors_txt, "r") as f:
            errors = f.read()

        assert errors == ""

        # Cleanup
        clear_outputs()
