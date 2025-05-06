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


def prepare_output_dirs(base_name: str):
    """Given a base name, define the paths for an output and a
    multirun directory. Delete these directories if they already exist.
    Also return the paths to the final output CSV and errors text file."""
    outdir = f"tests/outputs/outputs_{base_name}"
    multirun_dir = f"tests/outputs/multirun_{base_name}"

    output_csv = Path(outdir) / Path("complete.csv")
    errors_txt = Path(outdir) / Path("errors.txt")

    if Path(outdir).exists() and Path(outdir).is_dir():
        shutil.rmtree(outdir)

    if Path(multirun_dir).exists() and Path(multirun_dir).is_dir():
        shutil.rmtree(multirun_dir)

    return outdir, multirun_dir, output_csv, errors_txt


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


class TestLauncher:
    """Test the behavior of the launcher function"""

    def test_impulsive(self):
        """Test launching experiments with impulsive maneuvers"""

        outdir, multirun_dir, output_csv, errors_txt = prepare_output_dirs(
            "test_impulsive"
        )

        num_sim_pairs = 1
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
            multirun_root=multirun_dir,
        )

        # A single multirun experiment should exist
        new_experiment_count, _ = count_experiments(multirun_dir)
        assert new_experiment_count == 1

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
        shutil.rmtree(outdir)
        shutil.rmtree(multirun_dir)

    def test_impulsive_with_modified_sensor_dra_ddec(self):
        """Test launching experiments with impulsive maneuvers"""

        outdir, multirun_dir, output_csv, errors_txt = prepare_output_dirs(
            "test_impulsive_with_modified_sensor"
        )

        num_sim_pairs = 1
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
            multirun_root=multirun_dir,
        )

        # A single multirun experiment should exist
        new_experiment_count, _ = count_experiments(multirun_dir)
        assert new_experiment_count == 1

        multirun_yaml_path = Path(outdir) / Path("multirun.yaml")

        assert output_csv.exists()
        assert errors_txt.exists()

        results = pd.read_csv(output_csv)

        assert results["Sequence"].isin([0, 1]).all()
        assert len(results) > 0
        assert sum(results["Maneuver"] == 0) > 0
        assert sum(results["Maneuver"] == 1) > 0

        no_mnvr = results.loc[results["Maneuver"] == 0]

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
        shutil.rmtree(outdir)
        shutil.rmtree(multirun_dir)

    def test_continuous(self):
        """Test launching experiments with continuous maneuvers"""

        outdir, multirun_dir, output_csv, errors_txt = prepare_output_dirs(
            "test_continuous"
        )

        num_sim_pairs = 1
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
            multirun_root=multirun_dir,
        )

        # The number of multirun experiments should have increased by 1
        new_experiment_count, _ = count_experiments(multirun_dir)
        assert new_experiment_count == 1

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
        shutil.rmtree(outdir)
        shutil.rmtree(multirun_dir)

    def test_no_obs(self):
        """Test that a simulation with no observations creates an empty csv"""
        outdir, multirun_dir, output_csv, errors_txt = prepare_output_dirs(
            "test_no_obs"
        )

        num_sim_pairs = 1
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
            multirun_root=multirun_dir,
        )

        # The number of multirun experiments should have increased by 1
        new_experiment_count, _ = count_experiments(multirun_dir)
        assert new_experiment_count == 1

        assert output_csv.exists()
        assert errors_txt.exists()

        with open(output_csv, "r") as f:
            output = f.read()

        assert output.strip() == ""

        with open(errors_txt, "r") as f:
            errors = f.read()

        assert errors == ""

        # Cleanup
        shutil.rmtree(outdir)
        shutil.rmtree(multirun_dir)

    def test_auto_cleanup(self):
        """Test that the rm_multirun_root option will clean up multirun directories."""

        outdir, multirun_dir, output_csv, errors_txt = prepare_output_dirs(
            "test_auto_cleanup"
        )

        num_sim_pairs = 1
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
            multirun_root=multirun_dir,
        )

        # The number of multirun experiments should not have changed
        new_experiment_count, _ = count_experiments(multirun_dir)
        assert new_experiment_count == 0

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
        shutil.rmtree(outdir)

    def test_auto_cleanup_default(self):
        """Test that the rm_multirun_root option will clean up default multirun directory."""

        outdir, _, output_csv, errors_txt = prepare_output_dirs(
            "test_auto_cleanup_default"
        )

        multirun_dir = "multirun"

        num_sim_pairs = 1
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
        new_experiment_count, _ = count_experiments(multirun_dir)
        assert new_experiment_count == 0

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
        shutil.rmtree(outdir)

    def test_prediction_error(self):
        """Test launching experiments with orbit-misestimation"""

        outdir, multirun_dir, output_csv, errors_txt = prepare_output_dirs(
            "test_prediction_error"
        )

        num_sim_pairs = 1
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
            multirun_root=multirun_dir,
        )

        # The number of multirun experiments should have increased by 1
        new_experiment_count, _ = count_experiments(multirun_dir)
        assert new_experiment_count == 1

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
        shutil.rmtree(outdir)
        shutil.rmtree(multirun_dir)

    def test_auto_cleanup_with_error(self):
        """Test that the rm_multirun_root option will clean up multirun directories
        even if an error.txt file is produced."""

        outdir, multirun_dir, output_csv, errors_txt = prepare_output_dirs(
            "test_auto_cleanup_with_error"
        )

        num_sim_pairs = 1
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
            multirun_root=multirun_dir,
        )

        # The number of multirun experiments should not have changed
        new_experiment_count, _ = count_experiments("multirun")
        assert new_experiment_count == 0

        assert output_csv.exists()
        assert errors_txt.exists()

        with open(errors_txt, "r") as f:
            errors = f.read()

        assert errors != ""

        # Cleanup
        shutil.rmtree(outdir)

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
            outdir, multirun_dir, output_csv, errors_txt = prepare_output_dirs(
                "test_cleanup_without_hydra_dir"
            )

            num_sim_pairs = 1
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
                multirun_root=multirun_dir,
            )

            # The number of multirun experiments should have increased
            new_experiment_count, _ = count_experiments(multirun_dir)
            assert new_experiment_count == 1

            assert output_csv.exists()
            assert errors_txt.exists()

            with open(errors_txt, "r") as f:
                errors = f.read()

            assert errors != ""

            # Cleanup
            shutil.rmtree(outdir)

    def test_auto_cleanup_partial(self):
        """Check that directories are not deleted if they contain a non-hydra file."""
        outdir, multirun_dir, output_csv, errors_txt = prepare_output_dirs(
            "test_auto_cleanup_partial"
        )

        num_sim_pairs = 1
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
                multirun_root=multirun_dir,
                rm_multirun_root=True,
                random_seed=0,
            )

            # The number of multirun experiments should have increased
            new_experiment_count, _ = count_experiments(multirun_dir)
            assert new_experiment_count == 1

            assert output_csv.exists()
            assert errors_txt.exists()

            with open(errors_txt, "r") as f:
                errors = f.read()

            assert errors == ""

            # Cleanup
            shutil.rmtree(outdir)
            shutil.rmtree(multirun_dir)

    def test_yaml_copy_exception(self):
        """Check that code still runs as expected if multirun.yaml is not found at the end."""
        outdir, multirun_dir, output_csv, errors_txt = prepare_output_dirs(
            "test_yaml_copy_exception"
        )

        num_sim_pairs = 1
        mtype = "impulse"
        sim_duration_days = 0.5
        start_mjd = 60196.5

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
                multirun_root=multirun_dir,
                rm_multirun_root=True,
                random_seed=0,
            )

            # The number of multirun experiments should not have increased
            new_experiment_count, _ = count_experiments(multirun_dir)
            assert new_experiment_count == 0

            # The copied multirun.yaml should not exist
            assert not (Path(outdir) / "multirun.yaml").exists()

            assert output_csv.exists()
            assert errors_txt.exists()

            with open(errors_txt, "r") as f:
                errors = f.read()

            assert errors == ""

            # Cleanup
            shutil.rmtree(outdir)


class TestSubmitit:
    """These tests will check the behavior of the hydra submitit launcher"""

    def test_submitit(self):
        """Test that the submitit overrides work (NOTE: This will not actually run submitit)"""
        outdir, multirun_dir, output_csv, errors_txt = prepare_output_dirs(
            "test_submitit"
        )

        num_sim_pairs = 1
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
            multirun_root=multirun_dir,
        )

        # The number of multirun experiments should not have changed
        new_experiment_count, _ = count_experiments(multirun_dir)
        assert new_experiment_count == 0

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
        shutil.rmtree(outdir)

    def test_invalid(self):
        """Test that a submitit JSON with invalid format will be rejected"""
        outdir, multirun_dir, output_csv, errors_txt = prepare_output_dirs(
            "test_submitit_invalid"
        )

        num_sim_pairs = 1
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
                multirun_root=multirun_dir,
            )
        except MadlibException:
            failed = True

        assert failed

        # The number of multirun experiments should not have changed
        new_experiment_count, _ = count_experiments(multirun_dir)
        assert new_experiment_count == 0

        assert not output_csv.exists()
        assert not errors_txt.exists()

    def test_auto_cleanup(self):
        """Test that the rm_multirun_root option will clean up multirun directories."""

        outdir, multirun_dir, output_csv, errors_txt = prepare_output_dirs(
            "test_submitit_auto_cleanup"
        )

        num_sim_pairs = 1
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
            multirun_root=multirun_dir,
        )

        # The number of multirun experiments should not have changed
        new_experiment_count, _ = count_experiments(multirun_dir)
        assert new_experiment_count == 0

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
        shutil.rmtree(outdir)
