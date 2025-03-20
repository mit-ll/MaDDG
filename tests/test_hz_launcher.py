# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

"""
Test file: test_hz_launcher.py
Description: This file contains unit tests for the `hz_launcher` module
"""

import pathlib
import sys
import numpy as np
import pytest

# add parent directory of __file__ to sys.path, if isn't already included
if str(pathlib.Path(__file__).parents[1]) not in sys.path:
    sys.path.append(str(pathlib.Path(__file__).parents[1]))

from scripts.hz_launcher import simulator_task, parseArgs
from madlib._sensor_collection import SensorCollection

sensor_yaml_11 = "configs/sample_sensor_network.yaml"
sensor_params_11 = SensorCollection.paramsFromYAML(sensor_yaml_11)

sensor_yaml_blind = "tests/inputs/blind_sensor.yaml"
sensor_params_blind = SensorCollection.paramsFromYAML(sensor_yaml_blind)


class TestSimulator:
    """Test behavior of simulator_task function"""

    def test_single_non_maneuver(self):
        """Test the outputs of a simple satellite simulation with no maneuvers."""
        seq_id = 0
        maneuver_type = 0
        sim_duration_days = 2.0

        results = simulator_task(
            seq_id=seq_id,
            sensor_params=sensor_params_11,
            maneuver_type=maneuver_type,
            start_mjd=60196.5,
            sim_duration_days=sim_duration_days,
            random_seed=0,
        )

        assert results is not None

        max_mjd = max(results["MJD"])
        min_mjd = min(results["MJD"])

        assert isinstance(max_mjd, float)
        assert isinstance(min_mjd, float)

        # The following assertions must be true for the given inputs
        assert all(results["Maneuver"] == maneuver_type)
        assert all(results["Sequence"] == seq_id)
        assert max_mjd - min_mjd < sim_duration_days
        assert all(results["Maneuver_Start_MJD"].isna())
        assert all(results["Maneuver_End_MJD"].isna())
        assert all(results["Maneuver_DV_Radial_KmS"].isna())
        assert all(results["Maneuver_DV_InTrack_KmS"].isna())
        assert all(results["Maneuver_DV_CrossTrack_KmS"].isna())

    def test_single_maneuver(self):
        """Test the outputs of a simple satellite simulation with an impulsive maneuver."""
        seq_id = 0
        maneuver_type = 1
        sim_duration_days = 2.0

        results = simulator_task(
            seq_id=seq_id,
            sensor_params=sensor_params_11,
            maneuver_type=maneuver_type,
            start_mjd=60196.5,
            sim_duration_days=sim_duration_days,
            random_seed=0,
        )

        assert results is not None

        max_mjd = max(results["MJD"])
        min_mjd = min(results["MJD"])

        assert isinstance(max_mjd, float)
        assert isinstance(min_mjd, float)

        activated_sensors = list(set(results["SensorID"]))

        # The following assertions must be true for the given inputs
        assert all(results["Maneuver"] == maneuver_type)
        assert all(results["Sequence"] == seq_id)
        assert max_mjd - min_mjd < sim_duration_days
        np.testing.assert_allclose(
            results["Maneuver_Start_MJD"], 60197.930379, atol=1e-6
        )
        assert all(results["Maneuver_End_MJD"].isna())
        np.testing.assert_allclose(results["Maneuver_DV_Radial_KmS"], 0, atol=1e-6)
        assert all(np.abs(results["Maneuver_DV_InTrack_KmS"]) > 1e-6)
        assert all(np.abs(results["Maneuver_DV_CrossTrack_KmS"]) > 1e-6)

        # The following assertions must be true for the given seed and default sensors
        assert len(results) == 34
        assert len(activated_sensors) == 3

    def test_start_now(self):
        """Test the no-maneuver case when starting epoch is current time."""
        seq_id = 0
        maneuver_type = 0
        sim_duration_days = 1.0

        results = simulator_task(
            seq_id=seq_id,
            sensor_params=sensor_params_11,
            maneuver_type=maneuver_type,
            start_mjd=None,
            sim_duration_days=sim_duration_days,
            random_seed=0,
        )

        assert results is not None

        max_mjd = max(results["MJD"])
        min_mjd = min(results["MJD"])

        assert isinstance(max_mjd, float)
        assert isinstance(min_mjd, float)

        activated_sensors = list(set(results["SensorID"]))

        # The following assertions must be true for the given inputs
        assert all(results["Maneuver"] == maneuver_type)
        assert all(results["Sequence"] == seq_id)
        assert max_mjd - min_mjd < sim_duration_days

        # The following assertions must be true for the given seed and default sensors
        assert len(results) > 0
        assert len(activated_sensors) > 0

    def test_fixed_continuous(self):
        """Test the outputs of a continuous thrust with a fixed vector."""
        seq_id = 0
        maneuver_type = 2
        cont_thrust_duration_days = 0.25
        cont_thrust_model = 0
        cont_thrust_mag = 1e-9
        sim_duration_days = 2.0

        results = simulator_task(
            seq_id=seq_id,
            sensor_params=sensor_params_11,
            maneuver_type=maneuver_type,
            cont_thrust_duration_days=cont_thrust_duration_days,
            cont_thrust_mag=cont_thrust_mag,
            cont_thrust_model=cont_thrust_model,
            start_mjd=60196.5,
            sim_duration_days=sim_duration_days,
            random_seed=0,
        )

        assert results is not None

        max_mjd = max(results["MJD"])
        min_mjd = min(results["MJD"])

        assert isinstance(max_mjd, float)
        assert isinstance(min_mjd, float)

        activated_sensors = list(set(results["SensorID"]))

        final_ra = results["RA Arcsec"].iloc[-1]
        final_dec = results["DEC Arcsec"].iloc[-1]

        # The following assertions must be true for the given inputs
        assert all(results["Maneuver"] == maneuver_type)
        assert all(results["Sequence"] == seq_id)
        assert max_mjd - min_mjd < sim_duration_days
        np.testing.assert_allclose(results["Maneuver_Start_MJD"], 60196.5, atol=1e-6)
        np.testing.assert_allclose(results["Maneuver_End_MJD"], 60196.75, atol=1e-6)
        np.testing.assert_allclose(results["Maneuver_DV_Radial_KmS"], 0, atol=1e-6)
        np.testing.assert_allclose(
            results["Maneuver_DV_InTrack_KmS"], cont_thrust_mag, atol=1e-6
        )
        np.testing.assert_allclose(results["Maneuver_DV_CrossTrack_KmS"], 0, atol=1e-6)

    def test_random_continuous(self):
        """Test the outputs of a continuous thrust with a random vector."""
        seq_id = 0
        maneuver_type = 2
        cont_thrust_mag = 1e-9
        cont_thrust_duration_days = 0.5
        cont_thrust_model = 1
        sim_duration_days = 2.0

        results = simulator_task(
            seq_id=seq_id,
            sensor_params=sensor_params_11,
            maneuver_type=maneuver_type,
            cont_thrust_mag=cont_thrust_mag,
            cont_thrust_duration_days=cont_thrust_duration_days,
            cont_thrust_model=cont_thrust_model,
            start_mjd=60196.5,
            sim_duration_days=sim_duration_days,
            random_seed=0,
        )

        assert results is not None

        max_mjd = max(results["MJD"])
        min_mjd = min(results["MJD"])

        assert isinstance(max_mjd, float)
        assert isinstance(min_mjd, float)

        # The following assertions must be true for the given inputs
        assert all(results["Maneuver"] == maneuver_type)
        assert all(results["Sequence"] == seq_id)
        assert max_mjd - min_mjd < sim_duration_days
        np.testing.assert_allclose(results["Maneuver_Start_MJD"], 60196.5, atol=1e-6)
        # By default, continuous mnvr duration should be equal to simulation duration
        np.testing.assert_allclose(results["Maneuver_End_MJD"], 60197.0, atol=1e-6)
        np.testing.assert_allclose(
            results["Maneuver_DV_Radial_KmS"], 2.605e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            results["Maneuver_DV_InTrack_KmS"], 5.454e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            results["Maneuver_DV_CrossTrack_KmS"], -7.967e-10, atol=1e-12
        )

    def test_invalid_continuous(self):
        """Test the outputs of a continuous thrust with an invalid thrust mode."""
        seq_id = 0
        maneuver_type = 2
        cont_thrust_model = 3
        sim_duration_days = 0.5

        failed = False
        try:
            results = simulator_task(
                seq_id=seq_id,
                sensor_params=sensor_params_11,
                maneuver_type=maneuver_type,
                cont_thrust_model=cont_thrust_model,
                start_mjd=60196.5,
                sim_duration_days=sim_duration_days,
                random_seed=0,
            )
        except ValueError:
            failed = True

        assert failed

    def test_invalid_thrust_mode(self):
        """Test that an invalid thrust mode returns None"""
        seq_id = 0
        maneuver_type = 3
        cont_thrust_model = 1
        sim_duration_days = 0.5

        results = simulator_task(
            seq_id=seq_id,
            sensor_params=sensor_params_11,
            maneuver_type=maneuver_type,
            cont_thrust_model=cont_thrust_model,
            start_mjd=60196.5,
            sim_duration_days=sim_duration_days,
            random_seed=0,
        )

        assert results == None

    def test_no_obs(self):
        """Test that a simulation with no observations returns None"""
        seq_id = 0
        maneuver_type = 0
        sim_duration_days = 0.5

        results = simulator_task(
            seq_id=seq_id,
            sensor_params=sensor_params_blind,
            maneuver_type=maneuver_type,
            start_mjd=60196.5,
            sim_duration_days=sim_duration_days,
            random_seed=0,
        )

        assert results is None


class TestArgs:
    """Test behavior of argument parsing"""

    def test_defaults(self):
        """Test default argument values"""
        inputs = ["0", "sensor_yaml_path"]

        parser = parseArgs()
        args = parser.parse_args(inputs)

        # Count number of input arguments
        # (This is to make sure you update the tests when you add an arg)
        assert len(args.__dict__) == 19

        assert isinstance(args.num_pairs, int)
        assert args.num_pairs == 0
        assert isinstance(args.sensor_yaml, str)
        assert args.sensor_yaml == "sensor_yaml_path"
        assert isinstance(args.start_mjd, float)
        np.testing.assert_almost_equal(args.start_mjd, -1, decimal=7)
        assert isinstance(args.sim_duration_days, float)
        np.testing.assert_almost_equal(args.sim_duration_days, 3.0, decimal=7)
        assert isinstance(args.mtype, str)
        assert args.mtype == "impulse"
        assert args.cont_thrust_duration_days is None
        assert isinstance(args.cont_thrust_model, int)
        assert args.cont_thrust_model == 0
        assert isinstance(args.cont_thrust_mag, float)
        np.testing.assert_almost_equal(args.cont_thrust_mag, 1e-7, decimal=9)
        assert isinstance(args.outdir, str)
        assert args.outdir == "outputs"
        assert isinstance(args.submitit, str)
        assert args.submitit == ""
        assert isinstance(args.multirun_root, str)
        assert args.multirun_root == ""
        assert isinstance(args.pred_err, float)
        np.testing.assert_almost_equal(args.pred_err, 0, decimal=7)
        assert isinstance(args.rm_multirun_root, bool)
        assert not args.rm_multirun_root
        assert isinstance(args.overwrite, bool)
        assert not args.overwrite

        dv_mean = args.dv_ric_mean_kms
        dv_std = args.dv_ric_std_kms

        assert isinstance(dv_mean, list)
        assert isinstance(dv_std, list)
        assert len(dv_mean) == 3
        assert len(dv_std) == 3
        np.testing.assert_almost_equal(dv_mean[0], 0, decimal=7)
        np.testing.assert_almost_equal(dv_mean[1], 0, decimal=7)
        np.testing.assert_almost_equal(dv_mean[2], 0, decimal=7)
        np.testing.assert_almost_equal(dv_std[0], 0, decimal=7)
        np.testing.assert_almost_equal(dv_std[1], 0.1, decimal=7)
        np.testing.assert_almost_equal(dv_std[2], 1, decimal=7)

    def test_inputs(self):
        """Test that input values work as expected"""
        inputs = [
            "10",
            "sensor_yaml_path",
            "--start_mjd",
            "1000",
            "--sim_duration_days",
            "5.0",
            "--mtype",
            "continuous",
            "--dv_ric_mean_kms",
            "2",
            "2",
            "2",
            "--dv_ric_std_kms",
            "5",
            "5",
            "5",
            "--sensor_dra",
            "1",
            "--sensor_ddec",
            "1",
            "--cont_thrust_duration_days",
            "2.5",
            "--cont_thrust_model",
            "1",
            "--cont_thrust_mag",
            "1e-3",
            "--outdir",
            "test1",
            "--submitit",
            "test2",
            "--multirun_root",
            "test3",
            "--pred_err",
            "1e-3",
            "--rm_multirun_root",
            "--overwrite",
            "--sims_per_task",
            "10",
        ]

        parser = parseArgs()
        args = parser.parse_args(inputs)

        # Count number of input arguments
        # (This is to make sure you update the tests when you add an arg)
        assert len(args.__dict__) == 19

        assert isinstance(args.num_pairs, int)
        assert isinstance(args.sensor_yaml, str)
        assert args.sensor_yaml == "sensor_yaml_path"
        assert args.num_pairs == 10
        assert isinstance(args.start_mjd, float)
        np.testing.assert_almost_equal(args.start_mjd, 1000, decimal=7)
        assert isinstance(args.sim_duration_days, float)
        np.testing.assert_almost_equal(args.sim_duration_days, 5.0, decimal=7)
        assert isinstance(args.mtype, str)
        assert args.mtype == "continuous"
        assert isinstance(args.sensor_dra, float)
        np.testing.assert_almost_equal(args.sensor_dra, 1.0, decimal=7)
        assert isinstance(args.sensor_ddec, float)
        np.testing.assert_almost_equal(args.sensor_ddec, 1.0, decimal=7)
        assert isinstance(args.cont_thrust_duration_days, float)
        np.testing.assert_almost_equal(args.cont_thrust_duration_days, 2.5)
        assert isinstance(args.cont_thrust_model, int)
        assert args.cont_thrust_model == 1
        assert isinstance(args.cont_thrust_mag, float)
        np.testing.assert_almost_equal(args.cont_thrust_mag, 1e-3, decimal=9)
        assert isinstance(args.outdir, str)
        assert args.outdir == "test1"
        assert isinstance(args.submitit, str)
        assert args.submitit == "test2"
        assert isinstance(args.multirun_root, str)
        assert args.multirun_root == "test3"
        assert isinstance(args.pred_err, float)
        np.testing.assert_almost_equal(args.pred_err, 1e-3, decimal=9)
        assert isinstance(args.rm_multirun_root, bool)
        assert args.rm_multirun_root
        assert isinstance(args.overwrite, bool)
        assert args.overwrite
        assert isinstance(args.sims_per_task, int)
        assert args.sims_per_task == 10

        dv_mean = args.dv_ric_mean_kms
        dv_std = args.dv_ric_std_kms

        assert isinstance(dv_mean, list)
        assert isinstance(dv_std, list)
        assert len(dv_mean) == 3
        assert len(dv_std) == 3
        np.testing.assert_almost_equal(dv_mean[0], 2, decimal=7)
        np.testing.assert_almost_equal(dv_mean[1], 2, decimal=7)
        np.testing.assert_almost_equal(dv_mean[2], 2, decimal=7)
        np.testing.assert_almost_equal(dv_std[0], 5, decimal=7)
        np.testing.assert_almost_equal(dv_std[1], 5, decimal=7)
        np.testing.assert_almost_equal(dv_std[2], 5, decimal=7)

    class TestRicLengths:
        """Make sure the RIC inputs can only have length 3"""

        def test_mean_short(self):
            """dv_ric_mean must have 3 inputs"""
            inputs = [
                "10",
                "sensor_yaml_path",
                "--dv_ric_mean_kms",
                "2",
                "2",
            ]

            parser = parseArgs()

            with pytest.raises(SystemExit):
                parser.parse_args(inputs)

        def test_mean_long(self):
            """dv_ric_mean must have 3 inputs"""
            inputs = [
                "10",
                "sensor_yaml_path",
                "--dv_ric_mean_kms",
                "2",
                "2",
                "2",
                "2",
            ]

            parser = parseArgs()

            with pytest.raises(SystemExit):
                parser.parse_args(inputs)

        def test_std_short(self):
            """dv_ric_std must have 3 inputs"""
            inputs = [
                "10",
                "sensor_yaml_path",
                "--dv_std_mean_kms",
                "2",
                "2",
            ]

            parser = parseArgs()

            with pytest.raises(SystemExit):
                parser.parse_args(inputs)

        def test_std_long(self):
            """dv_ric_std must have 3 inputs"""
            inputs = [
                "10",
                "sensor_yaml_path",
                "--dv_std_mean_kms",
                "2",
                "2",
                "2",
                "2",
            ]

            parser = parseArgs()

            with pytest.raises(SystemExit):
                parser.parse_args(inputs)
