import pytest
import numpy as np

from madlib._utils import calc_separation_angle, MadlibException


def assert_vector_angle(V1, V2, angle_rad):
    sep_rad = calc_separation_angle(V1, V2)
    sep_deg = calc_separation_angle(V1, V2, in_deg=True)

    np.testing.assert_almost_equal(sep_rad, angle_rad, decimal=4)
    np.testing.assert_almost_equal(sep_deg, angle_rad * 180.0 / np.pi, decimal=4)


def assert_multiple_vector_angles(V1, V2, angle_rad_list):
    angle_rad_truth = np.array(angle_rad_list)

    sep_rad = calc_separation_angle(V1, V2)
    sep_deg = calc_separation_angle(V1, V2, in_deg=True)

    np.testing.assert_allclose(sep_rad, angle_rad_truth, atol=1e-4)
    np.testing.assert_allclose(sep_deg, angle_rad_truth * 180.0 / np.pi, atol=1e-4)


class TestSeparationAngle:
    """Test behavior of calc_separation_angle function"""

    def test_single_separations(self):
        """Test the separation between a few individual vectors in radians."""
        N1_1 = np.array(
            [
                [1, 0, 0],
            ]
        )
        N1_2 = np.array(
            [
                [0, 1, 0],
            ]
        )
        N1_3 = np.array(
            [
                [-1, 0, 0],
            ]
        )
        N1_4 = np.array(
            [
                [2, 0, 0],
            ]
        )
        N1_5 = np.array(
            [
                [1, 1, 0],
            ]
        )
        N1_6 = np.array(
            [
                [1, -1, 0],
            ]
        )
        N1_7 = np.array(
            [
                [-1, -1, 0],
            ]
        )
        N1_8 = np.array(
            [
                [1, 2, 3],
            ]
        )

        assert_vector_angle(N1_1, N1_1, 0.0)
        assert_vector_angle(N1_1, N1_2, np.pi / 2)
        assert_vector_angle(N1_2, N1_1, np.pi / 2)
        assert_vector_angle(N1_1, N1_3, np.pi)
        assert_vector_angle(N1_1, N1_4, 0.0)
        assert_vector_angle(N1_1, N1_5, np.pi / 4)
        assert_vector_angle(N1_1, N1_6, np.pi / 4)
        assert_vector_angle(N1_1, N1_7, 3 * np.pi / 4)
        assert_vector_angle(N1_1, N1_8, 1.30024656)

    def test_multiple_vectors(self):
        """Test the pairwise separation between arrays of vectors."""
        N2_1 = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
            ]
        )
        N2_2 = np.array(
            [
                [-1.2, -1.1, -1.0],
                [-0.9, -0.8, -0.7],
                [-0.6, -0.5, -0.4],
                [-0.3, -0.2, -0.1],
            ]
        )

        separations = [2.67990488, 2.87801221, 2.87801221, 2.67990488]

        assert_multiple_vector_angles(N2_1, N2_2, separations)

    def test_invalid_shapes(self):
        """Test invalid or incompatible vector shapes."""

        # Case: Neither array is two-dimensional
        B1 = np.array([0, 0, 0])
        B2 = np.array([0, 0, 0])

        with pytest.raises(MadlibException):
            calc_separation_angle(B1, B2)

        # Case: Only one of the arrays is two-dimensional
        B1 = np.array(
            [
                [0, 0, 0],
            ]
        )
        B2 = np.array([0, 0, 0])

        with pytest.raises(MadlibException):
            calc_separation_angle(B1, B2)

        with pytest.raises(MadlibException):
            calc_separation_angle(B2, B1)

        # Case: Neither array has three columns
        B1 = np.array(
            [
                [0, 0],
                [0, 0],
            ]
        )
        B2 = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        with pytest.raises(MadlibException):
            calc_separation_angle(B1, B2)

        # Case: Only one of the arrays has three columns
        B1 = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        B2 = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        with pytest.raises(MadlibException):
            calc_separation_angle(B1, B2)

        B1 = np.array(
            [
                [0, 0],
                [0, 0],
            ]
        )
        B2 = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        with pytest.raises(MadlibException):
            calc_separation_angle(B1, B2)

        # Case: The arrays have different numbers of rows
        B1 = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        B2 = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        with pytest.raises(MadlibException):
            calc_separation_angle(B1, B2)
