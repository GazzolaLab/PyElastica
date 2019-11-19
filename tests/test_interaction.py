__doc__ = "Interaction module tests"

import numpy as np
import pytest
from numpy.testing import assert_allclose
from elastica.utils import Tolerance, MaxDimension
from elastica.interaction import InteractionPlane
from test_rod import TestRod


class BaseRodClass(TestRod):
    def __init__(self, n_elem):
        """
        This class initialize a straight rod,
        which is for testing interaction functions.
        :param n_elem:
        """
        base_length = 1.0
        direction = np.array([0.0, 0.0, 1.0])
        start = np.array([0.0, 0.0, 0.0])

        end = start + direction * base_length
        self.position = np.zeros((MaxDimension.value(), n_elem + 1))
        for i in range(0, MaxDimension.value()):
            self.position[i, ...] = np.linspace(start[i], end[i], num=n_elem + 1)

        self.directors = np.repeat(np.identity(3)[:, :, np.newaxis], n_elem, axis=2)
        self.radius = np.repeat(np.array([0.25]), n_elem, axis=0)
        self.tangents = np.repeat(direction[:, np.newaxis], n_elem, axis=1)
        self.velocity = np.zeros((MaxDimension.value(), n_elem + 1))
        self.omega = np.zeros((MaxDimension.value(), n_elem))
        self.external_forces = np.zeros((MaxDimension.value(), n_elem + 1))
        self.external_torques = np.zeros((MaxDimension.value(), n_elem))
        self.internal_forces = np.zeros((MaxDimension.value(), n_elem + 1))


class TestInteractionPlane:
    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    def test_interaction_plane_without_k_and_nu(self, n_elem):
        """
        This function tests wall response  on the rod. Here
        wall stiffness coefficient and damping coefficient set
        to zero to check only sum of all forces on the rod.
        :param n_elem:
        :return:
        """

        k_w = 0.0
        nu_w = 0.0
        origin_plane = np.array([0.0, -0.25, 0.0])
        normal_plane = np.array([0.0, 1.0, 0.0])
        interaction_plane = InteractionPlane(k_w, nu_w, origin_plane, normal_plane)
        rod = BaseRodClass(n_elem)
        external_forces = np.zeros((MaxDimension.value(), n_elem + 1))
        external_forces[..., 1:-1] = np.repeat(
            np.array([0.0, -10.0, 0.0]).reshape(3, 1), n_elem - 1, axis=1
        )
        external_forces[..., 0] = 0.5 * external_forces[..., 1]
        external_forces[..., -1] = 0.5 * external_forces[..., -2]

        rod.external_forces = external_forces.copy()

        interaction_plane.apply_normal_force(rod)

        correct_external_forces = np.zeros((3, n_elem + 1))
        assert_allclose(
            correct_external_forces, rod.external_forces, atol=Tolerance.atol()
        )

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("k_w", [0.1, 0.5, 1.0, 2, 10])
    def test_interaction_plane_with_k_and_without_nu(self, n_elem, k_w):
        """
        This function tests wall response  on the rod. Here
        wall stiffness coefficient changed parametrically
        and damping coefficient set to zero .
        :param n_elem:
        :param k_w:
        :return:
        """

        nu_w = 0.0
        origin_plane = np.array([0.0, -0.24, 0.0])
        normal_plane = np.array([0.0, 1.0, 0.0])
        interaction_plane = InteractionPlane(k_w, nu_w, origin_plane, normal_plane)
        rod = BaseRodClass(n_elem)

        correct_external_forces = k_w * np.repeat(
            np.array([0.0, 0.01, 0.0]).reshape(3, 1), n_elem + 1, axis=1
        )
        correct_external_forces[..., 0] *= 0.5
        correct_external_forces[..., -1] *= 0.5

        interaction_plane.apply_normal_force(rod)

        assert_allclose(
            correct_external_forces, rod.external_forces, atol=Tolerance.atol()
        )

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("k_w", [0.1, 0.5, 1.0, 2.0, 10.0])
    @pytest.mark.parametrize("nu_w", [0.5, 1.0, 5.0, 7.0, 12.0])
    def test_interaction_plane_with_k_and_nu(self, n_elem, k_w, nu_w):
        """
        This function tests wall response on the rod. Here
        wall stiffness coefficient and damping coefficient are
        changed parametrically.
        :param n_elem:
        :param k_w:
        :param nu_w:
        :return:
        """

        origin_plane = np.array([0.0, -0.24, 0.0])
        normal_plane = np.array([0.0, 1.0, 0.0])
        interaction_plane = InteractionPlane(k_w, nu_w, origin_plane, normal_plane)
        rod = BaseRodClass(n_elem)
        correct_forces = np.zeros((MaxDimension.value(), n_elem + 1))
        correct_forces[..., 1:-1] = np.repeat(
            np.array([0.0, -10.0, 0.0]).reshape(3, 1), n_elem - 1, axis=1
        )
        correct_forces[..., 0] = 0.5 * correct_forces[..., 1]
        correct_forces[..., -1] = 0.5 * correct_forces[..., -2]

        rod.velocity[..., :] += np.array([0.0, -1.0, 0.0]).reshape(3, 1)
        correct_external_forces = np.repeat(
            (
                k_w * np.array([0.0, 0.01, 0.0]) + nu_w * np.array([0.0, 1.0, 0.0])
            ).reshape(3, 1),
            n_elem + 1,
            axis=1,
        )
        correct_external_forces[..., 0] *= 0.5
        correct_external_forces[..., -1] *= 0.5

        interaction_plane.apply_normal_force(rod)

        assert_allclose(
            correct_external_forces, rod.external_forces, atol=Tolerance.atol()
        )
