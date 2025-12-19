__doc__ = """ Interaction module import test"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from elastica.utils import Tolerance, MaxDimension


from elastica.interaction import (
    SlenderBodyTheory,
)
from elastica.contact_utils import (
    _node_to_element_mass_or_force,
)

from tests.test_rod.mock_rod import MockTestRod


class BaseRodClass(MockTestRod):
    def __init__(self, n_elem):
        """
        This class initialize a straight rod,
        which is for testing interaction functions.
        Parameters
        ----------
        n_elem
        """
        base_length = 1.0
        direction = np.array([0.0, 0.0, 1.0])
        start = np.array([0.0, 0.0, 0.0])

        end = start + direction * base_length
        self.n_elem = n_elem
        self.position_collection = np.zeros((MaxDimension.value(), n_elem + 1))
        for i in range(0, MaxDimension.value()):
            self.position_collection[i, ...] = np.linspace(
                start[i], end[i], num=n_elem + 1
            )

        self.director_collection = np.repeat(
            np.identity(3)[:, :, np.newaxis], n_elem, axis=2
        )
        self.radius = np.repeat(np.array([0.25]), n_elem, axis=0)
        self.tangents = np.repeat(direction[:, np.newaxis], n_elem, axis=1)
        self.velocity_collection = np.zeros((MaxDimension.value(), n_elem + 1))
        self.omega_collection = np.zeros((MaxDimension.value(), n_elem))
        self.external_forces = np.zeros((MaxDimension.value(), n_elem + 1))
        self.external_torques = np.zeros((MaxDimension.value(), n_elem))
        self.internal_forces = np.zeros((MaxDimension.value(), n_elem + 1))
        self.internal_torques = np.zeros((MaxDimension.value(), n_elem))
        self.lengths = np.ones(n_elem) * base_length / n_elem
        self.mass = np.ones(n_elem + 1)

    def _compute_internal_forces(self):
        return np.zeros((MaxDimension.value(), self.n_elem + 1))

    def _compute_internal_torques(self):
        return np.zeros((MaxDimension.value(), self.n_elem))


# Slender Body Theory Unit Tests
from elastica.interaction import (
    sum_over_elements,
)


# These functions are used in the case if Numba is available
class TestAuxiliaryFunctionsForSlenderBodyTheory:
    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    def test_sum_over_elements(self, n_elem, rng):
        """
        This function test sum over elements function with
        respect to default python function .sum(). We write
        this function because with numba we can get the sum
        faster.
        Parameters
        ----------
        n_elem

        Returns
        -------

        """

        input_variable = rng.random(n_elem)
        correct_output = input_variable.sum()
        output = sum_over_elements(input_variable)
        assert_allclose(correct_output, output, atol=Tolerance.atol())


class TestSlenderBody:
    def initializer(self, n_elem, dynamic_viscosity):
        rod = BaseRodClass(n_elem)
        slender_body_theory = SlenderBodyTheory(dynamic_viscosity)

        return rod, slender_body_theory

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("dynamic_viscosity", [2, 3, 5, 10, 20])
    def test_slender_body_theory_only_factor(self, n_elem, dynamic_viscosity):
        """
        This function test factors in front of the slender body theory equation

        factor = -4*pi*mu/log(L/r) * dL

        Parameters
        ----------
        n_elem
        dynamic_viscosity

        Returns
        -------

        """

        [rod, slender_body_theory] = self.initializer(n_elem, dynamic_viscosity)
        length = rod.lengths.sum()
        radius = rod.radius[0]
        factor = (
            -4 * np.pi * dynamic_viscosity / np.log(length / radius) * rod.lengths[0]
        )
        correct_forces = np.ones((3, n_elem + 1)) * factor
        correct_forces[..., 0] *= 0.5
        correct_forces[..., -1] *= 0.5

        rod.tangents = np.zeros((3, n_elem))
        rod.velocity_collection = np.ones((3, n_elem + 1))
        slender_body_theory.apply_forces(rod)

        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    def test_slender_body_matrix_product_only_xz(self, n_elem):
        """
        This function test x and z component of the hydrodynamic force. Here tangents are
        set such that y component of 1/2*t`t matrix removed, and only x and z component
        is left. Also non-diagonal components of matrix are zero, since we choose tangent
        vector to remove non-diagonal components
        Parameters
        ----------
        n_elem

        Returns
        -------

        """
        dynamic_viscosity = 0.1
        [rod, slender_body_theory] = self.initializer(n_elem, dynamic_viscosity)
        factor = -4 * np.pi * dynamic_viscosity / np.log(1 / 0.25) * rod.lengths[0]
        correct_forces = np.zeros((3, n_elem + 1))
        correct_forces[0, :] = factor
        correct_forces[2, :] = factor
        correct_forces[..., 0] *= 0.5
        correct_forces[..., -1] *= 0.5

        rod.tangents = np.repeat(
            np.array([0.0, np.sqrt(2.0), 0.0])[:, np.newaxis], n_elem, axis=1
        )
        rod.velocity_collection = np.ones((3, n_elem + 1))
        slender_body_theory.apply_forces(rod)

        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    def test_slender_body_matrix_product_only_yz(self, n_elem):
        """
        This function test y and z component of the hydrodynamic force. Here tangents are
        set such that x component of 1/2*t`t matrix removed, and only y and z component
        is left. Also non-diagonal components of matrix are zero, since we choose tangent
        vector to remove non-diagonal components
        Parameters
        ----------
        n_elem

        Returns
        -------

        """
        dynamic_viscosity = 0.1
        [rod, slender_body_theory] = self.initializer(n_elem, dynamic_viscosity)
        factor = -4 * np.pi * dynamic_viscosity / np.log(1 / 0.25) * rod.lengths[0]
        correct_forces = np.zeros((3, n_elem + 1))
        correct_forces[1, :] = factor
        correct_forces[2, :] = factor
        correct_forces[..., 0] *= 0.5
        correct_forces[..., -1] *= 0.5

        rod.tangents = np.repeat(
            np.array([np.sqrt(2.0), 0.0, 0.0])[:, np.newaxis], n_elem, axis=1
        )
        rod.velocity_collection = np.ones((3, n_elem + 1))
        slender_body_theory.apply_forces(rod)

        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    def test_slender_body_matrix_product_only_xy(self, n_elem):
        """
        This function test x and y component of the hydrodynamic force. Here tangents are
        set such that z component of 1/2*t`t matrix removed, and only x and y component
        is left. Also non-diagonal components of matrix are zero, since we choose tangent
        vector to remove non-diagonal components
        Parameters
        ----------
        n_elem

        Returns
        -------

        """
        dynamic_viscosity = 0.1
        [rod, slender_body_theory] = self.initializer(n_elem, dynamic_viscosity)
        factor = -4 * np.pi * dynamic_viscosity / np.log(1 / 0.25) * rod.lengths[0]
        correct_forces = np.zeros((3, n_elem + 1))
        correct_forces[0, :] = factor
        correct_forces[1, :] = factor
        correct_forces[..., 0] *= 0.5
        correct_forces[..., -1] *= 0.5

        rod.tangents = np.repeat(
            np.array([0.0, 0.0, np.sqrt(2.0)])[:, np.newaxis], n_elem, axis=1
        )
        rod.velocity_collection = np.ones((3, n_elem + 1))
        slender_body_theory.apply_forces(rod)

        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())
