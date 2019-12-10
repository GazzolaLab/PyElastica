__doc__ = "Interaction module tests"

import numpy as np
import pytest
from numpy.testing import assert_allclose
from elastica.utils import Tolerance, MaxDimension
from elastica.interaction import (
    InteractionPlane,
    linear_interpolation_slip,
    AnistropicFrictionalPlane,
    nodes_to_elements,
)
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
        self.internal_torques = np.zeros((MaxDimension.value(), n_elem))


class TestInteractionPlane:
    def initializer(
        self, n_elem, shift=0.0, k_w=0.0, nu_w=0.0,
    ):
        rod = BaseRodClass(n_elem)
        interaction_plane = InteractionPlane(
            k_w,
            nu_w,
            origin_plane=np.array([0.0, -rod.radius[0] + shift, 0.0]),
            normal_plane=np.array([0.0, 1.0, 0.0]),
        )

        external_forces = np.repeat(
            np.array([0.0, -10.0 * np.random.random_sample(1), 0.0]).reshape(3, 1),
            n_elem + 1,
            axis=1,
        )
        external_forces[..., 0] *= 0.5
        external_forces[..., -1] *= 0.5
        rod.external_forces = external_forces.copy()

        return rod, interaction_plane, external_forces

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    def test_interaction_without_contact(self, n_elem):
        """
        This test case tests the forces on rod, when there is no
        contact between rod and the plane.
        :param n_elem:
        :return:
        """

        shift = -1.0 * np.random.random_sample(1)  # we move plane away from rod

        [rod, interaction_plane, external_forces] = self.initializer(n_elem, shift)

        interaction_plane.apply_normal_force(rod)
        correct_forces = external_forces  # since no contact
        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    def test_interaction_plane_without_k_and_nu(self, n_elem):
        """
        This function tests wall response  on the rod. Here
        wall stiffness coefficient and damping coefficient set
        to zero to check only sum of all forces on the rod.
        :param n_elem:
        :return:
        """
        [rod, interaction_plane, external_forces] = self.initializer(n_elem)

        interaction_plane.apply_normal_force(rod)

        correct_forces = np.zeros((3, n_elem + 1))
        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("k_w", [0.1, 0.5, 1.0, 2, 10])
    def test_interaction_plane_with_k_without_nu(self, n_elem, k_w):
        """
        Here wall stiffness coefficient changed parametrically
        and damping coefficient set to zero .
        :param n_elem:
        :return:
        """
        shift = np.random.random_sample(1)  # we move plane towards to rod
        [rod, interaction_plane, external_forces] = self.initializer(
            n_elem, shift=shift, k_w=k_w
        )
        correct_forces = k_w * np.repeat(
            np.array([0.0, shift, 0.0]).reshape(3, 1), n_elem + 1, axis=1
        )
        correct_forces[..., 0] *= 0.5
        correct_forces[..., -1] *= 0.5

        interaction_plane.apply_normal_force(rod)

        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("nu_w", [0.5, 1.0, 5.0, 7.0, 12.0])
    def test_interaction_plane_without_k_with_nu(self, n_elem, nu_w):
        """
        Here wall damping coefficient are changed parametrically and
        wall response functions tested.
        :param n_elem:
        :param nu_w:
        :return:
        """

        [rod, interaction_plane, external_forces] = self.initializer(n_elem, nu_w=nu_w)

        normal_velocity = np.random.random_sample(1)
        rod.velocity[..., :] += np.array([0.0, -normal_velocity, 0.0]).reshape(3, 1)

        correct_forces = np.repeat(
            (nu_w * np.array([0.0, normal_velocity, 0.0])).reshape(3, 1),
            n_elem + 1,
            axis=1,
        )

        correct_forces[..., 0] *= 0.5
        correct_forces[..., -1] *= 0.5

        interaction_plane.apply_normal_force(rod)

        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())


class TestAuxiliaryFunctions:
    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    def test_linear_interpolation_slip(self, n_elem):
        velocity_threshold = 1.0

        # if slip velocity larger than threshold
        velocity_slip = np.repeat(
            np.array([0.0, 0.0, 2.0]).reshape(3, 1), n_elem, axis=1
        )
        slip_function = linear_interpolation_slip(velocity_slip, velocity_threshold)
        correct_slip_function = np.repeat(np.array([[0]]), n_elem, axis=1)
        assert_allclose(correct_slip_function, slip_function, atol=Tolerance.atol())

        # if slip velocity smaller than threshold
        velocity_slip = np.repeat(
            np.array([0.0, 0.0, 0.0]).reshape(3, 1), n_elem, axis=1
        )
        slip_function = linear_interpolation_slip(velocity_slip, velocity_threshold)
        correct_slip_function = np.repeat(np.array([[1]]), n_elem, axis=1)
        assert_allclose(correct_slip_function, slip_function, atol=Tolerance.atol())

        # if slip velocity smaller than threshold but very close to threshold
        velocity_slip = np.repeat(
            np.array([0.0, 0.0, 1.0 - 1e-6]).reshape(3, 1), n_elem, axis=1
        )
        slip_function = linear_interpolation_slip(velocity_slip, velocity_threshold)
        correct_slip_function = np.repeat(np.array([[1.0]]), n_elem, axis=1)
        assert_allclose(correct_slip_function, slip_function, atol=Tolerance.atol())

        # if slip velocity larger than threshold but very close to threshold
        velocity_slip = np.repeat(
            np.array([0.0, 0.0, 1.0 + 1e-6]).reshape(3, 1), n_elem, axis=1
        )
        slip_function = linear_interpolation_slip(velocity_slip, velocity_threshold)
        correct_slip_function = np.repeat(np.array([[1.0 - 1e-6]]), n_elem, axis=1)
        assert_allclose(correct_slip_function, slip_function, atol=Tolerance.atol())

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    def test_nodes_to_elements(self, n_elem):
        input = np.repeat(np.array([1.0, 3.0, 4.0]).reshape(3, 1), n_elem + 1, axis=1)
        input[..., 0] *= 0.5
        input[..., -1] *= 0.5
        correct_output = np.repeat(
            np.array([1.0, 3.0, 4.0]).reshape(3, 1), n_elem, axis=1
        )
        output = nodes_to_elements(input)
        assert_allclose(correct_output, output, atol=Tolerance.atol())


class TestAnisotropicFriction:
    def initializer(
        self,
        n_elem,
        static_mu_array=np.array([0.0, 0.0, 0.0]),
        kinetic_mu_array=np.array([0.0, 0.0, 0.0]),
        force_mag_long=0.0,  # forces along the rod
        force_mag_side=0.0,  # side forces on the rod
    ):

        rod = BaseRodClass(n_elem)

        origin_plane = np.array([0.0, -rod.radius[0], 0.0])
        normal_plane = np.array([0.0, 1.0, 0.0])
        slip_velocity_tol = 1e-2
        friction_plane = AnistropicFrictionalPlane(
            0.0,
            0.0,
            origin_plane,
            normal_plane,
            slip_velocity_tol,
            static_mu_array,  # forward, backward, sideways
            kinetic_mu_array,  # forward, backward, sideways
        )
        fnormal = (10.0 - 5.0) * np.random.random_sample(
            1
        ) + 5.0  # generates random numbers [5.0,10)
        external_forces = np.array([force_mag_side, -fnormal, force_mag_long])

        external_forces_collection = np.repeat(
            external_forces.reshape(3, 1), n_elem + 1, axis=1
        )
        external_forces_collection[..., 0] *= 0.5
        external_forces_collection[..., -1] *= 0.5
        rod.external_forces = external_forces_collection.copy()

        # Velocities has to be set to zero
        assert_allclose(np.zeros((3, n_elem)), rod.omega, atol=Tolerance.atol())
        assert_allclose(np.zeros((3, n_elem + 1)), rod.velocity, atol=Tolerance.atol())

        # We have not changed torques also, they have to be zero as well
        assert_allclose(
            np.zeros((3, n_elem)), rod.external_torques, atol=Tolerance.atol()
        )
        assert_allclose(
            np.zeros((3, n_elem)), rod.internal_torques, atol=Tolerance.atol()
        )

        return rod, friction_plane, external_forces_collection

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("velocity", [-1.0, -3.0, 1.0, 5.0, 2.0])
    def test_axial_kinetic_friction(self, n_elem, velocity):
        """
        This function tests kinetic friction in forward and backward direction.
        All other friction coefficients set to zero.
        :param n_elem:
        :param velocity:
        :return:
        """

        [rod, friction_plane, external_forces_collection] = self.initializer(
            n_elem, kinetic_mu_array=np.array([1.0, 1.0, 0.0])
        )

        rod.velocity += np.array([0.0, 0.0, velocity]).reshape(3, 1)

        friction_plane.apply_friction(rod)

        direction_collection = np.repeat(
            np.array([0.0, 0.0, 1.0]).reshape(3, 1), n_elem + 1, axis=1
        )
        correct_forces = (
            -1.0
            * np.sign(velocity)
            * np.linalg.norm(external_forces_collection, axis=0)
            * direction_collection
        )
        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("force_mag", [-1.0, -3.0, 1.0, 5.0, 2.0])
    def test_axial_static_friction_total_force_smaller_than_static_friction_force(
        self, n_elem, force_mag
    ):
        """
        This test is for static friction when total forces applied
        on the rod is smaller than the static friction force.
        Fx < F_normal*mu_s
        :param n_elem:
        :param force_mag:
        :return:
        """
        [rod, frictionplane, external_forces_collection] = self.initializer(
            n_elem, static_mu_array=np.array([1.0, 1.0, 0.0]), force_mag_long=force_mag
        )

        frictionplane.apply_friction(rod)
        correct_forces = np.zeros((3, n_elem + 1))
        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("force_mag", [-20.0, -15.0, 15.0, 20.0])
    def test_axial_static_friction_total_force_larger_than_static_friction_force(
        self, n_elem, force_mag
    ):
        """
        This test is for static friction when total forces applied
        on the rod is larger than the static friction force.
        Fx > F_normal*mu_s
        :param n_elem:
        :param force_mag:
        :return:
        """
        [rod, frictionplane, external_forces_collection] = self.initializer(
            n_elem, static_mu_array=np.array([1.0, 1.0, 0.0]), force_mag_long=force_mag
        )

        frictionplane.apply_friction(rod)
        correct_forces = np.zeros((3, n_elem + 1))
        if np.sign(force_mag) < 0:
            correct_forces[2] = (
                external_forces_collection[2]
            ) - 1.0 * external_forces_collection[1]
        else:
            correct_forces[2] = (
                external_forces_collection[2]
            ) + 1.0 * external_forces_collection[1]

        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("velocity", [-1.0, -3.0, 1.0, 2.0, 5.0])
    @pytest.mark.parametrize("omega", [-5.0, -2.0, 0.0, 4.0, 6.0])
    def test_kinetic_rolling_friction(self, n_elem, velocity, omega):
        """
        This test is for testing kinetic rolling friction,
        for different translational and angular velocities,
        we compute the final external forces and torques on the rod
        using apply friction function and compare results with
        analytical solutions.
        :param n_elem:
        :param velocity:
        :param omega:
        :return:
        """
        [rod, frictionplane, external_forces_collection] = self.initializer(
            n_elem, kinetic_mu_array=np.array([0.0, 0.0, 1.0]),
        )

        rod.velocity += np.array([velocity, 0.0, 0.0]).reshape(3, 1)
        rod.omega += np.array([0.0, 0.0, omega]).reshape(3, 1)

        frictionplane.apply_friction(rod)

        correct_forces = np.zeros((3, n_elem + 1))
        correct_forces[0] = (
            -1.0
            * np.sign(velocity + omega * rod.radius[0])
            * np.fabs(external_forces_collection[1])
        )

        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())
        forces_on_elements = nodes_to_elements(external_forces_collection)
        correct_torques = np.zeros((3, n_elem))
        correct_torques[2] += (
            -1.0
            * np.sign(velocity + omega * rod.radius[0])
            * np.fabs(forces_on_elements[1])
            * rod.radius
        )

        assert_allclose(correct_torques, rod.external_torques, atol=Tolerance.atol())

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("force_mag", [-20.0, -15.0, 15.0, 20.0])
    def test_static_rolling_friction_total_force_smaller_than_static_friction_force(
        self, n_elem, force_mag
    ):
        """
        In this test case static rolling friction force is tested. We set external and internal torques to
        zero and only changed the force in rolling direction. In this test case, total force in rolling direction
        is smaller than static friction force in rolling direction. Next test case will check what happens if
        total forces in rolling direction larger than static friction force.
        :param n_elem:
        :param force_mag:
        :return:
        """

        [rod, frictionplane, external_forces_collection] = self.initializer(
            n_elem, static_mu_array=np.array([0.0, 0.0, 10.0]), force_mag_side=force_mag
        )

        frictionplane.apply_friction(rod)

        correct_forces = np.zeros((3, n_elem + 1))
        correct_forces[0] = 2.0 / 3.0 * external_forces_collection[0]
        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

        forces_on_elements = nodes_to_elements(external_forces_collection)
        correct_torques = np.zeros((3, n_elem))
        correct_torques[2] += (
            -1.0
            * np.sign(forces_on_elements[0])
            * np.fabs(forces_on_elements[0])
            * rod.radius
            / 3.0
        )

        assert_allclose(correct_torques, rod.external_torques, atol=Tolerance.atol())

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("force_mag", [-100.0, -80.0, 65.0, 95.0])
    def test_static_rolling_friction_total_force_larger_than_static_friction_force(
        self, n_elem, force_mag
    ):
        """
        In this test case static rolling friction force is tested. We set external and internal torques to
        zero and only changed the force in rolling direction. In this test case, total force in rolling direction
        is larger than static friction force in rolling direction.
        :param n_elem:
        :param force_mag:
        :return:
        """
        [rod, frictionplane, external_forces_collection] = self.initializer(
            n_elem, static_mu_array=np.array([0.0, 0.0, 1.0]), force_mag_side=force_mag
        )

        frictionplane.apply_friction(rod)

        correct_forces = np.zeros((3, n_elem + 1))
        correct_forces[0] = external_forces_collection[0] - np.sign(
            external_forces_collection[0]
        ) * np.fabs(external_forces_collection[1])
        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

        forces_on_elements = nodes_to_elements(external_forces_collection)
        correct_torques = np.zeros((3, n_elem))
        correct_torques[2] += (
            -1.0
            * np.sign(forces_on_elements[0])
            * np.fabs(forces_on_elements[1])
            * rod.radius
        )

        assert_allclose(correct_torques, rod.external_torques, atol=Tolerance.atol())

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("torque_mag", [-3.0, -1.0, 2.0, 3.5])
    def test_static_rolling_friction_total_torque_smaller_than_static_friction_force(
        self, n_elem, torque_mag,
    ):
        """
        In this test case, static rolling friction force tested with zero internal and external force and
        with non-zero external torque. Here torque magnitude chosen such that total rolling force is
        always smaller than the static friction force.
        :param n_elem:
        :param torque_mag:
        :return:
        """
        [rod, frictionplane, external_forces_collection] = self.initializer(
            n_elem, static_mu_array=np.array([0.0, 0.0, 10.0])
        )

        external_torques = np.zeros((3, n_elem))
        external_torques[2] = torque_mag
        rod.external_torques = external_torques.copy()

        frictionplane.apply_friction(rod)

        correct_forces = np.zeros((3, n_elem + 1))
        correct_forces[0, :-1] += external_torques[2] / (3.0 * rod.radius)
        correct_forces[0, 1:] += external_torques[2] / (3.0 * rod.radius)
        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

        correct_torques = np.zeros((3, n_elem))
        correct_torques[2] += external_torques[2] + 2.0 / 3.0 * external_torques[2]

        assert_allclose(correct_torques, rod.external_torques, atol=Tolerance.atol())

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("torque_mag", [-10.0, -5.0, 6.0, 7.5])
    def test_static_rolling_friction_total_torque_larger_than_static_friction_force(
        self, n_elem, torque_mag
    ):
        """
        In this test case, static rolling friction force tested with zero internal and external force and
        with non-zero external torque. Here torque magnitude chosen such that total rolling force is
        always larger than the static friction force. Thus, lateral friction force will be equal to static
        friction force.
        :param self:
        :param n_elem:
        :param torque_mag:
        :return:
        """

        [rod, frictionplane, external_forces_collection] = self.initializer(
            n_elem, static_mu_array=np.array([0.0, 0.0, 1.0])
        )

        external_torques = np.zeros((3, n_elem))
        external_torques[2] = torque_mag
        rod.external_torques = external_torques.copy()

        frictionplane.apply_friction(rod)

        correct_forces = np.zeros((3, n_elem + 1))
        correct_forces[0] = np.sign(torque_mag) * np.fabs(external_forces_collection[1])
        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

        forces_on_elements = nodes_to_elements(external_forces_collection)
        correct_torques = external_torques
        correct_torques[2] += (
            np.sign(torque_mag) * np.fabs(forces_on_elements[1]) * rod.radius
        )

        assert_allclose(correct_torques, rod.external_torques, atol=Tolerance.atol())
