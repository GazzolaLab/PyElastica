__doc__ = """ Interaction module import test"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from elastica.utils import Tolerance, MaxDimension
from elastica.interaction import (
    InteractionPlane,
    find_slipping_elements,
    AnisotropicFrictionalPlane,
    nodes_to_elements,
    SlenderBodyTheory,
)

from tests.test_rod.test_rods import MockTestRod


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

    def _compute_internal_forces(self):
        return np.zeros((MaxDimension.value(), self.n_elem + 1))

    def _compute_internal_torques(self):
        return np.zeros((MaxDimension.value(), self.n_elem))


class TestInteractionPlane:
    def initializer(
        self,
        n_elem,
        shift=0.0,
        k_w=0.0,
        nu_w=0.0,
        plane_normal=np.array([0.0, 1.0, 0.0]),
    ):
        rod = BaseRodClass(n_elem)
        plane_origin = np.array([0.0, -rod.radius[0] + shift, 0.0])
        interaction_plane = InteractionPlane(k_w, nu_w, plane_origin, plane_normal)
        fnormal = -10.0 * np.sign(plane_normal[1]) * np.random.random_sample(1).item()
        external_forces = np.repeat(
            np.array([0.0, fnormal, 0.0]).reshape(3, 1), n_elem + 1, axis=1
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
        Parameters
        ----------
        n_elem

        Returns
        -------

        """

        shift = -(
            (2.0 - 1.0) * np.random.random_sample(1) + 1.0
        ).item()  # we move plane away from rod

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

        Parameters
        ----------
        n_elem

        Returns
        -------

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
        Parameters
        ----------
        n_elem
        k_w

        Returns
        -------

        """

        shift = np.random.random_sample(1).item()  # we move plane towards to rod
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
        Parameters
        ----------
        n_elem
        nu_w

        Returns
        -------

        """

        [rod, interaction_plane, external_forces] = self.initializer(n_elem, nu_w=nu_w)

        normal_velocity = np.random.random_sample(1).item()
        rod.velocity_collection[..., :] += np.array(
            [0.0, -normal_velocity, 0.0]
        ).reshape(3, 1)

        correct_forces = np.repeat(
            (nu_w * np.array([0.0, normal_velocity, 0.0])).reshape(3, 1),
            n_elem + 1,
            axis=1,
        )

        correct_forces[..., 0] *= 0.5
        correct_forces[..., -1] *= 0.5

        interaction_plane.apply_normal_force(rod)

        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    def test_interaction_when_rod_is_under_plane(self, n_elem):
        """
        This test case tests plane response forces on the rod
        in the case rod is under the plane and pushed towards
        the plane.
        Parameters
        ----------
        n_elem

        Returns
        -------

        """

        # we move plane on top of the rod. Note that 0.25 is radius of the rod.
        offset_of_plane_with_respect_to_rod = 2.0 * 0.25

        # plane normal changed, it is towards the negative direction, because rod
        # is under the rod.
        plane_normal = np.array([0.0, -1.0, 0.0])

        [rod, interaction_plane, external_forces] = self.initializer(
            n_elem, shift=offset_of_plane_with_respect_to_rod, plane_normal=plane_normal
        )

        interaction_plane.apply_normal_force(rod)
        correct_forces = np.zeros((3, n_elem + 1))
        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("k_w", [0.1, 0.5, 1.0, 2, 10])
    def test_interaction_when_rod_is_under_plane_with_k_without_nu(self, n_elem, k_w):
        """
        In this test case we move the rod under the plane.
        Here wall stiffness coefficient changed parametrically
        and damping coefficient set to zero .
        Parameters
        ----------
        n_elem
        k_w

        Returns
        -------

        """
        # we move plane on top of the rod. Note that 0.25 is radius of the rod.
        offset_of_plane_with_respect_to_rod = 2.0 * 0.25

        # we move plane towards to rod by random distance
        shift = offset_of_plane_with_respect_to_rod - np.random.random_sample(1).item()

        # plane normal changed, it is towards the negative direction, because rod
        # is under the rod.
        plane_normal = np.array([0.0, -1.0, 0.0])

        [rod, interaction_plane, external_forces] = self.initializer(
            n_elem, shift=shift, k_w=k_w, plane_normal=plane_normal
        )

        # we have to substract rod offset because top part
        correct_forces = k_w * np.repeat(
            np.array([0.0, shift - offset_of_plane_with_respect_to_rod, 0.0]).reshape(
                3, 1
            ),
            n_elem + 1,
            axis=1,
        )
        correct_forces[..., 0] *= 0.5
        correct_forces[..., -1] *= 0.5

        interaction_plane.apply_normal_force(rod)

        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("nu_w", [0.5, 1.0, 5.0, 7.0, 12.0])
    def test_interaction_when_rod_is_under_plane_without_k_with_nu(self, n_elem, nu_w):
        """
        In this test case we move under the plane and test damping force.
        Here wall damping coefficient are changed parametrically and
        wall response functions tested.
        Parameters
        ----------
        n_elem
        nu_w

        Returns
        -------

        """
        # we move plane on top of the rod. Note that 0.25 is radius of the rod.
        offset_of_plane_with_respect_to_rod = 2.0 * 0.25

        # plane normal changed, it is towards the negative direction, because rod
        # is under the rod.
        plane_normal = np.array([0.0, -1.0, 0.0])

        [rod, interaction_plane, external_forces] = self.initializer(
            n_elem,
            shift=offset_of_plane_with_respect_to_rod,
            nu_w=nu_w,
            plane_normal=plane_normal,
        )

        normal_velocity = np.random.random_sample(1).item()
        rod.velocity_collection[..., :] += np.array(
            [0.0, -normal_velocity, 0.0]
        ).reshape(3, 1)

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
        slip_function = find_slipping_elements(velocity_slip, velocity_threshold)
        correct_slip_function = np.zeros(n_elem)
        assert_allclose(correct_slip_function, slip_function, atol=Tolerance.atol())

        # if slip velocity smaller than threshold
        velocity_slip = np.repeat(
            np.array([0.0, 0.0, 0.0]).reshape(3, 1), n_elem, axis=1
        )
        slip_function = find_slipping_elements(velocity_slip, velocity_threshold)
        correct_slip_function = np.ones(n_elem)
        assert_allclose(correct_slip_function, slip_function, atol=Tolerance.atol())

        # if slip velocity smaller than threshold but very close to threshold
        velocity_slip = np.repeat(
            np.array([0.0, 0.0, 1.0 - 1e-6]).reshape(3, 1), n_elem, axis=1
        )
        slip_function = find_slipping_elements(velocity_slip, velocity_threshold)
        correct_slip_function = np.ones(n_elem)
        assert_allclose(correct_slip_function, slip_function, atol=Tolerance.atol())

        # if slip velocity larger than threshold but very close to threshold
        velocity_slip = np.repeat(
            np.array([0.0, 0.0, 1.0 + 1e-6]).reshape(3, 1), n_elem, axis=1
        )
        slip_function = find_slipping_elements(velocity_slip, velocity_threshold)
        correct_slip_function = np.ones(n_elem) - 1e-6
        assert_allclose(correct_slip_function, slip_function, atol=Tolerance.atol())

        # if half of the array slip velocity is larger than threshold and half of it
        # smaller than threshold
        velocity_slip = np.hstack(
            (
                np.repeat(np.array([0.0, 0.0, 2.0]).reshape(3, 1), n_elem, axis=1),
                np.repeat(np.array([0.0, 0.0, 0.0]).reshape(3, 1), n_elem, axis=1),
            )
        )
        slip_function = find_slipping_elements(velocity_slip, velocity_threshold)
        correct_slip_function = np.hstack((np.zeros(n_elem), np.ones(n_elem)))
        assert_allclose(correct_slip_function, slip_function, atol=Tolerance.atol())

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    def test_nodes_to_elements(self, n_elem):
        random_vector = np.random.rand(3).reshape(3, 1)
        input = np.repeat(random_vector, n_elem + 1, axis=1)
        input[..., 0] *= 0.5
        input[..., -1] *= 0.5
        correct_output = np.repeat(random_vector, n_elem, axis=1)
        output = nodes_to_elements(input)
        assert_allclose(correct_output, output, atol=Tolerance.atol())
        assert_allclose(np.sum(input), np.sum(output), atol=Tolerance.atol())


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
        friction_plane = AnisotropicFrictionalPlane(
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
        ).item() + 5.0  # generates random numbers [5.0,10)
        external_forces = np.array([force_mag_side, -fnormal, force_mag_long])

        external_forces_collection = np.repeat(
            external_forces.reshape(3, 1), n_elem + 1, axis=1
        )
        external_forces_collection[..., 0] *= 0.5
        external_forces_collection[..., -1] *= 0.5
        rod.external_forces = external_forces_collection.copy()

        # Velocities has to be set to zero
        assert_allclose(
            np.zeros((3, n_elem)), rod.omega_collection, atol=Tolerance.atol()
        )
        assert_allclose(
            np.zeros((3, n_elem + 1)), rod.velocity_collection, atol=Tolerance.atol()
        )

        # We have not changed torques also, they have to be zero as well
        assert_allclose(
            np.zeros((3, n_elem)), rod.external_torques, atol=Tolerance.atol()
        )
        assert_allclose(
            np.zeros((3, n_elem)),
            rod._compute_internal_torques(),
            atol=Tolerance.atol(),
        )

        return rod, friction_plane, external_forces_collection

    @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("velocity", [-1.0, -3.0, 1.0, 5.0, 2.0])
    def test_axial_kinetic_friction(self, n_elem, velocity):
        """
        This function tests kinetic friction in forward and backward direction.
        All other friction coefficients set to zero.
        Parameters
        ----------
        n_elem
        velocity

        Returns
        -------

        """

        [rod, friction_plane, external_forces_collection] = self.initializer(
            n_elem, kinetic_mu_array=np.array([1.0, 1.0, 0.0])
        )

        rod.velocity_collection += np.array([0.0, 0.0, velocity]).reshape(3, 1)

        friction_plane.apply_forces(rod)

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
        Parameters
        ----------
        n_elem
        force_mag

        Returns
        -------

        """
        [rod, frictionplane, external_forces_collection] = self.initializer(
            n_elem, static_mu_array=np.array([1.0, 1.0, 0.0]), force_mag_long=force_mag
        )

        frictionplane.apply_forces(rod)
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
        Parameters
        ----------
        n_elem
        force_mag

        Returns
        -------

        """

        [rod, frictionplane, external_forces_collection] = self.initializer(
            n_elem, static_mu_array=np.array([1.0, 1.0, 0.0]), force_mag_long=force_mag
        )

        frictionplane.apply_forces(rod)
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
        Parameters
        ----------
        n_elem
        velocity
        omega

        Returns
        -------

        """
        [rod, frictionplane, external_forces_collection] = self.initializer(
            n_elem, kinetic_mu_array=np.array([0.0, 0.0, 1.0])
        )

        rod.velocity_collection += np.array([velocity, 0.0, 0.0]).reshape(3, 1)
        rod.omega_collection += np.array([0.0, 0.0, omega]).reshape(3, 1)

        frictionplane.apply_forces(rod)

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
        Parameters
        ----------
        n_elem
        force_mag

        Returns
        -------

        """

        [rod, frictionplane, external_forces_collection] = self.initializer(
            n_elem, static_mu_array=np.array([0.0, 0.0, 10.0]), force_mag_side=force_mag
        )

        frictionplane.apply_forces(rod)

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
        Parameters
        ----------
        n_elem
        force_mag

        Returns
        -------

        """

        [rod, frictionplane, external_forces_collection] = self.initializer(
            n_elem, static_mu_array=np.array([0.0, 0.0, 1.0]), force_mag_side=force_mag
        )

        frictionplane.apply_forces(rod)

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
        self, n_elem, torque_mag
    ):
        """
        In this test case, static rolling friction force tested with zero internal and external force and
        with non-zero external torque. Here torque magnitude chosen such that total rolling force is
        always smaller than the static friction force.
        Parameters
        ----------
        n_elem
        torque_mag

        Returns
        -------

        """

        [rod, frictionplane, external_forces_collection] = self.initializer(
            n_elem, static_mu_array=np.array([0.0, 0.0, 10.0])
        )

        external_torques = np.zeros((3, n_elem))
        external_torques[2] = torque_mag
        rod.external_torques = external_torques.copy()

        frictionplane.apply_forces(rod)

        correct_forces = np.zeros((3, n_elem + 1))
        correct_forces[0, :-1] -= external_torques[2] / (3.0 * rod.radius)
        correct_forces[0, 1:] -= external_torques[2] / (3.0 * rod.radius)
        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

        correct_torques = np.zeros((3, n_elem))
        correct_torques[2] += external_torques[2] - 2.0 / 3.0 * external_torques[2]

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
        Parameters
        ----------
        n_elem
        torque_mag

        Returns
        -------

        """

        [rod, frictionplane, external_forces_collection] = self.initializer(
            n_elem, static_mu_array=np.array([0.0, 0.0, 1.0])
        )

        external_torques = np.zeros((3, n_elem))
        external_torques[2] = torque_mag
        rod.external_torques = external_torques.copy()

        frictionplane.apply_forces(rod)

        correct_forces = np.zeros((3, n_elem + 1))
        correct_forces[0] = (
            -1.0 * np.sign(torque_mag) * np.fabs(external_forces_collection[1])
        )
        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

        forces_on_elements = nodes_to_elements(external_forces_collection)
        correct_torques = external_torques
        correct_torques[2] += -(
            np.sign(torque_mag) * np.fabs(forces_on_elements[1]) * rod.radius
        )

        assert_allclose(correct_torques, rod.external_torques, atol=Tolerance.atol())


# Slender Body Theory Unit Tests

try:
    from elastica.interaction import sum_over_elements, node_to_element_pos_or_vel

    # These functions are used in the case if Numba is available
    class TestAuxiliaryFunctionsForSlenderBodyTheory:
        @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
        def test_sum_over_elements(self, n_elem):
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

            input_variable = np.random.rand(n_elem)
            correct_output = input_variable.sum()
            output = sum_over_elements(input_variable)
            assert_allclose(correct_output, output, atol=Tolerance.atol())

        @pytest.mark.parametrize("n_elem", [2, 3, 5, 10, 20])
        def test_node_to_elements(self, n_elem):
            """
            This function test node_to_element_velocity function. We are
            converting node velocities to element velocities. Here also
            we are using numba to speed up the process.

            Parameters
            ----------
            n_elem

            Returns
            -------

            """
            random = np.random.rand()  # Adding some random numbers
            input_variable = random * np.ones((3, n_elem + 1))
            correct_output = random * np.ones((3, n_elem))

            output = node_to_element_pos_or_vel(input_variable)
            assert_allclose(correct_output, output, atol=Tolerance.atol())

except ImportError:
    pass


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
