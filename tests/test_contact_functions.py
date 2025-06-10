__doc__ = """ Test specific functions used in contact in Elastica.contact_forces implementation"""

import numpy as np
from numpy.testing import assert_allclose
from elastica.rod import RodBase
from elastica.rigidbody import Cylinder, Sphere

from elastica._contact_functions import (
    _calculate_contact_forces_rod_cylinder,
    _calculate_contact_forces_rod_rod,
    _calculate_contact_forces_self_rod,
    _calculate_contact_forces_rod_sphere,
)


def mock_rod_init(self):

    "Initializing Rod"

    """
    This is a small rod with 2 elements;
    Initial Parameters:
    element's radius = 1, length = 1,
    tangent vector for both elements is (1, 0, 0),
    stationary rod i.e velocity vector of each node is (0, 0, 0),
    internal/external forces vectors are also (0, 0, 0)
    """

    self.n_elems = 2
    self.position_collection = np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0]])
    self.radius = np.array([1, 1])
    self.lengths = np.array([1, 1])
    self.tangents = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    self.velocity_collection = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    self.internal_forces = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    self.external_forces = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


def mock_cylinder_init(self):

    "Initializing Cylinder"

    """
    This is a rigid body cylinder;,
    Initial Parameters:
    radius = 1, length = 2,
    center positioned at origin i.e (0, 0, 0),
    cylinder's upright in x,y,z plane thus the director array,
    stationary cylinder i.e velocity vector is (0, 0, 0),
    external forces and torques vectors are also (0, 0, 0)
    """

    self.n_elems = 1
    self.position = np.array([[0], [0], [0]])
    self.director = np.array(
        [[[1.0], [0.0], [0.0]], [[0.0], [1.0], [0.0]], [[0.0], [0.0], [1.0]]]
    )
    self.radius = 1.0
    self.length = 2.0
    self.velocity_collection = np.array([[0.0], [0.0], [0.0]])
    self.external_forces = np.array([[0.0], [0.0], [0.0]])
    self.external_torques = np.array([[0.0], [0.0], [0.0]])


def mock_sphere_init(self):

    "Initializing Sphere"

    """
    This is a rigid body sphere;,
    Initial Parameters:
    radius = 1,
    center positioned at origin i.e (0, 0, 0),
    sphere's upright in x,y,z plane thus the director array,
    stationary sphere i.e velocity vector is (0, 0, 0),
    external forces and torques vectors are also (0, 0, 0)
    """

    self.n_elems = 1
    self.position = np.array([[0], [0], [0]])
    self.director = np.array(
        [[[1.0], [0.0], [0.0]], [[0.0], [1.0], [0.0]], [[0.0], [0.0], [1.0]]]
    )
    self.radius = 1.0
    self.velocity_collection = np.array([[0.0], [0.0], [0.0]])
    self.external_forces = np.array([[0.0], [0.0], [0.0]])
    self.external_torques = np.array([[0.0], [0.0], [0.0]])


MockRod = type("MockRod", (RodBase,), {"__init__": mock_rod_init})

MockCylinder = type("MockCylinder", (Cylinder,), {"__init__": mock_cylinder_init})

MockSphere = type("MockSphere", (Sphere,), {"__init__": mock_sphere_init})


class TestCalculateContactForcesRodCylinder:
    "Class to test the calculate contact forces rod cylinder function"

    "Testing function with handcrafted/calculated values"

    def test_calculate_contact_forces_rod_cylinder_with_k_without_nu_and_friction(
        self,
    ):

        "initializing rod parameters"
        rod = MockRod()
        rod_element_position = 0.5 * (
            rod.position_collection[..., 1:] + rod.position_collection[..., :-1]
        )

        "initializing cylinder parameters"
        cylinder = MockCylinder()
        x_cyl = (
            cylinder.position[..., 0]
            - 0.5 * cylinder.length * cylinder.director[2, :, 0]
        )

        "initializing constants"
        """
        Setting contact_k = 1 and other parameters to 0,
        so the net forces becomes a function of contact forces only.
        """
        k = 1.0
        nu = 0
        velocity_damping_coefficient = 0
        friction_coefficient = 0

        "Function call"
        _calculate_contact_forces_rod_cylinder(
            rod_element_position,
            rod.lengths * rod.tangents,
            cylinder.position[..., 0],
            x_cyl,
            cylinder.length * cylinder.director[2, :, 0],
            rod.radius + cylinder.radius,
            rod.lengths + cylinder.length,
            rod.internal_forces,
            rod.external_forces,
            cylinder.external_forces,
            cylinder.external_torques,
            cylinder.director[:, :, 0],
            rod.velocity_collection,
            cylinder.velocity_collection,
            k,
            nu,
            velocity_damping_coefficient,
            friction_coefficient,
        )

        "Test values"
        """
        The two systems were placed such that they are penetrating by 0.5 units and
        resulting forces act along the x-axis only.
        The net force was calculated by halving the contact force i.e
                                                net force = 0.5 * contact force = -0.25;
                                                    where, contact force = k(1) * min distance between colliding elements(-1) * gamma(0.5) = -0.5
        The net force is then divided to the nodes of the rod and the cylinder as per indices.
        """
        assert_allclose(
            cylinder.external_forces, np.array([[-0.5], [0], [0]]), atol=1e-6
        )
        assert_allclose(cylinder.external_torques, np.array([[0], [0], [0]]), atol=1e-6)
        assert_allclose(
            rod.external_forces,
            np.array([[0.166666, 0.333333, 0], [0, 0, 0], [0, 0, 0]]),
            atol=1e-6,
        )

    def test_calculate_contact_forces_rod_cylinder_with_nu_without_k_and_friction(
        self,
    ):

        "initializing rod parameters"
        rod = MockRod()
        "Moving rod towards the cylinder with a velocity of -1 in x-axis"
        rod.velocity_collection = np.array([[-1, 0, 0], [-1, 0, 0], [-1, 0, 0]])
        rod_element_position = 0.5 * (
            rod.position_collection[..., 1:] + rod.position_collection[..., :-1]
        )

        "initializing cylinder parameters"
        cylinder = MockCylinder()
        "Moving cylinder towards the rod with a velocity of 1 in x-axis"
        cylinder.velocity_collection = np.array([[1], [0], [0]])
        x_cyl = (
            cylinder.position[..., 0]
            - 0.5 * cylinder.length * cylinder.director[2, :, 0]
        )

        "initializing constants"
        """
        Setting contact_nu = 1 and other parameters to 0,
        so the net forces becomes a function of contact damping forces only.
        """
        k = 0.0
        nu = 1.0
        velocity_damping_coefficient = 0
        friction_coefficient = 0

        "Function call"
        _calculate_contact_forces_rod_cylinder(
            rod_element_position,
            rod.lengths * rod.tangents,
            cylinder.position[..., 0],
            x_cyl,
            cylinder.length * cylinder.director[2, :, 0],
            rod.radius + cylinder.radius,
            rod.lengths + cylinder.length,
            rod.internal_forces,
            rod.external_forces,
            cylinder.external_forces,
            cylinder.external_torques,
            cylinder.director[:, :, 0],
            rod.velocity_collection,
            cylinder.velocity_collection,
            k,
            nu,
            velocity_damping_coefficient,
            friction_coefficient,
        )

        "Test values"
        """
        The two systems were placed such that they are penetrating by 0.5 units and
        resulting forces act along the x-axis only.
        The net force was calculated by halving the contact damping force i.e
                                                net force = 0.5 * contact damping force = -0.75;
                                                    where, contact damping force = -nu(1) * penetration velocity(1.5)[x-axis] = -1.5
        The net force is then divided to the nodes of the rod and the cylinder as per indices.
        """
        assert_allclose(
            cylinder.external_forces, np.array([[-1.5], [0], [0]]), atol=1e-6
        )
        assert_allclose(cylinder.external_torques, np.array([[0], [0], [0]]), atol=1e-6)
        assert_allclose(
            rod.external_forces,
            np.array([[0.5, 1, 0], [0, 0, 0], [0, 0, 0]]),
            atol=1e-6,
        )

    def test_calculate_contact_forces_rod_cylinder_with_k_and_nu_without_friction(
        self,
    ):

        "initializing rod parameters"
        rod = MockRod()
        "Moving rod towards the cylinder with a velocity of -1 in x-axis"
        rod.velocity_collection = np.array([[-1, 0, 0], [-1, 0, 0], [-1, 0, 0]])
        rod_element_position = 0.5 * (
            rod.position_collection[..., 1:] + rod.position_collection[..., :-1]
        )

        "initializing cylinder parameters"
        cylinder = MockCylinder()
        "Moving cylinder towards the rod with a velocity of 1 in x-axis"
        cylinder.velocity_collection = np.array([[1], [0], [0]])
        x_cyl = (
            cylinder.position[..., 0]
            - 0.5 * cylinder.length * cylinder.director[2, :, 0]
        )

        "initializing constants"
        """
        Setting contact_nu = 1 and contact_k = 1,
        so the net forces becomes a function of contact damping and contact forces.
        """
        k = 1.0
        nu = 1.0
        velocity_damping_coefficient = 0
        friction_coefficient = 0

        "Function call"
        _calculate_contact_forces_rod_cylinder(
            rod_element_position,
            rod.lengths * rod.tangents,
            cylinder.position[..., 0],
            x_cyl,
            cylinder.length * cylinder.director[2, :, 0],
            rod.radius + cylinder.radius,
            rod.lengths + cylinder.length,
            rod.internal_forces,
            rod.external_forces,
            cylinder.external_forces,
            cylinder.external_torques,
            cylinder.director[:, :, 0],
            rod.velocity_collection,
            cylinder.velocity_collection,
            k,
            nu,
            velocity_damping_coefficient,
            friction_coefficient,
        )

        "Test values"
        """
        For nu and k dependent case, we just have to add both the forces that were generated above.
        """
        assert_allclose(cylinder.external_forces, np.array([[-2], [0], [0]]), atol=1e-6)
        assert_allclose(cylinder.external_torques, np.array([[0], [0], [0]]), atol=1e-6)
        assert_allclose(
            rod.external_forces,
            np.array([[0.666666, 1.333333, 0], [0, 0, 0], [0, 0, 0]]),
            atol=1e-6,
        )

    def test_calculate_contact_forces_rod_cylinder_with_k_and_nu_and_friction(self):

        "initializing rod parameters"
        rod = MockRod()
        "Moving rod towards the cylinder with a velocity of -1 in x-axis"
        rod.velocity_collection = np.array([[-1, 0, 0], [-1, 0, 0], [-1, 0, 0]])
        rod_element_position = 0.5 * (
            rod.position_collection[..., 1:] + rod.position_collection[..., :-1]
        )

        "initializing cylinder parameters"
        cylinder = MockCylinder()
        "Moving cylinder towards the rod with a velocity of 1 in x-axis"
        cylinder.velocity_collection = np.array([[1], [0], [0]])
        x_cyl = (
            cylinder.position[..., 0]
            - 0.5 * cylinder.length * cylinder.director[2, :, 0]
        )

        "initializing constants"
        k = 1.0
        nu = 1.0
        velocity_damping_coefficient = 0.1
        friction_coefficient = 0.1

        "Function call"
        _calculate_contact_forces_rod_cylinder(
            rod_element_position,
            rod.lengths * rod.tangents,
            cylinder.position[..., 0],
            x_cyl,
            cylinder.length * cylinder.director[2, :, 0],
            rod.radius + cylinder.radius,
            rod.lengths + cylinder.length,
            rod.internal_forces,
            rod.external_forces,
            cylinder.external_forces,
            cylinder.external_torques,
            cylinder.director[:, :, 0],
            rod.velocity_collection,
            cylinder.velocity_collection,
            k,
            nu,
            velocity_damping_coefficient,
            friction_coefficient,
        )

        "Test values"
        """
        With friction, we have to subtract the frictional forces from the net contact forces.
        from above values the frictional forces are calculated as:
            coulombic friction force = friction coefficient(0.1) * net contact force before friction(1) * slip_direction_velocity_unitized(0.5^-2) = 0.07071... (for y and z axis)
            slip direction friction = velocity damping coefficient(0.1) * slip_direction_velocity(0.5^-2) * slip_direction_velocity_unitized(0.5^-2) = 0.05 (for y and z axis)
        the minimum of the two is slip direction friciton, so the frictional force is that only:
            friction force = slip direction friction = 0.05
        after applying sign convention and dividing the force among the nodes of rod and cylinder,
        we get the following values.
        """
        assert_allclose(
            cylinder.external_forces, np.array([[-2], [-0.1], [-0.1]]), atol=1e-6
        )
        assert_allclose(cylinder.external_torques, np.array([[0], [0], [0]]), atol=1e-6)
        assert_allclose(
            rod.external_forces,
            np.array(
                [
                    [0.666666, 1.333333, 0],
                    [0.033333, 0.066666, 0],
                    [0.033333, 0.066666, 0],
                ]
            ),
            atol=1e-6,
        )


class TestCalculateContactForcesRodRod:
    "Function to test the calculate contact forces rod rod function"

    "Testing function with handcrafted/calculated values"

    def test_calculate_contact_forces_rod_rod_with_k_without_nu(self):

        rod_one = MockRod()
        rod_two = MockRod()
        """Placing rod two such that its first element just touches the last element of rod one."""
        rod_two.position_collection = np.array([[4, 5, 6], [0, 0, 0], [0, 0, 0]])

        "initializing constants"
        """
        Setting contact_k = 1 and nu to 0,
        so the net forces becomes a function of contact forces only.
        """
        k = 1.0
        nu = 0.0

        "Function call"
        _calculate_contact_forces_rod_rod(
            rod_one.position_collection[..., :-1],
            rod_one.radius,
            rod_one.lengths,
            rod_one.tangents,
            rod_one.velocity_collection,
            rod_one.internal_forces,
            rod_one.external_forces,
            rod_two.position_collection[..., :-1],
            rod_two.radius,
            rod_two.lengths,
            rod_two.tangents,
            rod_two.velocity_collection,
            rod_two.internal_forces,
            rod_two.external_forces,
            k,
            nu,
        )

        "Test values"
        """
        Resulting forces act along the x-axis only.
        The net force was calculated by halving the contact force i.e
                                                net force = 0.5 * contact force = 0.5;
                                                    where, contact force = k(1) * min distance between colliding elements(1) * gamma(1) = 1
        The net force is then divided to the nodes of the two rods as per indices.
        """
        assert_allclose(
            rod_one.external_forces,
            np.array(
                [[0, -0.666666, -0.333333], [0, 0, 0], [0, 0, 0]],
            ),
            atol=1e-6,
        )
        assert_allclose(
            rod_two.external_forces,
            np.array([[0.333333, 0.666666, 0], [0, 0, 0], [0, 0, 0]]),
            atol=1e-6,
        )

    def test_calculate_contact_forces_rod_rod_without_k_with_nu(self):

        rod_one = MockRod()
        rod_two = MockRod()
        """Placing rod two such that its first element just touches the last element of rod one."""
        rod_two.position_collection = np.array([[4, 5, 6], [0, 0, 0], [0, 0, 0]])

        """Moving the rods towards each other with a velocity of 1 along the x-axis."""
        rod_one.velocity_collection = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        rod_two.velocity_collection = np.array([[-1, 0, 0], [-1, 0, 0], [-1, 0, 0]])

        "initializing constants"
        """
        Setting contact_nu = 1 and nu to 0,
        so the net forces becomes a function of contact damping forces only.
        """
        k = 0.0
        nu = 1.0

        "Function call"
        _calculate_contact_forces_rod_rod(
            rod_one.position_collection[..., :-1],
            rod_one.radius,
            rod_one.lengths,
            rod_one.tangents,
            rod_one.velocity_collection,
            rod_one.internal_forces,
            rod_one.external_forces,
            rod_two.position_collection[..., :-1],
            rod_two.radius,
            rod_two.lengths,
            rod_two.tangents,
            rod_two.velocity_collection,
            rod_two.internal_forces,
            rod_two.external_forces,
            k,
            nu,
        )

        "Test values"
        """
        Resulting forces act along the x-axis only.
        The net force was calculated by halving the contact damping force i.e
                                                net force = 0.5 * contact damping force = 0.25;
                                                    where, contact damping force = nu(1) * penetration velocity(0.5)[x-axis] = 0.5
        The net force is then divided to the nodes of the two rods as per indices.
        """
        assert_allclose(
            rod_one.external_forces,
            np.array(
                [[0, -0.333333, -0.166666], [0, 0, 0], [0, 0, 0]],
            ),
            atol=1e-6,
        )
        assert_allclose(
            rod_two.external_forces,
            np.array([[0.166666, 0.333333, 0], [0, 0, 0], [0, 0, 0]]),
            atol=1e-6,
        )

    def test_calculate_contact_forces_rod_rod_with_k_and_nu(self):

        rod_one = MockRod()
        rod_two = MockRod()
        """Placing rod two such that its first element just touches the last element of rod one."""
        rod_two.position_collection = np.array([[4, 5, 6], [0, 0, 0], [0, 0, 0]])

        """Moving the rods towards each other with a velocity of 1 along the x-axis."""
        rod_one.velocity_collection = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        rod_two.velocity_collection = np.array([[-1, 0, 0], [-1, 0, 0], [-1, 0, 0]])

        "initializing constants"
        """
        Setting contact_nu = 1 and contact_k = 1,
        so the net forces becomes a function of contact damping and contact forces.
        """
        k = 1.0
        nu = 1.0

        "Function call"
        _calculate_contact_forces_rod_rod(
            rod_one.position_collection[..., :-1],
            rod_one.radius,
            rod_one.lengths,
            rod_one.tangents,
            rod_one.velocity_collection,
            rod_one.internal_forces,
            rod_one.external_forces,
            rod_two.position_collection[..., :-1],
            rod_two.radius,
            rod_two.lengths,
            rod_two.tangents,
            rod_two.velocity_collection,
            rod_two.internal_forces,
            rod_two.external_forces,
            k,
            nu,
        )

        "Test values"
        """
        For nu and k dependent case, we just have to add both the forces that were generated above.
        """
        assert_allclose(
            rod_one.external_forces,
            np.array(
                [[0, -1, -0.5], [0, 0, 0], [0, 0, 0]],
            ),
            atol=1e-6,
        )
        assert_allclose(
            rod_two.external_forces,
            np.array([[0.5, 1, 0], [0, 0, 0], [0, 0, 0]]),
            atol=1e-6,
        )


def test_calculate_contact_forces_self_rod():
    "Function to test the calculate contact forces self rod function"

    "Testing function with handcrafted/calculated values"

    rod = MockRod()
    """Changing rod parameters to establish self contact in rod;
    elements are placed such that the a 'U' rod is formed in the x-y plane,
    where the rod is penetrating itself by 0.5 units by radius."""
    rod.n_elems = 3
    rod.position_collection = np.array([[1, 4, 4, 1], [0, 0, 1, 1], [0, 0, 0, 0]])
    rod.radius = np.array([1, 1, 1])
    rod.lengths = np.array([3, 1, 3])
    rod.tangents = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    rod.velocity_collection = np.array(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    rod.internal_forces = np.array(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    rod.external_forces = np.array(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )

    "initializing constants"
    k = 1.0
    nu = 1.0

    "Function call"
    _calculate_contact_forces_self_rod(
        rod.position_collection[..., :-1],
        rod.radius,
        rod.lengths,
        rod.tangents,
        rod.velocity_collection,
        rod.external_forces,
        k,
        nu,
    )

    "Test values"
    """Resulting forces act along the y-axis only.
    Since the rod is stationary i.e velocity = 0, the net force is a function of contact force only.
    The net force was calculated by halving the contact force i.e
                                        net force = 0.5 * contact force = -0.5;
                                            where, contact force = k(1) * minimum distance between colliding elements centres(-1) * gamma(1) = -1
    The net force is then divided to the nodes of the rod as per indices."""
    assert_allclose(
        rod.external_forces,
        np.array(
            [[0, 0, 0, 0], [-0.333333, -0.666666, 0.666666, 0.333333], [0, 0, 0, 0]]
        ),
        atol=1e-6,
    )


class TestCalculateContactForcesRodSphere:
    "Class to test the calculate contact forces rod sphere function"

    "Testing function with handcrafted/calculated values"

    def test_calculate_contact_forces_rod_sphere_with_k_without_nu_and_friction(
        self,
    ):

        "initializing rod parameters"
        rod = MockRod()
        rod_element_position = 0.5 * (
            rod.position_collection[..., 1:] + rod.position_collection[..., :-1]
        )

        "initializing sphere parameters"
        sphere = MockSphere()
        x_sph = sphere.position[..., 0] - sphere.radius * sphere.director[2, :, 0]

        "initializing constants"
        """
        Setting contact_k = 1 and other parameters to 0,
        so the net forces becomes a function of contact forces only.
        """
        k = 1.0
        nu = 0
        velocity_damping_coefficient = 0
        friction_coefficient = 0

        "Function call"
        _calculate_contact_forces_rod_sphere(
            rod_element_position,
            rod.lengths * rod.tangents,
            sphere.position[..., 0],
            x_sph,
            sphere.radius * sphere.director[2, :, 0],
            rod.radius + sphere.radius,
            rod.lengths + sphere.radius * 2,
            rod.internal_forces,
            rod.external_forces,
            sphere.external_forces,
            sphere.external_torques,
            sphere.director[:, :, 0],
            rod.velocity_collection,
            sphere.velocity_collection,
            k,
            nu,
            velocity_damping_coefficient,
            friction_coefficient,
        )

        "Test values"
        """
        The two systems were placed such that they are penetrating by 0.5 units and
        resulting forces act along the x-axis only.
        The net force was calculated by halving the contact force i.e
                                                net force = 0.5 * contact force = -0.25;
                                                    where, contact force = k(1) * min distance between colliding elements(-1) * gamma(0.5) = -0.5
        The net force is then divided to the nodes of the rod and the sphere as per indices.
        """
        assert_allclose(sphere.external_forces, np.array([[-0.5], [0], [0]]), atol=1e-6)
        assert_allclose(sphere.external_torques, np.array([[0], [0], [0]]), atol=1e-6)
        assert_allclose(
            rod.external_forces,
            np.array([[0.166666, 0.333333, 0], [0, 0, 0], [0, 0, 0]]),
            atol=1e-6,
        )
