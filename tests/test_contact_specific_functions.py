__doc__ = """ Test specific functions used in contact in Elastica.joint implementation, should be removed along with contact in joint"""

import numpy as np
from numpy.testing import assert_allclose
from elastica.typing import RodBase, RigidBodyBase
from elastica.joint import (
    _prune_using_aabbs_rod_rigid_body,
    _prune_using_aabbs_rod_rod,
    _calculate_contact_forces_rod_rigid_body,
    _calculate_contact_forces_rod_rod,
    _calculate_contact_forces_self_rod,
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
    self.radius_collection = np.array([1, 1])
    self.length_collection = np.array([1, 1])
    self.tangent_collection = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    self.velocity_collection = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    self.internal_forces = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    self.external_forces = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


def mock_rigid_body_init(self):

    "Initializing Rigid Body as a cylinder"

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


MockRod = type("MockRod", (RodBase,), {"__init__": mock_rod_init})

MockRigidBody = type(
    "MockRigidBody", (RigidBodyBase,), {"__init__": mock_rigid_body_init}
)


def test_prune_using_aabbs_rod_rigid_body():
    "Function to test the prune using aabbs rod rigid body function"

    "Testing function with analytically verified values"

    "Intersecting rod and cylinder"

    """
    Since both the initialized rod and cylinder are overlapping in 3D space at (1, 1, 1);
    Hence they are intersectiong and the function should return 0
    """
    rod = MockRod()
    cylinder = MockRigidBody()
    assert (
        _prune_using_aabbs_rod_rigid_body(
            rod.position_collection,
            rod.radius_collection,
            rod.length_collection,
            cylinder.position,
            cylinder.director,
            cylinder.radius,
            cylinder.length,
        )
        == 0
    )

    "Non - Intersecting rod and cylinder"
    rod = MockRod()
    cylinder = MockRigidBody()

    """
    Changing the position of cylinder in 3D space so the rod and cylinder don't overlap/intersect.
    """
    cylinder.position = np.array([[20], [3], [4]])
    assert (
        _prune_using_aabbs_rod_rigid_body(
            rod.position_collection,
            rod.radius_collection,
            rod.length_collection,
            cylinder.position,
            cylinder.director,
            cylinder.radius,
            cylinder.length,
        )
        == 1
    )


def test_prune_using_aabbs_rod_rod():
    "Function to test the prune using aabbs rod rod function"

    "Testing function with analytically verified values"

    "Intersecting rod and rod"
    """
    Since both the rods have same position, node's radius and length, they are overlapping/intersecting in 3D space.
    """
    rod_one = MockRod()
    rod_two = MockRod()
    assert (
        _prune_using_aabbs_rod_rod(
            rod_one.position_collection,
            rod_one.radius_collection,
            rod_one.length_collection,
            rod_two.position_collection,
            rod_two.radius_collection,
            rod_two.length_collection,
        )
        == 0
    )

    "Non - Intersecting rod and rod"
    """
    Changing the position of rod_two in 3D space so the rod_one and rod_two don't overlap/intersect.
    """
    rod_two.position_collection = np.array([[20, 21, 22], [0, 0, 0], [0, 0, 0]])
    assert (
        _prune_using_aabbs_rod_rod(
            rod_one.position_collection,
            rod_one.radius_collection,
            rod_one.length_collection,
            rod_two.position_collection,
            rod_two.radius_collection,
            rod_two.length_collection,
        )
        == 1
    )


class TestCalculateContactForcesRodRigidBody:
    "Class to test the calculate contact forces rod rigid body function"

    "Testing function with handcrafted/calculated values"

    def test_calculate_contact_forces_rod_rigid_body_with_k_without_nu_and_friction(
        self,
    ):

        "initializing rod parameters"
        rod = MockRod()
        rod_element_position = 0.5 * (
            rod.position_collection[..., 1:] + rod.position_collection[..., :-1]
        )

        "initializing cylinder parameters"
        cylinder = MockRigidBody()
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
        _calculate_contact_forces_rod_rigid_body(
            rod_element_position,
            rod.length_collection * rod.tangent_collection,
            cylinder.position[..., 0],
            x_cyl,
            cylinder.length * cylinder.director[2, :, 0],
            rod.radius_collection + cylinder.radius,
            rod.length_collection + cylinder.length,
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

    def test_calculate_contact_forces_rod_rigid_body_with_nu_without_k_and_friction(
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
        cylinder = MockRigidBody()
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
        _calculate_contact_forces_rod_rigid_body(
            rod_element_position,
            rod.length_collection * rod.tangent_collection,
            cylinder.position[..., 0],
            x_cyl,
            cylinder.length * cylinder.director[2, :, 0],
            rod.radius_collection + cylinder.radius,
            rod.length_collection + cylinder.length,
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

    def test_calculate_contact_forces_rod_rigid_body_with_k_and_nu_without_friction(
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
        cylinder = MockRigidBody()
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
        _calculate_contact_forces_rod_rigid_body(
            rod_element_position,
            rod.length_collection * rod.tangent_collection,
            cylinder.position[..., 0],
            x_cyl,
            cylinder.length * cylinder.director[2, :, 0],
            rod.radius_collection + cylinder.radius,
            rod.length_collection + cylinder.length,
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

    def test_calculate_contact_forces_rod_rigid_body_with_k_and_nu_and_friction(self):

        "initializing rod parameters"
        rod = MockRod()
        "Moving rod towards the cylinder with a velocity of -1 in x-axis"
        rod.velocity_collection = np.array([[-1, 0, 0], [-1, 0, 0], [-1, 0, 0]])
        rod_element_position = 0.5 * (
            rod.position_collection[..., 1:] + rod.position_collection[..., :-1]
        )

        "initializing cylinder parameters"
        cylinder = MockRigidBody()
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
        _calculate_contact_forces_rod_rigid_body(
            rod_element_position,
            rod.length_collection * rod.tangent_collection,
            cylinder.position[..., 0],
            x_cyl,
            cylinder.length * cylinder.director[2, :, 0],
            rod.radius_collection + cylinder.radius,
            rod.length_collection + cylinder.length,
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
            rod_one.radius_collection,
            rod_one.length_collection,
            rod_one.tangent_collection,
            rod_one.velocity_collection,
            rod_one.internal_forces,
            rod_one.external_forces,
            rod_two.position_collection[..., :-1],
            rod_two.radius_collection,
            rod_two.length_collection,
            rod_two.tangent_collection,
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
            rod_one.radius_collection,
            rod_one.length_collection,
            rod_one.tangent_collection,
            rod_one.velocity_collection,
            rod_one.internal_forces,
            rod_one.external_forces,
            rod_two.position_collection[..., :-1],
            rod_two.radius_collection,
            rod_two.length_collection,
            rod_two.tangent_collection,
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
            rod_one.radius_collection,
            rod_one.length_collection,
            rod_one.tangent_collection,
            rod_one.velocity_collection,
            rod_one.internal_forces,
            rod_one.external_forces,
            rod_two.position_collection[..., :-1],
            rod_two.radius_collection,
            rod_two.length_collection,
            rod_two.tangent_collection,
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
    rod.radius_collection = np.array([1, 1, 1])
    rod.length_collection = np.array([3, 1, 3])
    rod.tangent_collection = np.array(
        [[1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    )
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
        rod.radius_collection,
        rod.length_collection,
        rod.tangent_collection,
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
