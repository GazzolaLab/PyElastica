__doc__ = """ Test Wrapper Classes used in contact in Elastica.contact_forces implementation"""

import numpy as np
from numpy.testing import assert_allclose
from elastica.utils import Tolerance
from elastica.contact_forces import (
    RodRodContact,
    RodCylinderContact,
    RodSelfContact,
    RodSphereContact,
    RodPlaneContact,
    RodPlaneContactWithAnisotropicFriction,
    CylinderPlaneContact,
)
from elastica.rod import RodBase
from elastica.rigidbody import Cylinder, Sphere
from elastica.surface import Plane
import pytest
from elastica.contact_utils import (
    _node_to_element_mass_or_force,
)


def mock_rod_init(self):

    "Initializing Rod"
    "Details of initialization are given in test_contact_specific_functions.py"

    self.n_elem = 2
    self.position_collection = np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0]])
    self.mass = np.ones(self.n_elem + 1)
    self.radius = np.array([1, 1])
    self.lengths = np.array([1, 1])
    self.tangents = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    self.internal_forces = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    self.external_forces = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    self.velocity_collection = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    self.omega_collection = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    self.external_torques = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    self.internal_torques = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])


def mock_cylinder_init(self):

    "Initializing Cylinder"
    "Details of initialization are given in test_contact_specific_functions.py"

    self.n_elems = 1
    self.position_collection = np.array([[0], [0], [0]])
    self.director_collection = np.array(
        [[[1.0], [0.0], [0.0]], [[0.0], [1.0], [0.0]], [[0.0], [0.0], [1.0]]]
    )
    self.radius = 1.0
    self.length = 2.0
    self.external_forces = np.array([[0.0], [0.0], [0.0]])
    self.external_torques = np.array([[0.0], [0.0], [0.0]])
    self.velocity_collection = np.array([[0.0], [0.0], [0.0]])


def mock_sphere_init(self):

    "Initializing Sphere"
    "Details of initialization are given in test_contact_specific_functions.py"

    self.n_elems = 1
    self.position_collection = np.array([[0], [0], [0]])
    self.director_collection = np.array(
        [[[1.0], [0.0], [0.0]], [[0.0], [1.0], [0.0]], [[0.0], [0.0], [1.0]]]
    )
    self.radius = 1.0
    self.velocity_collection = np.array([[0.0], [0.0], [0.0]])
    self.external_forces = np.array([[0.0], [0.0], [0.0]])
    self.external_torques = np.array([[0.0], [0.0], [0.0]])


def mock_plane_init(self):

    "Initializing Plane"

    self.normal = np.asarray([1.0, 0.0, 0.0]).reshape(3)
    self.origin = np.asarray([0.0, 0.0, 0.0]).reshape(3, 1)


MockRod = type("MockRod", (RodBase,), {"__init__": mock_rod_init})

MockCylinder = type("MockCylinder", (Cylinder,), {"__init__": mock_cylinder_init})

MockSphere = type("MockSphere", (Sphere,), {"__init__": mock_sphere_init})

MockPlane = type("MockPlane", (Plane,), {"__init__": mock_plane_init})


class TestRodCylinderContact:
    def test_check_systems_validity_with_invalid_systems(
        self,
    ):
        mock_rod = MockRod()
        mock_list = [1, 2, 3]
        mock_cylinder = MockCylinder()
        rod_cylinder_contact = RodCylinderContact(k=1.0, nu=0.0)

        "Testing Rod Cylinder Contact wrapper with incorrect type for second argument"
        with pytest.raises(TypeError) as excinfo:
            rod_cylinder_contact._check_systems_validity(mock_rod, mock_list)
        assert "System provided (list) must be derived from ['Cylinder']." == str(
            excinfo.value
        )

        with pytest.raises(TypeError) as excinfo:
            rod_cylinder_contact._check_systems_validity(mock_list, mock_rod)
        assert "System provided (list) must be derived from ['RodBase']." == str(
            excinfo.value
        )

        with pytest.raises(TypeError) as excinfo:
            rod_cylinder_contact._check_systems_validity(mock_cylinder, mock_rod)
        assert (
            "System provided (MockCylinder) must be derived from ['RodBase']."
            == str(excinfo.value)
        )

        with pytest.raises(TypeError) as excinfo:
            rod_cylinder_contact._check_systems_validity(mock_list, mock_cylinder)
        assert "System provided (list) must be derived from ['RodBase']." == str(
            excinfo.value
        )

    def test_contact_rod_cylinder_with_collision_with_k_without_nu_and_friction(
        self,
    ):

        # Testing Rod Cylinder Contact wrapper with Collision with analytical verified values

        mock_rod = MockRod()
        mock_cylinder = MockCylinder()
        rod_cylinder_contact = RodCylinderContact(k=1.0, nu=0.0)
        rod_cylinder_contact.apply_contact(mock_rod, mock_cylinder)

        # Details and reasoning about the values are given in 'test_contact_specific_functions.py/test_calculate_contact_forces_rod_cylinder()'
        assert_allclose(
            mock_rod.external_forces,
            np.array([[0.166666, 0.333333, 0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            atol=1e-6,
        )

        assert_allclose(
            mock_cylinder.external_forces, np.array([[-0.5], [0.0], [0.0]]), atol=1e-6
        )

        assert_allclose(
            mock_cylinder.external_torques, np.array([[0.0], [0.0], [0.0]]), atol=1e-6
        )

    def test_contact_rod_cylinder_with_collision_with_nu_without_k_and_friction(
        self,
    ):

        "Testing Rod Cylinder Contact wrapper with Collision with analytical verified values"

        mock_rod = MockRod()
        "Moving rod towards the cylinder with a velocity of -1 in x-axis"
        mock_rod.velocity_collection = np.array([[-1, 0, 0], [-1, 0, 0], [-1, 0, 0]])
        mock_cylinder = MockCylinder()
        "Moving cylinder towards the rod with a velocity of 1 in x-axis"
        mock_cylinder.velocity_collection = np.array([[1], [0], [0]])
        rod_cylinder_contact = RodCylinderContact(k=0.0, nu=1.0)
        rod_cylinder_contact.apply_contact(mock_rod, mock_cylinder)

        """Details and reasoning about the values are given in 'test_contact_specific_functions.py/test_calculate_contact_forces_rod_cylinder()'"""
        assert_allclose(
            mock_rod.external_forces,
            np.array([[0.5, 1, 0], [0, 0, 0], [0, 0, 0]]),
            atol=1e-6,
        )

        assert_allclose(
            mock_cylinder.external_forces, np.array([[-1.5], [0], [0]]), atol=1e-6
        )

        assert_allclose(
            mock_cylinder.external_torques, np.array([[0.0], [0.0], [0.0]]), atol=1e-6
        )

    def test_contact_rod_cylinder_with_collision_with_k_and_nu_without_friction(
        self,
    ):

        "Testing Rod Cylinder Contact wrapper with Collision with analytical verified values"

        mock_rod = MockRod()
        "Moving rod towards the cylinder with a velocity of -1 in x-axis"
        mock_rod.velocity_collection = np.array([[-1, 0, 0], [-1, 0, 0], [-1, 0, 0]])
        mock_cylinder = MockCylinder()
        "Moving cylinder towards the rod with a velocity of 1 in x-axis"
        mock_cylinder.velocity_collection = np.array([[1], [0], [0]])
        rod_cylinder_contact = RodCylinderContact(k=1.0, nu=1.0)
        rod_cylinder_contact.apply_contact(mock_rod, mock_cylinder)

        """Details and reasoning about the values are given in 'test_contact_specific_functions.py/test_calculate_contact_forces_rod_cylinder()'"""
        assert_allclose(
            mock_rod.external_forces,
            np.array([[0.666666, 1.333333, 0], [0, 0, 0], [0, 0, 0]]),
            atol=1e-6,
        )

        assert_allclose(
            mock_cylinder.external_forces, np.array([[-2], [0], [0]]), atol=1e-6
        )

        assert_allclose(
            mock_cylinder.external_torques, np.array([[0.0], [0.0], [0.0]]), atol=1e-6
        )

    def test_contact_rod_cylinder_with_collision_with_k_and_nu_and_friction(
        self,
    ):

        "Testing Rod Cylinder Contact wrapper with Collision with analytical verified values"

        mock_rod = MockRod()
        "Moving rod towards the cylinder with a velocity of -1 in x-axis"
        mock_rod.velocity_collection = np.array([[-1, 0, 0], [-1, 0, 0], [-1, 0, 0]])
        mock_cylinder = MockCylinder()
        "Moving cylinder towards the rod with a velocity of 1 in x-axis"
        mock_cylinder.velocity_collection = np.array([[1], [0], [0]])
        rod_cylinder_contact = RodCylinderContact(
            k=1.0, nu=1.0, velocity_damping_coefficient=0.1, friction_coefficient=0.1
        )
        rod_cylinder_contact.apply_contact(mock_rod, mock_cylinder)

        """Details and reasoning about the values are given in 'test_contact_specific_functions.py/test_calculate_contact_forces_rod_cylinder()'"""
        assert_allclose(
            mock_rod.external_forces,
            np.array(
                [
                    [0.666666, 1.333333, 0],
                    [0.033333, 0.066666, 0],
                    [0.033333, 0.066666, 0],
                ]
            ),
            atol=1e-6,
        )

        assert_allclose(
            mock_cylinder.external_forces, np.array([[-2], [-0.1], [-0.1]]), atol=1e-6
        )

        assert_allclose(
            mock_cylinder.external_torques, np.array([[0.0], [0.0], [0.0]]), atol=1e-6
        )

    def test_contact_rod_cylinder_without_collision(self):

        "Testing Rod Cylinder Contact wrapper without Collision with analytical verified values"

        mock_rod = MockRod()
        mock_cylinder = MockCylinder()
        rod_cylinder_contact = RodCylinderContact(k=1.0, nu=0.5)

        """Setting cylinder position such that there is no collision"""
        mock_cylinder.position_collection = np.array([[400], [500], [600]])
        mock_rod_external_forces_before_execution = mock_rod.external_forces.copy()
        mock_cylinder_external_forces_before_execution = (
            mock_cylinder.external_forces.copy()
        )
        mock_cylinder_external_torques_before_execution = (
            mock_cylinder.external_torques.copy()
        )
        rod_cylinder_contact.apply_contact(mock_rod, mock_cylinder)

        assert_allclose(
            mock_rod.external_forces, mock_rod_external_forces_before_execution
        )
        assert_allclose(
            mock_cylinder.external_forces,
            mock_cylinder_external_forces_before_execution,
        )
        assert_allclose(
            mock_cylinder.external_torques,
            mock_cylinder_external_torques_before_execution,
        )


class TestRodRodContact:
    def test_check_systems_validity_with_invalid_systems(
        self,
    ):
        mock_rod_one = MockRod()
        mock_list = [1, 2, 3]
        rod_rod_contact = RodRodContact(k=1.0, nu=0.0)

        # Testing Rod Rod Contact wrapper with incorrect type for second argument
        with pytest.raises(TypeError) as excinfo:
            rod_rod_contact._check_systems_validity(mock_rod_one, mock_list)
        assert "System provided (list) must be derived from ['RodBase']." == str(
            excinfo.value
        )

        # Testing Rod Rod Contact wrapper with incorrect type for first argument
        with pytest.raises(TypeError) as excinfo:
            rod_rod_contact._check_systems_validity(mock_list, mock_rod_one)
        assert "System provided (list) must be derived from ['RodBase']." == str(
            excinfo.value
        )

        # Testing Rod Rod Contact wrapper with same rod for both arguments
        with pytest.raises(TypeError) as excinfo:
            rod_rod_contact._check_systems_validity(mock_rod_one, mock_rod_one)
        assert (
            "First system is identical to second system. Systems must be distinct for contact."
            == str(excinfo.value)
        )

    def test_contact_with_two_rods_with_collision_with_k_without_nu(self):

        "Testing Rod Rod Contact wrapper with two rods with analytical verified values"
        "Test values have been copied from 'test_contact_specific_functions.py/test_calculate_contact_forces_rod_rod()'"

        mock_rod_one = MockRod()
        mock_rod_two = MockRod()
        mock_rod_two.position_collection = np.array([[4, 5, 6], [0, 0, 0], [0, 0, 0]])
        rod_rod_contact = RodRodContact(k=1.0, nu=0.0)
        rod_rod_contact.apply_contact(mock_rod_one, mock_rod_two)

        assert_allclose(
            mock_rod_one.external_forces,
            np.array([[0, -0.666666, -0.333333], [0, 0, 0], [0, 0, 0]]),
            atol=1e-6,
        )
        assert_allclose(
            mock_rod_two.external_forces,
            np.array([[0.333333, 0.666666, 0], [0, 0, 0], [0, 0, 0]]),
            atol=1e-6,
        )

    def test_contact_with_two_rods_with_collision_without_k_with_nu(self):

        "Testing Rod Rod Contact wrapper with two rods with analytical verified values"
        "Test values have been copied from 'test_contact_specific_functions.py/test_calculate_contact_forces_rod_rod()'"

        mock_rod_one = MockRod()
        mock_rod_two = MockRod()

        """Moving the rods towards each other with a velocity of 1 along the x-axis."""
        mock_rod_one.velocity_collection = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        mock_rod_two.velocity_collection = np.array(
            [[-1, 0, 0], [-1, 0, 0], [-1, 0, 0]]
        )
        mock_rod_two.position_collection = np.array([[4, 5, 6], [0, 0, 0], [0, 0, 0]])
        rod_rod_contact = RodRodContact(k=0.0, nu=1.0)
        rod_rod_contact.apply_contact(mock_rod_one, mock_rod_two)

        assert_allclose(
            mock_rod_one.external_forces,
            np.array(
                [[0, -0.333333, -0.166666], [0, 0, 0], [0, 0, 0]],
            ),
            atol=1e-6,
        )
        assert_allclose(
            mock_rod_two.external_forces,
            np.array([[0.166666, 0.333333, 0], [0, 0, 0], [0, 0, 0]]),
            atol=1e-6,
        )

    def test_contact_with_two_rods_with_collision_with_k_and_nu(self):

        "Testing RodRod Contact wrapper with two rods with analytical verified values"
        "Test values have been copied from 'test_contact_specific_functions.py/test_calculate_contact_forces_rod_rod()'"

        mock_rod_one = MockRod()
        mock_rod_two = MockRod()

        """Moving the rods towards each other with a velocity of 1 along the x-axis."""
        mock_rod_one.velocity_collection = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        mock_rod_two.velocity_collection = np.array(
            [[-1, 0, 0], [-1, 0, 0], [-1, 0, 0]]
        )
        mock_rod_two.position_collection = np.array([[4, 5, 6], [0, 0, 0], [0, 0, 0]])
        rod_rod_contact = RodRodContact(k=1.0, nu=1.0)
        rod_rod_contact.apply_contact(mock_rod_one, mock_rod_two)

        assert_allclose(
            mock_rod_one.external_forces,
            np.array(
                [[0, -1, -0.5], [0, 0, 0], [0, 0, 0]],
            ),
            atol=1e-6,
        )
        assert_allclose(
            mock_rod_two.external_forces,
            np.array([[0.5, 1, 0], [0, 0, 0], [0, 0, 0]]),
            atol=1e-6,
        )

    def test_contact_with_two_rods_without_collision(self):

        "Testing Rod Rod Contact wrapper with two rods with analytical verified values"

        mock_rod_one = MockRod()
        mock_rod_two = MockRod()

        "Setting rod two position such that there is no collision"
        mock_rod_two.position_collection = np.array(
            [[100, 101, 102], [0, 0, 0], [0, 0, 0]]
        )
        rod_rod_contact = RodRodContact(k=1.0, nu=1.0)
        mock_rod_one_external_forces_before_execution = (
            mock_rod_one.external_forces.copy()
        )
        mock_rod_two_external_forces_before_execution = (
            mock_rod_two.external_forces.copy()
        )
        rod_rod_contact.apply_contact(mock_rod_one, mock_rod_two)

        assert_allclose(
            mock_rod_one.external_forces, mock_rod_one_external_forces_before_execution
        )
        assert_allclose(
            mock_rod_two.external_forces, mock_rod_two_external_forces_before_execution
        )


class TestRodSelfContact:
    def test_check_systems_validity_with_invalid_systems(
        self,
    ):
        mock_rod_one = MockRod()
        mock_rod_two = MockRod()
        mock_list = [1, 2, 3]
        self_contact = RodSelfContact(k=1.0, nu=0.0)

        # Testing Self Contact wrapper with incorrect type for second argument
        with pytest.raises(TypeError) as excinfo:
            self_contact._check_systems_validity(mock_rod_one, mock_list)
        assert "System provided (list) must be derived from ['RodBase']." == str(
            excinfo.value
        )

        # Testing Self Contact wrapper with incorrect type for first argument
        with pytest.raises(TypeError) as excinfo:
            self_contact._check_systems_validity(mock_list, mock_rod_one)
        assert "System provided (list) must be derived from ['RodBase']." == str(
            excinfo.value
        )

        # Testing Self Contact wrapper with different rods
        with pytest.raises(TypeError) as excinfo:
            self_contact._check_systems_validity(mock_rod_one, mock_rod_two)
        assert "First system must be identical to the second system." == str(
            excinfo.value
        )

    def test_self_contact_with_rod_self_collision(self):

        "Testing Self Contact wrapper rod self collision with analytical verified values"

        mock_rod = MockRod()

        "Test values have been copied from 'test_contact_specific_functions.py/test_calculate_contact_forces_self_rod()'"
        mock_rod.n_elems = 3
        mock_rod.position_collection = np.array(
            [[1, 4, 4, 1], [0, 0, 1, 1], [0, 0, 0, 0]]
        )
        mock_rod.radius = np.array([1, 1, 1])
        mock_rod.lengths = np.array([3, 1, 3])
        mock_rod.tangents = np.array(
            [[1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )
        mock_rod.velocity_collection = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )
        mock_rod.internal_forces = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )
        mock_rod.external_forces = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )
        self_contact = RodSelfContact(k=1.0, nu=0.0)
        self_contact.apply_contact(mock_rod, mock_rod)

        assert_allclose(
            mock_rod.external_forces,
            np.array(
                [[0, 0, 0, 0], [-0.333333, -0.666666, 0.666666, 0.333333], [0, 0, 0, 0]]
            ),
            atol=1e-6,
        )

    def test_self_contact_with_rod_no_self_collision(self):

        "Testing Self Contact wrapper rod no self collision with analytical verified values"

        mock_rod = MockRod()

        "the initially set rod does not have self collision"
        mock_rod_external_forces_before_execution = mock_rod.external_forces.copy()
        self_contact = RodSelfContact(k=1.0, nu=1.0)
        self_contact.apply_contact(mock_rod, mock_rod)

        assert_allclose(
            mock_rod.external_forces, mock_rod_external_forces_before_execution
        )


class TestRodSphereContact:
    def test_check_systems_validity_with_invalid_systems(
        self,
    ):
        mock_rod = MockRod()
        mock_list = [1, 2, 3]
        mock_sphere = MockSphere()
        rod_sphere_contact = RodSphereContact(k=1.0, nu=0.0)

        # Testing Rod Sphere Contact wrapper with incorrect type for second argument
        with pytest.raises(TypeError) as excinfo:
            rod_sphere_contact._check_systems_validity(mock_rod, mock_list)
        assert "System provided (list) must be derived from ['Sphere']." == str(
            excinfo.value
        )

        # Testing Rod Sphere Contact wrapper with incorrect type for first argument
        with pytest.raises(TypeError) as excinfo:
            rod_sphere_contact._check_systems_validity(mock_list, mock_rod)
        assert "System provided (list) must be derived from ['RodBase']." == str(
            excinfo.value
        )

        # Testing Rod Sphere Contact wrapper with incorrect order
        with pytest.raises(TypeError) as excinfo:
            rod_sphere_contact._check_systems_validity(mock_sphere, mock_rod)
        assert "System provided (MockSphere) must be derived from ['RodBase']." == str(
            excinfo.value
        )

    def test_contact_rod_sphere_with_collision_with_k_without_nu_and_friction(
        self,
    ):

        # "Testing Rod Sphere Contact wrapper with Collision with analytical verified values

        mock_rod = MockRod()
        mock_sphere = MockSphere()
        rod_sphere_contact = RodSphereContact(k=1.0, nu=0.0)
        rod_sphere_contact.apply_contact(mock_rod, mock_sphere)

        # Details and reasoning about the values are given in 'test_contact_specific_functions.py/test_calculate_contact_forces_rod_sphere_with_k_without_nu_and_friction()'
        assert_allclose(
            mock_sphere.external_forces, np.array([[-0.5], [0], [0]]), atol=1e-6
        )
        assert_allclose(
            mock_sphere.external_torques, np.array([[0], [0], [0]]), atol=1e-6
        )
        assert_allclose(
            mock_rod.external_forces,
            np.array([[0.166666, 0.333333, 0], [0, 0, 0], [0, 0, 0]]),
            atol=1e-6,
        )


class TestRodPlaneContact:
    def initializer(
        self,
        shift=0.0,
        k_w=0.0,
        nu_w=0.0,
        plane_normal=np.array([0.0, 1.0, 0.0]),
    ):
        # create rod
        rod = MockRod()
        start = np.array([0.0, 0.0, 0.0])
        direction = np.array([0.0, 0.0, 1.0])
        end = start + 1.0 * direction
        rod.position_collection = np.zeros((3, 3))
        for i in range(0, 3):
            rod.position_collection[i, ...] = np.linspace(start[i], end[i], num=3)
        rod.director_collection = np.repeat(np.identity(3)[:, :, np.newaxis], 2, axis=2)
        rod.lengths = np.ones(2) * 1.0 / 2
        rod.radius = np.repeat(np.array([0.25]), 2, axis=0)
        rod.tangents = np.repeat(direction[:, np.newaxis], 2, axis=1)

        # create plane
        plane = MockPlane()
        plane.origin = np.array([0.0, -rod.radius[0] + shift, 0.0]).reshape(3, 1)
        plane.normal = plane_normal.reshape(
            3,
        )
        rod_plane_contact = RodPlaneContact(k_w, nu_w)

        fnormal = -10.0 * np.sign(plane_normal[1]) * np.random.random_sample(1).item()
        external_forces = np.repeat(
            np.array([0.0, fnormal, 0.0]).reshape(3, 1), 3, axis=1
        )
        external_forces[..., 0] *= 0.5
        external_forces[..., -1] *= 0.5
        rod.external_forces = external_forces.copy()

        return rod, plane, rod_plane_contact, external_forces

    def test_check_systems_validity_with_invalid_systems(
        self,
    ):
        mock_rod = MockRod()
        mock_plane = MockPlane()
        mock_list = [1, 2, 3]
        rod_plane_contact = RodPlaneContact(k=1.0, nu=0.0)

        # Testing Rod Plane Contact wrapper with incorrect type for second argument
        with pytest.raises(TypeError) as excinfo:
            rod_plane_contact._check_systems_validity(mock_rod, mock_list)
        assert "System provided (list) must be derived from ['SurfaceBase']." == str(
            excinfo.value
        )

        # Testing Rod Plane Contact wrapper with incorrect type for first argument
        with pytest.raises(TypeError) as excinfo:
            rod_plane_contact._check_systems_validity(mock_list, mock_plane)
        assert "System provided (list) must be derived from ['RodBase']." == str(
            excinfo.value
        )

    def test_rod_plane_contact_without_contact(self):
        """
        This test case tests the forces on rod, when there is no
        contact between rod and the plane.

        """

        shift = -(
            (2.0 - 1.0) * np.random.random_sample(1) + 1.0
        ).item()  # we move plane away from rod
        print("q")
        [rod, plane, rod_plane_contact, external_forces] = self.initializer(shift)

        rod_plane_contact.apply_contact(rod, plane)
        correct_forces = external_forces  # since no contact
        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

    def test_rod_plane_contact_without_k_and_nu(self):
        """
        This function tests wall response  on the rod. Here
        wall stiffness coefficient and damping coefficient set
        to zero to check only sum of all forces on the rod.

        """

        [rod, plane, rod_plane_contact, external_forces] = self.initializer()

        rod_plane_contact.apply_contact(rod, plane)

        correct_forces = np.zeros((3, 3))
        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("k_w", [0.1, 0.5, 1.0, 2, 10])
    def test_rod_plane_contact_with_k_without_nu(self, k_w):
        """
        Here wall stiffness coefficient changed parametrically
        and damping coefficient set to zero .
        Parameters
        ----------
        k_w


        """

        shift = np.random.random_sample(1).item()  # we move plane towards to rod
        [rod, plane, rod_plane_contact, external_forces] = self.initializer(
            shift=shift, k_w=k_w
        )
        correct_forces = k_w * np.repeat(
            np.array([0.0, shift, 0.0]).reshape(3, 1), 3, axis=1
        )
        correct_forces[..., 0] *= 0.5
        correct_forces[..., -1] *= 0.5

        rod_plane_contact.apply_contact(rod, plane)

        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("nu_w", [0.5, 1.0, 5.0, 7.0, 12.0])
    def test_rod_plane_contact_without_k_with_nu(self, nu_w):
        """
        Here wall damping coefficient are changed parametrically and
        wall response functions tested.
        Parameters
        ----------
        nu_w
        """

        [rod, plane, rod_plane_contact, external_forces] = self.initializer(nu_w=nu_w)

        normal_velocity = np.random.random_sample(1).item()
        rod.velocity_collection[..., :] += np.array(
            [0.0, -normal_velocity, 0.0]
        ).reshape(3, 1)

        correct_forces = np.repeat(
            (nu_w * np.array([0.0, normal_velocity, 0.0])).reshape(3, 1),
            3,
            axis=1,
        )

        correct_forces[..., 0] *= 0.5
        correct_forces[..., -1] *= 0.5

        rod_plane_contact.apply_contact(rod, plane)

        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

    def test_rod_plane_contact_when_rod_is_under_plane(self):
        """
        This test case tests plane response forces on the rod
        in the case rod is under the plane and pushed towards
        the plane.

        """

        # we move plane on top of the rod. Note that 0.25 is radius of the rod.
        offset_of_plane_with_respect_to_rod = 2.0 * 0.25

        # plane normal changed, it is towards the negative direction, because rod
        # is under the plane.
        plane_normal = np.array([0.0, -1.0, 0.0])

        [rod, plane, rod_plane_contact, external_forces] = self.initializer(
            shift=offset_of_plane_with_respect_to_rod, plane_normal=plane_normal
        )

        rod_plane_contact.apply_contact(rod, plane)
        correct_forces = np.zeros((3, 3))
        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("k_w", [0.1, 0.5, 1.0, 2, 10])
    def test_rod_plane_contact_when_rod_is_under_plane_with_k_without_nu(self, k_w):
        """
        In this test case we move the rod under the plane.
        Here wall stiffness coefficient changed parametrically
        and damping coefficient set to zero .
        Parameters
        ----------
        k_w

        """
        # we move plane on top of the rod. Note that 0.25 is radius of the rod.
        offset_of_plane_with_respect_to_rod = 2.0 * 0.25

        # we move plane towards to rod by random distance
        shift = offset_of_plane_with_respect_to_rod - np.random.random_sample(1).item()

        # plane normal changed, it is towards the negative direction, because rod
        # is under the plane.
        plane_normal = np.array([0.0, -1.0, 0.0])

        [rod, plane, rod_plane_contact, external_forces] = self.initializer(
            shift=shift, k_w=k_w, plane_normal=plane_normal
        )

        # we have to substract rod offset because top part
        correct_forces = k_w * np.repeat(
            np.array([0.0, shift - offset_of_plane_with_respect_to_rod, 0.0]).reshape(
                3, 1
            ),
            3,
            axis=1,
        )
        correct_forces[..., 0] *= 0.5
        correct_forces[..., -1] *= 0.5

        rod_plane_contact.apply_contact(rod, plane)

        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("nu_w", [0.5, 1.0, 5.0, 7.0, 12.0])
    def test_rod_plane_contact_when_rod_is_under_plane_without_k_with_nu(self, nu_w):
        """
        In this test case we move under the plane and test damping force.
        Here wall damping coefficient are changed parametrically and
        wall response functions tested.
        Parameters
        ----------
        nu_w

        """
        # we move plane on top of the rod. Note that 0.25 is radius of the rod.
        offset_of_plane_with_respect_to_rod = 2.0 * 0.25

        # plane normal changed, it is towards the negative direction, because rod
        # is under the plane.
        plane_normal = np.array([0.0, -1.0, 0.0])

        [rod, plane, rod_plane_contact, external_forces] = self.initializer(
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
            3,
            axis=1,
        )

        correct_forces[..., 0] *= 0.5
        correct_forces[..., -1] *= 0.5

        rod_plane_contact.apply_contact(rod, plane)

        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())


class TestRodPlaneWithAnisotropicFriction:
    def initializer(
        self,
        static_mu_array=np.array([0.0, 0.0, 0.0]),
        kinetic_mu_array=np.array([0.0, 0.0, 0.0]),
        force_mag_long=0.0,  # forces along the rod
        force_mag_side=0.0,  # side forces on the rod
    ):

        # create rod
        rod = MockRod()
        start = np.array([0.0, 0.0, 0.0])
        direction = np.array([0.0, 0.0, 1.0])
        end = start + 1.0 * direction
        rod.position_collection = np.zeros((3, 3))
        for i in range(0, 3):
            rod.position_collection[i, ...] = np.linspace(start[i], end[i], num=3)
        rod.director_collection = np.repeat(np.identity(3)[:, :, np.newaxis], 2, axis=2)
        rod.lengths = np.ones(2) * 1.0 / 2
        rod.radius = np.repeat(np.array([0.25]), 2, axis=0)
        rod.tangents = np.repeat(direction[:, np.newaxis], 2, axis=1)

        # create plane
        plane = MockPlane()
        plane.origin = np.array([0.0, -rod.radius[0], 0.0]).reshape(3, 1)
        plane.normal = np.array([0.0, 1.0, 0.0]).reshape(
            3,
        )
        slip_velocity_tol = 1e-2
        rod_plane_contact = RodPlaneContactWithAnisotropicFriction(
            0.0,
            0.0,
            slip_velocity_tol,
            static_mu_array,  # forward, backward, sideways
            kinetic_mu_array,  # forward, backward, sideways
        )

        fnormal = (10.0 - 5.0) * np.random.random_sample(
            1
        ).item() + 5.0  # generates random numbers [5.0,10)
        external_forces = np.array([force_mag_side, -fnormal, force_mag_long])

        external_forces_collection = np.repeat(external_forces.reshape(3, 1), 3, axis=1)
        external_forces_collection[..., 0] *= 0.5
        external_forces_collection[..., -1] *= 0.5
        rod.external_forces = external_forces_collection.copy()

        return rod, plane, rod_plane_contact, external_forces_collection

    def test_check_systems_validity_with_invalid_systems(
        self,
    ):
        mock_rod = MockRod()
        mock_plane = MockPlane()
        mock_list = [1, 2, 3]
        rod_plane_contact = RodPlaneContactWithAnisotropicFriction(
            0.0,
            0.0,
            1e-2,
            np.array([0.0, 0.0, 0.0]),  # forward, backward, sideways
            np.array([0.0, 0.0, 0.0]),  # forward, backward, sideways
        )

        # Testing Rod Plane Contact wrapper with incorrect type for second argument
        with pytest.raises(TypeError) as excinfo:
            rod_plane_contact._check_systems_validity(mock_rod, mock_list)
        assert "System provided (list) must be derived from ['SurfaceBase']." == str(
            excinfo.value
        )

        # Testing Rod Plane wrapper with incorrect type for first argument
        with pytest.raises(TypeError) as excinfo:
            rod_plane_contact._check_systems_validity(mock_list, mock_plane)
        assert "System provided (list) must be derived from ['RodBase']." == str(
            excinfo.value
        )

    @pytest.mark.parametrize("velocity", [-1.0, -3.0, 1.0, 5.0, 2.0])
    def test_axial_kinetic_friction(self, velocity):
        """
        This function tests kinetic friction in forward and backward direction.
        All other friction coefficients set to zero.
        Parameters
        ----------
        velocity



        """

        [rod, plane, rod_plane_contact, external_forces_collection] = self.initializer(
            kinetic_mu_array=np.array([1.0, 1.0, 0.0])
        )

        rod.velocity_collection += np.array([0.0, 0.0, velocity]).reshape(3, 1)

        rod_plane_contact.apply_contact(rod, plane)

        direction_collection = np.repeat(
            np.array([0.0, 0.0, 1.0]).reshape(3, 1), 3, axis=1
        )
        correct_forces = (
            -1.0
            * np.sign(velocity)
            * np.linalg.norm(external_forces_collection, axis=0)
            * direction_collection
        )
        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("force_mag", [-1.0, -3.0, 1.0, 5.0, 2.0])
    def test_axial_static_friction_total_force_smaller_than_static_friction_force(
        self, force_mag
    ):
        """
        This test is for static friction when total forces applied
        on the rod is smaller than the static friction force.
        Fx < F_normal*mu_s
        Parameters
        ----------
        force_mag
        """
        [rod, plane, rod_plane_contact, external_forces_collection] = self.initializer(
            static_mu_array=np.array([1.0, 1.0, 0.0]), force_mag_long=force_mag
        )

        rod_plane_contact.apply_contact(rod, plane)
        correct_forces = np.zeros((3, 3))
        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("force_mag", [-20.0, -15.0, 15.0, 20.0])
    def test_axial_static_friction_total_force_larger_than_static_friction_force(
        self, force_mag
    ):
        """
        This test is for static friction when total forces applied
        on the rod is larger than the static friction force.
        Fx > F_normal*mu_s
        Parameters
        ----------
        force_mag


        """

        [rod, plane, rod_plane_contact, external_forces_collection] = self.initializer(
            static_mu_array=np.array([1.0, 1.0, 0.0]), force_mag_long=force_mag
        )

        rod_plane_contact.apply_contact(rod, plane)
        correct_forces = np.zeros((3, 3))
        if np.sign(force_mag) < 0:
            correct_forces[2] = (
                external_forces_collection[2]
            ) - 1.0 * external_forces_collection[1]
        else:
            correct_forces[2] = (
                external_forces_collection[2]
            ) + 1.0 * external_forces_collection[1]

        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("velocity", [-1.0, -3.0, 1.0, 2.0, 5.0])
    @pytest.mark.parametrize("omega", [-5.0, -2.0, 0.0, 4.0, 6.0])
    def test_kinetic_rolling_friction(self, velocity, omega):
        """
        This test is for testing kinetic rolling friction,
        for different translational and angular velocities,
        we compute the final external forces and torques on the rod
        using apply friction function and compare results with
        analytical solutions.
        Parameters
        ----------
        velocity
        omega

        """
        [rod, plane, rod_plane_contact, external_forces_collection] = self.initializer(
            kinetic_mu_array=np.array([0.0, 0.0, 1.0])
        )

        rod.velocity_collection += np.array([velocity, 0.0, 0.0]).reshape(3, 1)
        rod.omega_collection += np.array([0.0, 0.0, omega]).reshape(3, 1)

        rod_plane_contact.apply_contact(rod, plane)

        correct_forces = np.zeros((3, 3))
        correct_forces[0] = (
            -1.0
            * np.sign(velocity + omega * rod.radius[0])
            * np.fabs(external_forces_collection[1])
        )

        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())
        forces_on_elements = _node_to_element_mass_or_force(external_forces_collection)
        correct_torques = np.zeros((3, 2))
        correct_torques[2] += (
            -1.0
            * np.sign(velocity + omega * rod.radius[0])
            * np.fabs(forces_on_elements[1])
            * rod.radius
        )

        assert_allclose(correct_torques, rod.external_torques, atol=Tolerance.atol())

    @pytest.mark.parametrize("force_mag", [-20.0, -15.0, 15.0, 20.0])
    def test_static_rolling_friction_total_force_smaller_than_static_friction_force(
        self, force_mag
    ):
        """
        In this test case static rolling friction force is tested. We set external and internal torques to
        zero and only changed the force in rolling direction. In this test case, total force in rolling direction
        is smaller than static friction force in rolling direction. Next test case will check what happens if
        total forces in rolling direction larger than static friction force.
        Parameters
        ----------
        force_mag


        """

        [rod, plane, rod_plane_contact, external_forces_collection] = self.initializer(
            static_mu_array=np.array([0.0, 0.0, 10.0]), force_mag_side=force_mag
        )

        rod_plane_contact.apply_contact(rod, plane)

        correct_forces = np.zeros((3, 3))
        correct_forces[0] = 2.0 / 3.0 * external_forces_collection[0]
        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

        forces_on_elements = _node_to_element_mass_or_force(external_forces_collection)
        correct_torques = np.zeros((3, 2))
        correct_torques[2] += (
            -1.0
            * np.sign(forces_on_elements[0])
            * np.fabs(forces_on_elements[0])
            * rod.radius
            / 3.0
        )

        assert_allclose(correct_torques, rod.external_torques, atol=Tolerance.atol())

    @pytest.mark.parametrize("force_mag", [-100.0, -80.0, 65.0, 95.0])
    def test_static_rolling_friction_total_force_larger_than_static_friction_force(
        self, force_mag
    ):
        """
        In this test case static rolling friction force is tested. We set external and internal torques to
        zero and only changed the force in rolling direction. In this test case, total force in rolling direction
        is larger than static friction force in rolling direction.
        Parameters
        ----------
        force_mag


        """

        [rod, plane, rod_plane_contact, external_forces_collection] = self.initializer(
            static_mu_array=np.array([0.0, 0.0, 1.0]), force_mag_side=force_mag
        )

        rod_plane_contact.apply_contact(rod, plane)

        correct_forces = np.zeros((3, 3))
        correct_forces[0] = external_forces_collection[0] - np.sign(
            external_forces_collection[0]
        ) * np.fabs(external_forces_collection[1])
        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

        forces_on_elements = _node_to_element_mass_or_force(external_forces_collection)
        correct_torques = np.zeros((3, 2))
        correct_torques[2] += (
            -1.0
            * np.sign(forces_on_elements[0])
            * np.fabs(forces_on_elements[1])
            * rod.radius
        )

        assert_allclose(correct_torques, rod.external_torques, atol=Tolerance.atol())

    @pytest.mark.parametrize("torque_mag", [-3.0, -1.0, 2.0, 3.5])
    def test_static_rolling_friction_total_torque_smaller_than_static_friction_force(
        self, torque_mag
    ):
        """
        In this test case, static rolling friction force tested with zero internal and external force and
        with non-zero external torque. Here torque magnitude chosen such that total rolling force is
        always smaller than the static friction force.
        Parameters
        ----------
        torque_mag
        """

        [rod, plane, rod_plane_contact, external_forces_collection] = self.initializer(
            static_mu_array=np.array([0.0, 0.0, 10.0])
        )

        external_torques = np.zeros((3, 2))
        external_torques[2] = torque_mag
        rod.external_torques = external_torques.copy()

        rod_plane_contact.apply_contact(rod, plane)

        correct_forces = np.zeros((3, 3))
        correct_forces[0, :-1] -= external_torques[2] / (3.0 * rod.radius)
        correct_forces[0, 1:] -= external_torques[2] / (3.0 * rod.radius)
        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

        correct_torques = np.zeros((3, 2))
        correct_torques[2] += external_torques[2] - 2.0 / 3.0 * external_torques[2]

        assert_allclose(correct_torques, rod.external_torques, atol=Tolerance.atol())

    @pytest.mark.parametrize("torque_mag", [-10.0, -5.0, 6.0, 7.5])
    def test_static_rolling_friction_total_torque_larger_than_static_friction_force(
        self, torque_mag
    ):
        """
        In this test case, static rolling friction force tested with zero internal and external force and
        with non-zero external torque. Here torque magnitude chosen such that total rolling force is
        always larger than the static friction force. Thus, lateral friction force will be equal to static
        friction force.
        Parameters
        ----------
        torque_mag

        """

        [rod, plane, rod_plane_contact, external_forces_collection] = self.initializer(
            static_mu_array=np.array([0.0, 0.0, 1.0])
        )

        external_torques = np.zeros((3, 2))
        external_torques[2] = torque_mag
        rod.external_torques = external_torques.copy()

        rod_plane_contact.apply_contact(rod, plane)

        correct_forces = np.zeros((3, 3))
        correct_forces[0] = (
            -1.0 * np.sign(torque_mag) * np.fabs(external_forces_collection[1])
        )
        assert_allclose(correct_forces, rod.external_forces, atol=Tolerance.atol())

        forces_on_elements = _node_to_element_mass_or_force(external_forces_collection)
        correct_torques = external_torques
        correct_torques[2] += -(
            np.sign(torque_mag) * np.fabs(forces_on_elements[1]) * rod.radius
        )

        assert_allclose(correct_torques, rod.external_torques, atol=Tolerance.atol())


class TestCylinderPlaneContact:
    def initializer(
        self,
        shift=0.0,
        k_w=0.0,
        nu_w=0.0,
        plane_normal=np.array([0.0, 1.0, 0.0]),
    ):
        # create cylinder
        cylinder = MockCylinder()

        # create plane
        plane = MockPlane()
        plane.origin = np.array([0.0, -cylinder.radius + shift, 0.0]).reshape(3, 1)
        plane.normal = plane_normal.reshape(
            3,
        )
        cylinder_plane_contact = CylinderPlaneContact(k_w, nu_w)

        fnormal = -10.0 * np.sign(plane_normal[1]) * np.random.random_sample(1).item()
        external_forces = np.array([0.0, fnormal, 0.0]).reshape(3, 1)
        cylinder.external_forces = external_forces.copy()

        return cylinder, plane, cylinder_plane_contact, external_forces

    def test_check_systems_validity_with_invalid_systems(
        self,
    ):
        mock_cylinder = MockCylinder()
        mock_plane = MockPlane()
        mock_list = [1, 2, 3]
        cylinder_plane_contact = CylinderPlaneContact(k=1.0, nu=0.0)

        # Testing Cylinder Plane Contact wrapper with incorrect type for second argument
        with pytest.raises(TypeError) as excinfo:
            cylinder_plane_contact._check_systems_validity(mock_cylinder, mock_list)
        assert "System provided (list) must be derived from ['SurfaceBase']." == str(
            excinfo.value
        )

        # Testing Cylinder Plane wrapper with incorrect type for first argument
        with pytest.raises(TypeError) as excinfo:
            cylinder_plane_contact._check_systems_validity(mock_list, mock_plane)
        assert "System provided (list) must be derived from ['Cylinder']." == str(
            excinfo.value
        )

    def test_cylinder_plane_contact_without_contact(self):
        """
        This test case tests the forces on cylinder, when there is no
        contact between cylinder and the plane.

        """

        shift = -(
            (2.0 - 1.0) * np.random.random_sample(1) + 1.0
        ).item()  # we move plane away from cylinder
        print("q")
        [cylinder, plane, cylinder_plane_contact, external_forces] = self.initializer(
            shift
        )

        cylinder_plane_contact.apply_contact(cylinder, plane)
        correct_forces = external_forces  # since no contact
        assert_allclose(correct_forces, cylinder.external_forces, atol=Tolerance.atol())

    def test_cylinder_plane_contact_without_k_and_nu(self):
        """
        This function tests wall response on the cylinder. Here
        wall stiffness coefficient and damping coefficient set
        to zero to check only sum of all forces on the cylinder.

        """

        [cylinder, plane, cylinder_plane_contact, external_forces] = self.initializer()

        cylinder_plane_contact.apply_contact(cylinder, plane)

        correct_forces = np.zeros((3, 1))
        assert_allclose(correct_forces, cylinder.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("k_w", [0.1, 0.5, 1.0, 2, 10])
    def test_cylinder_plane_contact_with_k_without_nu(self, k_w):
        """
        Here wall stiffness coefficient changed parametrically
        and damping coefficient set to zero .
        Parameters
        ----------
        k_w


        """

        shift = np.random.random_sample(1).item()  # we move plane towards to cylinder
        [cylinder, plane, cylinder_plane_contact, external_forces] = self.initializer(
            shift=shift, k_w=k_w
        )
        correct_forces = k_w * np.array([0.0, shift, 0.0]).reshape(3, 1)

        cylinder_plane_contact.apply_contact(cylinder, plane)

        assert_allclose(correct_forces, cylinder.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("nu_w", [0.5, 1.0, 5.0, 7.0, 12.0])
    def test_cylinder_plane_contact_without_k_with_nu(self, nu_w):
        """
        Here wall damping coefficient are changed parametrically and
        wall response functions tested.
        Parameters
        ----------
        nu_w
        """

        [cylinder, plane, cylinder_plane_contact, external_forces] = self.initializer(
            nu_w=nu_w
        )

        normal_velocity = np.random.random_sample(1).item()
        cylinder.velocity_collection[..., :] += np.array(
            [0.0, -normal_velocity, 0.0]
        ).reshape(3, 1)

        correct_forces = nu_w * np.array([0.0, normal_velocity, 0.0]).reshape(3, 1)

        cylinder_plane_contact.apply_contact(cylinder, plane)

        assert_allclose(correct_forces, cylinder.external_forces, atol=Tolerance.atol())

    def test_cylinder_plane_contact_when_cylinder_is_under_plane(self):
        """
        This test case tests plane response forces on the cylinder
        in the case cylinder is under the plane and pushed towards
        the plane.

        """

        # we move plane on top of the cylinder. Note that 1.0 is radius of the cylinder.
        offset_of_plane_with_respect_to_cylinder = 2.0 * 1.0

        # plane normal changed, it is towards the negative direction, because cylinder
        # is under the plane.
        plane_normal = np.array([0.0, -1.0, 0.0])

        [cylinder, plane, cylinder_plane_contact, external_forces] = self.initializer(
            shift=offset_of_plane_with_respect_to_cylinder, plane_normal=plane_normal
        )

        cylinder_plane_contact.apply_contact(cylinder, plane)
        correct_forces = np.zeros((3, 1))
        assert_allclose(correct_forces, cylinder.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("k_w", [0.1, 0.5, 1.0, 2, 10])
    def test_cylinder_plane_contact_when_cylinder_is_under_plane_with_k_without_nu(
        self, k_w
    ):
        """
        In this test case we move the cylinder under the plane.
        Here wall stiffness coefficient changed parametrically
        and damping coefficient set to zero .
        Parameters
        ----------
        k_w

        """
        # we move plane on top of the cylinder. Note that 1.0 is radius of the cylinder.
        offset_of_plane_with_respect_to_cylinder = 2.0 * 1.0

        # we move plane towards to cylinder by random distance
        shift = (
            offset_of_plane_with_respect_to_cylinder - np.random.random_sample(1).item()
        )

        # plane normal changed, it is towards the negative direction, because cylinder
        # is under the plane.
        plane_normal = np.array([0.0, -1.0, 0.0])

        [cylinder, plane, cylinder_plane_contact, external_forces] = self.initializer(
            shift=shift, k_w=k_w, plane_normal=plane_normal
        )

        # we have to substract cylinder offset because top part
        correct_forces = k_w * np.array(
            [0.0, shift - offset_of_plane_with_respect_to_cylinder, 0.0]
        ).reshape(3, 1)

        cylinder_plane_contact.apply_contact(cylinder, plane)

        assert_allclose(correct_forces, cylinder.external_forces, atol=Tolerance.atol())

    @pytest.mark.parametrize("nu_w", [0.5, 1.0, 5.0, 7.0, 12.0])
    def test_cylinder_plane_contact_when_cylinder_is_under_plane_without_k_with_nu(
        self, nu_w
    ):
        """
        In this test case we move under the plane and test damping force.
        Here wall damping coefficient are changed parametrically and
        wall response functions tested.
        Parameters
        ----------
        nu_w

        """
        # we move plane on top of the cylinder. Note that 1.0 is radius of the cylinder.
        offset_of_plane_with_respect_to_cylinder = 2.0 * 1.0

        # plane normal changed, it is towards the negative direction, because cylinder
        # is under the plane.
        plane_normal = np.array([0.0, -1.0, 0.0])

        [cylinder, plane, cylinder_plane_contact, external_forces] = self.initializer(
            shift=offset_of_plane_with_respect_to_cylinder,
            nu_w=nu_w,
            plane_normal=plane_normal,
        )

        normal_velocity = np.random.random_sample(1).item()
        cylinder.velocity_collection[..., :] += np.array(
            [0.0, -normal_velocity, 0.0]
        ).reshape(3, 1)

        correct_forces = nu_w * np.array([0.0, normal_velocity, 0.0]).reshape(3, 1)
        cylinder_plane_contact.apply_contact(cylinder, plane)

        assert_allclose(correct_forces, cylinder.external_forces, atol=Tolerance.atol())
