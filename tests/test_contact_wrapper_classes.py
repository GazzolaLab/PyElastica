__doc__ = """ Test Wrapper Classes used in contact in Elastica.contact_forces implementation"""

import numpy as np
from numpy.testing import assert_allclose
from elastica.contact_forces import RodRodContact, RodCylinderContact, RodSelfContact
from elastica.typing import RodBase
from elastica.rigidbody import Cylinder
import pytest


def mock_rod_init(self):

    "Initializing Rod"
    "Details of initialization are given in test_contact_specific_functions.py"

    self.n_elems = 2
    self.position_collection = np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0]])
    self.radius = np.array([1, 1])
    self.lengths = np.array([1, 1])
    self.tangents = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    self.internal_forces = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    self.external_forces = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    self.velocity_collection = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )


def mock_cylinder_init(self):

    "Initializing Cylinder"
    "Details of initialization are given in test_contact_specific_functions.py"

    self.n_elems = 1
    self.position_collection = np.array([[0], [0], [0]])
    self.director_collection = np.array(
        [[[1.0], [0.0], [0.0]], [[0.0], [1.0], [0.0]], [[0.0], [0.0], [1.0]]]
    )
    self.radius = np.array([1.0])
    self.length = np.array([2.0])
    self.external_forces = np.array([[0.0], [0.0], [0.0]])
    self.external_torques = np.array([[0.0], [0.0], [0.0]])
    self.velocity_collection = np.array([[0.0], [0.0], [0.0]])


MockRod = type("MockRod", (RodBase,), {"__init__": mock_rod_init})

MockCylinder = type("MockCylinder", (Cylinder,), {"__init__": mock_cylinder_init})


class TestRodCylinderContact:
    def test_check_incorrect_order_type(
        self,
    ):
        mock_rod = MockRod()
        mock_list = [1, 2, 3]
        mock_cylinder = MockCylinder()
        rod_cylinder_contact = RodCylinderContact(k=1.0, nu=0.0)

        "Testing Rod Cylinder Contact wrapper with incorrect type for second argument"
        with pytest.raises(TypeError) as excinfo:
            rod_cylinder_contact._check_order_and_type(mock_rod, mock_list)
        assert (
            "Systems provided to the contact class have incorrect order/type. \n"
            " First system is {0} and second system is {1}. \n"
            " First system should be a rod, second should be a cylinder"
        ).format(mock_rod.__class__, mock_list.__class__) == str(excinfo.value)

        "Testing Rod Cylinder Contact wrapper with incorrect type for first argument"
        with pytest.raises(TypeError) as excinfo:
            rod_cylinder_contact._check_order_and_type(mock_list, mock_rod)
        assert (
            "Systems provided to the contact class have incorrect order/type. \n"
            " First system is {0} and second system is {1}. \n"
            " First system should be a rod, second should be a cylinder"
        ).format(mock_list.__class__, mock_rod.__class__) == str(excinfo.value)

        "Testing Rod Cylinder Contact wrapper with incorrect order"
        with pytest.raises(TypeError) as excinfo:
            rod_cylinder_contact._check_order_and_type(mock_cylinder, mock_rod)
            print(excinfo.value)
        assert (
            "Systems provided to the contact class have incorrect order/type. \n"
            " First system is {0} and second system is {1}. \n"
            " First system should be a rod, second should be a cylinder"
        ).format(mock_cylinder.__class__, mock_rod.__class__) == str(excinfo.value)

    def test_contact_rod_Cylinder_with_collision_with_k_without_nu_and_friction(
        self,
    ):

        "Testing Rod Cylinder Contact wrapper with Collision with analytical verified values"

        mock_rod = MockRod()
        mock_cylinder = MockCylinder()
        rod_cylinder_contact = RodCylinderContact(k=1.0, nu=0.0)
        rod_cylinder_contact.apply_contact(mock_rod, mock_cylinder)

        """Details and reasoning about the values are given in 'test_contact_specific_functions.py/test_calculate_contact_forces_rod_cylinder()'"""
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

        """Details and reasoning about the values are given in 'test_contact_specific_functions.py/test_claculate_contact_forces_rod_cylinder()'"""
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
    def test_check_incorrect_order_type(
        self,
    ):
        mock_rod_one = MockRod()
        mock_list = [1, 2, 3]
        rod_rod_contact = RodRodContact(k=1.0, nu=0.0)

        "Testing Rod Rod Contact wrapper with incorrect type for second argument"
        with pytest.raises(TypeError) as excinfo:
            rod_rod_contact._check_order_and_type(mock_rod_one, mock_list)
        assert (
            "Systems provided to the contact class have incorrect order. \n"
            " First system is {0} and second system is {1}. \n"
            " Both systems must be distinct rods"
        ).format(mock_rod_one.__class__, mock_list.__class__) == str(excinfo.value)

        "Testing Rod Rod Contact wrapper with incorrect type for first argument"
        with pytest.raises(TypeError) as excinfo:
            rod_rod_contact._check_order_and_type(mock_list, mock_rod_one)
        assert (
            "Systems provided to the contact class have incorrect order. \n"
            " First system is {0} and second system is {1}. \n"
            " Both systems must be distinct rods"
        ).format(mock_list.__class__, mock_rod_one.__class__) == str(excinfo.value)

        "Testing Rod Rod Contact wrapper with same rod for both arguments"
        with pytest.raises(TypeError) as excinfo:
            rod_rod_contact._check_order_and_type(mock_rod_one, mock_rod_one)
        assert (
            "First rod is identical to second rod. \n"
            "Rods must be distinct for RodRodConact. \n"
            "If you want self contact, use RodSelfContact instead"
        ) == str(excinfo.value)

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
    def test_check_incorrect_order_type(
        self,
    ):
        mock_rod_one = MockRod()
        mock_rod_two = MockRod()
        mock_list = [1, 2, 3]
        self_contact = RodSelfContact(k=1.0, nu=0.0)

        "Testing Self Contact wrapper with incorrect type for second argument"
        with pytest.raises(TypeError) as excinfo:
            self_contact._check_order_and_type(mock_rod_one, mock_list)
        assert (
            "Systems provided to the contact class have incorrect order/type. \n"
            " First system is {0} and second system is {1}. \n"
            " First system and second system should be the same rod \n"
            " If you want rod rod contact, use RodRodContact instead"
        ).format(mock_rod_one.__class__, mock_list.__class__) == str(excinfo.value)

        "Testing Self Contact wrapper with incorrect type for first argument"
        with pytest.raises(TypeError) as excinfo:
            self_contact._check_order_and_type(mock_list, mock_rod_one)
        assert (
            "Systems provided to the contact class have incorrect order/type. \n"
            " First system is {0} and second system is {1}. \n"
            " First system and second system should be the same rod \n"
            " If you want rod rod contact, use RodRodContact instead"
        ).format(mock_list.__class__, mock_rod_one.__class__) == str(excinfo.value)

        "Testing Self Contact wrapper with different rods"
        with pytest.raises(TypeError) as excinfo:
            self_contact._check_order_and_type(mock_rod_one, mock_rod_two)
        assert (
            "Systems provided to the contact class have incorrect order/type. \n"
            " First system is {0} and second system is {1}. \n"
            " First system and second system should be the same rod \n"
            " If you want rod rod contact, use RodRodContact instead"
        ).format(mock_rod_one.__class__, mock_rod_two.__class__) == str(excinfo.value)

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
