__doc__ = """ Test Wrapper Classes used in contact in Elastica.joint implementation"""

import numpy as np
from numpy.testing import assert_allclose
from elastica.joint import ExternalContact, SelfContact
from elastica.typing import RodBase, RigidBodyBase


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


def mock_rigid_body_init(self):

    "Initializing Rigid Body"
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

MockRigidBody = type(
    "MockRigidBody", (RigidBodyBase,), {"__init__": mock_rigid_body_init}
)


def test_external_contact_rod_rigid_body_with_collision():

    "Testing External Contact wrapper with Collision with analytical verified values"

    mock_rod = MockRod()
    mock_rigid_body = MockRigidBody()
    ext_contact = ExternalContact(k=1.0, nu=0.0)
    ext_contact.apply_forces(mock_rod, 0, mock_rigid_body, 1)

    """Details and reasoning about the values are given in 'test_contact_specific_functions.py/test_claculate_contact_forces_rod_rigid_body()'"""
    assert_allclose(
        mock_rod.external_forces,
        np.array([[0.166666, 0.333333, 0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        atol=1e-6,
    )

    assert_allclose(
        mock_rigid_body.external_forces, np.array([[-0.5], [0.0], [0.0]]), atol=1e-6
    )

    assert_allclose(
        mock_rigid_body.external_torques, np.array([[0.0], [0.0], [0.0]]), atol=1e-6
    )


def test_external_contact_rod_rigid_body_without_collision():

    "Testing External Contact wrapper without Collision with analytical verified values"

    mock_rod = MockRod()
    mock_rigid_body = MockRigidBody()
    ext_contact = ExternalContact(k=1.0, nu=0.5)

    """Setting rigid body position such that there is no collision"""
    mock_rigid_body.position_collection = np.array([[400], [500], [600]])
    mock_rod_external_forces_before_execution = mock_rod.external_forces.copy()
    mock_rigid_body_external_forces_before_execution = (
        mock_rigid_body.external_forces.copy()
    )
    mock_rigid_body_external_torques_before_execution = (
        mock_rigid_body.external_torques.copy()
    )
    ext_contact.apply_forces(mock_rod, 0, mock_rigid_body, 1)

    assert_allclose(mock_rod.external_forces, mock_rod_external_forces_before_execution)
    assert_allclose(
        mock_rigid_body.external_forces,
        mock_rigid_body_external_forces_before_execution,
    )
    assert_allclose(
        mock_rigid_body.external_torques,
        mock_rigid_body_external_torques_before_execution,
    )


def test_external_contact_with_two_rods_with_collision():

    "Testing External Contact wrapper with two rods with analytical verified values"
    "Test values have been copied from 'test_contact_specific_functions.py/test_calculate_contact_forces_rod_rod()'"

    mock_rod_one = MockRod()
    mock_rod_two = MockRod()
    mock_rod_two.position_collection = np.array([[4, 5, 6], [0, 0, 0], [0, 0, 0]])
    ext_contact = ExternalContact(k=1.0, nu=0.0)
    ext_contact.apply_forces(mock_rod_one, 0, mock_rod_two, 0)

    assert_allclose(
        mock_rod_one.external_forces,
        np.array([[0, -0.5, -0.5], [0, 0, 0], [0, 0, 0]]),
        atol=1e-6,
    )
    assert_allclose(
        mock_rod_two.external_forces,
        np.array([[0.333333, 0.666666, 0], [0, 0, 0], [0, 0, 0]]),
        atol=1e-6,
    )


def test_external_contact_with_two_rods_without_collision():

    "Testing External Contact wrapper with two rods with analytical verified values"

    mock_rod_one = MockRod()
    mock_rod_two = MockRod()

    "Setting rod two position such that there is no collision"
    mock_rod_two.position_collection = np.array([[100, 101, 102], [0, 0, 0], [0, 0, 0]])
    ext_contact = ExternalContact(k=1.0, nu=1.0)
    mock_rod_one_external_forces_before_execution = mock_rod_one.external_forces.copy()
    mock_rod_two_external_forces_before_execution = mock_rod_two.external_forces.copy()
    ext_contact.apply_forces(mock_rod_one, 0, mock_rod_two, 0)

    assert_allclose(
        mock_rod_one.external_forces, mock_rod_one_external_forces_before_execution
    )
    assert_allclose(
        mock_rod_two.external_forces, mock_rod_two_external_forces_before_execution
    )


def test_self_contact_with_rod_self_collision():

    "Testing Self Contact wrapper rod self collision with analytical verified values"

    mock_rod = MockRod()

    "Test values have been copied from 'test_contact_specific_functions.py/test_calculate_contact_forces_self_rod()'"
    mock_rod.n_elems = 3
    mock_rod.position_collection = np.array([[1, 4, 4, 1], [0, 0, 1, 1], [0, 0, 0, 0]])
    mock_rod.radius = np.array([1, 1, 1])
    mock_rod.lengths = np.array([3, 3, 3])
    mock_rod.tangents = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    mock_rod.velocity_collection = np.array(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    mock_rod.internal_forces = np.array(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    mock_rod.external_forces = np.array(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    sel_contact = SelfContact(k=1.0, nu=0.0)
    sel_contact.apply_forces(mock_rod, 0, mock_rod, 0)

    assert_allclose(
        mock_rod.external_forces,
        np.array([[0, 0, 0, 0], [-0.333333, -0.666666, 0.5, 0.5], [0, 0, 0, 0]]),
        atol=1e-6,
    )


def test_self_contact_with_rod_no_self_collision():

    "Testing Self Contact wrapper rod no self collision with analytical verified values"

    mock_rod = MockRod()

    "the initially set rod does not have self collision"
    mock_rod_external_forces_before_execution = mock_rod.external_forces.copy()
    sel_contact = SelfContact(k=1.0, nu=1.0)
    sel_contact.apply_forces(mock_rod, 0, mock_rod, 0)

    assert_allclose(mock_rod.external_forces, mock_rod_external_forces_before_execution)
