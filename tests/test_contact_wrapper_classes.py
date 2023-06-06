__doc__ = """ Test Wrapper Classes used in contact in Elastica.joint implementation"""

import numpy as np
from numpy.testing import assert_allclose
from elastica.joint import ExternalContact


def mock_rod_init(self):

    "Initializing Rod"

    self.n_elems = 2
    self.position_collection = np.array(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    )
    self.radius = np.array([5.12952263, 0.19493561, 0.72904683, 0.21676892, 0.18705351])
    self.lengths = np.array([0.89050271, 0.7076813, 0.21078658, 0.95372826, 0.86037329])
    self.tangents = np.array(
        [
            [-0.92511524, 0.19778438, 0.42897242, 0.28430662, -0.57637184],
            [1.73622348, 1.55757074, -0.4606567, -1.30228854, -0.34647765],
            [0.61561226, -0.86529598, -0.9180072, 0.99279484, -2.09424394],
        ]
    )

    self.internal_forces = np.array(
        [
            [-0.77306421, -0.25648047, -0.93419262, -0.77665042, -0.33345937],
            [-1.04834225, -1.92250527, -1.46505411, -0.71192403, -0.99256648],
            [0.33465609, 1.22871475, 0.06250578, -0.49531749, 0.58044695],
        ]
    )

    self.external_forces = np.array(
        [
            [-0.77306421, -0.25648047, -0.93419262, -0.77665042, -0.33345937],
            [-1.04834225, -1.92250527, -1.46505411, -0.71192403, -0.99256648],
            [0.33465609, 1.22871475, 0.06250578, -0.49531749, 0.58044695],
        ]
    )

    self.velocity_collection = np.array(
        [
            [0.70544082, 0.05240655, -1.93283144, -0.35381074, -0.1305802],
            [-0.15193337, -0.16199143, 0.94085659, 0.53076711, 2.15766298],
            [-0.60888955, 0.36362709, 1.31370542, -0.7457939, -0.78005834],
        ]
    )


def mock_rigid_body_init(self):

    "Initializing Rigid Body"

    self.position_collection = np.array([[4], [5], [6]])
    self.director_collection = np.array(
        [[[1], [0], [0]], [[0], [0.707], [-0.707]], [[0], [0.707], [0.707]]]
    )
    self.radius = np.array([1.5])
    self.length = np.array([5.0])
    self.n_elems = 1
    self.external_forces = np.array([[-0.27817918], [-0.04400299], [1.36401515]])
    self.external_torques = np.array([[-0.2338623], [-1.39748107], [0.31085926]])
    self.velocity_collection = np.array(
        [
            [0.63276313, -0.32444142, 0.61402734],
            [-0.01528792, -0.28025795, 0.32799382],
            [-2.22331567, -0.80881859, -0.82109278],
        ]
    )


MockRod = type("MockRod", (object,), {"__init__": mock_rod_init})

MockRigidBody = type("MockRigidBody", (object,), {"__init__": mock_rigid_body_init})


def test_external_contact_with_collision():

    "Testing External Contact wrapper with Collision with analytical verified values"

    tol = 1e-6
    mock_rod = MockRod()
    mock_rigid_body = MockRigidBody()
    ext_contact = ExternalContact(k=1.0, nu=0.5)
    ext_contact.apply_forces(mock_rod, 0, mock_rigid_body, 1)

    assert_allclose(
        mock_rod.external_forces,
        np.array(
            [
                [-0.992182, -0.694716, -0.934193, -0.77665, -0.333459],
                [-1.071788, -1.969396, -1.465054, -0.711924, -0.992566],
                [0.661799, 1.883001, 0.062506, -0.495317, 0.580447],
            ]
        ),
        rtol=tol,
        atol=tol,
    )

    assert_allclose(
        mock_rigid_body.external_forces,
        np.array([[0.379174], [0.026334], [0.382586]]),
        rtol=tol,
        atol=tol,
    )

    assert_allclose(
        mock_rigid_body.external_torques,
        np.array([[-2.092858], [0.245406], [0.310859]]),
        rtol=tol,
        atol=tol,
    )


def test_external_contact_without_collision():

    "Testing External Contact wrapper without Collision with analytical verified values"

    mock_rod = MockRod()
    mock_rigid_body = MockRigidBody()
    ext_contact = ExternalContact(k=1.0, nu=0.5)
    mock_rigid_body.position_collection = np.array([[400], [500], [600]])
    ext_contact.apply_forces(mock_rod, 0, mock_rigid_body, 1)

    assert_allclose(mock_rod.external_forces, mock_rod.external_forces)
    assert_allclose(mock_rigid_body.external_forces, mock_rigid_body.external_forces)
    assert_allclose(mock_rigid_body.external_torques, mock_rigid_body.external_torques)


def test_external_contact_with_two_rods():

    "Testing External Contact wrapper with two rods with analytical verified values"

    tol = 1e-6
    mock_rod_one = MockRod()
    mock_rod_two = MockRod()
    mock_rod_two.position_collection = np.array(
        [[1.2, 2.2, 3.2, 4.2, 5.2], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    )
    ext_contact = ExternalContact(k=1.0, nu=0.5)
    ext_contact.apply_forces(mock_rod_one, 0, mock_rod_two, 0)

    assert_allclose(
        mock_rod_one.external_forces,
        np.array(
            [
                [-5.848848, -7.696102, 2.569813, 0.257867, -0.091002],
                [-3.177381, -2.169013, 2.890443, -0.240067, -0.864637],
                [-1.309157, 4.732951, 7.427895, 0.333335, 0.835572],
            ]
        ),
        rtol=tol,
        atol=tol,
    )

    assert_allclose(
        mock_rod_two.external_forces,
        np.array(
            [
                [-6.167549, -1.298545, 10.66082, 1.435384, 0.030468],
                [-2.17321, -4.425968, -1.173107, -0.060764, -0.887081],
                [-4.672525, -7.453212, 2.279982, 0.528223, 0.718947],
            ]
        ),
        rtol=tol,
        atol=tol,
    )
