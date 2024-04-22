__doc__ = """Tests for mesh rigid body module"""
import numpy as np
from numpy.testing import assert_allclose

from elastica.utils import Tolerance
from elastica.rigidbody import MeshRigidBody
from elastica._linalg import (
    _batch_cross,
)


class MockMesh:
    def __init__(self, n_faces, faces, face_centers, face_normals):
        self.n_faces = n_faces
        self.faces = faces
        self.face_centers = face_centers
        self.face_normals = face_normals


# tests Initialisation of mesh rigid body
def test_MeshRigidBody_initialization():
    """
    This test case is for testing initialization of mesh rigid body and it checks the
    validity of the members of MeshRigidBody class.

    Returns
    -------

    """

    faces = np.array(
        [
            [
                [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
                [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        ]
    )

    face_normals = np.array(
        [
            [0, 0, -1.0, -1.0, 0, 0, 1.0, 1.0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, -1.0, -1.0],
            [-1.0, -1.0, 0, 0, 1.0, 1.0, 0, 0, 0, 0, 0, 0],
        ]
    )
    face_centers = np.array(
        [
            [
                -0.333333,
                0.333333,
                -1.0,
                -1.0,
                0.333333,
                -0.333333,
                1.0,
                1.0,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
            ],
            [
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                1.0,
                1.0,
                -1.0,
                -1.0,
            ],
            [
                -1.0,
                -1.0,
                0.333333,
                -0.333333,
                1.0,
                1.0,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
            ],
        ]
    )
    n_faces = 12

    # setting up test params
    test_mesh = MockMesh(n_faces, faces, face_centers, face_normals)
    center_of_mass = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0]).reshape(3, 1)
    normal = np.array([1.0, 0.0, 0.0]).reshape(3, 1)
    binormal = np.array([0.0, 1.0, 0.0]).reshape(3, 1)
    base_length = 2
    volume = base_length ** 3
    density = np.random.uniform(1.0, 10)
    mass = density * volume
    # Mass second moment of inertia for cube
    mass_second_moment_of_inertia = np.zeros((3, 3), np.float64)
    np.fill_diagonal(mass_second_moment_of_inertia, (mass * base_length ** 2) / 6)
    # Inverse mass second of inertia
    inv_mass_second_moment_of_inertia = np.linalg.inv(mass_second_moment_of_inertia)

    test_mesh_rigid_body = MeshRigidBody(
        test_mesh, center_of_mass, mass_second_moment_of_inertia, density, volume
    )
    # checking origin and length of rod
    assert_allclose(
        test_mesh_rigid_body.position_collection[..., -1],
        center_of_mass,
        atol=Tolerance.atol(),
    )

    # element lengths are equal for all rod.
    # checking velocities, omegas and rest strains
    # density and mass
    assert_allclose(
        test_mesh_rigid_body.velocity_collection,
        np.zeros((3, 1)),
        atol=Tolerance.atol(),
    )

    correct_director_collection = np.zeros((3, 3, 1))
    correct_director_collection[0] = normal
    correct_director_collection[1] = binormal
    correct_director_collection[2] = direction
    assert_allclose(
        test_mesh_rigid_body.director_collection,
        correct_director_collection,
        atol=Tolerance.atol(),
    )

    assert_allclose(
        test_mesh_rigid_body.omega_collection, np.zeros((3, 1)), atol=Tolerance.atol()
    )

    assert_allclose(test_mesh_rigid_body.density, density, atol=Tolerance.atol())

    # Check mass at each node. Note that, node masses is
    # half of element mass at the first and last node.
    assert_allclose(test_mesh_rigid_body.mass, mass, atol=Tolerance.atol())

    # checking directors, rest length
    # and shear, bend matrices and moment of inertia
    assert_allclose(
        test_mesh_rigid_body.inv_mass_second_moment_of_inertia[..., -1],
        inv_mass_second_moment_of_inertia,
        atol=Tolerance.atol(),
    )


def test_mesh_rigid_body_update_accelerations():
    """
    This test is testing the update acceleration method of MeshRigidBody class.

    Returns
    -------

    """

    faces = np.array(
        [
            [
                [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
                [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        ]
    )

    face_normals = np.array(
        [
            [0, 0, -1.0, -1.0, 0, 0, 1.0, 1.0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, -1.0, -1.0],
            [-1.0, -1.0, 0, 0, 1.0, 1.0, 0, 0, 0, 0, 0, 0],
        ]
    )
    face_centers = np.array(
        [
            [
                -0.333333,
                0.333333,
                -1.0,
                -1.0,
                0.333333,
                -0.333333,
                1.0,
                1.0,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
            ],
            [
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                1.0,
                1.0,
                -1.0,
                -1.0,
            ],
            [
                -1.0,
                -1.0,
                0.333333,
                -0.333333,
                1.0,
                1.0,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
            ],
        ]
    )
    n_faces = 12

    test_mesh = MockMesh(n_faces, faces, face_centers, face_normals)
    center_of_mass = np.zeros(
        3,
    )
    base_length = 2
    volume = base_length ** 3
    density = np.random.uniform(1.0, 10)
    mass = density * volume

    # Mass second moment of inertia for cube
    mass_second_moment_of_inertia = np.zeros((3, 3), np.float64)
    np.fill_diagonal(mass_second_moment_of_inertia, (mass * base_length ** 2) / 6)

    test_mesh_rigid_body = MeshRigidBody(
        test_mesh, center_of_mass, mass_second_moment_of_inertia, density, volume
    )

    inv_mass_second_moment_of_inertia = (
        test_mesh_rigid_body.inv_mass_second_moment_of_inertia.reshape(3, 3)
    )

    external_forces = np.random.randn(3).reshape(3, 1)
    external_torques = np.random.randn(3).reshape(3, 1)

    correct_acceleration = external_forces / mass
    correct_alpha = inv_mass_second_moment_of_inertia @ external_torques.reshape(3)
    correct_alpha = correct_alpha.reshape(3, 1)

    test_mesh_rigid_body.external_forces[:] = external_forces
    test_mesh_rigid_body.external_torques[:] = external_torques

    test_mesh_rigid_body.update_accelerations(time=0)

    assert_allclose(
        correct_acceleration,
        test_mesh_rigid_body.acceleration_collection,
        atol=Tolerance.atol(),
    )
    assert_allclose(
        correct_alpha, test_mesh_rigid_body.alpha_collection, atol=Tolerance.atol()
    )


def test_compute_position_center_of_mass():
    """
    This test is testing compute position center of mass method of MeshRigidBody class.

    Returns
    -------

    """

    faces = np.array(
        [
            [
                [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
                [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        ]
    )

    face_normals = np.array(
        [
            [0, 0, -1.0, -1.0, 0, 0, 1.0, 1.0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, -1.0, -1.0],
            [-1.0, -1.0, 0, 0, 1.0, 1.0, 0, 0, 0, 0, 0, 0],
        ]
    )
    face_centers = np.array(
        [
            [
                -0.333333,
                0.333333,
                -1.0,
                -1.0,
                0.333333,
                -0.333333,
                1.0,
                1.0,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
            ],
            [
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                1.0,
                1.0,
                -1.0,
                -1.0,
            ],
            [
                -1.0,
                -1.0,
                0.333333,
                -0.333333,
                1.0,
                1.0,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
            ],
        ]
    )
    n_faces = 12

    test_mesh = MockMesh(n_faces, faces, face_centers, face_normals)
    center_of_mass = np.zeros(
        3,
    )
    base_length = 2
    volume = base_length ** 3
    density = np.random.uniform(1.0, 10)
    mass = density * volume
    # Mass second moment of inertia for cube
    mass_second_moment_of_inertia = np.zeros((3, 3), np.float64)
    np.fill_diagonal(mass_second_moment_of_inertia, (mass * base_length ** 2) / 6)

    test_mesh_rigid_body = MeshRigidBody(
        test_mesh, center_of_mass, mass_second_moment_of_inertia, density, volume
    )

    correct_position = center_of_mass

    test_position = test_mesh_rigid_body.compute_position_center_of_mass()

    assert_allclose(correct_position, test_position, atol=Tolerance.atol())


def test_compute_translational_energy():
    """
    This test is testing compute translational energy function.

    Returns
    -------

    """

    faces = np.array(
        [
            [
                [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
                [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        ]
    )

    face_normals = np.array(
        [
            [0, 0, -1.0, -1.0, 0, 0, 1.0, 1.0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, -1.0, -1.0],
            [-1.0, -1.0, 0, 0, 1.0, 1.0, 0, 0, 0, 0, 0, 0],
        ]
    )
    face_centers = np.array(
        [
            [
                -0.333333,
                0.333333,
                -1.0,
                -1.0,
                0.333333,
                -0.333333,
                1.0,
                1.0,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
            ],
            [
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                1.0,
                1.0,
                -1.0,
                -1.0,
            ],
            [
                -1.0,
                -1.0,
                0.333333,
                -0.333333,
                1.0,
                1.0,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
            ],
        ]
    )
    n_faces = 12

    test_mesh = MockMesh(n_faces, faces, face_centers, face_normals)
    center_of_mass = np.zeros(
        3,
    )
    base_length = 2
    volume = base_length ** 3
    density = np.random.uniform(1.0, 10)
    mass = density * volume
    # Mass second moment of inertia for cube
    mass_second_moment_of_inertia = np.zeros((3, 3), np.float64)
    np.fill_diagonal(mass_second_moment_of_inertia, (mass * base_length ** 2) / 6)

    test_mesh_rigid_body = MeshRigidBody(
        test_mesh, center_of_mass, mass_second_moment_of_inertia, density, volume
    )

    speed = np.random.randn()
    test_mesh_rigid_body.velocity_collection[2] = speed

    correct_energy = 0.5 * mass * speed ** 2
    test_energy = test_mesh_rigid_body.compute_translational_energy()

    assert_allclose(correct_energy, test_energy, atol=Tolerance.atol())


def test_update_faces_translation():
    """
    This test is testing compute position center of mass method of MeshRigidBody class.

    Returns
    -------

    """

    faces = np.array(
        [
            [
                [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
                [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        ]
    )

    face_normals = np.array(
        [
            [0, 0, -1.0, -1.0, 0, 0, 1.0, 1.0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, -1.0, -1.0],
            [-1.0, -1.0, 0, 0, 1.0, 1.0, 0, 0, 0, 0, 0, 0],
        ]
    )
    face_centers = np.array(
        [
            [
                -0.333333,
                0.333333,
                -1.0,
                -1.0,
                0.333333,
                -0.333333,
                1.0,
                1.0,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
            ],
            [
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                1.0,
                1.0,
                -1.0,
                -1.0,
            ],
            [
                -1.0,
                -1.0,
                0.333333,
                -0.333333,
                1.0,
                1.0,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
            ],
        ]
    )
    n_faces = 12

    test_mesh = MockMesh(n_faces, faces, face_centers, face_normals)
    center_of_mass = np.zeros(
        3,
    )
    base_length = 2
    volume = base_length ** 3
    density = np.random.uniform(1.0, 10)
    mass = density * volume
    # Mass second moment of inertia for cube
    mass_second_moment_of_inertia = np.zeros((3, 3), np.float64)
    np.fill_diagonal(mass_second_moment_of_inertia, (mass * base_length ** 2) / 6)

    test_mesh_rigid_body = MeshRigidBody(
        test_mesh, center_of_mass, mass_second_moment_of_inertia, density, volume
    )

    test_mesh_rigid_body.position_collection[:, 0] = np.ones(
        3,
    )

    correct_face_centers = face_centers + 1.0 * np.ones_like(face_centers)
    correct_faces = faces + 1.0 * np.ones_like(faces)
    correct_face_normals = face_normals
    test_mesh_rigid_body.update_faces()
    test_face_centers = test_mesh_rigid_body.face_centers
    test_face_normals = test_mesh_rigid_body.face_normals
    test_faces = test_mesh_rigid_body.faces

    assert_allclose(correct_face_centers, test_face_centers, atol=Tolerance.atol())
    assert_allclose(correct_faces, test_faces, atol=Tolerance.atol())
    assert_allclose(correct_face_normals, test_face_normals, atol=Tolerance.atol())


def test_update_faces_rotation():
    """
    This test is testing compute position center of mass method of MeshRigidBody class.

    Returns
    -------

    """

    faces = np.array(
        [
            [
                [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
                [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        ]
    )

    face_normals = np.array(
        [
            [0, 0, -1.0, -1.0, 0, 0, 1.0, 1.0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, -1.0, -1.0],
            [-1.0, -1.0, 0, 0, 1.0, 1.0, 0, 0, 0, 0, 0, 0],
        ]
    )
    face_centers = np.array(
        [
            [
                -0.333333,
                0.333333,
                -1.0,
                -1.0,
                0.333333,
                -0.333333,
                1.0,
                1.0,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
            ],
            [
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                1.0,
                1.0,
                -1.0,
                -1.0,
            ],
            [
                -1.0,
                -1.0,
                0.333333,
                -0.333333,
                1.0,
                1.0,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
            ],
        ]
    )
    n_faces = 12

    test_mesh = MockMesh(n_faces, faces, face_centers, face_normals)
    center_of_mass = np.zeros(
        3,
    )
    base_length = 2
    volume = base_length ** 3
    density = np.random.uniform(1.0, 10)
    mass = density * volume
    # Mass second moment of inertia for cube
    mass_second_moment_of_inertia = np.zeros((3, 3), np.float64)
    np.fill_diagonal(mass_second_moment_of_inertia, (mass * base_length ** 2) / 6)

    test_mesh_rigid_body = MeshRigidBody(
        test_mesh, center_of_mass, mass_second_moment_of_inertia, density, volume
    )

    normal = np.array([0.0, 0.0, 1.0]).reshape(3, 1)
    tangent = np.array([-1.0, 0.0, 0.0]).reshape(3, 1)
    binormal = _batch_cross(tangent, normal)

    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[0, ...] = normal.reshape(
        3,
    )
    rotation_matrix[1, ...] = binormal.reshape(
        3,
    )
    rotation_matrix[2, ...] = tangent.reshape(
        3,
    )
    correct_faces = np.zeros_like(faces)

    correct_face_centers = rotation_matrix @ face_centers
    for i in range(3):
        correct_faces[:, i, :] = rotation_matrix @ faces[:, i, :]
    correct_face_normals = rotation_matrix @ face_normals

    test_mesh_rigid_body.director_collection[0, ...] = normal
    test_mesh_rigid_body.director_collection[1, ...] = binormal
    test_mesh_rigid_body.director_collection[2, ...] = tangent
    test_mesh_rigid_body.update_faces()

    test_face_centers = test_mesh_rigid_body.face_centers
    test_face_normals = test_mesh_rigid_body.face_normals
    test_faces = test_mesh_rigid_body.faces

    assert_allclose(correct_face_normals, test_face_normals, atol=Tolerance.atol())
    assert_allclose(correct_face_centers, test_face_centers, atol=Tolerance.atol())
    assert_allclose(correct_faces, test_faces, atol=Tolerance.atol())


def test_update_faces_rotation_and_translation():
    """
    This test is testing compute position center of mass method of MeshRigidBody class.

    Returns
    -------

    """

    faces = np.array(
        [
            [
                [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
                [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        ]
    )

    face_normals = np.array(
        [
            [0, 0, -1.0, -1.0, 0, 0, 1.0, 1.0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, -1.0, -1.0],
            [-1.0, -1.0, 0, 0, 1.0, 1.0, 0, 0, 0, 0, 0, 0],
        ]
    )
    face_centers = np.array(
        [
            [
                -0.333333,
                0.333333,
                -1.0,
                -1.0,
                0.333333,
                -0.333333,
                1.0,
                1.0,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
            ],
            [
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                1.0,
                1.0,
                -1.0,
                -1.0,
            ],
            [
                -1.0,
                -1.0,
                0.333333,
                -0.333333,
                1.0,
                1.0,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
            ],
        ]
    )
    n_faces = 12

    test_mesh = MockMesh(n_faces, faces, face_centers, face_normals)
    center_of_mass = np.zeros(
        3,
    )
    base_length = 2
    volume = base_length ** 3
    density = np.random.uniform(1.0, 10)
    mass = density * volume
    # Mass second moment of inertia for cube
    mass_second_moment_of_inertia = np.zeros((3, 3), np.float64)
    np.fill_diagonal(mass_second_moment_of_inertia, (mass * base_length ** 2) / 6)

    test_mesh_rigid_body = MeshRigidBody(
        test_mesh, center_of_mass, mass_second_moment_of_inertia, density, volume
    )

    normal = np.array([1.0 / np.sqrt(2), -1.0 / np.sqrt(2), 0.0]).reshape(3, 1)
    tangent = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0]).reshape(3, 1)
    binormal = _batch_cross(tangent, normal)

    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[0, ...] = normal.reshape(
        3,
    )
    rotation_matrix[1, ...] = binormal.reshape(
        3,
    )
    rotation_matrix[2, ...] = tangent.reshape(
        3,
    )

    translation = np.ones_like(face_centers)
    correct_faces = np.zeros_like(faces)
    for i in range(3):
        correct_faces[:, i, :] = translation + rotation_matrix @ faces[:, i, :]
    correct_face_centers = translation + rotation_matrix @ face_centers
    correct_face_normals = rotation_matrix @ face_normals

    test_mesh_rigid_body.director_collection[0, ...] = normal
    test_mesh_rigid_body.director_collection[1, ...] = binormal
    test_mesh_rigid_body.director_collection[2, ...] = tangent
    test_mesh_rigid_body.position_collection[:, 0] = np.ones(
        3,
    )
    print(test_mesh_rigid_body.face_normals)
    test_mesh_rigid_body.update_faces()

    test_face_centers = test_mesh_rigid_body.face_centers.copy()
    test_face_normals = test_mesh_rigid_body.face_normals.copy()
    print(test_face_normals)
    test_faces = test_mesh_rigid_body.faces.copy()

    assert_allclose(correct_face_normals, test_face_normals, atol=Tolerance.atol())
    assert_allclose(correct_face_centers, test_face_centers, atol=Tolerance.atol())
    assert_allclose(correct_faces, test_faces, atol=Tolerance.atol())


def test_update_faces_translation_with_mesh_initializer():
    """
    This test is testing compute position center of mass method of MeshRigidBody class.

    Returns
    -------

    """

    faces = np.array(
        [
            [
                [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
                [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        ]
    )

    face_normals = np.array(
        [
            [0, 0, -1.0, -1.0, 0, 0, 1.0, 1.0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, -1.0, -1.0],
            [-1.0, -1.0, 0, 0, 1.0, 1.0, 0, 0, 0, 0, 0, 0],
        ]
    )
    face_centers = np.array(
        [
            [
                -0.333333,
                0.333333,
                -1.0,
                -1.0,
                0.333333,
                -0.333333,
                1.0,
                1.0,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
            ],
            [
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                1.0,
                1.0,
                -1.0,
                -1.0,
            ],
            [
                -1.0,
                -1.0,
                0.333333,
                -0.333333,
                1.0,
                1.0,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
            ],
        ]
    )
    from elastica import Mesh

    test_mesh = Mesh(r"tests/cube.stl")
    center_of_mass = np.zeros(
        3,
    )
    base_length = 2
    volume = base_length ** 3
    density = np.random.uniform(1.0, 10)
    mass = density * volume
    # Mass second moment of inertia for cube
    mass_second_moment_of_inertia = np.zeros((3, 3), np.float64)
    np.fill_diagonal(mass_second_moment_of_inertia, (mass * base_length ** 2) / 6)

    test_mesh_rigid_body = MeshRigidBody(
        test_mesh, center_of_mass, mass_second_moment_of_inertia, density, volume
    )

    translation = np.ones_like(face_centers)
    correct_faces = np.zeros_like(faces)
    for i in range(3):
        correct_faces[:, i, :] = translation + faces[:, i, :]
    correct_face_centers = translation + face_centers
    correct_face_normals = face_normals

    test_mesh_rigid_body.position_collection[:, 0] = np.ones(
        3,
    )
    test_mesh_rigid_body.update_faces()

    test_face_centers = test_mesh_rigid_body.face_centers
    test_face_normals = test_mesh_rigid_body.face_normals
    test_faces = test_mesh_rigid_body.faces

    assert_allclose(
        correct_face_normals,
        test_face_normals,
        atol=Tolerance.atol(),
        rtol=Tolerance.rtol(),
    )
    assert_allclose(
        correct_face_centers,
        test_face_centers,
        atol=Tolerance.atol(),
        rtol=Tolerance.rtol(),
    )
    assert_allclose(
        correct_faces, test_faces, atol=Tolerance.atol(), rtol=Tolerance.rtol()
    )


def test_update_faces_rotation_with_mesh_initializer():
    """
    This test is testing compute position center of mass method of MeshRigidBody class.

    Returns
    -------

    """

    faces = np.array(
        [
            [
                [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
                [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        ]
    )

    face_normals = np.array(
        [
            [0, 0, -1.0, -1.0, 0, 0, 1.0, 1.0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, -1.0, -1.0],
            [-1.0, -1.0, 0, 0, 1.0, 1.0, 0, 0, 0, 0, 0, 0],
        ]
    )
    face_centers = np.array(
        [
            [
                -0.333333,
                0.333333,
                -1.0,
                -1.0,
                0.333333,
                -0.333333,
                1.0,
                1.0,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
            ],
            [
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                1.0,
                1.0,
                -1.0,
                -1.0,
            ],
            [
                -1.0,
                -1.0,
                0.333333,
                -0.333333,
                1.0,
                1.0,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
            ],
        ]
    )
    from elastica import Mesh

    test_mesh = Mesh(r"tests/cube.stl")
    center_of_mass = np.zeros(
        3,
    )
    base_length = 2
    volume = base_length ** 3
    density = np.random.uniform(1.0, 10)
    mass = density * volume
    # Mass second moment of inertia for cube
    mass_second_moment_of_inertia = np.zeros((3, 3), np.float64)
    np.fill_diagonal(mass_second_moment_of_inertia, (mass * base_length ** 2) / 6)

    test_mesh_rigid_body = MeshRigidBody(
        test_mesh, center_of_mass, mass_second_moment_of_inertia, density, volume
    )

    normal = np.array([0.0, 0.0, 1.0]).reshape(3, 1)
    tangent = np.array([-1.0, 0.0, 0.0]).reshape(3, 1)
    binormal = _batch_cross(tangent, normal)

    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[0, ...] = normal.reshape(
        3,
    )
    rotation_matrix[1, ...] = binormal.reshape(
        3,
    )
    rotation_matrix[2, ...] = tangent.reshape(
        3,
    )

    correct_faces = np.zeros_like(faces)
    for i in range(3):
        correct_faces[:, i, :] = rotation_matrix @ faces[:, i, :]
    correct_face_centers = rotation_matrix @ test_mesh.face_centers
    correct_face_normals = rotation_matrix @ test_mesh.face_normals

    test_mesh_rigid_body.director_collection[0, ...] = normal
    test_mesh_rigid_body.director_collection[1, ...] = binormal
    test_mesh_rigid_body.director_collection[2, ...] = tangent
    test_mesh_rigid_body.update_faces()

    test_face_centers = test_mesh_rigid_body.face_centers
    test_face_normals = test_mesh_rigid_body.face_normals
    test_faces = test_mesh_rigid_body.faces

    assert_allclose(
        correct_face_normals,
        test_face_normals,
        atol=Tolerance.atol(),
        rtol=Tolerance.rtol(),
    )
    assert_allclose(
        correct_face_centers,
        test_face_centers,
        atol=Tolerance.atol(),
        rtol=Tolerance.rtol(),
    )
    assert_allclose(
        correct_faces, test_faces, atol=Tolerance.atol(), rtol=Tolerance.rtol()
    )


def test_update_faces_rotation_and_translation_with_mesh_initializer():
    """
    This test is testing compute position center of mass method of MeshRigidBody class.

    Returns
    -------

    """

    faces = np.array(
        [
            [
                [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
                [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        ]
    )

    face_normals = np.array(
        [
            [0, 0, -1.0, -1.0, 0, 0, 1.0, 1.0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, -1.0, -1.0],
            [-1.0, -1.0, 0, 0, 1.0, 1.0, 0, 0, 0, 0, 0, 0],
        ]
    )
    face_centers = np.array(
        [
            [
                -0.333333,
                0.333333,
                -1.0,
                -1.0,
                0.333333,
                -0.333333,
                1.0,
                1.0,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
            ],
            [
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                1.0,
                1.0,
                -1.0,
                -1.0,
            ],
            [
                -1.0,
                -1.0,
                0.333333,
                -0.333333,
                1.0,
                1.0,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
                -0.333333,
                0.333333,
            ],
        ]
    )
    from elastica import Mesh

    test_mesh = Mesh(r"tests/cube.stl")
    center_of_mass = np.zeros(
        3,
    )
    base_length = 2
    volume = base_length ** 3
    density = np.random.uniform(1.0, 10)
    mass = density * volume
    # Mass second moment of inertia for cube
    mass_second_moment_of_inertia = np.zeros((3, 3), np.float64)
    np.fill_diagonal(mass_second_moment_of_inertia, (mass * base_length ** 2) / 6)

    test_mesh_rigid_body = MeshRigidBody(
        test_mesh, center_of_mass, mass_second_moment_of_inertia, density, volume
    )

    normal = np.array([0.0, 0.0, 1.0]).reshape(3, 1)
    tangent = np.array([-1.0, 0.0, 0.0]).reshape(3, 1)
    binormal = _batch_cross(tangent, normal)

    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[0, ...] = normal.reshape(
        3,
    )
    rotation_matrix[1, ...] = binormal.reshape(
        3,
    )
    rotation_matrix[2, ...] = tangent.reshape(
        3,
    )

    translation = np.ones_like(face_centers)
    correct_faces = np.zeros_like(faces)
    for i in range(3):
        correct_faces[:, i, :] = translation + rotation_matrix @ faces[:, i, :]
    correct_face_centers = translation + rotation_matrix @ face_centers
    correct_face_normals = rotation_matrix @ face_normals

    test_mesh_rigid_body.director_collection[0, ...] = normal
    test_mesh_rigid_body.director_collection[1, ...] = binormal
    test_mesh_rigid_body.director_collection[2, ...] = tangent
    test_mesh_rigid_body.position_collection[:, 0] = np.ones(
        3,
    )
    test_mesh_rigid_body.update_faces()

    test_face_centers = test_mesh_rigid_body.face_centers
    test_face_normals = test_mesh_rigid_body.face_normals
    test_faces = test_mesh_rigid_body.faces

    assert_allclose(
        correct_face_normals,
        test_face_normals,
        atol=Tolerance.atol(),
        rtol=Tolerance.rtol(),
    )
    assert_allclose(
        correct_face_centers,
        test_face_centers,
        atol=Tolerance.atol(),
        rtol=Tolerance.rtol(),
    )
    assert_allclose(
        correct_faces, test_faces, atol=Tolerance.atol(), rtol=Tolerance.rtol()
    )


if __name__ == "__main__":
    from pytest import main

    main([__file__])
