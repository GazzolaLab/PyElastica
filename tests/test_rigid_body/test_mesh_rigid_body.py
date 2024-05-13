__doc__ = """Tests for mesh rigid body module"""
import numpy as np
from numpy.testing import assert_allclose

from elastica.utils import Tolerance
from elastica.rigidbody import MeshRigidBody
from elastica import Mesh
from elastica._linalg import (
    _batch_norm,
)


def initialize_cube_rigid_body():
    """
    This function is to initialize the cube rigid body from the cube.stl.
    """
    cube_mesh = Mesh(r"tests/cube.stl")
    center_of_mass = np.array([0.0, 0.0, 0.0])
    base_length = 2
    volume = base_length**3
    density = 1.0
    mass = density * volume
    # Mass second moment of inertia for cube
    mass_second_moment_of_inertia = np.zeros((3, 3), np.float64)
    np.fill_diagonal(mass_second_moment_of_inertia, (mass * base_length**2) / 6)
    cube_mesh_rigid_body = MeshRigidBody(
        cube_mesh, center_of_mass, mass_second_moment_of_inertia, density, volume
    )
    return (
        cube_mesh_rigid_body,
        cube_mesh,
        center_of_mass,
        mass,
        mass_second_moment_of_inertia,
    )


# tests Initialization of mesh rigid body
def test_MeshRigidBody_initialization():
    """
    This test case is for testing initialization of mesh rigid body and it checks the
    validity of the members of MeshRigidBody class.
    """
    (
        cube_mesh_rigid_body,
        cube_mesh,
        correct_center_of_mass,
        correct_mass,
        correct_mass_second_moment_of_inertia,
    ) = initialize_cube_rigid_body()

    # checking mesh rigid body center
    assert_allclose(
        cube_mesh_rigid_body.position_collection[..., -1],
        correct_center_of_mass,
        atol=Tolerance.atol(),
    )

    # checking velocities, omegas, accelerations and alphas
    assert_allclose(
        cube_mesh_rigid_body.velocity_collection,
        np.zeros((3, 1)),
        atol=Tolerance.atol(),
    )
    assert_allclose(
        cube_mesh_rigid_body.omega_collection,
        np.zeros((3, 1)),
        atol=Tolerance.atol(),
    )
    assert_allclose(
        cube_mesh_rigid_body.acceleration_collection,
        np.zeros((3, 1)),
        atol=Tolerance.atol(),
    )
    assert_allclose(
        cube_mesh_rigid_body.alpha_collection,
        np.zeros((3, 1)),
        atol=Tolerance.atol(),
    )

    # check mass and density initalization
    cube_base_length = 2
    cube_volume = cube_base_length**3
    correct_density = 1.0
    assert_allclose(
        cube_mesh_rigid_body.density, correct_density, atol=Tolerance.atol()
    )
    assert_allclose(cube_mesh_rigid_body.volume, cube_volume, atol=Tolerance.atol())
    assert_allclose(cube_mesh_rigid_body.mass, correct_mass, atol=Tolerance.atol())

    # check mass second moment of inertia initalization
    correct_inv_mass_second_moment_of_inertia = np.linalg.inv(
        correct_mass_second_moment_of_inertia
    )
    assert_allclose(
        cube_mesh_rigid_body.inv_mass_second_moment_of_inertia[..., -1],
        correct_inv_mass_second_moment_of_inertia,
        atol=Tolerance.atol(),
    )
    assert_allclose(
        cube_mesh_rigid_body.mass_second_moment_of_inertia[..., -1],
        correct_mass_second_moment_of_inertia,
        atol=Tolerance.atol(),
    )

    # check director initalization
    correct_director_collection = np.zeros((3, 3, 1))
    correct_director_collection[0] = np.array([1.0, 0.0, 0.0]).reshape(3, 1)
    correct_director_collection[1] = np.array([0.0, 1.0, 0.0]).reshape(3, 1)
    correct_director_collection[2] = np.array([0.0, 0.0, 1.0]).reshape(3, 1)
    assert_allclose(
        cube_mesh_rigid_body.director_collection,
        correct_director_collection,
        atol=Tolerance.atol(),
    )

    # check faces, n_faces, face_centers, face_normals
    correct_faces = cube_mesh.faces
    correct_n_faces = cube_mesh.faces.shape[-1]
    correct_face_centers = cube_mesh.face_centers
    correct_face_normals = cube_mesh.face_normals
    assert_allclose(cube_mesh_rigid_body.faces, correct_faces, atol=Tolerance.atol())
    assert_allclose(
        cube_mesh_rigid_body.n_faces, correct_n_faces, atol=Tolerance.atol()
    )
    assert_allclose(
        cube_mesh_rigid_body.face_centers, correct_face_centers, atol=Tolerance.atol()
    )
    assert_allclose(
        cube_mesh_rigid_body.face_normals, correct_face_normals, atol=Tolerance.atol()
    )

    # check distance to faces, direction to faces, distance to face center, direction to faces centers, face normals in material frame
    correct_distance_to_face_centers_from_center_of_mass = _batch_norm(
        correct_face_centers - correct_center_of_mass.reshape(3, 1)
    )
    correct_direction_to_face_centers_from_center_of_mass_in_material_frame = (
        correct_face_centers - correct_center_of_mass.reshape(3, 1)
    ) / correct_distance_to_face_centers_from_center_of_mass

    correct_distance_to_faces_from_center_of_mass = np.zeros((3, correct_n_faces))
    correct_direction_to_faces_from_center_of_mass_in_material_frame = np.zeros(
        (3, 3, correct_n_faces)
    )
    for i in range(3):
        for k in range(correct_n_faces):
            correct_distance_to_faces_from_center_of_mass[i, k] = np.linalg.norm(
                correct_faces[:, i, k] - correct_center_of_mass
            )
            correct_direction_to_faces_from_center_of_mass_in_material_frame[
                :, i, k
            ] = (
                correct_faces[:, i, k] - correct_center_of_mass
            ) / correct_distance_to_faces_from_center_of_mass[
                i, k
            ]

    assert_allclose(
        cube_mesh_rigid_body.distance_to_face_centers_from_center_of_mass,
        correct_distance_to_face_centers_from_center_of_mass,
        atol=Tolerance.atol(),
    )
    assert_allclose(
        cube_mesh_rigid_body.direction_to_face_centers_from_center_of_mass_in_material_frame,
        correct_direction_to_face_centers_from_center_of_mass_in_material_frame,
        atol=Tolerance.atol(),
    )
    assert_allclose(
        cube_mesh_rigid_body.distance_to_faces_from_center_of_mass,
        correct_distance_to_faces_from_center_of_mass,
        atol=Tolerance.atol(),
    )
    assert_allclose(
        cube_mesh_rigid_body.direction_to_faces_from_center_of_mass_in_material_frame,
        correct_direction_to_faces_from_center_of_mass_in_material_frame,
        atol=Tolerance.atol(),
    )


def test_mesh_rigid_body_update_accelerations():
    """
    This test is testing the update acceleration method of MeshRigidBody class.
    """
    (
        cube_mesh_rigid_body,
        cube_mesh,
        correct_center_of_mass,
        correct_mass,
        correct_mass_second_moment_of_inertia,
    ) = initialize_cube_rigid_body()

    # Mass second moment of inertia for cube
    inv_mass_second_moment_of_inertia = np.linalg.inv(
        correct_mass_second_moment_of_inertia
    )
    external_forces = np.random.randn(3).reshape(3, 1)
    external_torques = np.random.randn(3).reshape(3, 1)

    correct_acceleration = external_forces / correct_mass
    correct_alpha = inv_mass_second_moment_of_inertia @ external_torques.reshape(3)
    correct_alpha = correct_alpha.reshape(3, 1)

    cube_mesh_rigid_body.external_forces[:] = external_forces
    cube_mesh_rigid_body.external_torques[:] = external_torques

    cube_mesh_rigid_body.update_accelerations(time=0)

    assert_allclose(
        correct_acceleration,
        cube_mesh_rigid_body.acceleration_collection,
        atol=Tolerance.atol(),
    )
    assert_allclose(
        correct_alpha, cube_mesh_rigid_body.alpha_collection, atol=Tolerance.atol()
    )


def test_compute_position_center_of_mass():
    """
    This test is testing compute position center of mass method of MeshRigidBody class.
    """

    (
        cube_mesh_rigid_body,
        cube_mesh,
        correct_center_of_mass,
        correct_mass,
        correct_mass_second_moment_of_inertia,
    ) = initialize_cube_rigid_body()
    assert_allclose(
        correct_center_of_mass,
        cube_mesh_rigid_body.compute_position_center_of_mass(),
        atol=Tolerance.atol(),
    )


def test_compute_translational_energy():
    """
    This test is testing compute translational energy function.

    """
    (
        cube_mesh_rigid_body,
        cube_mesh,
        correct_center_of_mass,
        correct_mass,
        correct_mass_second_moment_of_inertia,
    ) = initialize_cube_rigid_body()
    speed = np.random.randn()
    cube_mesh_rigid_body.velocity_collection[2] = speed
    assert_allclose(
        0.5 * correct_mass * speed**2,
        cube_mesh_rigid_body.compute_translational_energy(),
        atol=Tolerance.atol(),
    )


def test_update_faces_translation():
    """
    This test is testing update_faces method of MeshRigidBody class (translation only).
    """
    (
        cube_mesh_rigid_body,
        cube_mesh,
        correct_center_of_mass,
        correct_mass,
        correct_mass_second_moment_of_inertia,
    ) = initialize_cube_rigid_body()

    cube_mesh_rigid_body.position_collection[:, 0] = np.ones(
        3,
    )

    correct_face_centers = cube_mesh.face_centers + 1.0 * np.ones_like(
        cube_mesh.face_centers
    )
    correct_faces = cube_mesh.faces + 1.0 * np.ones_like(cube_mesh.faces)
    correct_face_normals = cube_mesh.face_normals
    cube_mesh_rigid_body.update_faces()

    assert_allclose(
        correct_face_normals, cube_mesh_rigid_body.face_normals, atol=Tolerance.atol()
    )
    assert_allclose(
        correct_face_centers, cube_mesh_rigid_body.face_centers, atol=Tolerance.atol()
    )
    assert_allclose(correct_faces, cube_mesh_rigid_body.faces, atol=Tolerance.atol())


def test_update_faces_rotation():
    """
    This test is testing update_faces method of MeshRigidBody class (rotation only).
    """
    (
        cube_mesh_rigid_body,
        cube_mesh,
        correct_center_of_mass,
        correct_mass,
        correct_mass_second_moment_of_inertia,
    ) = initialize_cube_rigid_body()
    normal = np.array([0.0, 0.0, 1.0])
    tangent = np.array([-1.0, 0.0, 0.0])
    binormal = np.cross(tangent, normal)

    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[0, ...] = normal
    rotation_matrix[1, ...] = binormal
    rotation_matrix[2, ...] = tangent

    correct_faces = np.zeros_like(cube_mesh.faces)
    for i in range(3):
        correct_faces[:, i, :] = rotation_matrix @ cube_mesh.faces[:, i, :]
    correct_face_centers = rotation_matrix @ cube_mesh.face_centers
    correct_face_normals = rotation_matrix @ cube_mesh.face_normals

    cube_mesh_rigid_body.director_collection[0, ...] = normal.reshape(3, 1)
    cube_mesh_rigid_body.director_collection[1, ...] = binormal.reshape(3, 1)
    cube_mesh_rigid_body.director_collection[2, ...] = tangent.reshape(3, 1)
    cube_mesh_rigid_body.update_faces()

    assert_allclose(
        correct_face_normals, cube_mesh_rigid_body.face_normals, atol=Tolerance.atol()
    )
    assert_allclose(
        correct_face_centers, cube_mesh_rigid_body.face_centers, atol=Tolerance.atol()
    )
    assert_allclose(correct_faces, cube_mesh_rigid_body.faces, atol=Tolerance.atol())


def test_update_faces_rotation_and_translation():
    """
    This test is testing update_faces method of MeshRigidBody class (rotation and translation).
    """
    (
        cube_mesh_rigid_body,
        cube_mesh,
        correct_center_of_mass,
        correct_mass,
        correct_mass_second_moment_of_inertia,
    ) = initialize_cube_rigid_body()

    normal = np.array([1.0 / np.sqrt(2), -1.0 / np.sqrt(2), 0.0])
    tangent = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0])
    binormal = np.cross(tangent, normal)

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

    translation = np.ones_like(cube_mesh.face_centers)
    correct_faces = np.zeros_like(cube_mesh.faces)
    for i in range(3):
        correct_faces[:, i, :] = (
            translation + rotation_matrix @ cube_mesh.faces[:, i, :]
        )
    correct_face_centers = translation + rotation_matrix @ cube_mesh.face_centers
    correct_face_normals = rotation_matrix @ cube_mesh.face_normals

    cube_mesh_rigid_body.director_collection[0, ...] = normal.reshape(3, 1)
    cube_mesh_rigid_body.director_collection[1, ...] = binormal.reshape(3, 1)
    cube_mesh_rigid_body.director_collection[2, ...] = tangent.reshape(3, 1)
    cube_mesh_rigid_body.position_collection[:, 0] = np.ones(
        3,
    )
    cube_mesh_rigid_body.update_faces()

    assert_allclose(
        correct_face_normals, cube_mesh_rigid_body.face_normals, atol=Tolerance.atol()
    )
    assert_allclose(
        correct_face_centers, cube_mesh_rigid_body.face_centers, atol=Tolerance.atol()
    )
    assert_allclose(correct_faces, cube_mesh_rigid_body.faces, atol=Tolerance.atol())


if __name__ == "__main__":
    from pytest import main

    main([__file__])
