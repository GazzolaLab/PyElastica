__doc__ = """ Joint between rods test module """

# System imports
from elastica.joint import (
    FreeJoint,
    HingeJoint,
    FixedJoint,
)
from numpy.testing import assert_allclose
from elastica._rotations import _inv_rotate
from elastica.rod.cosserat_rod import CosseratRod
from elastica.utils import Tolerance
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# TODO: change tests and made them independent of rod, at least assigin hardcoded values for forces and torques


# seed random number generator
rng = np.random.default_rng(0)


def test_freejoint():

    # Some rod properties. We need them for constructer, they are not used.
    normal = np.array([0.0, 1.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    base_length = 1
    base_radius = 0.2
    density = 1
    nu = 0.1

    # Youngs Modulus [Pa]
    youngs_modulus = 1e6
    # poisson ratio
    poisson_ratio = 0.5
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

    # Origin of the rod
    origin1 = np.array([0.0, 0.0, 0.0])
    origin2 = np.array([1.0, 0.0, 0.0])

    # Number of elements
    n = 4

    # create rod classes
    rod1 = CosseratRod.straight_rod(
        n,
        origin1,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )
    rod2 = CosseratRod.straight_rod(
        n,
        origin2,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )

    # Stiffness between points
    k = 1e8

    # Damping between two points
    nu = 1

    # Rod indexes
    rod1_index = -1
    rod2_index = 0

    # Rod velocity
    v1 = np.array([-1, 0, 0]).reshape(3, 1)
    v2 = v1 * -1

    rod1.velocity_collection = np.tile(v1, (1, n + 1))
    rod2.velocity_collection = np.tile(v2, (1, n + 1))

    # Compute the free joint forces
    distance = (
        rod2.position_collection[..., rod2_index]
        - rod1.position_collection[..., rod1_index]
    )
    elasticforce = k * distance
    relative_vel = (
        rod2.velocity_collection[..., rod2_index]
        - rod1.velocity_collection[..., rod1_index]
    )
    dampingforce = nu * relative_vel
    contactforce = elasticforce + dampingforce

    frjt = FreeJoint(k, nu)
    frjt.apply_forces(rod1, rod1_index, rod2, rod2_index)
    frjt.apply_torques(rod1, rod1_index, rod2, rod2_index)

    assert_allclose(
        rod1.external_forces[..., rod1_index], contactforce, atol=Tolerance.atol()
    )
    assert_allclose(
        rod2.external_forces[..., rod2_index], -1 * contactforce, atol=Tolerance.atol()
    )


def test_hingejoint():
    # Define the rod for testing
    # Some rod properties. We need them for constructer, they are not used.
    normal1 = np.array([0.0, 1.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal2 = np.array([0.0, 0.0, 1.0])
    direction2 = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0])
    # direction2 = np.array([0.,1.0,0.])
    base_length = 1
    base_radius = 0.2
    density = 1
    nu = 0.1

    # Youngs Modulus [Pa]
    youngs_modulus = 1e6
    # poisson ratio
    poisson_ratio = 0.5
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

    # Origin of the rod
    origin1 = np.array([0.0, 0.0, 0.0])
    origin2 = np.array([1.0, 0.0, 0.0])

    # Number of elements
    n = 2

    # create rod classes
    rod1 = CosseratRod.straight_rod(
        n,
        origin1,
        direction,
        normal1,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )
    rod2 = CosseratRod.straight_rod(
        n,
        origin2,
        direction2,
        normal2,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )

    # Rod velocity
    v1 = np.array([-1, 0, 0]).reshape(3, 1)
    v2 = v1 * -1

    rod1.velocity_collection = np.tile(v1, (1, n + 1))
    rod2.velocity_collection = np.tile(v2, (1, n + 1))

    # Stiffness between points
    k = 1e8
    kt = 1e6
    # Damping between two points
    nu = 1

    # Rod indexes
    rod1_index = -1
    rod2_index = 0

    # Compute the free joint forces
    distance = (
        rod2.position_collection[..., rod2_index]
        - rod1.position_collection[..., rod1_index]
    )
    elasticforce = k * distance
    relative_vel = (
        rod2.velocity_collection[..., rod2_index]
        - rod1.velocity_collection[..., rod1_index]
    )
    dampingforce = nu * relative_vel
    contactforce = elasticforce + dampingforce

    hgjt = HingeJoint(k, nu, kt, normal1)

    hgjt.apply_forces(rod1, rod1_index, rod2, rod2_index)
    hgjt.apply_torques(rod1, rod1_index, rod2, rod2_index)

    assert_allclose(
        rod1.external_forces[..., rod1_index], contactforce, atol=Tolerance.atol()
    )
    assert_allclose(
        rod2.external_forces[..., rod2_index], -1 * contactforce, atol=Tolerance.atol()
    )

    system_two_tangent = rod2.director_collection[2, :, rod2_index]
    force_direction = np.dot(system_two_tangent, normal1) * normal1
    torque = -kt * np.cross(system_two_tangent, force_direction)

    torque_rod1 = -rod1.director_collection[..., rod1_index] @ torque
    torque_rod2 = rod2.director_collection[..., rod2_index] @ torque

    assert_allclose(
        rod1.external_torques[..., rod1_index], torque_rod1, atol=Tolerance.atol()
    )
    assert_allclose(
        rod2.external_torques[..., rod2_index], torque_rod2, atol=Tolerance.atol()
    )


rest_euler_angles = [
    np.array([0.0, 0.0, 0.0]),
    np.array([np.pi / 2, 0.0, 0.0]),
    np.array([0.0, np.pi / 2, 0.0]),
    np.array([0.0, 0.0, np.pi / 2]),
    2 * np.pi * rng.random(size=3),
]


@pytest.mark.parametrize("rest_euler_angle", rest_euler_angles)
def test_fixedjoint(rest_euler_angle):
    # Define the rod for testing
    # Some rod properties. We need them for constructor, they are not used.
    normal1 = np.array([0.0, 1.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal2 = np.array([0.0, 0.0, 1.0])
    direction2 = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0])
    # direction2 = np.array([0.,1.0,0.])
    base_length = 1
    base_radius = 0.2
    density = 1
    nu = 0.1

    # Youngs Modulus [Pa]
    youngs_modulus = 1e6
    # poisson ratio
    poisson_ratio = 0.5
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

    # Origin of the rod
    origin1 = np.array([0.0, 0.0, 0.0])
    origin2 = np.array([1.0, 0.0, 0.0])

    # Number of elements
    n = 2

    # create rod classes
    rod1 = CosseratRod.straight_rod(
        n,
        origin1,
        direction,
        normal1,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )
    rod2 = CosseratRod.straight_rod(
        n,
        origin2,
        direction2,
        normal2,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )

    # Rod velocity
    v1 = np.array([-1, 0, 0]).reshape(3, 1)
    v2 = v1 * -1

    rod1.velocity_collection = np.tile(v1, (1, n + 1))
    rod2.velocity_collection = np.tile(v2, (1, n + 1))

    # Rod angular velocity
    omega1 = 1 / 180 * np.pi * np.array([0, 0, 1]).reshape(3, 1)
    omega2 = -omega1
    rod1.omega_collection = np.tile(omega1, (1, n + 1))
    rod2.omega_collection = np.tile(omega2, (1, n + 1))

    # Positional and rotational stiffness between systems
    k = 1e8
    kt = 1e6
    # Positional and rotational damping between systems
    nu = 1
    nut = 1e2

    # Rod indexes
    rod1_index = -1
    rod2_index = 0

    # Compute the free joint forces
    distance = (
        rod2.position_collection[..., rod2_index]
        - rod1.position_collection[..., rod1_index]
    )
    elasticforce = k * distance
    relative_vel = (
        rod2.velocity_collection[..., rod2_index]
        - rod1.velocity_collection[..., rod1_index]
    )
    dampingforce = nu * relative_vel
    contactforce = elasticforce + dampingforce

    rest_rotation_matrix = Rotation.from_euler(
        "xyz", rest_euler_angle, degrees=False
    ).as_matrix()
    fxjt = FixedJoint(k, nu, kt, nut, rest_rotation_matrix=rest_rotation_matrix)

    fxjt.apply_forces(rod1, rod1_index, rod2, rod2_index)
    fxjt.apply_torques(rod1, rod1_index, rod2, rod2_index)

    assert_allclose(
        rod1.external_forces[..., rod1_index], contactforce, atol=Tolerance.atol()
    )
    assert_allclose(
        rod2.external_forces[..., rod2_index], -1 * contactforce, atol=Tolerance.atol()
    )

    # collect directors of systems one and two
    # note that systems can be either rods or rigid bodies
    rod1_director = rod1.director_collection[..., rod1_index]
    rod2_director = rod2.director_collection[..., rod2_index]

    # rel_rot: C_12 = C_1I @ C_I2
    # C_12 is relative rotation matrix from system 1 to system 2
    # C_1I is the rotation from system 1 to the inertial frame (i.e. the world frame)
    # C_I2 is the rotation from the inertial frame to system 2 frame (inverse of system_two_director)
    rel_rot = rod1_director @ rod2_director.T
    # error_rot: C_22* = C_21 @ C_12*
    # C_22* is rotation matrix from current orientation of system 2 to desired orientation of system 2
    # C_21 is the inverse of C_12, which describes the relative (current) rotation from system 1 to system 2
    # C_12* is the desired rotation between systems one and two, which is saved in the static_rotation attribute
    dev_rot = rel_rot.T @ rest_rotation_matrix

    # compute rotation vectors based on C_22*
    # scipy implementation
    rot_vec = _inv_rotate(np.dstack([np.eye(3), dev_rot.T])).squeeze()

    # rotate rotation vector into inertial frame
    rot_vec_inertial_frame = rod2_director.T @ rot_vec

    # deviation in rotation velocity between system 1 and system 2
    # first convert to inertial frame, then take differences
    dev_omega = (
        rod2_director.T @ rod2.omega_collection[..., rod2_index]
        - rod1_director.T @ rod1.omega_collection[..., rod1_index]
    )

    # we compute the constraining torque using a rotational spring - damper system in the inertial frame
    torque = kt * rot_vec_inertial_frame - nut * dev_omega

    # The opposite torques will be applied to system one and two after rotating the torques into the local frame
    torque_rod1 = -rod1_director @ torque
    torque_rod2 = rod2_director @ torque

    assert_allclose(
        rod1.external_torques[..., rod1_index], torque_rod1, atol=Tolerance.atol()
    )
    assert_allclose(
        rod2.external_torques[..., rod2_index], torque_rod2, atol=Tolerance.atol()
    )
