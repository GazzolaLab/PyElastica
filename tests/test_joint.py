__doc__ = """ Joint between rods test module """

# System imports
import numpy as np
from elastica.joint import FreeJoint, HingeJoint, FixedJoint
from numpy.testing import assert_allclose
from elastica.utils import Tolerance
from elastica.rod.cosserat_rod import CosseratRod
import importlib
import elastica

# TODO: change tests and made them independent of rod, at least assigin hardcoded values for forces and torques


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
        nu,
        youngs_modulus,
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
        nu,
        youngs_modulus,
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
    end_distance = np.linalg.norm(distance)
    if end_distance <= Tolerance.atol():
        end_distance = 1.0
    elasticforce = k * distance
    relative_vel = (
        rod2.velocity_collection[..., rod2_index]
        - rod1.velocity_collection[..., rod1_index]
    )
    normal_relative_vel = np.dot(relative_vel, distance) / end_distance
    dampingforce = nu * normal_relative_vel * distance / end_distance
    contactforce = elasticforce - dampingforce

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
        nu,
        youngs_modulus,
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
        nu,
        youngs_modulus,
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
    end_distance = np.linalg.norm(distance)
    if end_distance == 0:
        end_distance = 1
    elasticforce = k * distance
    relative_vel = (
        rod2.velocity_collection[..., rod2_index]
        - rod1.velocity_collection[..., rod1_index]
    )
    normal_relative_vel = np.dot(relative_vel, distance) / end_distance
    dampingforce = nu * normal_relative_vel * distance / end_distance
    contactforce = elasticforce - dampingforce

    hgjt = HingeJoint(k, nu, kt, normal1)

    hgjt.apply_forces(rod1, rod1_index, rod2, rod2_index)
    hgjt.apply_torques(rod1, rod1_index, rod2, rod2_index)

    assert_allclose(
        rod1.external_forces[..., rod1_index], contactforce, atol=Tolerance.atol()
    )
    assert_allclose(
        rod2.external_forces[..., rod2_index], -1 * contactforce, atol=Tolerance.atol()
    )

    linkdirection = (
        rod2.position_collection[..., rod2_index + 1]
        - rod2.position_collection[..., rod2_index]
    )
    forcedirection = np.dot(linkdirection, normal1) * normal1
    torque = -kt * np.cross(linkdirection, forcedirection)

    torque_rod1 = -rod1.director_collection[..., rod1_index] @ torque
    torque_rod2 = rod2.director_collection[..., rod2_index] @ torque

    assert_allclose(
        rod1.external_torques[..., rod1_index], torque_rod1, atol=Tolerance.atol()
    )
    assert_allclose(
        rod2.external_torques[..., rod2_index], torque_rod2, atol=Tolerance.atol()
    )


def test_fixedjoint():
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
        nu,
        youngs_modulus,
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
        nu,
        youngs_modulus,
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
    end_distance = np.linalg.norm(distance)
    if end_distance == 0:
        end_distance = 1
    elasticforce = k * distance
    relative_vel = (
        rod2.velocity_collection[..., rod2_index]
        - rod1.velocity_collection[..., rod1_index]
    )
    normal_relative_vel = np.dot(relative_vel, distance) / end_distance
    dampingforce = nu * normal_relative_vel * distance / end_distance
    contactforce = elasticforce - dampingforce

    fxjt = FixedJoint(k, nu, kt)

    fxjt.apply_forces(rod1, rod1_index, rod2, rod2_index)
    fxjt.apply_torques(rod1, rod1_index, rod2, rod2_index)

    assert_allclose(
        rod1.external_forces[..., rod1_index], contactforce, atol=Tolerance.atol()
    )
    assert_allclose(
        rod2.external_forces[..., rod2_index], -1 * contactforce, atol=Tolerance.atol()
    )

    linkdirection = (
        rod2.position_collection[..., rod2_index + 1]
        - rod2.position_collection[..., rod2_index]
    )

    positiondiff = (
        rod1.position_collection[..., rod1_index]
        - rod1.position_collection[..., rod1_index - 1]
    )
    tangent = positiondiff / np.sqrt(np.dot(positiondiff, positiondiff))

    # rod 2 has to be alligned with rod 1
    check1 = (
        rod1.position_collection[..., rod1_index]
        + rod2.rest_lengths[rod2_index] * tangent
    )
    check2 = rod2.position_collection[..., rod2_index + 1]
    forcedirection = -kt * (check2 - check1)
    torque = np.cross(linkdirection, forcedirection)

    torque_rod1 = -rod1.director_collection[..., rod1_index] @ torque
    torque_rod2 = rod2.director_collection[..., rod2_index] @ torque

    assert_allclose(
        rod1.external_torques[..., rod1_index], torque_rod1, atol=Tolerance.atol()
    )
    assert_allclose(
        rod2.external_torques[..., rod2_index], torque_rod2, atol=Tolerance.atol()
    )


if __name__ == "__main__":
    from pytest import main

    main([__file__])
