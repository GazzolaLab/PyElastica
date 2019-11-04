__doc__ = """ Joint between rods test module """

# System imports
import numpy as np
from elastica.joint import FreeJoint, HingeJoint, FixedJoint
from numpy.testing import assert_allclose
from elastica.utils import Tolerance
from elastica._rod import CosseratRod


def test_freejoint():

    # Some rod properties. We need them for constructer, they are not used.
    normal = np.array([0.0, 1.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    base_length = 1
    base_radius = 0.2
    density = 1
    shear_matrix = np.zeros((3, 3))
    mass_second_moment_of_inertia = np.zeros((3, 3))
    bend_matrix = np.zeros((3, 3))
    nu = 0.1

    # Origin of the rod
    origin1 = np.array([0.0, 0.0, 0.0])
    origin2 = np.array([1.1, 0.0, 0.0])

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
        mass_second_moment_of_inertia,
        shear_matrix,
        bend_matrix,
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
        mass_second_moment_of_inertia,
        shear_matrix,
        bend_matrix,
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

    rod1.velocity = np.tile(v1, (1, n + 1))
    rod2.velocity = np.tile(v2, (1, n + 1))

    # Compute the free joint forces
    distance = rod2.position[..., rod2_index] - rod1.position[..., rod1_index]
    end_distance = np.linalg.norm(distance)
    if end_distance == 0:
        end_distance = 1
    elasticforce = k * distance
    relative_vel = rod2.velocity[..., rod2_index] - rod1.velocity[..., rod1_index]
    normal_relative_vel = np.dot(relative_vel, distance) / end_distance
    dampingforce = nu * normal_relative_vel * distance / end_distance
    contactforce = elasticforce - dampingforce

    frjt = FreeJoint(k, nu, rod1, rod2, rod1_index, rod2_index)

    frjt.apply_force()

    assert_allclose(
        frjt.rod_one.external_forces[..., rod1_index],
        contactforce,
        atol=Tolerance.atol(),
    )
    assert_allclose(
        frjt.rod_two.external_forces[..., rod2_index],
        -1 * contactforce,
        atol=Tolerance.atol(),
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
    shear_matrix = np.zeros((3, 3))
    mass_second_moment_of_inertia = np.zeros((3, 3))
    bend_matrix = np.zeros((3, 3))
    nu = 0.1

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
        mass_second_moment_of_inertia,
        shear_matrix,
        bend_matrix,
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
        mass_second_moment_of_inertia,
        shear_matrix,
        bend_matrix,
    )

    # Rod velocity
    v1 = np.array([-1, 0, 0]).reshape(3, 1)
    v2 = v1 * -1

    rod1.velocity = np.tile(v1, (1, n + 1))
    rod2.velocity = np.tile(v2, (1, n + 1))

    # Stiffness between points
    k = 1e8
    kt = 1e6
    # Damping between two points
    nu = 1

    # Rod indexes
    rod1_index = -1
    rod2_index = 0

    # Compute the free joint forces
    distance = rod2.position[..., rod2_index] - rod1.position[..., rod1_index]
    end_distance = np.linalg.norm(distance)
    if end_distance == 0:
        end_distance = 1
    elasticforce = k * distance
    relative_vel = rod2.velocity[..., rod2_index] - rod1.velocity[..., rod1_index]
    normal_relative_vel = np.dot(relative_vel, distance) / end_distance
    dampingforce = nu * normal_relative_vel * distance / end_distance
    contactforce = elasticforce - dampingforce

    hgjt = HingeJoint(k, nu, rod1, rod2, rod1_index, rod2_index, kt, normal1)

    hgjt.apply_force()
    hgjt.apply_torque()

    assert_allclose(
        hgjt.rod_one.external_forces[..., rod1_index],
        contactforce,
        atol=Tolerance.atol(),
    )
    assert_allclose(
        hgjt.rod_two.external_forces[..., rod2_index],
        -1 * contactforce,
        atol=Tolerance.atol(),
    )

    linkdirection = rod2.position[..., rod2_index + 1] - rod2.position[..., rod2_index]
    forcedirection = np.dot(linkdirection, normal1) * normal1
    torque = -kt * np.cross(linkdirection, forcedirection)

    assert_allclose(hgjt.torque, torque, atol=Tolerance.atol())


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
    shear_matrix = np.zeros((3, 3))
    mass_second_moment_of_inertia = np.zeros((3, 3))
    bend_matrix = np.zeros((3, 3))
    nu = 0.1

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
        mass_second_moment_of_inertia,
        shear_matrix,
        bend_matrix,
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
        mass_second_moment_of_inertia,
        shear_matrix,
        bend_matrix,
    )

    # Rod velocity
    v1 = np.array([-1, 0, 0]).reshape(3, 1)
    v2 = v1 * -1

    rod1.velocity = np.tile(v1, (1, n + 1))
    rod2.velocity = np.tile(v2, (1, n + 1))

    # Stiffness between points
    k = 1e8
    kt = 1e6
    # Damping between two points
    nu = 1

    # Rod indexes
    rod1_index = -1
    rod2_index = 0

    # Compute the free joint forces
    distance = rod2.position[..., rod2_index] - rod1.position[..., rod1_index]
    end_distance = np.linalg.norm(distance)
    if end_distance == 0:
        end_distance = 1
    elasticforce = k * distance
    relative_vel = rod2.velocity[..., rod2_index] - rod1.velocity[..., rod1_index]
    normal_relative_vel = np.dot(relative_vel, distance) / end_distance
    dampingforce = nu * normal_relative_vel * distance / end_distance
    contactforce = elasticforce - dampingforce

    fxjt = FixedJoint(k, nu, rod1, rod2, rod1_index, rod2_index, kt)

    fxjt.apply_force()
    fxjt.apply_torque()

    assert_allclose(
        fxjt.rod_one.external_forces[..., rod1_index],
        contactforce,
        atol=Tolerance.atol(),
    )
    assert_allclose(
        fxjt.rod_two.external_forces[..., rod2_index],
        -1 * contactforce,
        atol=Tolerance.atol(),
    )

    linkdirection = rod2.position[..., rod2_index + 1] - rod2.position[..., rod2_index]

    positiondiff = rod1.position[..., rod1_index] - rod1.position[..., rod1_index - 1]
    tangent = positiondiff / np.sqrt(np.dot(positiondiff, positiondiff))

    # rod 2 has to be alligned with rod 1
    check1 = rod1.position[..., rod1_index] + rod2.rest_lengths[rod2_index] * tangent
    check2 = rod2.position[..., rod2_index + 1]
    forcedirection = -kt * (check2 - check1)
    torque = np.cross(linkdirection, forcedirection)

    assert_allclose(fxjt.torque, torque, atol=Tolerance.atol())


if __name__ == "__main__":
    from pytest import main

    main([__file__])
