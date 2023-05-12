__doc__ = """Tests for cylinder module"""
import numpy as np
from numpy.testing import assert_allclose

from elastica.utils import Tolerance


from elastica.rigidbody import Cylinder


# tests Initialisation of cylinder
def test_cylinder_initialization():
    """
    This test case is for testing initialization of rigid cylinder and it checks the
    validity of the members of Cylinder class.

    Returns
    -------

    """
    # setting up test params
    start = np.random.rand(3)
    direction = 5 * np.random.rand(3)
    direction_norm = np.linalg.norm(direction)
    direction /= direction_norm
    normal = np.array((direction[1], -direction[0], 0))
    base_length = 10
    base_radius = np.random.uniform(1, 10)
    density = np.random.uniform(1, 10)
    mass = density * np.pi * base_radius ** 2 * base_length

    # Second moment of inertia
    A0 = np.pi * base_radius * base_radius
    I0_1 = A0 * A0 / (4.0 * np.pi)
    I0_2 = I0_1
    I0_3 = 2.0 * I0_2
    I0 = np.array([I0_1, I0_2, I0_3])
    # Mass second moment of inertia for disk cross-section
    mass_second_moment_of_inertia = np.zeros((3, 3), np.float64)
    np.fill_diagonal(mass_second_moment_of_inertia, I0 * density * base_length)
    # Inverse mass second of inertia
    inv_mass_second_moment_of_inertia = np.linalg.inv(mass_second_moment_of_inertia)

    test_rod = Cylinder(start, direction, normal, base_length, base_radius, density)
    # checking origin and length of rod
    assert_allclose(
        test_rod.position_collection[..., -1],
        start + base_length / 2 * direction,
        atol=Tolerance.atol(),
    )

    # element lengths are equal for all rod.
    # checking velocities, omegas and rest strains
    # density and mass
    rod_length = np.linalg.norm(test_rod.length)
    assert_allclose(rod_length, base_length, atol=Tolerance.atol())
    assert_allclose(
        test_rod.velocity_collection, np.zeros((3, 1)), atol=Tolerance.atol()
    )
    assert_allclose(test_rod.omega_collection, np.zeros((3, 1)), atol=Tolerance.atol())

    assert_allclose(test_rod.density, density, atol=Tolerance.atol())

    # Check mass at each node. Note that, node masses is
    # half of element mass at the first and last node.
    assert_allclose(test_rod.mass, mass, atol=Tolerance.atol())

    # checking directors, rest length
    # and shear, bend matrices and moment of inertia
    assert_allclose(
        test_rod.inv_mass_second_moment_of_inertia[..., -1],
        inv_mass_second_moment_of_inertia,
        atol=Tolerance.atol(),
    )


def test_cylinder_update_accelerations():
    """
    This test is testing the update acceleration method of cylinder class.

    Returns
    -------

    """

    start = np.random.rand(3)
    direction = 5 * np.random.rand(3)
    direction_norm = np.linalg.norm(direction)
    direction /= direction_norm
    normal = np.array((direction[1], -direction[0], 0))
    base_length = 10
    base_radius = np.random.uniform(1, 10)
    volume = np.pi * base_radius ** 2 * base_length
    density = np.random.uniform(1, 10)
    mass = volume * density
    test_cylinder = Cylinder(
        start, direction, normal, base_length, base_radius, density
    )

    inv_mass_second_moment_of_inertia = (
        test_cylinder.inv_mass_second_moment_of_inertia.reshape(3, 3)
    )

    external_forces = np.random.randn(3).reshape(3, 1)
    external_torques = np.random.randn(3).reshape(3, 1)

    correct_acceleration = external_forces / mass
    correct_alpha = inv_mass_second_moment_of_inertia @ external_torques.reshape(3)
    correct_alpha = correct_alpha.reshape(3, 1)

    test_cylinder.external_forces[:] = external_forces
    test_cylinder.external_torques[:] = external_torques

    test_cylinder.update_accelerations(time=0)

    assert_allclose(
        correct_acceleration,
        test_cylinder.acceleration_collection,
        atol=Tolerance.atol(),
    )
    assert_allclose(
        correct_alpha, test_cylinder.alpha_collection, atol=Tolerance.atol()
    )


def test_compute_position_center_of_mass():
    """
    This test is testing compute position center of mass method of Cylinder class.

    Returns
    -------

    """

    start = np.random.rand(3)
    direction = 5 * np.random.rand(3)
    direction_norm = np.linalg.norm(direction)
    direction /= direction_norm
    normal = np.array((direction[1], -direction[0], 0))
    base_length = 10
    base_radius = np.random.uniform(1, 10)
    density = np.random.uniform(1, 10)
    test_cylinder = Cylinder(
        start, direction, normal, base_length, base_radius, density
    )

    correct_position = start + direction * base_length / 2

    test_position = test_cylinder.compute_position_center_of_mass()

    assert_allclose(correct_position, test_position, atol=Tolerance.atol())


def test_compute_translational_energy():
    """
    This test is testing compute translational energy function.

    Returns
    -------

    """
    start = np.random.rand(3)
    direction = 5 * np.random.rand(3)
    direction_norm = np.linalg.norm(direction)
    direction /= direction_norm
    normal = np.array((direction[1], -direction[0], 0))
    base_length = 10
    base_radius = np.random.uniform(1, 10)
    volume = np.pi * base_radius ** 2 * base_length
    density = np.random.uniform(1, 10)
    mass = volume * density

    test_cylinder = Cylinder(
        start, direction, normal, base_length, base_radius, density
    )

    speed = np.random.randn()
    test_cylinder.velocity_collection[2] = speed

    correct_energy = 0.5 * mass * speed ** 2
    test_energy = test_cylinder.compute_translational_energy()

    assert_allclose(correct_energy, test_energy, atol=Tolerance.atol())


if __name__ == "__main__":
    from pytest import main

    main([__file__])
