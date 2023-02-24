__doc__ = """Tests for sphere module"""
import numpy as np
from numpy.testing import assert_allclose

from elastica.utils import Tolerance


from elastica.rigidbody import Sphere


# tests Initialisation of sphere
def test_sphere_initialization():
    """
    This test case is for testing initialization of rigid sphere and it checks the
    validity of the members of sphere class.

    Returns
    -------

    """
    # setting up test params
    start = np.random.rand(3)
    direction = np.array([0.0, 0.0, 1.0]).reshape(3, 1)
    normal = np.array([1.0, 0.0, 0.0]).reshape(3, 1)
    binormal = np.cross(direction[..., 0], normal[..., 0]).reshape(3, 1)
    base_radius = np.random.uniform(1, 10)
    volume = 4.0 / 3.0 * np.pi * base_radius ** 3
    density = np.random.uniform(1, 10)
    mass = density * volume

    # Mass second moment of inertia for disk cross-section
    mass_second_moment_of_inertia = np.zeros((3, 3), np.float64)
    np.fill_diagonal(mass_second_moment_of_inertia, 2.0 / 5.0 * mass * base_radius ** 2)
    # Inverse mass second of inertia
    inv_mass_second_moment_of_inertia = np.linalg.inv(mass_second_moment_of_inertia)

    test_sphere = Sphere(start, base_radius, density)
    # checking origin and length of rod
    assert_allclose(
        test_sphere.position_collection[..., -1],
        start,
        atol=Tolerance.atol(),
    )

    # element lengths are equal for all rod.
    # checking velocities, omegas and rest strains
    # density and mass
    assert_allclose(
        test_sphere.velocity_collection, np.zeros((3, 1)), atol=Tolerance.atol()
    )

    correct_director_collection = np.zeros((3, 3, 1))
    correct_director_collection[0] = normal
    correct_director_collection[1] = binormal
    correct_director_collection[2] = direction
    assert_allclose(
        test_sphere.director_collection,
        correct_director_collection,
        atol=Tolerance.atol(),
    )

    assert_allclose(
        test_sphere.omega_collection, np.zeros((3, 1)), atol=Tolerance.atol()
    )

    assert_allclose(test_sphere.density, density, atol=Tolerance.atol())

    # Check mass at each node. Note that, node masses is
    # half of element mass at the first and last node.
    assert_allclose(test_sphere.mass, mass, atol=Tolerance.atol())

    # checking directors, rest length
    # and shear, bend matrices and moment of inertia
    assert_allclose(
        test_sphere.inv_mass_second_moment_of_inertia[..., -1],
        inv_mass_second_moment_of_inertia,
        atol=Tolerance.atol(),
    )


def test_cylinder_update_accelerations():
    """
    This test is testing the update acceleration method of Sphere class.

    Returns
    -------

    """

    start = np.random.rand(3)
    base_radius = np.random.uniform(1, 10)
    volume = 4.0 / 3.0 * np.pi * base_radius ** 3
    density = np.random.uniform(1, 10)
    mass = density * volume
    test_sphere = Sphere(start, base_radius, density)

    inv_mass_second_moment_of_inertia = (
        test_sphere.inv_mass_second_moment_of_inertia.reshape(3, 3)
    )

    external_forces = np.random.randn(3).reshape(3, 1)
    external_torques = np.random.randn(3).reshape(3, 1)

    correct_acceleration = external_forces / mass
    correct_alpha = inv_mass_second_moment_of_inertia @ external_torques.reshape(3)
    correct_alpha = correct_alpha.reshape(3, 1)

    test_sphere.external_forces[:] = external_forces
    test_sphere.external_torques[:] = external_torques

    test_sphere.update_accelerations(time=0)

    assert_allclose(
        correct_acceleration, test_sphere.acceleration_collection, atol=Tolerance.atol()
    )
    assert_allclose(correct_alpha, test_sphere.alpha_collection, atol=Tolerance.atol())


def test_compute_position_center_of_mass():
    """
    This test is testing compute position center of mass method of Sphere class.

    Returns
    -------

    """

    start = np.random.rand(3)
    base_radius = np.random.uniform(1, 10)
    density = np.random.uniform(1, 10)
    test_sphere = Sphere(start, base_radius, density)

    correct_position = start

    test_position = test_sphere.compute_position_center_of_mass()

    assert_allclose(correct_position, test_position, atol=Tolerance.atol())


def test_compute_translational_energy():
    """
    This test is testing compute translational energy function.

    Returns
    -------

    """
    start = np.random.rand(3)
    base_radius = np.random.uniform(1, 10)
    volume = 4.0 / 3.0 * np.pi * base_radius ** 3
    density = np.random.uniform(1, 10)
    mass = density * volume
    test_sphere = Sphere(start, base_radius, density)

    speed = np.random.randn()
    test_sphere.velocity_collection[2] = speed

    correct_energy = 0.5 * mass * speed ** 2
    test_energy = test_sphere.compute_translational_energy()

    assert_allclose(correct_energy, test_energy, atol=Tolerance.atol())


if __name__ == "__main__":
    from pytest import main

    main([__file__])
