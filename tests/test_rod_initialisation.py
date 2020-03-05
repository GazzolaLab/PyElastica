__doc__ = """ Initialisation for rod test module """

# System imports
import numpy as np
from elastica.rod.cosserat_rod import CosseratRod
from elastica.rod.rigid_body import RigidBodyCyclinder
from numpy.testing import assert_allclose
from elastica.utils import Tolerance
from pytest import main


# tests Initialisation of straight rod
def test_straight_rod():

    # setting up test params
    n = 30
    start = np.random.rand(3)
    direction = 5 * np.random.rand(3)
    direction_norm = np.linalg.norm(direction)
    direction /= direction_norm
    normal = np.array((direction[1], -direction[0], 0))
    base_length = 10
    base_radius = np.random.uniform(1, 10)
    density = np.random.uniform(1, 10)
    mass = density * np.pi * base_radius ** 2 * base_length / n

    nu = 0.1
    # Youngs Modulus [Pa]
    E = 1e6
    # poisson ratio
    poisson_ratio = 0.5
    # Shear Modulus [Pa]
    G = E / (1.0 + poisson_ratio)
    # alpha c, constant for circular cross-sections
    # Second moment of inertia
    A0 = np.pi * base_radius * base_radius
    I0_1 = A0 * A0 / (4.0 * np.pi)
    I0_2 = I0_1
    I0_3 = 2.0 * I0_2
    I0 = np.array([I0_1, I0_2, I0_3])
    # Mass second moment of inertia for disk cross-section
    mass_second_moment_of_inertia = np.zeros((3, 3), np.float64)
    np.fill_diagonal(mass_second_moment_of_inertia, I0 * density * base_length / n)
    # Inverse mass second of inertia
    inv_mass_second_moment_of_inertia = np.linalg.inv(mass_second_moment_of_inertia)
    # Shear/Stretch matrix
    shear_matrix = np.zeros((3, 3), np.float64)
    np.fill_diagonal(shear_matrix, [4.0 * G * A0 / 3.0, 4.0 * G * A0 / 3.0, E * A0])
    # Bend/Twist matrix
    bend_matrix = np.zeros((3, 3), np.float64)
    np.fill_diagonal(bend_matrix, [E * I0_1, E * I0_2, G * I0_3])

    test_rod = CosseratRod.straight_rod(
        n,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        poisson_ratio,
    )
    # checking origin and length of rod
    assert_allclose(test_rod.position_collection[..., 0], start, atol=Tolerance.atol())
    rod_length = np.linalg.norm(
        test_rod.position_collection[..., -1] - test_rod.position_collection[..., 0]
    )
    rest_voronoi_lengths = 0.5 * (
        base_length / n + base_length / n
    )  # element lengths are equal for all rod.
    # checking velocities, omegas and rest strains
    # density and mass
    assert_allclose(rod_length, base_length, atol=Tolerance.atol())
    assert_allclose(
        test_rod.velocity_collection, np.zeros((3, n + 1)), atol=Tolerance.atol()
    )
    assert_allclose(test_rod.omega_collection, np.zeros((3, n)), atol=Tolerance.atol())
    assert_allclose(test_rod.rest_sigma, np.zeros((3, n)), atol=Tolerance.atol())
    assert_allclose(test_rod.rest_kappa, np.zeros((3, n - 1)), atol=Tolerance.atol())
    assert_allclose(test_rod.density, density, atol=Tolerance.atol())
    assert_allclose(test_rod.nu, nu, atol=Tolerance.atol())
    assert_allclose(
        rest_voronoi_lengths, test_rod.rest_voronoi_lengths, atol=Tolerance.atol()
    )
    # Check mass at each node. Note that, node masses is
    # half of element mass at the first and last node.
    for i in range(1, n):
        assert_allclose(test_rod.mass[i], mass, atol=Tolerance.atol())
    assert_allclose(test_rod.mass[0], 0.5 * mass, atol=Tolerance.atol())
    assert_allclose(test_rod.mass[-1], 0.5 * mass, atol=Tolerance.atol())
    # checking directors, rest length
    # and shear, bend matrices and moment of inertia
    for i in range(n):
        assert_allclose(
            test_rod.director_collection[0, :, i], normal, atol=Tolerance.atol()
        )
        assert_allclose(
            test_rod.director_collection[1, :, i],
            np.cross(direction, normal),
            atol=Tolerance.atol(),
        )
        assert_allclose(
            test_rod.director_collection[2, :, i], direction, atol=Tolerance.atol()
        )
        assert_allclose(test_rod.rest_lengths, base_length / n, atol=Tolerance.atol())
        assert_allclose(
            test_rod.shear_matrix[..., i], shear_matrix, atol=Tolerance.atol()
        )
        assert_allclose(
            test_rod.mass_second_moment_of_inertia[..., i],
            mass_second_moment_of_inertia,
            atol=Tolerance.atol(),
        )
        assert_allclose(
            test_rod.inv_mass_second_moment_of_inertia[..., i],
            inv_mass_second_moment_of_inertia,
            atol=Tolerance.atol(),
        )
    for i in range(n - 1):
        assert_allclose(
            test_rod.bend_matrix[..., i], bend_matrix, atol=Tolerance.atol()
        )


# tests Initialisation of straight rigid body rod
def test_straight_rigid_rod():
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

    test_rod = RigidBodyCyclinder(
        start, direction, normal, base_length, base_radius, density,
    )
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


if __name__ == "__main__":
    main([__file__])
