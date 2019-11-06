__doc__ = """ Initialisation for rod test module """

import sys

# System imports
import numpy as np
from elastica._rod import CosseratRod
from numpy.testing import assert_allclose, assert_array_equal
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
    base_radius = np.random.rand(1)
    density = np.random.rand(1)
    mass = density * np.pi * base_radius ** 2 * base_length / n
    mass_second_moment_of_inertia = np.random.rand(3, 3)
    # shear_matrix = np.random.rand(3, 3)
    shear_matrix = np.ones((3, 3))
    shear_matrix[0, 1] = 2
    bend_matrix = np.random.rand(3, 3)
    nu = 0.1

    test_rod = CosseratRod.straight_rod(
        n,
        start,
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
    # checking origin and length of rod
    assert_allclose(test_rod.position[..., 0], start, atol=Tolerance.atol())
    rod_length = np.linalg.norm(test_rod.position[..., -1] - test_rod.position[..., 0])
    # checking velocities, omegas and rest strains
    # density and mass
    assert_allclose(rod_length, base_length, atol=Tolerance.atol())
    assert_allclose(test_rod.velocity, np.zeros((3, n + 1)), atol=Tolerance.atol())
    assert_allclose(test_rod.omega, np.zeros((3, n)), atol=Tolerance.atol())
    assert_allclose(test_rod.rest_sigma, np.zeros((3, n)), atol=Tolerance.atol())
    assert_allclose(test_rod.rest_kappa, np.zeros((3, n - 1)), atol=Tolerance.atol())
    assert_allclose(test_rod.density, density, atol=Tolerance.atol())
    assert_allclose(test_rod.nu, nu, atol=Tolerance.atol())
    # checking directors, rest length
    # and shear, bend matrices and moment of inertia
    for i in range(n):
        assert_allclose(test_rod.mass[i], mass, atol=Tolerance.atol())
        assert_allclose(test_rod.directors[0, :, i], normal, atol=Tolerance.atol())
        assert_allclose(test_rod.directors[1, :, i], direction, atol=Tolerance.atol())
        assert_allclose(
            test_rod.directors[2, :, i],
            np.cross(direction, normal),
            atol=Tolerance.atol(),
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
    for i in range(n - 1):
        assert_allclose(
            test_rod.bend_matrix[..., i], bend_matrix, atol=Tolerance.atol()
        )


if __name__ == "__main__":
    main([__file__])
