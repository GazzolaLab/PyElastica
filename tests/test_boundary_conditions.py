__doc__ = """ Boundary conditions for rod test module """
import sys
sys.path.append("..")

# System imports
import numpy as np
from test_rod import TestRod
from elastica.boundary_conditions import FreeRod, OneEndFixedRod
from numpy.testing import assert_allclose, assert_array_equal
from elastica.utils import Tolerance
from elastica._linalg import _batch_matmul, _batch_matvec, _batch_cross


# tests free rod boundary conditions
def test_free_rod():

    test_rod = TestRod()
    free_rod = FreeRod(test_rod)
    test_position = np.random.rand(3, 20)
    test_rod.position = test_position
    test_directors = np.random.rand(3, 3, 20)
    test_rod.directors = test_directors
    free_rod.dirichlet()
    assert_allclose(test_position, test_rod.position,
                    atol=Tolerance.atol())
    assert_allclose(test_directors, test_rod.directors,
                    atol=Tolerance.atol())

    test_velocity = np.random.rand(3, 20)
    test_rod.velocity = test_velocity
    test_omega = np.random.rand(3, 20)
    test_rod.omega = test_omega
    free_rod.neumann()
    assert_allclose(test_velocity, test_rod.velocity,
                    atol=Tolerance.atol())
    assert_allclose(test_omega, test_rod.omega,
                    atol=Tolerance.atol())


def test_one_end_fixed_rod():

    test_rod = TestRod()
    start_position = np.random.rand(3)
    start_directors = np.random.rand(3, 3)
    fixed_rod = (OneEndFixedRod(test_rod, start_position,
                 start_directors))
    test_position = np.random.rand(3, 20)
    test_rod.position = test_position
    test_directors = np.random.rand(3, 3, 20)
    test_rod.directors = test_directors
    fixed_rod.dirichlet()
    test_position[..., 0] = start_position
    test_directors[..., 0] = start_directors
    assert_allclose(test_position, test_rod.position,
                    atol=Tolerance.atol())
    assert_allclose(test_directors, test_rod.directors,
                    atol=Tolerance.atol())

    test_velocity = np.random.rand(3, 20)
    test_rod.velocity = test_velocity
    test_omega = np.random.rand(3, 20)
    test_rod.omega = test_omega
    fixed_rod.neumann()
    test_velocity[..., 0] = np.array((0, 0, 0))
    test_omega[..., 0] = np.array((0, 0, 0))
    assert_allclose(test_velocity, test_rod.velocity,
                    atol=Tolerance.atol())
    assert_allclose(test_omega, test_rod.omega,
                    atol=Tolerance.atol())


if __name__ == "__main__":
    from pytest import main

    main([__file__])
