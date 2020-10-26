__doc__ = (
    """ Boundary conditions for rod test module in Elastica Numba implementation"""
)
import sys

# System imports
import numpy as np
from test_rod_nb import TestRod
from elastica._elastica_numba._boundary_conditions import (
    FreeRod,
    OneEndFixedRod,
    HelicalBucklingBC,
)
from numpy.testing import assert_allclose
from elastica.utils import Tolerance
from pytest import main


# tests free rod boundary conditions
def test_free_rod():
    test_rod = TestRod()
    free_rod = FreeRod()
    test_position_collection = np.random.rand(3, 20)
    test_rod.position_collection = (
        test_position_collection.copy()
    )  # We need copy of the list not a reference to this array
    test_director_collection = np.random.rand(3, 3, 20)
    test_rod.director_collection = (
        test_director_collection.copy()
    )  # We need copy of the list not a reference to this array
    free_rod.constrain_values(test_rod, time=0)
    assert_allclose(
        test_position_collection, test_rod.position_collection, atol=Tolerance.atol()
    )
    assert_allclose(
        test_director_collection, test_rod.director_collection, atol=Tolerance.atol()
    )

    test_velocity_collection = np.random.rand(3, 20)
    test_rod.velocity_collection = (
        test_velocity_collection.copy()
    )  # We need copy of the list not a reference to this array
    test_omega_collection = np.random.rand(3, 20)
    test_rod.omega_collection = (
        test_omega_collection.copy()
    )  # We need copy of the list not a reference to this array
    free_rod.constrain_rates(test_rod, time=0)
    assert_allclose(
        test_velocity_collection, test_rod.velocity_collection, atol=Tolerance.atol()
    )
    assert_allclose(
        test_omega_collection, test_rod.omega_collection, atol=Tolerance.atol()
    )


def test_one_end_fixed_rod():

    test_rod = TestRod()
    start_position_collection = np.random.rand(3)
    start_director_collection = np.random.rand(3, 3)
    fixed_rod = OneEndFixedRod(start_position_collection, start_director_collection)
    test_position_collection = np.random.rand(3, 20)
    test_rod.position_collection = (
        test_position_collection.copy()
    )  # We need copy of the list not a reference to this array
    test_director_collection = np.random.rand(3, 3, 20)
    test_rod.director_collection = (
        test_director_collection.copy()
    )  # We need copy of the list not a reference to this array
    fixed_rod.constrain_values(test_rod, time=0)
    test_position_collection[..., 0] = start_position_collection
    test_director_collection[..., 0] = start_director_collection
    assert_allclose(
        test_position_collection, test_rod.position_collection, atol=Tolerance.atol()
    )
    assert_allclose(
        test_director_collection, test_rod.director_collection, atol=Tolerance.atol()
    )

    test_velocity_collection = np.random.rand(3, 20)
    test_rod.velocity_collection = (
        test_velocity_collection.copy()
    )  # We need copy of the list not a reference to this array
    test_omega_collection = np.random.rand(3, 20)
    test_rod.omega_collection = (
        test_omega_collection.copy()
    )  # We need copy of the list not a reference to this array
    fixed_rod.constrain_rates(test_rod, time=0)
    test_velocity_collection[..., 0] = np.array((0, 0, 0))
    test_omega_collection[..., 0] = np.array((0, 0, 0))
    assert_allclose(
        test_velocity_collection, test_rod.velocity_collection, atol=Tolerance.atol()
    )
    assert_allclose(
        test_omega_collection, test_rod.omega_collection, atol=Tolerance.atol()
    )


def test_helical_buckling_bc():

    twisting_time = 500.0
    slack = 3.0
    number_of_rotations = 27.0  # number of 2pi rotations
    start_position_collection = np.array([0.0, 0.0, 0.0])
    start_director_collection = np.identity(3, float)
    end_position_collection = np.array([100.0, 0.0, 0.0])
    end_director_collection = np.identity(3, float)

    test_rod = TestRod()

    test_position_collection = np.random.rand(3, 20)
    test_position_collection[..., 0] = start_position_collection
    test_position_collection[..., -1] = end_position_collection
    test_rod.position_collection = (
        test_position_collection.copy()
    )  # We need copy of the list not a reference to this array
    test_director_collection = np.tile(np.identity(3, float), 20).reshape(3, 3, 20)
    test_director_collection[..., 0] = start_director_collection
    test_director_collection[..., -1] = end_director_collection
    test_rod.director_collection = (
        test_director_collection.copy()
    )  # We need copy of the list not a reference to this array

    test_velocity_collection = np.random.rand(3, 20)
    test_rod.velocity_collection = (
        test_velocity_collection.copy()
    )  # We need copy of the list not a reference to this array
    test_omega_collection = np.random.rand(3, 20)
    test_rod.omega_collection = (
        test_omega_collection.copy()
    )  # We need copy of the list not a reference to this array
    position_collection_start = test_rod.position_collection[..., 0]
    position_collection_end = test_rod.position_collection[..., -1]
    director_start = test_rod.director_collection[..., 0]
    director_end = test_rod.director_collection[..., -1]
    helicalbuckling_rod = HelicalBucklingBC(
        position_collection_start,
        position_collection_end,
        director_start,
        director_end,
        twisting_time,
        slack,
        number_of_rotations,
    )

    # Check Neumann BC
    # time < twisting time
    time = twisting_time - 1.0

    helicalbuckling_rod.constrain_rates(test_rod, time=time)
    test_velocity_collection[..., 0] = np.array([0.003, 0.0, 0.0])
    test_velocity_collection[..., -1] = -np.array([0.003, 0.0, 0.0])
    test_omega_collection[..., 0] = np.array([0.169646, 0.0, 0.0])
    test_omega_collection[..., -1] = -np.array([0.169646, 0.0, 0.0])

    assert_allclose(
        test_velocity_collection, test_rod.velocity_collection, atol=Tolerance.atol()
    )
    assert_allclose(
        test_omega_collection, test_rod.omega_collection, atol=Tolerance.atol()
    )

    # time > twisting time
    time = twisting_time + 1
    helicalbuckling_rod.constrain_rates(test_rod, time=time)
    test_velocity_collection[..., 0] = np.array((0, 0, 0))
    test_velocity_collection[..., -1] = np.array((0, 0, 0))
    test_omega_collection[..., 0] = np.array((0, 0, 0))
    test_omega_collection[..., -1] = np.array((0, 0, 0))
    assert_allclose(
        test_velocity_collection, test_rod.velocity_collection, atol=Tolerance.atol()
    )
    assert_allclose(
        test_omega_collection, test_rod.omega_collection, atol=Tolerance.atol()
    )

    # Check Dirichlet BC

    helicalbuckling_rod.constrain_values(test_rod, time=time)

    test_position_collection[..., 0] = np.array([1.5, 0.0, 0.0])
    test_position_collection[..., -1] = np.array([98.5, 0.0, 0.0])

    test_director_collection[..., 0] = np.array(
        [[1.0, 0.0, 0.0], [0.0, -1.0, -6.85926004e-15], [0.0, 6.85926004e-15, -1.0]]
    )

    test_director_collection[..., -1] = np.array(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 6.85926004e-15], [0.0, -6.85926004e-15, -1.0]]
    )

    assert_allclose(
        test_position_collection, test_rod.position_collection, atol=Tolerance.atol()
    )
    assert_allclose(
        test_director_collection, test_rod.director_collection, atol=Tolerance.atol()
    )


if __name__ == "__main__":
    main([__file__])
