import numpy as np
from numpy.testing import assert_allclose

from elastica._synchronize_periodic_boundary import (
    _synchronize_periodic_boundary_of_matrix_collection,
    _synchronize_periodic_boundary_of_vector_collection,
    _synchronize_periodic_boundary_of_scalar_collection,
    _ConstrainPeriodicBoundaries,
)
from elastica.utils import Tolerance
import pytest
from tests.test_rod.test_rods import MockTestRingRod


@pytest.mark.parametrize("n_elems", [10, 30, 40])
def test_synchronize_periodic_boundary_vector(n_elems):
    """
    Testing the validity of _synchronize_periodic_boundary_of_vector_collection function.

    Parameters
    ----------
    n_elems

    Returns
    -------

    """

    input_vector = np.random.random((3, n_elems + 3))

    periodic_idx = np.zeros((2, 3), dtype=np.int64)
    periodic_idx[0, 0] = 0
    periodic_idx[0, 1] = -2
    periodic_idx[0, 2] = -1

    periodic_idx[1, 0] = -3
    periodic_idx[1, 1] = 1
    periodic_idx[1, 2] = 2

    correct_vector = input_vector.copy()
    correct_vector[..., 0] = input_vector[..., -3]
    correct_vector[..., -2] = input_vector[..., 1]
    correct_vector[..., -1] = input_vector[..., 2]

    _synchronize_periodic_boundary_of_vector_collection(input_vector, periodic_idx)

    assert_allclose(correct_vector, input_vector, atol=Tolerance.atol())


@pytest.mark.parametrize("n_elems", [10, 30, 40])
def test_synchronize_periodic_boundary_matrix(n_elems):
    """
    Testing the validity of _synchronize_periodic_boundary_of_matrix_collection function.

    Parameters
    ----------
    n_elems

    Returns
    -------

    """

    input_matrix = np.random.random((3, 3, n_elems + 3))

    periodic_idx = np.zeros((2, 3), dtype=np.int64)
    periodic_idx[0, 0] = 0
    periodic_idx[0, 1] = -2
    periodic_idx[0, 2] = -1

    periodic_idx[1, 0] = -3
    periodic_idx[1, 1] = 1
    periodic_idx[1, 2] = 2

    correct_matrix = input_matrix.copy()
    correct_matrix[..., 0] = input_matrix[..., -3]
    correct_matrix[..., -2] = input_matrix[..., 1]
    correct_matrix[..., -1] = input_matrix[..., 2]

    _synchronize_periodic_boundary_of_matrix_collection(input_matrix, periodic_idx)

    assert_allclose(correct_matrix, input_matrix, atol=Tolerance.atol())


@pytest.mark.parametrize("n_elems", [10, 30, 40])
def test_synchronize_periodic_boundary_scalar(n_elems):
    """
    Testing the validity of _synchronize_periodic_boundary_of_scalar_collection function.

    Parameters
    ----------
    n_elems

    Returns
    -------

    """

    input_matrix = np.random.random((n_elems + 3))

    periodic_idx = np.zeros((2, 3), dtype=np.int64)
    periodic_idx[0, 0] = 0
    periodic_idx[0, 1] = -2
    periodic_idx[0, 2] = -1

    periodic_idx[1, 0] = -3
    periodic_idx[1, 1] = 1
    periodic_idx[1, 2] = 2

    correct_matrix = input_matrix.copy()
    correct_matrix[..., 0] = input_matrix[..., -3]
    correct_matrix[..., -2] = input_matrix[..., 1]
    correct_matrix[..., -1] = input_matrix[..., 2]

    _synchronize_periodic_boundary_of_scalar_collection(input_matrix, periodic_idx)

    assert_allclose(correct_matrix, input_matrix, atol=Tolerance.atol())


def test_ConstrainPeriodicBoundaries():
    n_elem = 29
    test_rod = MockTestRingRod()
    fixed_rod = _ConstrainPeriodicBoundaries(_system=test_rod)
    test_position_collection = np.random.rand(3, n_elem + 3)
    test_rod.position_collection = (
        test_position_collection.copy()
    )  # We need copy of the list not a reference to this array
    test_director_collection = np.random.rand(3, 3, n_elem + 2)
    test_rod.director_collection = (
        test_director_collection.copy()
    )  # We need copy of the list not a reference to this array
    fixed_rod.constrain_values(test_rod, time=0)

    periodic_boundary_node_idx = np.array([[0, n_elem + 1, n_elem + 2], [n_elem, 1, 2]])
    periodic_boundary_elems_idx = np.array([[0, n_elem + 1], [n_elem, 1]])

    for i in range(3):
        for k in range(periodic_boundary_node_idx.shape[1]):
            test_position_collection[
                i, periodic_boundary_node_idx[0, k]
            ] = test_position_collection[i, periodic_boundary_node_idx[1, k]]

    for i in range(3):
        for j in range(3):
            for k in range(periodic_boundary_elems_idx.shape[1]):
                test_director_collection[
                    i, j, periodic_boundary_elems_idx[0, k]
                ] = test_director_collection[i, j, periodic_boundary_elems_idx[1, k]]

    assert_allclose(
        test_position_collection, test_rod.position_collection, atol=Tolerance.atol()
    )
    assert_allclose(
        test_director_collection, test_rod.director_collection, atol=Tolerance.atol()
    )

    test_velocity_collection = np.random.rand(3, n_elem + 3)
    test_rod.velocity_collection = (
        test_velocity_collection.copy()
    )  # We need copy of the list not a reference to this array
    test_omega_collection = np.random.rand(3, n_elem + 2)
    test_rod.omega_collection = (
        test_omega_collection.copy()
    )  # We need copy of the list not a reference to this array
    fixed_rod.constrain_rates(test_rod, time=0)

    for i in range(3):
        for k in range(periodic_boundary_node_idx.shape[1]):
            test_velocity_collection[
                i, periodic_boundary_node_idx[0, k]
            ] = test_velocity_collection[i, periodic_boundary_node_idx[1, k]]

    for i in range(3):
        for k in range(periodic_boundary_elems_idx.shape[1]):
            test_omega_collection[
                i, periodic_boundary_elems_idx[0, k]
            ] = test_omega_collection[i, periodic_boundary_elems_idx[1, k]]

    assert_allclose(
        test_velocity_collection, test_rod.velocity_collection, atol=Tolerance.atol()
    )
    assert_allclose(
        test_omega_collection, test_rod.omega_collection, atol=Tolerance.atol()
    )
