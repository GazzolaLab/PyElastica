__doc__ = """ Test scripts for rotation kernels in Elastica Numpy implementaiton """
# System imports
import numpy as np
import pytest
from numpy.testing import assert_allclose
import sys

from elastica._rotations import (
    _generate_skew_map,  # noqa
    _get_diag_map,  # noqa
    _get_inv_skew_map,  # noqa
    _get_skew_symmetric_pair,  # noqa
    _get_skew_map,  # noqa
    _inv_skew_symmetrize,  # noqa
    _skew_symmetrize,  # noqa
    _skew_symmetrize_sq,  # noqa
)


###############################################################################
##################### Implementation tests start ##############################
###############################################################################

# Cross products only make sense till dim = 3 (dim = 7 is an exception)
@pytest.mark.parametrize("dim", [2, 3])
def test_skewmap_integrity(dim):
    maps = _generate_skew_map(dim)
    for a_map in maps:
        src, tgt_i, tgt_j = a_map
        assert src < dim
        assert tgt_i < dim
        assert tgt_j < dim


skewmap_correctness_data = [(2, [(1, 1, 0)]), (3, [(2, 1, 0), (1, 0, 2), (0, 2, 1)])]


@pytest.mark.parametrize("dim, correct_list", skewmap_correctness_data)
def test_skewmap_correctness(dim, correct_list):
    assert _generate_skew_map(dim) == correct_list


matrix_to_vector_data = [(2, ((1, 1, 0),)), (3, ((0, 2, 1), (1, 0, 2), (2, 1, 0)))]


@pytest.mark.parametrize("dim, matrix_to_vector_map", matrix_to_vector_data)
def test_matrix_to_vector_map(dim, matrix_to_vector_map):
    assert _get_skew_map(dim) == matrix_to_vector_map


vector_to_matrix_data = [
    (2, ((1, 0, 1),)),  # Gotcha! tuple always has an extra , at the end
    (3, ((1, 0, 2), (0, 2, 1), (2, 1, 0))),
]


@pytest.mark.parametrize("dim, vector_to_matrix_map", vector_to_matrix_data)
def test_vector_to_matrix_map(dim, vector_to_matrix_map):
    assert _get_inv_skew_map(dim) == vector_to_matrix_map


matrix_diag_data = [(2, (0, 3)), (3, (0, 4, 8)), (4, (0, 5, 10, 15))]


@pytest.mark.parametrize("dim, diag_map", matrix_diag_data)
def test_matrix_diag_map(dim, diag_map):
    assert _get_diag_map(dim) == diag_map


def test_skew_symmetrize_correctness_in_two_dimensions():
    dim = 3
    vector = np.hstack((np.random.randn(2), 1))

    test_matrix = _skew_symmetrize(vector[:, np.newaxis])

    # Reshape for checking correctness and drop the last dimension
    test_matrix = test_matrix.reshape(dim, dim, -1)
    test_matrix = test_matrix[:-1, :-1, 0]

    correct_matrix = np.array([[0.0, -1.0], [1.0, 0.0]])

    assert_allclose(test_matrix, correct_matrix)


@pytest.mark.parametrize("blocksize", [1, 32, 128, 512])
def test_skew_symmetrize_correctness_in_three_dimensions(blocksize):
    dim = 3
    vector = np.random.randn(dim, blocksize)
    correct_matrix = np.zeros((dim * dim, blocksize))

    correct_matrix[1] = -vector[2]
    correct_matrix[2] = vector[1]
    correct_matrix[3] = vector[2]
    correct_matrix[5] = -vector[0]
    correct_matrix[6] = -vector[1]
    correct_matrix[7] = vector[0]

    correct_matrix = correct_matrix.reshape(dim, dim, -1)

    # reshape and squeeze because we are testing a single vector
    test_matrix = _skew_symmetrize(vector)

    assert test_matrix.shape == (3, 3, blocksize)
    assert_allclose(test_matrix, correct_matrix)


@pytest.mark.parametrize("blocksize", [1, 32, 128, 512])
def test_skew_symmetrize_sq_correctness_in_three_dimensions(blocksize):
    dim = 3
    vector = np.random.randn(dim, blocksize)
    correct_matrix = _skew_symmetrize(vector).reshape(dim, dim, -1)
    correct_matrix = np.einsum("ijk,jlk->ilk", correct_matrix, correct_matrix)

    # reshape and squeeze because we are testing a single vector
    test_matrix = _skew_symmetrize_sq(vector)

    assert test_matrix.shape == (3, 3, blocksize)
    assert_allclose(test_matrix, correct_matrix)


def test_get_skew_symmetric_pair_correctness():
    dim = 3
    blocksize = 8
    vector_collection = np.random.randn(dim, blocksize)
    u, u_sq = _get_skew_symmetric_pair(vector_collection)
    assert u_sq.shape == (3, 3, blocksize)
    assert_allclose(u_sq, _skew_symmetrize_sq(vector_collection))


@pytest.mark.parametrize("blocksize", [1, 32, 128, 512])
def test_inv_skew_symmetrize_correctness(blocksize):
    dim = 3
    vector = np.random.randn(dim, blocksize)
    input_matrix = _skew_symmetrize(vector)

    # reshape and squeeze because we are testing a single vector
    test_vector = _inv_skew_symmetrize(input_matrix)

    assert test_vector.shape == (3, blocksize)
    assert_allclose(test_vector, vector)


###############################################################################
##################### Implementation tests finis ##############################
###############################################################################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])
