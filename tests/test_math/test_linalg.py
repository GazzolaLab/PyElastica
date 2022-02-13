#!/usr/bin/env python3
__doc__ = (
    """ Test scripts for linear algebra helpers in Elastica Numba implementation"""
)
# System imports
import numpy as np
import pytest
from numpy.testing import assert_allclose
from elastica._linalg import (
    _batch_matvec,
    _batch_matmul,
    _batch_cross,
    _batch_vec_oneD_vec_cross,
    _batch_dot,
    _batch_norm,
    _batch_product_i_k_to_ik,
    _batch_product_i_ik_to_k,
    _batch_product_k_ik_to_ik,
    _batch_vector_sum,
    _batch_matrix_transpose,
)


@pytest.mark.parametrize("blocksize", [8, 32])
def test_batch_matvec(blocksize):
    input_matrix_collection = np.random.randn(3, 3, blocksize)
    input_vector_collection = np.random.randn(3, blocksize)

    test_vector_collection = _batch_matvec(
        input_matrix_collection, input_vector_collection
    )

    correct_vector_collection = [
        np.dot(input_matrix_collection[..., i], input_vector_collection[..., i])
        for i in range(blocksize)
    ]
    correct_vector_collection = np.array(correct_vector_collection).T

    assert_allclose(test_vector_collection, correct_vector_collection)


@pytest.mark.parametrize("blocksize", [8, 32])
def test_batch_matmul(blocksize):
    input_first_matrix_collection = np.random.randn(3, 3, blocksize)
    input_second_matrix_collection = np.random.randn(3, 3, blocksize)

    test_matrix_collection = _batch_matmul(
        input_first_matrix_collection, input_second_matrix_collection
    )

    correct_matrix_collection = np.empty((3, 3, blocksize))
    for i in range(blocksize):
        correct_matrix_collection[..., i] = np.dot(
            input_first_matrix_collection[..., i],
            input_second_matrix_collection[..., i],
        )

    assert_allclose(test_matrix_collection, correct_matrix_collection)


# TODO : Generalize to two dimensions
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("blocksize", [8, 32])
def test_batch_cross(dim, blocksize):
    input_first_vector_collection = np.random.randn(dim, blocksize)
    input_second_vector_collection = np.random.randn(dim, blocksize)

    test_vector_collection = _batch_cross(
        input_first_vector_collection, input_second_vector_collection
    )
    correct_vector_collection = np.cross(
        input_first_vector_collection, input_second_vector_collection, axisa=0, axisb=0
    ).T

    assert_allclose(test_vector_collection, correct_vector_collection)


@pytest.mark.parametrize("blocksize", [8, 32])
def test_batch_vec_oneD_vec_cross(blocksize):
    input_first_vector_collection = np.random.randn(3, blocksize)
    input_second_vector = np.random.randn(3)

    test_vector_collection = _batch_vec_oneD_vec_cross(
        input_first_vector_collection, input_second_vector
    )

    correct_vector_collection = np.cross(
        input_first_vector_collection, input_second_vector, axisa=0, axisb=0
    ).T

    assert_allclose(test_vector_collection, correct_vector_collection)


@pytest.mark.parametrize("blocksize", [8, 32])
def test_batch_dot(blocksize):
    input_first_vector_collection = np.random.randn(3, blocksize)
    input_second_vector_collection = np.random.randn(3, blocksize)

    test_vector_collection = _batch_dot(
        input_first_vector_collection, input_second_vector_collection
    )

    correct_vector_collection = np.einsum(
        "ij,ij->j", input_first_vector_collection, input_second_vector_collection
    )

    assert_allclose(test_vector_collection, correct_vector_collection)


@pytest.mark.parametrize("blocksize", [8, 32])
def test_batch_norm(blocksize):
    input_first_vector_collection = np.random.randn(3, blocksize)

    test_vector_collection = _batch_norm(input_first_vector_collection)

    correct_vector_collection = np.sqrt(
        np.einsum(
            "ij,ij->j", input_first_vector_collection, input_first_vector_collection
        )
    )

    assert_allclose(test_vector_collection, correct_vector_collection)


@pytest.mark.parametrize("blocksize", [8, 32])
def test_batch_product_i_k_to_ik(blocksize):
    input_first_vector_collection = np.random.randn(3)
    input_second_vector_collection = np.random.randn(blocksize)

    test_vector_collection = _batch_product_i_k_to_ik(
        input_first_vector_collection, input_second_vector_collection
    )

    correct_vector_collection = np.einsum(
        "i,j->ij", input_first_vector_collection, input_second_vector_collection
    )

    assert_allclose(test_vector_collection, correct_vector_collection)


@pytest.mark.parametrize("blocksize", [8, 32])
def test_batch_product_i_ik_to_k(blocksize):
    input_first_vector_collection = np.random.randn(3)
    input_second_vector_collection = np.random.randn(3, blocksize)

    test_vector_collection = _batch_product_i_ik_to_k(
        input_first_vector_collection, input_second_vector_collection
    )

    correct_vector_collection = np.einsum(
        "i,ij->j", input_first_vector_collection, input_second_vector_collection
    )

    assert_allclose(test_vector_collection, correct_vector_collection)


@pytest.mark.parametrize("blocksize", [8, 32])
def test_batch_product_k_ik_to_ik(blocksize):
    input_first_vector_collection = np.random.randn(blocksize)
    input_second_vector_collection = np.random.randn(3, blocksize)

    test_vector_collection = _batch_product_k_ik_to_ik(
        input_first_vector_collection, input_second_vector_collection
    )

    correct_vector_collection = np.einsum(
        "j,ij->ij", input_first_vector_collection, input_second_vector_collection
    )

    assert_allclose(test_vector_collection, correct_vector_collection)


@pytest.mark.parametrize("blocksize", [8, 32])
def test_batch_vector_sum(blocksize):
    input_first_vector_collection = np.random.randn(3, blocksize)
    input_second_vector_collection = np.random.randn(3, blocksize)

    test_vector_collection = _batch_vector_sum(
        input_first_vector_collection, input_second_vector_collection
    )

    correct_vector_collection = (
        input_first_vector_collection + input_second_vector_collection
    )

    assert_allclose(test_vector_collection, correct_vector_collection)


@pytest.mark.parametrize("blocksize", [8, 32])
def test_batch_matrix_transpose(blocksize):
    input_matrix_collection = np.random.randn(3, 3, blocksize)

    test_matrix_collection = _batch_matrix_transpose(input_matrix_collection)

    correct_matrix_collection = np.einsum("ijk->jik", input_matrix_collection)

    assert_allclose(test_matrix_collection, correct_matrix_collection)
