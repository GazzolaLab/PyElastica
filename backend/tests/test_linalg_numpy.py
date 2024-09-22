#!/usr/bin/env python3

# This file is based on pyelastica tests/test_math/test_linalg.py

# System imports
import numpy as np
import pytest
from numpy.testing import assert_allclose
from elasticapp._linalg_numpy import (
    difference_kernel,
    batch_matvec,
    batch_matmul,
    batch_cross,
    batch_dot,
    batch_norm,
)


@pytest.mark.parametrize("blocksize", [1, 2, 8, 32])
def test_difference_kernel(blocksize: int):
    input_vector_collection = np.random.randn(3, blocksize)
    output_vector_collection = difference_kernel(input_vector_collection)

    correct_vector_collection = (
        input_vector_collection[:, 1:] - input_vector_collection[:, :-1]
    )

    assert_allclose(output_vector_collection, correct_vector_collection)


@pytest.mark.parametrize("blocksize", [1, 2, 8, 32])
def test_batch_matvec(blocksize: int):
    input_matrix_collection = np.random.randn(3, 3, blocksize)
    input_vector_collection = np.random.randn(3, blocksize)

    test_vector_collection = batch_matvec(
        input_matrix_collection, input_vector_collection
    )

    correct_vector_collection = [
        np.dot(input_matrix_collection[..., i], input_vector_collection[..., i])
        for i in range(blocksize)
    ]
    correct_vector_collection = np.array(correct_vector_collection).T

    assert_allclose(test_vector_collection, correct_vector_collection)


@pytest.mark.parametrize("blocksize", [1, 2, 8, 32])
def test_batch_matmul(blocksize: int):
    input_first_matrix_collection = np.random.randn(3, 3, blocksize)
    input_second_matrix_collection = np.random.randn(3, 3, blocksize)

    test_matrix_collection = batch_matmul(
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
@pytest.mark.parametrize("blocksize", [1, 2, 8, 32])
def test_batch_cross(dim, blocksize: int):
    input_first_vector_collection = np.random.randn(dim, blocksize)
    input_second_vector_collection = np.random.randn(dim, blocksize)

    test_vector_collection = batch_cross(
        input_first_vector_collection, input_second_vector_collection
    )
    correct_vector_collection = np.cross(
        input_first_vector_collection, input_second_vector_collection, axisa=0, axisb=0
    ).T

    assert_allclose(test_vector_collection, correct_vector_collection)


@pytest.mark.parametrize("blocksize", [1, 2, 8, 32])
def test_batch_dot(blocksize: int):
    input_first_vector_collection = np.random.randn(3, blocksize)
    input_second_vector_collection = np.random.randn(3, blocksize)

    test_vector_collection = batch_dot(
        input_first_vector_collection, input_second_vector_collection
    )

    correct_vector_collection = np.einsum(
        "ij,ij->j", input_first_vector_collection, input_second_vector_collection
    )

    assert_allclose(test_vector_collection, correct_vector_collection)


@pytest.mark.parametrize("blocksize", [1, 2, 8, 32])
def test_batch_norm(blocksize: int):
    input_first_vector_collection = np.random.randn(3, blocksize)

    test_vector_collection = batch_norm(input_first_vector_collection)

    correct_vector_collection = np.sqrt(
        np.einsum(
            "ij,ij->j", input_first_vector_collection, input_first_vector_collection
        )
    )

    assert_allclose(test_vector_collection, correct_vector_collection)
