#!/usr/bin/env python3
""" Test scripts for transformation interface in elastica
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
import sys

from elastica.transformations import skew_symmetrize, inv_skew_symmetrize

from elastica._rotations import _skew_symmetrize


###############################################################################
########################## Interface tests start ##############################
###############################################################################
# We need this test to avoid confusion with situations where blocksize is small
# For example, with a block of (32, 3) the user clearly wants dim=3, bs=32
# But with a block of (3, 1) the user would have intended dim=3, bs=1
# To add more confusion a block of (1, 3) means we consider dim=3, bs=1 again
# This function tests for all blocksizes and see if we handle it sanely/
@pytest.mark.parametrize("blocksize", [(1), (2), (3), (4), (32), (128)])
def test_skew_symmetrize_handles_blocksizes(blocksize):
    dim = 3  # decides when to transpose
    vector = np.random.randn(dim, blocksize)
    input_matrix = skew_symmetrize(vector)
    test_matrix = _skew_symmetrize(vector)

    assert_allclose(input_matrix, test_matrix)


@pytest.mark.parametrize("ndim", [(1), (2)])
def test_skew_symmetrize_against_input_shapes(ndim):
    dim = 3
    blocksize = 32
    if ndim == 1:
        in_vector = np.random.randn(dim)
        vector = in_vector.reshape(dim, 1)
    elif ndim == 2:
        in_vector = np.random.randn(dim, blocksize)
        vector = in_vector

    input_matrix = skew_symmetrize(in_vector)
    test_matrix = _skew_symmetrize(vector)

    assert_allclose(input_matrix, test_matrix)


@pytest.mark.parametrize("ndim", [(1), (2)])
def test_skew_symmetrize_raises_dimension_error(ndim):
    dim = 4  # one greater than allowed dimension
    blocksize = 8
    if ndim == 1:
        incorrect_in_vector = np.random.randn(dim)
    elif ndim == 2:
        incorrect_in_vector = np.random.randn(dim, blocksize)

    with pytest.raises(AssertionError) as excinfo:
        skew_symmetrize(incorrect_in_vector)
    assert excinfo.type == AssertionError


def test_skew_symmetrize_raises_ndim_error():
    incorrect_in_vector = np.random.randn(2, 3, 5)

    with pytest.raises(RuntimeError) as excinfo:
        skew_symmetrize(incorrect_in_vector)
    assert excinfo.type == RuntimeError


def test_skew_symmetrize_transposes_for_two_dimensions():
    dim = 3
    blocksize = 32
    in_vector = np.random.randn(blocksize, dim)  # switched order
    vector = in_vector.T

    input_matrix = skew_symmetrize(in_vector)
    test_matrix = _skew_symmetrize(vector)

    assert_allclose(input_matrix, test_matrix)


# See the other handles_blocksizes function
@pytest.mark.parametrize("blocksize", [(1), (2), (3), (4), (32), (128)])
def test_inv_skew_symmetrize_handles_blocksizes(blocksize):
    dim = 3  # decides when to transpose
    input_vector = np.random.randn(dim, blocksize)
    input_matrix = skew_symmetrize(input_vector)  # tested

    # reshape and squeeze because we are testing a single vector
    test_vector = inv_skew_symmetrize(input_matrix)

    assert_allclose(test_vector, input_vector)


@pytest.mark.parametrize("ndim", [(1), (2), (3)])
def test_inv_skew_symmetrize_against_input_shapes(ndim):
    dim = 3

    # First generate a skew-symmetric matrix from vector
    if ndim == 1:
        blocksize = 1
    else:
        blocksize = 16

    input_vector = np.random.randn(dim, blocksize)

    input_matrix = _skew_symmetrize(input_vector)

    if ndim == 1:
        input_matrix = input_matrix.reshape(dim ** 2)
    elif ndim == 2:
        input_matrix = input_matrix.reshape(dim ** 2, blocksize)
    elif ndim == 3:
        input_matrix = input_matrix.reshape(dim, dim, blocksize)

    test_vector = inv_skew_symmetrize(input_matrix)

    assert_allclose(test_vector, input_vector)


@pytest.mark.parametrize("ndim", [(1), (2), (3)])
def test_inv_skew_symmetrize_throws_if_dim_not_square(ndim):
    dim = 3
    blocksize = 16

    # Matrix not necessary to be sk-symm in this case
    if ndim == 1:
        incorrect_input_matrix = np.random.randn(dim + dim)
    elif ndim == 2:
        incorrect_input_matrix = np.random.randn(dim + dim, blocksize)
    elif ndim == 3:
        incorrect_input_matrix = np.random.randn(dim, dim + 1, blocksize)

    with pytest.raises(AssertionError) as excinfo:
        inv_skew_symmetrize(incorrect_input_matrix)
    assert excinfo.type == AssertionError


@pytest.mark.parametrize("ndim", [(1), (2), (3)])
def test_inv_skew_symmetrize_raises_dimension_error(ndim):
    dim = 4  # one greater than allowed dimension
    blocksize = 8
    # Matrix not necessary to be sk-symm in this case
    if ndim == 1:
        incorrect_input_matrix = np.random.randn(dim ** 2)
    elif ndim == 2:
        incorrect_input_matrix = np.random.randn(dim ** 2, blocksize)
    elif ndim == 3:
        incorrect_input_matrix = np.random.randn(dim, dim, blocksize)

    # Matrix not necessary to be sk-symm in this case
    with pytest.raises(AssertionError) as excinfo:
        inv_skew_symmetrize(incorrect_input_matrix)
    assert excinfo.type == AssertionError


def test_inv_skew_symmetrize_raises_ndim_error():
    incorrect_in_matrix = np.random.randn(2, 2, 3, 4)

    with pytest.raises(RuntimeError) as excinfo:
        skew_symmetrize(incorrect_in_matrix)
    assert excinfo.type == RuntimeError


@pytest.mark.parametrize("blocksize", [(32)])
def test_inv_skew_symmetrize_checks_skew_symmetry(blocksize):
    incorrect_input_matrix = np.random.randn(3, 3, blocksize)

    with pytest.raises(ValueError) as excinfo:
        inv_skew_symmetrize(incorrect_input_matrix)
    assert excinfo.type == ValueError


###############################################################################
########################## Interface tests finis ##############################
###############################################################################

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])
