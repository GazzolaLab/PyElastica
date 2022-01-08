#!/usr/bin/env python3
__doc__ = """ Test scripts for calculus kernels in Elastica Numba implementation"""
# System imports
import numpy as np
import pytest
from numpy.testing import assert_allclose
from elastica.utils import Tolerance
from elastica._calculus import (
    _trapezoidal,
    _two_point_difference,
    _clip_array,
    _isnan_check,
    _trapezoidal_for_block_structure,
    _two_point_difference_for_block_structure,
)


class Trapezoidal:
    kernel = _trapezoidal

    @staticmethod
    def oned_setup():
        blocksize = 32
        input_vector = np.random.randn(blocksize)

        first_element = 0.5 * input_vector[0]
        last_element = 0.5 * input_vector[-1]
        correct_vector = np.hstack(
            (first_element, 0.5 * (input_vector[1:] + input_vector[:-1]), last_element)
        )

        return input_vector, correct_vector


class Difference:
    kernel = _two_point_difference

    @staticmethod
    def oned_setup():
        blocksize = 32
        input_vector = np.random.randn(blocksize)

        first_element = input_vector[0]
        last_element = -input_vector[-1]
        correct_vector = np.hstack(
            (first_element, (input_vector[1:] - input_vector[:-1]), last_element)
        )

        return input_vector, correct_vector


@pytest.mark.parametrize("Setup", [Trapezoidal, Difference])
@pytest.mark.parametrize("ndim", [3])
def test_two_point_difference_integrity(Setup, ndim):
    input_vector_oned, correct_vector_oned = Setup.oned_setup()

    # Above tests where failing, because, new Numba kernels only works for 3,n
    # matrices.
    input_vector = np.repeat(input_vector_oned[np.newaxis, :], ndim, axis=0)
    test_vector = Setup.kernel(input_vector)
    correct_vector = np.repeat(correct_vector_oned[np.newaxis, :], ndim, axis=0)

    assert test_vector.shape == input_vector.shape[:-1] + (input_vector.shape[-1] + 1,)
    assert_allclose(test_vector, correct_vector)


def test_trapezoidal_correctness():
    r"""
    Tests integral of a function :math:`f : [a,b] \rightarrow \mathbb{R}`,
         :math:`\int_{a}^{b} f \rightarrow \mathbb{R}`
    where f satisfies the conditions f(a) = f(b) = 0.0
    """
    blocksize = 64
    a = 0.0
    b = np.pi
    dh = (b - a) / (blocksize - 1)

    # Should integrate this well, as end
    input_vector = np.repeat(
        np.sin(np.linspace(a, b, blocksize))[np.newaxis, :], 3, axis=0
    )
    test_vector = _trapezoidal(input_vector[..., 1:-1]) * dh

    # Sampling for the analytical derivative needs to be done
    # one a grid that lies in between the actual function for
    # second-order accuracy!
    interior_a = a + 0.5 * dh
    interior_b = b - 0.5 * dh
    correct_vector = (
        np.repeat(
            np.sin(np.linspace(interior_a, interior_b, blocksize - 1))[np.newaxis, :],
            3,
            axis=0,
        )
        * dh
    )

    # Pathetic error of 1e-2 :(
    assert_allclose(np.sum(test_vector[0]), 2.0, atol=1e-3)
    assert_allclose(np.sum(test_vector[1]), 2.0, atol=1e-3)
    assert_allclose(np.sum(test_vector[2]), 2.0, atol=1e-3)
    assert_allclose(test_vector, correct_vector, atol=1e-4)


def test_trapezoidal_for_block_structure_correctness():
    """
    This test is testing the validity of trapezoidal rule for block structure implementation.
    Here we are using the _trapezoidal for validity check, because it is already tested in the
    above test function.

    Returns
    -------

    """

    blocksize = 30
    ghost_idx = np.array([14, 15])
    input_vector = np.random.randn(3, blocksize)

    correct_vector = np.hstack(
        (
            _trapezoidal(input_vector[..., : ghost_idx[0]]),
            np.zeros((3, 1)),
            _trapezoidal(input_vector[..., ghost_idx[1] + 1 :]),
        )
    )

    test_vector = _trapezoidal_for_block_structure(input_vector, ghost_idx)

    assert_allclose(test_vector, correct_vector, atol=Tolerance.atol())


def test_two_point_difference_correctness():
    """
    Tests difference of a function f:[a,b]-> R, i.e
        D f[a,b] -> df[a,b]
    where f satisfies the conditions f(a) = f(b) = 0.0
    """
    blocksize = 128
    a = 0.0
    b = np.pi
    dh = (b - a) / (blocksize - 1)

    # Sampling for the analytical derivative needs to be done
    # one a grid that lies in between the actual function for
    # second-order accuracy!
    interior_a = a + 0.5 * dh
    interior_b = b - 0.5 * dh

    # Should integrate this well
    input_vector = np.repeat(
        np.sin(np.linspace(a, b, blocksize))[np.newaxis, :], 3, axis=0
    )
    test_vector = _two_point_difference(input_vector[..., 1:-1]) / dh
    correct_vector = np.repeat(
        np.cos(np.linspace(interior_a, interior_b, blocksize - 1))[np.newaxis, :],
        3,
        axis=0,
    )

    # Pathetic error of 1e-2 :(
    assert_allclose(test_vector, correct_vector, atol=1e-4)


def test_two_point_difference_for_block_structure_correctness():
    """
    This test is testing the validity of two_point_difference rule for block structure implementation.
    Here we are using the _two_point_difference for validity check, because it is already tested in the
    above test function.

    Returns
    -------

    """

    blocksize = 30
    ghost_idx = np.array([14, 15])
    input_vector = np.random.randn(3, blocksize)

    correct_vector = np.hstack(
        (
            _two_point_difference(input_vector[..., : ghost_idx[0]]),
            np.zeros((3, 1)),
            _two_point_difference(input_vector[..., ghost_idx[1] + 1 :]),
        )
    )

    test_vector = _two_point_difference_for_block_structure(input_vector, ghost_idx)

    assert_allclose(test_vector, correct_vector, atol=Tolerance.atol())


def test_clip_array():
    """
    Test _clip array function.
    Returns
    -------

    """

    blocksize = 100
    input_array = np.hstack(
        (10 * np.random.rand(blocksize), -10 * np.random.rand(blocksize))
    )

    vmin = -1
    vmax = 1

    correct_vector = np.clip(input_array, vmin, vmax)
    test_vector = _clip_array(input_array, vmin, vmax)

    assert_allclose(test_vector, correct_vector, atol=Tolerance.atol())


def test_isnan_check():
    """
    Test _isnan_check function
    Returns
    -------

    """

    blocksize = 100

    input_vector = np.repeat(np.array([np.NaN]), blocksize)
    correct_vector = np.repeat(np.array([np.True_]), blocksize)
    test_vector = _isnan_check(input_vector)

    assert_allclose(test_vector, correct_vector)
