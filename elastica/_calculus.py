__doc__ = """ Quadrature and difference kernels """

import numpy as np
import functools
import numba
from numba import njit
from numpy import zeros, empty

# TODO Check feasiblity of other quadrature / difference rules


@functools.lru_cache(maxsize=2)
def _get_zero_array(dim, ndim):
    if ndim == 1:
        return 0.0
    if ndim == 2:
        return np.zeros((dim, 1))


# def _trapezoidal(array_collection):
#     """
#     Simple trapezoidal quadrature rule with zero at end-points, in a dimension agnostic way
#
#     Parameters
#     ----------
#     array_collection
#
#     Returns
#     -------
#
#     Note
#     ----
#     Not using numpy.pad, numpy.hstack for performance reasons
#     with pad : 23.3 µs ± 1.65 µs per loop
#     without pad (previous version, see git history) : 9.73 µs ± 168 ns per loop
#     without pad and hstack (this version) : 6.52 µs ± 118 ns per loop
#
#     - Getting the array shape and manipulating them seems to add ±0.5 µs difference
#     - As an added bonus, this works for n-dimensions as long as last dimension
#     is preserved
#     """
#     temp_collection = np.empty(
#         array_collection.shape[:-1] + (array_collection.shape[-1] + 1,)
#     )
#     temp_collection[..., 0] = array_collection[..., 0]
#     temp_collection[..., -1] = array_collection[..., -1]
#     temp_collection[..., 1:-1] = array_collection[..., 1:] + array_collection[..., :-1]
#     return 0.5 * temp_collection


@njit()
def _trapezoidal(array_collection):
    blocksize = array_collection.shape[1]
    temp_collection = empty((3, blocksize + 1))

    temp_collection[0, 0] = 0.5 * array_collection[0, 0]
    temp_collection[1, 0] = 0.5 * array_collection[1, 0]
    temp_collection[2, 0] = 0.5 * array_collection[2, 0]

    temp_collection[0, blocksize] = 0.5 * array_collection[0, blocksize - 1]
    temp_collection[1, blocksize] = 0.5 * array_collection[1, blocksize - 1]
    temp_collection[2, blocksize] = 0.5 * array_collection[2, blocksize - 1]

    for k in range(1, blocksize):
        temp_collection[0, k] = 0.5 * (
            array_collection[0, k] + array_collection[0, k - 1]
        )
        temp_collection[1, k] = 0.5 * (
            array_collection[1, k] + array_collection[1, k - 1]
        )
        temp_collection[2, k] = 0.5 * (
            array_collection[2, k] + array_collection[2, k - 1]
        )

    return temp_collection


# def _two_point_difference(array_collection):
#     """
#
#     Parameters
#     ----------
#     array_collection
#
#     Returns
#     -------
#
#     Note
#     ----
#     Not using numpy.pad, numpy.diff, numpy.hstack for performance reasons
#     with pad : 23.3 µs ± 1.65 µs per loop
#     without pad (previous version, see git history) : 8.3 µs ± 195 ns per loop
#     without pad, hstack (this version) : 5.73 µs ± 216 ns per loop
#
#     - Getting the array shape and ndim seems to add ±0.5 µs difference
#     - Diff also seems to add only ±3.0 µs
#     - As an added bonus, this works for n-dimensions as long as last dimension
#     is preserved
#     """
#     temp_collection = np.empty(
#         array_collection.shape[:-1] + (array_collection.shape[-1] + 1,)
#     )
#     temp_collection[..., 0] = array_collection[..., 0]
#     temp_collection[..., -1] = -array_collection[..., -1]
#     temp_collection[..., 1:-1] = array_collection[..., 1:] - array_collection[..., :-1]
#     return temp_collection


@njit()
def _two_point_difference(array_collection):
    blocksize = array_collection.shape[1]
    temp_collection = empty((3, blocksize + 1))

    temp_collection[0, 0] = array_collection[0, 0]
    temp_collection[1, 0] = array_collection[1, 0]
    temp_collection[2, 0] = array_collection[2, 0]

    temp_collection[0, blocksize] = -array_collection[0, blocksize - 1]
    temp_collection[1, blocksize] = -array_collection[1, blocksize - 1]
    temp_collection[2, blocksize] = -array_collection[2, blocksize - 1]

    for k in range(1, blocksize):
        temp_collection[0, k] = array_collection[0, k] - array_collection[0, k - 1]
        temp_collection[1, k] = array_collection[1, k] - array_collection[1, k - 1]
        temp_collection[2, k] = array_collection[2, k] - array_collection[2, k - 1]

    return temp_collection


quadrature_kernel = _trapezoidal
difference_kernel = _two_point_difference


@njit()
def _difference(vector):
    blocksize = vector.shape[1] - 1
    output_vector = zeros((3, blocksize))

    for i in range(3):
        for k in range(blocksize):
            output_vector[i, k] += vector[i, k + 1] - vector[i, k]

    return output_vector


@njit()
def _average(vector):
    blocksize = vector.shape[0] - 1
    output_vector = empty((blocksize))

    for k in range(blocksize):
        output_vector[k] = 0.5 * (vector[k + 1] + vector[k])

    return output_vector


position_difference_kernel = _difference
position_average = _average
