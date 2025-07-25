__doc__ = """ Quadrature and difference kernels """
import numpy as np
from numpy import zeros, empty
from numpy.typing import NDArray
import numba
from numba import njit
from elastica.reset_functions_for_block_structure._reset_ghost_vector_or_scalar import (
    _reset_vector_ghost,
)


@njit(cache=True)  # type: ignore
def _trapezoidal(array_collection: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Simple trapezoidal quadrature rule with zero at end-points, in a dimension agnostic way

    Parameters
    ----------
    array_collection : numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.

    Returns
    -------
    result: numpy.ndarray
        2D (dim, blocksize+1) array containing data with 'float' type.

    Notes
    -----
    Micro benchmark results, for a block size of 100, using timeit
    Python version: 8.14 µs ± 1.42 µs per loop
    This version: 781 ns ± 18.3 ns per loop
    """
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


@njit(cache=True)  # type: ignore
def _trapezoidal_for_block_structure(
    array_collection: NDArray[np.float64], ghost_idx: NDArray[np.int32]
) -> NDArray[np.float64]:
    """
    Simple trapezoidal quadrature rule with zero at end-points, in a dimension agnostic way. This form
    specifically for the block structure implementation and there is a reset function call, to reset
    ghosts.

    Parameters
    ----------
    array_collection : numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
    ghost_idx : numpy.ndarray
        1D (n_ghost) array containing data with 'int' type.

    Returns
    -------
    result: numpy.ndarray
        2D (dim, blocksize+1) array containing data with 'float' type.

    Notes
    -----
    Micro benchmark results, for a block size of 100, using timeit
    Python version: 8.21 µs per loop
    This version: 1.03 µs per loop

    User should use this function with extreme care, since this function is rewritten for
    block structure.

    """

    _reset_vector_ghost(array_collection, ghost_idx)

    blocksize = array_collection.shape[1]
    temp_collection = np.empty((3, blocksize + 1))

    temp_collection[0, 0] = 0.5 * array_collection[0, 0]
    temp_collection[1, 0] = 0.5 * array_collection[1, 0]
    temp_collection[2, 0] = 0.5 * array_collection[2, 0]

    temp_collection[0, blocksize] = 0.5 * array_collection[0, blocksize - 1]
    temp_collection[1, blocksize] = 0.5 * array_collection[1, blocksize - 1]
    temp_collection[2, blocksize] = 0.5 * array_collection[2, blocksize - 1]

    for i in range(3):
        for k in range(1, blocksize):
            temp_collection[i, k] = 0.5 * (
                array_collection[i, k] + array_collection[i, k - 1]
            )

    return temp_collection


@njit(cache=True)  # type: ignore
def _two_point_difference(
    array_collection: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    This function does differentiation.

    Parameters
    ----------
    array_collection : numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.

    Returns
    -------
    result: numpy.ndarray
        2D (dim, blocksize-1) array containing data with 'float' type.


    Notes
    -----
    Micro benchmark results showed that for a block size of 100, using timeit
    Python version: 9.07 µs ± 2.15 µs per loop
    This version: 952 ns ± 91.1 ns per loop
    """
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


@njit(cache=True)  # type: ignore
def _two_point_difference_for_block_structure(
    array_collection: NDArray[np.float64], ghost_idx: NDArray[np.int32]
) -> NDArray[np.float64]:
    """
    This function does the differentiation, for Cosserat rod model equations. This form
    specifically for the block structure implementation and there is a reset function call, to
    reset ghosts.

    Parameters
    ----------
    array_collection : numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
    ghost_idx : numpy.ndarray
        1D (n_ghost) array containing data with 'int' type.

    Returns
    -------
    result: numpy.ndarray
        2D (dim, blocksize-1) array containing data with 'float' type.


    Notes
    -----
    Micro benchmark results showed that for a block size of 100, using timeit
    Python version: 7.1 µs per loop
    This version: 1.01 µs per loop

    User should use this function with extreme care, since this function is rewritten for
    block structure.

    """

    _reset_vector_ghost(array_collection, ghost_idx)

    blocksize = array_collection.shape[1]
    temp_collection = np.empty((3, blocksize + 1))

    temp_collection[0, 0] = array_collection[0, 0]
    temp_collection[1, 0] = array_collection[1, 0]
    temp_collection[2, 0] = array_collection[2, 0]

    temp_collection[0, blocksize] = -array_collection[0, blocksize - 1]
    temp_collection[1, blocksize] = -array_collection[1, blocksize - 1]
    temp_collection[2, blocksize] = -array_collection[2, blocksize - 1]

    for i in range(3):
        for k in range(1, blocksize):
            temp_collection[i, k] = array_collection[i, k] - array_collection[i, k - 1]

    return temp_collection


@njit(cache=True)  # type: ignore
def _difference(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    This function computes difference between elements of a batch vector.

    Parameters
    ----------
    vector: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.

    Returns
    -------
    result: numpy.ndarray
        2D (dim, blocksize-1) array containing data with 'float' type.

    Notes
    -----
    Micro benchmark results showed that for a block size of 100, using timeit
    Python version: 3.29 µs ± 767 ns per loop
    This version: 840 ns ± 14.5 ns per loop
    """
    blocksize = vector.shape[1] - 1
    output_vector = zeros((3, blocksize))

    for i in range(3):
        for k in range(blocksize):
            output_vector[i, k] += vector[i, k + 1] - vector[i, k]

    return output_vector


@njit(cache=True)  # type: ignore
def _average(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    This function computes the average between elements of a vector.

    Parameters
    ----------
    vector : numpy.ndarray
        1D (blocksize) array containing data with 'float' type.

    Returns
    -------
    result: numpy.ndarray
        1D (blocksize-1) array containing data with 'float' type.

    Notes
    -----
    Micro benchmark results showed that for a block size of 100, using timeit
    Python version: 2.37 µs ± 764 ns per loop
    This version: 713 ns ± 3.69 ns per loop
    """
    blocksize = vector.shape[0] - 1
    output_vector = empty((blocksize))

    for k in range(blocksize):
        output_vector[k] = 0.5 * (vector[k + 1] + vector[k])

    return output_vector


@njit(cache=True)  # type: ignore
def _clip_array(
    input_array: NDArray[np.float64], vmin: np.float64, vmax: np.float64
) -> NDArray[np.float64]:
    """
    This function clips an array values
    between user defined minimum and maximum
    values.

    Parameters
    ----------
    input_array: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
    vmin: float
    vmax: float

    Returns
    -------
    result: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.

    Notes
    -----
    Micro benchmark results showed that for a block size of 100, using timeit
    Python version: 18.2 µs ± 845 ns per loop
    This version: 357 ns ± 7.29 ns per loop
    """
    blocksize = input_array.shape[0]

    for k in range(blocksize):
        if input_array[k] < vmin:
            input_array[k] = vmin
        if input_array[k] > vmax:
            input_array[k] = vmax

    return input_array


@njit(cache=True)  # type: ignore
def _isnan_check(array: NDArray) -> bool:
    """
    This function checks if there is any nan inside the array.
    If there is nan, it returns True boolean.

    Parameters
    ----------
    array : numpy.ndarray
        Any size of array.

    Returns
    -------
    result: boolean

    Notes
    -----
    Micro benchmark results showed that for a block size of 100, using timeit
    Python version: 2.24 µs ± 96.1 ns per loop
    This version: 479 ns ± 6.49 ns per loop
    """
    return bool(np.isnan(array).any())


position_difference_kernel = _difference
position_average = _average
quadrature_kernel = _trapezoidal
difference_kernel = _two_point_difference
quadrature_kernel_for_block_structure = _trapezoidal_for_block_structure
difference_kernel_for_block_structure = _two_point_difference_for_block_structure
