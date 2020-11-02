__doc__ = """ Quadrature and difference kernels Numpy implementation"""
import numpy as np
import functools


@functools.lru_cache(maxsize=2)
def _get_zero_array(dim, ndim):
    if ndim == 1:
        return 0.0
    if ndim == 2:
        return np.zeros((dim, 1))


def _trapezoidal(array_collection):
    """
    Simple trapezoidal quadrature rule with zero at end-points, in a dimension agnostic way

    Parameters
    ----------
    array_collection

    Returns
    -------

    Note
    ----
    Not using numpy.pad, numpy.hstack for performance reasons
    with pad : 23.3 µs ± 1.65 µs per loop
    without pad (previous version, see git history) : 9.73 µs ± 168 ns per loop
    without pad and hstack (this version) : 6.52 µs ± 118 ns per loop

    - Getting the array shape and manipulating them seems to add ±0.5 µs difference
    - As an added bonus, this works for n-dimensions as long as last dimension
    is preserved
    """
    temp_collection = np.empty(
        array_collection.shape[:-1] + (array_collection.shape[-1] + 1,)
    )
    temp_collection[..., 0] = array_collection[..., 0]
    temp_collection[..., -1] = array_collection[..., -1]
    temp_collection[..., 1:-1] = array_collection[..., 1:] + array_collection[..., :-1]
    return 0.5 * temp_collection


def _two_point_difference(array_collection):
    """

    Parameters
    ----------
    array_collection

    Returns
    -------

    Note
    ----
    Not using numpy.pad, numpy.diff, numpy.hstack for performance reasons
    with pad : 23.3 µs ± 1.65 µs per loop
    without pad (previous version, see git history) : 8.3 µs ± 195 ns per loop
    without pad, hstack (this version) : 5.73 µs ± 216 ns per loop

    - Getting the array shape and ndim seems to add ±0.5 µs difference
    - Diff also seems to add only ±3.0 µs
    - As an added bonus, this works for n-dimensions as long as last dimension
    is preserved
    """
    temp_collection = np.empty(
        array_collection.shape[:-1] + (array_collection.shape[-1] + 1,)
    )
    temp_collection[..., 0] = array_collection[..., 0]
    temp_collection[..., -1] = -array_collection[..., -1]
    temp_collection[..., 1:-1] = array_collection[..., 1:] - array_collection[..., :-1]
    return temp_collection


def _clip_array(input_array, vmin, vmax):
    """
    This function clips an array values
    between user defined minimum and maximum
    values.
    Parameters
    ----------
    input_array
    vmin
    vmax

    Returns
    -------
    Notes
    -----
    Micro benchmark results showed that for a block size of 100, using timeit
    Python high level function version: 18.2 µs ± 845 ns per loop
    This version: 2.87 µs ± 202 ns per loop
    """
    return np.core.umath.maximum(np.core.umath.minimum(input_array, vmax), vmin)


def _isnan_check(array):
    """
    This function checks if there is any nan inside the array.
    If there is nan, it returns True boolean
    Parameters
    ----------
    array

    Returns
    -------
    Notes
    -----
    Micro benchmark results showed that for a block size of 100, using timeit
    Numba version: 479 ns ± 6.49 ns per loop
    This version: 2.24 µs ± 96.1 ns per loop
    """
    return np.isnan(array).any()
