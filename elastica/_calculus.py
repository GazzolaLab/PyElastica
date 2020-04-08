__doc__ = """ Quadrature and difference kernels """

import numpy as np
import functools


# TODO Check feasiblity of other quadrature / difference rules


@functools.lru_cache(maxsize=2)
def _get_zero_array(dim, ndim):
    """
    Returns zeros float or array depending on ndim
    Parameters
    ----------
    dim: int
    ndim : int

    Returns
    -------

    """
    if ndim == 1:
        return 0.0
    if ndim == 2:
        return np.zeros((dim, 1))


def _trapezoidal(array_collection):
    """
    Simple trapezoidal quadrature rule with zero at end-points, in a dimension agnostic way

    Parameters
    ----------
    array_collection: ndarray
        2D (dim, blocksize) array containing data with 'float' type.

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
    array_collection: ndarray
        2D (dim, blocksize) array containing data with 'float' type.

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


quadrature_kernel = _trapezoidal
difference_kernel = _two_point_difference
