__doc__ = """ Quadrature and difference kernels """

import numpy as np
import functools

# TODO Check feasiblity of other quadrature / difference rules


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
    Not using numpy.pad for performance reasons
    with pad : 23.3 µs ± 1.65 µs per loop
    without pad (this version) : 9.73 µs ± 168 ns per loop

    getting the array shape and ndim seems to add ±0.5 µs difference
    """
    zero_array = _get_zero_array(array_collection.shape[0], array_collection.ndim)
    padded_collection = np.hstack((zero_array, array_collection, zero_array))
    return 0.5 * (padded_collection[..., :-1] + padded_collection[..., 1:])


