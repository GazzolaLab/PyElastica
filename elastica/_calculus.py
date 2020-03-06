__doc__ = """ Quadrature and difference kernels """

import numpy as np
import functools
from numpy import zeros, empty

# TODO Check feasiblity of other quadrature / difference rules


@functools.lru_cache(maxsize=2)
def _get_zero_array(dim, ndim):
    if ndim == 1:
        return 0.0
    if ndim == 2:
        return np.zeros((dim, 1))


try:
    import numba
    from numba import njit

    @njit(cache=True)
    def _trapezoidal(array_collection):
        """
        Simple trapezoidal quadrature rule with zero at end-points, in a dimension agnostic way

        Parameters
        ----------
        array_collection

        Returns
        -------
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

    @njit(cache=True)
    def _two_point_difference(array_collection):
        """
        This function does differentiation.
        Parameters
        ----------
        array_collection

        Returns
        -------
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

    quadrature_kernel = _trapezoidal
    difference_kernel = _two_point_difference

    @njit(cache=True)
    def _difference(vector):
        """
        This function computes difference between elements of a batch vector
        Parameters
        ----------
        vector

        Returns
        -------
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

    @njit(cache=True)
    def _average(vector):
        """
        This function computes the average between elements of a vector
        Parameters
        ----------
        vector

        Returns
        -------
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

    position_difference_kernel = _difference
    position_average = _average

    @njit(cache=True)
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

    @njit(cache=True)
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
        Python version: 2.24 µs ± 96.1 ns per loop
        This version: 479 ns ± 6.49 ns per loop
        """
        return np.isnan(array).any()


except ImportError:

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
        temp_collection[..., 1:-1] = (
            array_collection[..., 1:] + array_collection[..., :-1]
        )
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
        temp_collection[..., 1:-1] = (
            array_collection[..., 1:] - array_collection[..., :-1]
        )
        return temp_collection

    quadrature_kernel = _trapezoidal
    difference_kernel = _two_point_difference

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
