"""Handy utilities"""

from typing import Generator, Iterable, Any, Literal, TypeVar
import functools
import numpy as np
from numpy import finfo, float64
from itertools import islice
from scipy.interpolate import BSpline

from numpy.typing import NDArray


# Slower than the python3.8 isqrt implementation for small ints
# python isqrt : ~130 ns
# this : 621 ns ± 17.7 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
# with lru cache it drops to 80-100ns, which is anyway our common use case
@functools.lru_cache(maxsize=1)
def isqrt(num: int) -> int:
    """
    Efficiently computes sqrt for integer values

    Dropin replacement for python3.8's isqrt function
    Credits : https://stackoverflow.com/a/53983683

    Parameters
    ----------
    num : int, input

    Returns
    -------
    sqrt_num : int, rounded down sqrt of num

    Notes
    -----
    - Doesn't handle edge-cases of negative numbers by design
    - Doesn't type-check for integers by design, although it is hinted at

    Examples
    --------

    """
    if num > 0:
        x_iterate = 1 << (num.bit_length() + 1 >> 1)
        while True:
            f_iterate = (x_iterate + num // x_iterate) >> 1
            if f_iterate >= x_iterate:
                return x_iterate
            x_iterate = f_iterate
    elif num == 0:
        return 0

    raise ValueError("num must be a positive number")


class MaxDimension:
    """
    Contains maximum space dimensions. Typically useful for range-checking
    """

    @staticmethod
    def value() -> Literal[3]:
        """
        Returns spatial dimension

        Returns
        -------
        3, static value
        """
        return 3


class Tolerance:
    @staticmethod
    def atol() -> float:
        """
        Static absolute tolerance method

        Returns
        -------
        atol : library-wide set absolute tolerance for kernels
        """
        return float(finfo(float64).eps * 1e4)

    @staticmethod
    def rtol() -> float:
        """
        Static relative tolerance method

        Returns
        -------
        tol : library-wide set relative tolerance for kernels
        """
        return float(finfo(float64).eps * 1e11)


def perm_parity(lst: list[int]) -> int:
    """
    Given a permutation of the digits 0..N in order as a list,
    returns its parity (or sign): +1 for even parity; -1 for odd.

    Parameters
    ----------
    lst

    Returns
    -------

    Credits
    -------
    Code obtained with thanks from https://code.activestate.com/recipes/578227-generate-the-parity-or-sign-of-a-permutation/
    licensed with a MIT License
    """
    parity = 1
    for i in range(0, len(lst) - 1):
        if lst[i] != i:
            parity *= -1
            mn = min(range(i, len(lst)), key=lst.__getitem__)
            lst[i], lst[mn] = lst[mn], lst[i]
    return parity


_T = TypeVar("_T")


def grouper(iterable: Iterable[_T], n: int) -> Generator[tuple[_T, ...], None, None]:
    """Collect data into fixed-length chunks or blocks"

    Parameters
    ----------
    iterable : input collection
    n : size of chunk

    Returns
    -------

    Example
    -------
    grouper('ABCDEFG', 3) --> ABC DEF G"

    Credits
    -------
    https://docs.python.org/3/library/itertools.html#itertools-recipes
    https://stackoverflow.com/a/10791887
    """

    it = iter(iterable)
    while True:
        group = tuple(islice(it, None, n))
        if not group:
            break
        yield group


def extend_instance(obj: Any, cls: Any) -> None:
    """

    Apply mixins to a class instance after creation

    Parameters
    ----------
    obj : object (not class!) targeted for interface extension
          Interface carries throughout its lifetime.
    cls : class (not object!) to dynamically mixin

    Returns
    -------
    None

    Credits
    -------
    https://stackoverflow.com/a/31075641

    """
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    # Putting class first to override default behavior
    obj.__class__ = type(base_cls_name, (cls, base_cls), {})


def _bspline(  # type: ignore[no-any-unimported]
    t_coeff: NDArray, l_centerline: np.float64 = np.float64(1.0)
) -> tuple[BSpline, NDArray, NDArray]:
    """Generates a bspline object that plots the spline interpolant for
    any vector x. Optionally takes in a centerline length, set to 1.0 by
    default and keep_pts for keeping record of control points

    Parameters
    ----------
    t_coeff : np.array
        The spline coefficients, denoted by :math:`beta_i`. Note that the first
        and the last values are set to zero by default.
    l_centreline : float
        The length of the centerline in meters.

    Returns
    -------
    spline : scipy.interpolate.Bspline class
        A spline class that can be called as spline(x), where x are the points at
        which the spline needs to be evaluated.
    """
    # Divide into n_control_pts number of points (n_ctr_pts-1) regions
    control_pts = l_centerline * np.linspace(0.0, 1.0, t_coeff.shape[0] - 2)

    # Degree of b-spline required. Set to cubic
    degree = 3

    return __bspline_impl__(control_pts, t_coeff, degree)


def __bspline_impl__(  # type: ignore[no-any-unimported]
    x_pts: NDArray, t_c: NDArray, degree: int
) -> tuple[BSpline, NDArray, NDArray]:
    """"""

    # Update the knots
    n_upd = t_c.shape[0] + (degree + 1)

    # Generate the knots
    knots_updated = np.zeros(n_upd)
    # Fix first degree-1 knots into head
    knots_updated[:degree] = x_pts[0]
    # Middle knot locations correspond to x_locations
    knots_updated[degree : n_upd - (degree)] = x_pts
    # Fix first degree-1 knots into tail
    knots_updated[n_upd - (degree) :] = x_pts[-1]

    return BSpline(knots_updated, t_c, degree, extrapolate=False), x_pts, t_c
