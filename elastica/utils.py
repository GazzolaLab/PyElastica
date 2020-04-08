""" Handy utilities
"""
__all__ = ["isqrt"]
import functools
from numpy import finfo, float64
from itertools import islice


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

    Note
    ----
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


class MaxDimension:
    """
    Contains maximum space dimensions. Typically useful for range-checking
    """

    @staticmethod
    def value():
        """
        Returns spatial dimension

        Returns
        -------
        3, static value
        """
        return 3


class Tolerance:
    @staticmethod
    def atol():
        """
        Static absolute tolerance method

        Returns
        -------
        atol : library-wide set absolute tolerance for kernels
        """
        return finfo(float64).eps * 1e4

    @staticmethod
    def rtol():
        """
        Static relative tolerance method

        Returns
        -------
        tol : library-wide set relative tolerance for kernels
        """
        return finfo(float64).eps * 1e11


def perm_parity(lst):
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


def grouper(iterable, n):
    """ Collect data into fixed-length chunks or blocks"

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


def extend_instance(obj, cls):
    """ Apply mixins to a class instance after creation
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
