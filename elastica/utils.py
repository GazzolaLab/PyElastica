""" Handy utilities
"""
__all__ = ["isqrt"]
import functools


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

    Caveats
    -------
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
