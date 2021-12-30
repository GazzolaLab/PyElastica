__doc__ = """Reset the ghost vectors or scalar variables using functions implemented in Numba"""

import numpy as np
import numba
from numba import njit


@njit(cache=True)
def _reset_vector_ghost(input, ghost_idx, reset_value=0.0):
    """
    This function resets the ghost of an input vector collection. Default reset value is 0.0.

    Parameters
    ----------
    input
    ghost_idx
    reset_value

    Returns
    -------

    Note
    ----
    Micro benchmark results, 10 rods each with 10 elements block structure is used.

    For ghost nodes:
    python version: 2.31 µs ± 243 ns per loop
    this version: 647 ns ± 5.88 ns per loop

    For ghost elements:
    python version: 2.75 µs ± 206 ns per loop
    this version: 673 ns ± 33.7 ns per loop

    For voronoi elements:
    python version: 2.86 µs ± 477 ns per loop
    this version: 703 ns ± 40.1 ns per loop

    """
    for i in range(3):
        for k in ghost_idx:
            input[i, k] = reset_value


@njit(cache=True)
def _reset_scalar_ghost(input, ghost_idx, reset_value=0.0):
    """
    This function resets the ghost of a scalar collection. Default reset value is 0.0.

    Parameters
    ----------
    input
    ghost_idx
    reset_value

    Returns
    -------

    Note
    ----
    Micro benchmark results, 10 rods each with 10 elements block structure is used.

    For ghost nodes:
    python version: 2.7 µs ± 72.1 ns per loop
    this version: 603 ns ± 11.2 ns per loop

    For ghost elements:
    python version: 4.83 µs ± 236 ns per loop
    this version: 618 ns ± 13.2 ns per loop

    For voronoi elements:
    python version: 6.31 µs ± 190 ns per loop
    this version: 646 ns ± 28.5 ns per loop

    """
    for k in ghost_idx:
        input[k] = reset_value
