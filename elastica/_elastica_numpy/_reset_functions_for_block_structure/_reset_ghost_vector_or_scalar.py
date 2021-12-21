__doc__ = """Reset the ghost vectors or scalar variables using functions implemented in Numpy"""

import numpy as np


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

    """
    input[..., ghost_idx] = reset_value


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

    """
    for k in ghost_idx:
        input[k] = reset_value
