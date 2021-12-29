import numpy as np
from numpy.testing import assert_allclose
from elastica.reset_functions_for_block_structure._reset_ghost_vector_or_scalar import (
    _reset_scalar_ghost,
    _reset_vector_ghost,
)
from elastica.utils import Tolerance
import pytest


@pytest.mark.parametrize("n_elems", [10, 30, 41])
def test_reset_vector_ghosts(n_elems):
    """
    Test resetting of ghosts on vector collection, for Numba implementation.

    Returns
    -------

    """

    ghosts = np.random.randint(1, n_elems, int(n_elems / 4))

    input_vector = np.random.randn(3, n_elems)

    _reset_vector_ghost(input_vector, ghosts)

    for k in ghosts:
        assert_allclose(input_vector[..., k], 0.0, atol=Tolerance.atol())


@pytest.mark.parametrize("n_elems", [10, 30, 41])
def test_reset_scalar_ghosts(n_elems):
    """
    Test resetting of ghosts on scalar collection, for Numba implementation.

    Returns
    -------

    """

    ghosts = np.random.randint(1, n_elems, int(n_elems / 4))

    input_vector = np.random.randn(n_elems)

    _reset_scalar_ghost(input_vector, ghosts)

    for k in ghosts:
        assert_allclose(input_vector[k], 0.0, atol=Tolerance.atol())


if __name__ == "__main__":
    from pytest import main

    main([__file__])
