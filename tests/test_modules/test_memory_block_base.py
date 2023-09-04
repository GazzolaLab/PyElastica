__doc__ = """ Test make_block_memory_metadata and make_block_memory_periodic_boundary_metadata functions """

import pytest
import numpy as np
from elastica.memory_block.memory_block_rod_base import (
    make_block_memory_metadata,
    make_block_memory_periodic_boundary_metadata,
)


@pytest.mark.module
@pytest.mark.parametrize(
    "n_elems_in_rods, outputs",
    [
        (
            np.array([5], dtype=np.int64),
            [
                np.int64(5),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            ],
        ),
        (
            np.array([5, 5], dtype=np.int64),
            [
                np.int64(12),
                np.array([6], dtype=np.int64),
                np.array([5, 6], dtype=np.int64),
                np.array([4, 5, 6], dtype=np.int64),
            ],
        ),
        (
            np.array([1, 1, 1], dtype=np.int64),
            [
                np.int64(7),
                np.array([2, 5], dtype=np.int64),
                np.array([1, 2, 4, 5], dtype=np.int64),
                np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
            ],
        ),
        (
            np.array([1, 2, 3], dtype=np.int64),
            [
                np.int64(10),
                np.array([2, 6], dtype=np.int64),
                np.array([1, 2, 5, 6], dtype=np.int64),
                np.array([0, 1, 2, 4, 5, 6], dtype=np.int64),
            ],
        ),
        (
            np.array([10, 10, 5, 5], dtype=np.int64),
            [
                np.int64(36),
                np.array([11, 23, 30], dtype=np.int64),
                np.array([10, 11, 22, 23, 29, 30], dtype=np.int64),
                np.array([9, 10, 11, 21, 22, 23, 28, 29, 30], dtype=np.int64),
            ],
        ),
    ],
)
def test_make_block_memory_metadata(n_elems_in_rods, outputs):
    [
        n_elems_with_ghosts,
        ghost_nodes_idx,
        ghost_elems_idx,
        ghost_voronoi_idx,
    ] = make_block_memory_metadata(n_elems_in_rods)

    assert np.all(n_elems_with_ghosts == outputs[0])
    assert np.all(ghost_nodes_idx == outputs[1])
    assert np.all(ghost_elems_idx == outputs[2])
    assert np.all(ghost_voronoi_idx == outputs[3])
