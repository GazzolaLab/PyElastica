__doc__ = """ Test make_block_memory_metadata and make_block_memory_periodic_boundary_metadata functions """

import pytest
import numpy as np
from numpy.testing import assert_array_equal
from elastica.memory_block.utils import (
    make_block_memory_metadata,
    make_block_memory_periodic_boundary_metadata,
)


@pytest.mark.parametrize(
    "n_elems_in_rods, outputs",
    [
        (
            np.array([5], dtype=np.int32),
            [
                np.int64(5),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
            ],
        ),
        (
            np.array([5, 5], dtype=np.int32),
            [
                np.int64(12),
                np.array([6], dtype=np.int32),
                np.array([5, 6], dtype=np.int32),
                np.array([4, 5, 6], dtype=np.int32),
            ],
        ),
        (
            np.array([1, 1, 1], dtype=np.int32),
            [
                np.int64(7),
                np.array([2, 5], dtype=np.int32),
                np.array([1, 2, 4, 5], dtype=np.int32),
                np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
            ],
        ),
        (
            np.array([1, 2, 3], dtype=np.int32),
            [
                np.int64(10),
                np.array([2, 6], dtype=np.int32),
                np.array([1, 2, 5, 6], dtype=np.int32),
                np.array([0, 1, 2, 4, 5, 6], dtype=np.int32),
            ],
        ),
        (
            np.array([10, 10, 5, 5], dtype=np.int32),
            [
                np.int64(36),
                np.array([11, 23, 30], dtype=np.int32),
                np.array([10, 11, 22, 23, 29, 30], dtype=np.int32),
                np.array([9, 10, 11, 21, 22, 23, 28, 29, 30], dtype=np.int32),
            ],
        ),
    ],
)
def test_make_block_memory_metadata(n_elems_in_rods, outputs):
    (
        n_elems_with_ghosts,
        ghost_nodes_idx,
        ghost_elems_idx,
        ghost_voronoi_idx,
    ) = make_block_memory_metadata(n_elems_in_rods)

    assert_array_equal(n_elems_with_ghosts, outputs[0])
    assert_array_equal(ghost_nodes_idx, outputs[1])
    assert_array_equal(ghost_elems_idx, outputs[2])
    assert_array_equal(ghost_voronoi_idx, outputs[3])


@pytest.mark.parametrize(
    "n_elems_in_ring_rods",
    [
        np.array([10], dtype=np.int32),
        np.array([2, 4], dtype=np.int32),
        np.array([4, 5, 7], dtype=np.int32),
        np.array([10, 10, 10, 10], dtype=np.int32),
    ],
)
def test_make_block_memory_periodic_boundary_metadata(n_elems_in_ring_rods):
    (
        n_elem,
        periodic_boundary_node_idx,
        periodic_boundary_elems_idx,
        periodic_boundary_voronoi_idx,
    ) = make_block_memory_periodic_boundary_metadata(n_elems_in_ring_rods)

    n_ring_rods = len(n_elems_in_ring_rods)
    expected_n_elem = n_elems_in_ring_rods + 2
    expected_node_idx = np.empty((2, 3 * n_ring_rods), dtype=np.int32)
    expected_element_idx = np.empty((2, 2 * n_ring_rods), dtype=np.int32)
    expected_voronoi_idx = np.empty((2, n_ring_rods), dtype=np.int32)

    accumulation = np.hstack((0, np.cumsum(n_elems_in_ring_rods[:-1] + 4)))

    expected_node_idx[0, 0::3] = accumulation
    expected_node_idx[0, 1::3] = accumulation + n_elems_in_ring_rods + 1
    expected_node_idx[0, 2::3] = accumulation + n_elems_in_ring_rods + 2
    expected_node_idx[1, 0::3] = accumulation + n_elems_in_ring_rods
    expected_node_idx[1, 1::3] = accumulation + 1
    expected_node_idx[1, 2::3] = accumulation + 2

    expected_element_idx[0, 0::2] = accumulation
    expected_element_idx[0, 1::2] = accumulation + n_elems_in_ring_rods + 1
    expected_element_idx[1, 0::2] = accumulation + n_elems_in_ring_rods
    expected_element_idx[1, 1::2] = accumulation + 1

    expected_voronoi_idx[0, :] = accumulation
    expected_voronoi_idx[1, :] = accumulation + n_elems_in_ring_rods

    assert_array_equal(n_elem, expected_n_elem)
    assert_array_equal(periodic_boundary_node_idx, expected_node_idx)
    assert_array_equal(periodic_boundary_elems_idx, expected_element_idx)
    assert_array_equal(periodic_boundary_voronoi_idx, expected_voronoi_idx)
