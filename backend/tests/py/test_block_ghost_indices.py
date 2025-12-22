import numpy as np
import pytest

from elasticapp import BlockRodSystem


def test_ghost_nodes_idx():
    """Test ghost nodes indices computation."""
    block = BlockRodSystem([3, 5, 2])
    # Rod 0: 3 elems -> 4 nodes (indices 0-3)
    # Ghost node at 4
    # Rod 1: 5 elems -> 6 nodes (indices 5-10)
    # Ghost node at 11
    # Rod 2: 2 elems -> 3 nodes (indices 12-14)

    ghost_nodes = block.ghost_nodes_idx
    assert ghost_nodes == [4, 11]


def test_ghost_elems_idx():
    """Test ghost elements indices computation."""
    block = BlockRodSystem([3, 5, 2])

    ghost_elems = block.ghost_elems_idx
    assert ghost_elems == [3, 4, 10, 11]


def test_ghost_voronoi_idx():
    """Test ghost voronoi indices computation."""
    block = BlockRodSystem([3, 5, 2])

    ghost_voronoi = block.ghost_voronoi_idx
    assert ghost_voronoi == [2, 3, 4, 9, 10, 11]


def test_ghost_indices_single_rod():
    """Test ghost indices with single rod (should be empty)."""
    block = BlockRodSystem([5])

    ghost_nodes = block.ghost_nodes_idx
    ghost_elems = block.ghost_elems_idx
    ghost_voronoi = block.ghost_voronoi_idx

    assert ghost_nodes == []
    assert ghost_elems == []
    assert ghost_voronoi == []


def test_ghost_indices_two_rods():
    """Test ghost indices with two rods."""
    block = BlockRodSystem([2, 3])
    # Rod 0: 2 elems -> 3 nodes (indices 0-2)
    # Ghost node at 3
    # Rod 1: 3 elems -> 4 nodes (indices 4-7)

    ghost_nodes = block.ghost_nodes_idx
    assert ghost_nodes == [3]
    ghost_elems = block.ghost_elems_idx
    assert ghost_elems == [2, 3]
    ghost_voronoi = block.ghost_voronoi_idx
    assert ghost_voronoi == [1, 2, 3]


def test_ghost_indices_empty_block():
    """Test ghost indices with empty block (should be empty)."""
    block = BlockRodSystem([])

    ghost_nodes = block.ghost_nodes_idx
    ghost_elems = block.ghost_elems_idx
    ghost_voronoi = block.ghost_voronoi_idx

    assert ghost_nodes == []
    assert ghost_elems == []
    assert ghost_voronoi == []
