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
    # Rod 0: 3 elems -> elements 0-2, ghost elements at 3, 4
    # Rod 1: 5 elems -> elements 5-9, ghost elements at 10, 11
    # Rod 2: 2 elems -> elements 12-13, ghost element at 14 (last element)
    assert ghost_elems == [3, 4, 10, 11, 14]


def test_ghost_voronoi_idx():
    """Test ghost voronoi indices computation."""
    block = BlockRodSystem([3, 5, 2])

    ghost_voronoi = block.ghost_voronoi_idx
    # Rod 0: 3 elems -> voronoi 0-1, ghost voronoi at 2, 3, 4
    # Rod 1: 5 elems -> voronoi 5-8, ghost voronoi at 9, 10, 11
    # Rod 2: 2 elems -> voronoi 12, ghost voronoi at 13, 14 (includes last voronoi)
    assert ghost_voronoi == [2, 3, 4, 9, 10, 11, 13, 14]


def test_ghost_indices_single_rod():
    """Test ghost indices with single rod (includes last element/voronoi)."""
    block = BlockRodSystem([6])

    ghost_nodes = block.ghost_nodes_idx
    ghost_elems = block.ghost_elems_idx
    ghost_voronoi = block.ghost_voronoi_idx

    assert ghost_nodes == []
    assert ghost_elems == [6]  # Last element is included as ghost
    assert ghost_voronoi == [5, 6]  # Last two voronoi are included as ghosts


def test_ghost_indices_two_rods():
    """Test ghost indices with two rods."""
    block = BlockRodSystem([3, 3])
    # Rod 0: 3 elems -> 4 nodes (indices 0-3)
    # Ghost node at 4
    # Rod 1: 3 elems -> 4 nodes (indices 5-8)

    ghost_nodes = block.ghost_nodes_idx
    assert ghost_nodes == [4]
    ghost_elems = block.ghost_elems_idx
    assert ghost_elems == [3, 4, 8]  # Includes last element of rod 1
    ghost_voronoi = block.ghost_voronoi_idx
    assert ghost_voronoi == [2, 3, 4, 7, 8]  # Includes last voronoi of rod 1


def test_ghost_indices_empty_block():
    """Test ghost indices with empty block (should raise error)."""
    with pytest.raises(ValueError, match="n_elems_per_rod cannot be empty"):
        block = BlockRodSystem([])
