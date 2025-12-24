import numpy as np
import pytest

from elasticapp import BlockRodSystem


def test_at_returns_block_view():
    """Test that at() returns a BlockRodSystemView object."""
    n_elems_per_system = [3, 5, 2]
    block = BlockRodSystem(n_elems_per_system)
    expected_depth = 103
    # Rod 0: 3 elems -> 4 nodes, Rod 1: 5 elems -> 6 nodes, Rod 2: 2 elems -> 3 nodes
    # Ghost nodes: 2 (between 3 rods)
    # Total width = 4 + 6 + 3 + 2 = 15
    expected_shape = (expected_depth, 15)

    assert block.n_systems == len(n_elems_per_system)
    assert block.shape == expected_shape

    for sys_index in range(block.n_systems):
        block_view = block.at(sys_index)
        assert block_view.shape == (expected_depth, n_elems_per_system[sys_index] + 1)
