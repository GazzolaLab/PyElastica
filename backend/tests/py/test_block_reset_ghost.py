import numpy as np
import pytest

from elasticapp import BlockRodSystem


def test_reset_ghost_for_variable_onnode():
    """Test reset_ghost_for_variable for OnNode variable."""
    block = BlockRodSystem([3, 5])
    # Ghost nodes at indices 4, 11

    # Get position variable and modify ghost positions
    position = block.get("position")
    position[:] = 0.0
    position[0, 4] = 999.0  # Modify ghost node
    position[1, 4] = 888.0
    position[2, 4] = 777.0

    # Reset ghost for position variable
    block.reset_ghost_for_variable("position")

    # Verify ghost values are reset (position uses Vector, ghost_value = Zero(3,1))
    assert position[0, 4] == 0.0
    assert position[1, 4] == 0.0
    assert position[2, 4] == 0.0


def test_reset_ghost_for_variable_onelement():
    """Test reset_ghost_for_variable for OnElement variable."""
    block = BlockRodSystem([3, 5])
    # Ghost elements at indices 3, 4, 10, 11

    # Get director variable and modify ghost positions
    director = block.get("director")
    director[0, 0, 3] = 999.0  # Modify ghost element
    director[1, 0, 3] = 888.0
    director[2, 0, 3] = 777.0

    # Reset ghost for director variable
    block.reset_ghost_for_variable("director")

    # Verify ghost values are reset (director uses Matrix, ghost_value = Zero(3,3))
    print(director.flags)
    assert director[0, 0, 3] == 0.0
    assert director[1, 0, 3] == 0.0
    assert director[2, 0, 3] == 0.0


def test_reset_ghost_for_variable_onvoronoi():
    """Test reset_ghost_for_variable for OnVoronoi variable."""
    block = BlockRodSystem([3, 5])
    # Ghost voronoi at indices 2, 3, 4, 9, 10, 11

    # Get kappa variable and modify ghost positions
    kappa = block.get("kappa")
    kappa[:, 2] = 999.0  # Modify ghost voronoi

    # Reset ghost for kappa variable
    block.reset_ghost_for_variable("kappa")

    # Verify ghost values are reset (kappa uses Vector, ghost_value = Zero(3,1))
    assert kappa[0, 2] == 0.0
    assert kappa[1, 2] == 0.0
    assert kappa[2, 2] == 0.0


def test_reset_ghost_resets_all_variables():
    """Test that reset_ghost resets all variables."""
    block = BlockRodSystem([2, 3])
    # Ghost nodes at index 3

    # Modify ghost positions for multiple variables
    lengths = block.get("lengths")
    position = block.get("position")
    velocity = block.get("velocity")
    director = block.get("director")

    lengths[3] = 999.0
    position[0, 3] = 999.0
    velocity[0, 3] = 888.0
    director[0, 0, 3] = 777.0  # Ghost element at 2 (before ghost node at 3)

    # Reset all ghosts
    block.reset_ghost()

    # Verify all ghost values are reset
    assert lengths[3] == 0.0
    assert position[0, 3] == 0.0
    assert velocity[0, 3] == 0.0
    assert director[0, 0, 3] == 0.0


def test_reset_ghost_called_in_constructor():
    """Test that reset_ghost is called in constructor."""
    block = BlockRodSystem([3, 5])
    # Constructor should have called reset_ghost()

    # Check that ghost values are already initialized
    position = block.get("position")
    ghost_nodes = block.ghost_nodes_idx

    if len(ghost_nodes) > 0:
        ghost_col = ghost_nodes[0]
        # Ghost values should be zero (default ghost_value for Vector)
        assert position[0, ghost_col] == 0.0
        assert position[1, ghost_col] == 0.0
        assert position[2, ghost_col] == 0.0


def test_reset_ghost_with_single_rod():
    """Test reset_ghost with single rod (no ghosts)."""
    block = BlockRodSystem([5])

    # Should not crash even with no ghost nodes
    block.reset_ghost()
    block.reset_ghost_for_variable("position")


def test_reset_ghost_with_empty_block():
    """Test reset_ghost with empty block."""
    block = BlockRodSystem([])

    # Should not crash even with empty block
    block.reset_ghost()
    block.reset_ghost_for_variable("position")


def test_reset_ghost_for_variable_invalid_name():
    """Test reset_ghost_for_variable with invalid variable name."""
    block = BlockRodSystem([3, 5])

    with pytest.raises(RuntimeError, match="Unknown variable name"):
        block.reset_ghost_for_variable("invalid_variable")


def test_reset_ghost_multiple_ghost_positions():
    """Test reset_ghost with multiple rods (multiple ghost positions)."""
    block = BlockRodSystem([2, 3, 4])
    # Ghost nodes at indices 3, 8
    # Rod 0: 2 elems -> 3 nodes (0-2), ghost at 3
    # Rod 1: 3 elems -> 4 nodes (4-7), ghost at 8
    # Rod 2: 4 elems -> 5 nodes (9-13)

    # Modify multiple ghost positions
    position = block.get("position")
    position[:, 3] = 999.0  # First ghost node
    position[:, 8] = 888.0  # Second ghost node

    # Reset all ghosts
    block.reset_ghost()

    # Verify all ghost positions are reset
    np.testing.assert_array_equal(position[:, 3], 0.0)
    np.testing.assert_array_equal(position[:, 8], 0.0)
