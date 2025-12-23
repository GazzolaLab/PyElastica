import numpy as np
import pytest

from elasticapp import BlockRodSystem


def test_block_construction():
    """Test that Block can be constructed with shape."""
    block = BlockRodSystem([3, 4])
    # Rod 0: 3 elems -> 4 nodes, Rod 1: 4 elems -> 5 nodes
    # Ghost nodes: 1 (between 2 rods)
    # Total width = 4 + 5 + 1 = 10
    shape = block.shape
    assert shape == (103, 10)


def test_block_as_ref_returns_numpy_array():
    """Test that as_ref() returns a numpy array."""
    block = BlockRodSystem([2, 3])
    # Rod 0: 2 elems -> 3 nodes, Rod 1: 3 elems -> 4 nodes
    # Ghost nodes: 1 (between 2 rods)
    # Total width = 3 + 4 + 1 = 8
    arr = block.as_ref()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (103, 8)
    assert arr.shape == block.shape
    assert arr.flags['OWNDATA'] is False
    assert arr.flags['WRITEABLE'] is True



def test_block_as_ref_modifies_underlying_memory():
    """Test that modifying the numpy array modifies the underlying C++ memory."""
    block = BlockRodSystem([2, 3])
    arr1 = block.as_ref()

    # Try to modify
    arr1[0, 0] = 1.5
    assert arr1[0, 0] == 1.5
    arr1[0, 0] = 2.5
    assert arr1[0, 0] == 2.5


def test_block_as_ref_bidirectional_modification():
    """Test that modifications persist across multiple as_ref() calls."""
    block = BlockRodSystem([3, 3])
    arr1 = block.as_ref()

    # Fill with some values
    vals = np.empty(block.shape)
    arr1[:] = vals
    arr2 = block.as_ref()
    np.testing.assert_array_equal(arr2, vals)

    # Modify through second reference
    arr2[0, 0] = 100.0
    assert arr1[0, 0] == 100.0


def test_block_rod_start_indices():
    """Test that Block stores correct starting indices for each rod."""
    block = BlockRodSystem([3, 5, 2])

    # Rod 0: starts at 0, has 3+1=4 nodes
    assert block.system_start_index(0) == 0
    # Rod 1: starts at 4, has 5+1=6 nodes
    assert block.system_start_index(1) == 4
    # Rod 2: starts at 4+6=10, has 2+1=3 nodes
    assert block.system_start_index(2) == 10

def test_block_view_variable_query():
    """Test that BlockRodSystemView can query variables."""
    block = BlockRodSystem([3, 5, 2])
    view = block.at(0)
    assert view.get("position") is not None
    assert view.get("position").shape == (3, 4)
    assert view.get("velocity") is not None
    assert view.get("velocity").shape == (3, 4)
    assert view.get("director") is not None

    print("")
    print(type(view))
    print(type(view.get("director")))
    print(view.get("director"))
    print(view.get("director").flags)
    print(view.get("director").shape)
    assert view.get("director").shape == (3, 3, 3)


def test_block_get_shape():
    """Test that Block.get() returns correct shapes for different variable types."""
    block = BlockRodSystem([3, 4])
    # Block has 2 rods: rod 0 has 3 elems (4 nodes), rod 1 has 4 elems (5 nodes)
    # Ghost nodes: 1 (between 2 rods)
    # Total width = 4 + 5 + 1 = 10

    # OnNode variables: should have shape (dimension, full width)
    position = block.get("position")
    assert position.shape == (3, 10)  # Vector (3D) x total nodes (full width)
    assert position.flags['OWNDATA'] is False
    assert position.flags['WRITEABLE'] is True

    velocity = block.get("velocity")
    assert velocity.shape == (3, 10)  # OnNode: full width

    mass = block.get("mass")
    assert mass.shape == (10,)  # OnNode: full width

    # OnElement variables: should have shape (dimension, width - 1)
    director = block.get("director")
    assert director.shape == (3, 3, 9)  # Matrix (9D) x (width - 1) = 10 - 1 = 9
    assert director.flags['OWNDATA'] is False
    assert director.flags['WRITEABLE'] is True

    omega = block.get("omega")
    assert omega.shape == (3, 9)  # Vector (3D) x (width - 1) = 10 - 1 = 9

    # OnVoronoi variables: should have shape (dimension, width - 2)
    kappa = block.get("kappa")
    assert kappa.shape == (3, 8)  # Vector (3D) x (width - 2) = 10 - 2 = 8


def test_block_get_contents():
    """Test that Block.get() returns writable views that modify underlying data."""
    block = BlockRodSystem([2, 3])
    # Rod 0: 2 elems -> 3 nodes, Rod 1: 3 elems -> 4 nodes
    # Ghost nodes: 1 (between 2 rods)
    # Total width = 3 + 4 + 1 = 8

    # Get position variable (OnNode: full width)
    position = block.get("position")
    assert position.shape == (3, 8)

    # Modify through the view
    position[0, 0] = 1.5
    position[1, 0] = 2.5
    position[2, 0] = 3.5

    # Verify modifications persist
    assert position[0, 0] == 1.5
    assert position[1, 0] == 2.5
    assert position[2, 0] == 3.5

    # Get another view - should see the same data
    position2 = block.get("position")
    assert position2[0, 0] == 1.5
    assert position2[1, 0] == 2.5
    assert position2[2, 0] == 3.5

    # Modify through second view
    position2[0, 1] = 10.0
    assert position[0, 1] == 10.0  # Should be reflected in first view


def test_block_get_different_variable_types():
    """Test that Block.get() works for different variable types (OnNode, OnElement, OnVoronoi)."""
    block = BlockRodSystem([3, 2])
    # Rod 0: 3 elems -> 4 nodes, Rod 1: 2 elems -> 3 nodes
    # Ghost nodes: 1 (between 2 rods)
    # Total width = 4 + 3 + 1 = 8

    # OnNode variable (Vector) - full width
    velocity = block.get("velocity")
    assert velocity.shape == (3, 8)  # OnNode: full width
    velocity[0, 0] = 100.0
    assert velocity[0, 0] == 100.0

    # OnNode variable (Scalar) - full width
    mass = block.get("mass")
    assert mass.shape == (8,)  # OnNode: full width
    mass[0] = 5.0
    assert mass[0] == 5.0

    # OnElement variable (Matrix) - width - 1
    director = block.get("director")
    assert director.shape == (3, 3, 7)  # OnElement: width - 1 = 8 - 1 = 7
    director[0, 0, 0] = 0.5
    assert director[0, 0, 0] == 0.5

    # OnElement variable (Vector)
    omega = block.get("omega")
    assert omega.shape == (3, 7)
    omega[0, 0] = 1.0
    assert omega[0, 0] == 1.0

    # OnVoronoi variable (Vector) - width - 2
    kappa = block.get("kappa")
    assert kappa.shape == (3, 6)  # OnVoronoi: width - 2 = 8 - 2 = 6
    kappa[0, 0] = 2.0
    assert kappa[0, 0] == 2.0


def test_block_get_invalid_variable():
    """Test that Block.get() raises error for invalid variable names."""
    block = BlockRodSystem([3, 4])

    with pytest.raises(RuntimeError, match="Unknown variable name"):
        block.get("invalid_variable")
