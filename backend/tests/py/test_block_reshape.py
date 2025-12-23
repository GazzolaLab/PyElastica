"""
Tests for Block variable reshaping behavior.

This module tests that:
- Scalar variables return 1D arrays
- Vector variables return 2D arrays
- Matrix variables return 3D arrays that are views (not copies)
"""
import numpy as np
import pytest

from elasticapp import BlockRodSystem


class TestScalarVariableReshaping:
    """Test that Scalar variables return 1D arrays."""

    def test_scalar_variables_are_1d(self):
        """Test that scalar variables return 1D arrays."""
        block = BlockRodSystem([6, 6])
        # Scalar variables: mass, density, volume, etc.
        mass = block.get("mass")
        density = block.get("density")
        volume = block.get("volume")

        # Should be 1D arrays
        assert mass.ndim == 1
        assert density.ndim == 1
        assert volume.ndim == 1

        # Check shapes (accounting for ghost nodes)
        # 2 rods with 6 elements each = 7 nodes each = 14 nodes + 1 ghost = 15 total
        assert mass.shape == (15,)
        # OnElement variables: width - 1 = 15 - 1 = 14
        assert density.shape == (14,)
        assert volume.shape == (14,)

    def test_scalar_variables_are_writable(self):
        """Test that scalar variables are writable."""
        block = BlockRodSystem([6])
        mass = block.get("mass")

        assert mass.flags.writeable is True
        assert mass.flags.owndata is False  # Should be a view

        # Modify and verify
        original_value = mass[0]
        mass[0] = 999.0
        assert mass[0] == 999.0

        # Verify persistence
        mass_new = block.get("mass")
        assert mass_new[0] == 999.0


class TestVectorVariableReshaping:
    """Test that Vector variables return 2D arrays."""

    def test_vector_variables_are_2d(self):
        """Test that vector variables return 2D arrays."""
        block = BlockRodSystem([6, 6])
        # Vector variables: position, velocity, acceleration, etc.
        position = block.get("position")
        velocity = block.get("velocity")
        acceleration = block.get("acceleration")

        # Should be 2D arrays
        assert position.ndim == 2
        assert velocity.ndim == 2
        assert acceleration.ndim == 2

        # Check shapes: (3, n_cols)
        # 2 rods with 6 elements each = 7 nodes each = 14 nodes + 1 ghost = 15 total
        assert position.shape == (3, 15)
        assert velocity.shape == (3, 15)
        assert acceleration.shape == (3, 15)

    def test_vector_variables_are_writable(self):
        """Test that vector variables are writable."""
        block = BlockRodSystem([6])
        position = block.get("position")

        assert position.flags.writeable is True
        assert position.flags.owndata is False  # Should be a view

        # Modify and verify
        original_value = position[0, 0].copy()
        position[0, 0] = 999.0
        assert position[0, 0] == 999.0

        # Verify persistence
        position_new = block.get("position")
        assert position_new[0, 0] == 999.0


class TestMatrixVariableReshaping:
    """Test that Matrix variables return 3D arrays that are views."""

    def test_matrix_variables_are_3d(self):
        """Test that matrix variables return 3D arrays."""
        block = BlockRodSystem([6, 6])
        # Matrix variables: director, shear_matrix, bend_matrix, etc.
        director = block.get("director")
        shear_matrix = block.get("shear_matrix")
        bend_matrix = block.get("bend_matrix")
        mass_second_moment = block.get("mass_second_moment_of_inertia")

        # Should be 3D arrays
        assert director.ndim == 3
        assert shear_matrix.ndim == 3
        assert bend_matrix.ndim == 3
        assert mass_second_moment.ndim == 3

        # Check shapes: (3, 3, n_cols)
        # OnElement variables: width - 1 = 15 - 1 = 14
        assert director.shape == (3, 3, 14)
        assert shear_matrix.shape == (3, 3, 14)
        assert bend_matrix.shape == (3, 3, 13)  # OnVoronoi: width - 2 = 15 - 2 = 13
        assert mass_second_moment.shape == (3, 3, 14)

    def test_matrix_variables_are_views_not_copies(self):
        """Test that matrix variables are views, not copies."""
        block = BlockRodSystem([6])
        director = block.get("director")

        # Should be a view, not a copy
        assert director.flags.owndata is False, "Matrix variables should be views, not copies"
        assert director.flags.writeable is True
        assert director.base is not None, "View should have a base array"

    def test_matrix_variables_modifications_persist(self):
        """Test that modifications to matrix variables persist."""
        block = BlockRodSystem([6])
        director = block.get("director")

        # Modify a value
        original_value = director[0, 0, 0].copy()
        director[0, 0, 0] = 123.456

        # Verify the modification
        assert director[0, 0, 0] == 123.456

        # Get a new view and verify it sees the same modification
        director_new = block.get("director")
        assert director_new[0, 0, 0] == 123.456, "Modifications should persist across views"

    def test_matrix_variables_reference_same_memory(self):
        """Test that matrix variables reference the same underlying memory."""
        block = BlockRodSystem([6])
        director1 = block.get("director")
        director2 = block.get("director")

        # Modify through first view
        director1[0, 0, 0] = 789.0

        # Check through second view
        assert director2[0, 0, 0] == 789.0, "Both views should reference the same memory"

    def test_matrix_variables_correct_strides(self):
        """Test that matrix variables have correct strides for non-contiguous view."""
        block = BlockRodSystem([6])
        director = block.get("director")

        # Strides should be non-contiguous (not C or F contiguous)
        # For a (3, 3, 6) view of a (9, 6) column-major matrix:
        # - Page stride: outer_stride (distance between columns in original)
        # - Row stride: inner_stride (distance between rows)
        # - Col stride: 3 * inner_stride (distance between columns in 3x3)
        assert len(director.strides) == 3

        # Strides should be in bytes
        # Col stride should be 3 * 8 = 24 bytes
        # inner_stride should be sizeof(double) = 8 bytes
        assert director.strides[0] == 24, "Col stride should be 24 bytes (3 * sizeof(double))"
        assert director.strides[1] == 8, "Row stride should be 8 bytes (sizeof(double))"


        # Page stride should be outer_stride (depends on underlying matrix)
        assert director.strides[0] > 0, "Page stride should be positive"


class TestBlockRodSystemViewReshaping:
    """Test reshaping behavior for BlockRodSystemView."""

    def test_blockview_scalar_is_1d(self):
        """Test that BlockRodSystemView scalar variables are 1D."""
        block = BlockRodSystem([6])
        view = block.at(0)

        mass = view.get("mass")
        assert mass.ndim == 1
        assert mass.shape == (7,)  # 6 elements + 1 = 7 nodes

    def test_blockview_vector_is_2d(self):
        """Test that BlockRodSystemView vector variables are 2D."""
        block = BlockRodSystem([6])
        view = block.at(0)

        position = view.get("position")
        assert position.ndim == 2
        assert position.shape == (3, 7)  # 3 rows, 7 nodes

    def test_blockview_matrix_is_3d(self):
        """Test that BlockRodSystemView matrix variables are 3D views."""
        block = BlockRodSystem([6])
        view = block.at(0)

        director = view.get("director")
        assert director.ndim == 3
        assert director.shape == (3, 3, 6)  # 3x3 matrices, 6 elements
        assert director.flags.owndata is False, "Should be a view, not a copy"

    def test_blockview_matrix_modifications_persist(self):
        """Test that BlockRodSystemView matrix modifications persist."""
        block = BlockRodSystem([6])
        view = block.at(0)

        director = view.get("director")
        director[0, 0, 0] = 456.789

        # Get new view
        director_new = view.get("director")
        assert director_new[0, 0, 0] == 456.789


class TestReshapingEdgeCases:
    """Test edge cases for reshaping."""

    def test_single_rod_block(self):
        """Test reshaping with a single rod."""
        block = BlockRodSystem([5])

        # Scalar
        mass = block.get("mass")
        assert mass.ndim == 1
        assert mass.shape == (6,)  # 5 elements + 1 = 6 nodes

        # Vector
        position = block.get("position")
        assert position.ndim == 2
        assert position.shape == (3, 6)

        # Matrix
        director = block.get("director")
        assert director.ndim == 3
        assert director.shape == (3, 3, 5)  # 5 elements
        assert director.flags.owndata is False

    def test_multiple_rods_different_sizes(self):
        """Test reshaping with multiple rods of different sizes."""
        block = BlockRodSystem([3, 5, 2])

        # Scalar (OnNode)
        mass = block.get("mass")
        assert mass.ndim == 1
        # 3 elems -> 4 nodes, 5 elems -> 6 nodes, 2 elems -> 3 nodes
        # Ghost nodes: 2 (between 3 rods)
        # Total: 4 + 6 + 3 + 2 = 15
        assert mass.shape == (15,)

        # Vector (OnNode)
        position = block.get("position")
        assert position.ndim == 2
        assert position.shape == (3, 15)

        # Matrix (OnElement)
        director = block.get("director")
        assert director.ndim == 3
        # OnElement: width - 1 = 15 - 1 = 14
        assert director.shape == (3, 3, 14)
        assert director.flags.owndata is False

    def test_voronoi_variables(self):
        """Test reshaping for OnVoronoi variables."""
        block = BlockRodSystem([6, 6])

        # OnVoronoi variables should have width - 2
        kappa = block.get("kappa")
        assert kappa.ndim == 2  # Vector, not Matrix
        # OnVoronoi: width - 2 = 15 - 2 = 13
        assert kappa.shape == (3, 13)

        # OnVoronoi Matrix variable
        # Note: There might not be OnVoronoi Matrix variables in the system
        # This test verifies the width adjustment works correctly
