import numpy as np
import pytest

from elasticapp import BlockRodSystem


def test_compute_internal_forces_and_torques():
    """Test that compute_internal_forces_and_torques can be called."""
    block = BlockRodSystem([3, 4])
    # Should not raise an exception
    block.compute_internal_forces_and_torques()


def test_update_accelerations():
    """Test that update_accelerations can be called."""
    block = BlockRodSystem([3, 4])
    # Should not raise an exception
    block.update_accelerations()


def test_zeroed_out_external_forces_and_torques():
    """Test that zeroed_out_external_forces_and_torques can be called."""
    block = BlockRodSystem([3, 4])
    # Should not raise an exception
    block.zeroed_out_external_forces_and_torques()


def test_update_kinematics():
    """Test that update_kinematics can be called."""
    block = BlockRodSystem([3, 4])
    # Should not raise an exception
    block.update_kinematics()


def test_update_dynamics():
    """Test that update_dynamics can be called."""
    block = BlockRodSystem([3, 4])
    # Should not raise an exception
    block.update_dynamics()


def test_all_operations_in_sequence():
    """Test that all operations can be called in sequence."""
    block = BlockRodSystem([5, 6, 7])

    # Call all operations in a typical simulation sequence
    block.zeroed_out_external_forces_and_torques()
    block.compute_internal_forces_and_torques()
    block.update_accelerations()
    block.update_kinematics()
    block.update_dynamics()

    # Verify block is still valid after operations
    assert block.shape[0] > 0  # depth
    assert block.shape[1] > 0  # width


def test_operations_with_different_block_sizes():
    """Test that operations work with different block configurations."""
    # Single rod
    block1 = BlockRodSystem([10])
    block1.compute_internal_forces_and_torques()
    block1.update_accelerations()

    # Multiple rods
    block2 = BlockRodSystem([3, 5, 7, 9])
    block2.compute_internal_forces_and_torques()
    block2.update_accelerations()

    # Many rods
    block3 = BlockRodSystem([2] * 10)
    block3.compute_internal_forces_and_torques()
    block3.update_accelerations()


def test_operations_are_callable():
    """Test that all operation methods exist and are callable."""
    block = BlockRodSystem([3, 4])

    # Verify all methods exist
    assert hasattr(block, 'compute_internal_forces_and_torques')
    assert hasattr(block, 'update_accelerations')
    assert hasattr(block, 'zeroed_out_external_forces_and_torques')
    assert hasattr(block, 'update_kinematics')
    assert hasattr(block, 'update_dynamics')

    # Verify they are callable
    assert callable(block.compute_internal_forces_and_torques)
    assert callable(block.update_accelerations)
    assert callable(block.zeroed_out_external_forces_and_torques)
    assert callable(block.update_kinematics)
    assert callable(block.update_dynamics)
