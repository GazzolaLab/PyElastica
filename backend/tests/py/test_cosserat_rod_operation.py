import numpy as np
import pytest

from elasticapp import BlockRodSystem


def test_compute_internal_forces_and_torques():
    """Test that compute_internal_forces_and_torques can be called."""
    block = BlockRodSystem([3, 4])
    # Should not raise an exception
    block.compute_internal_forces_and_torques(0.0)


def test_update_accelerations():
    """Test that update_accelerations can be called."""
    block = BlockRodSystem([3, 4])
    # Should not raise an exception
    block.update_accelerations(0.0)


def test_zeroed_out_external_forces_and_torques():
    """Test that zeroed_out_external_forces_and_torques can be called."""
    block = BlockRodSystem([3, 4])
    # Should not raise an exception
    block.zeroed_out_external_forces_and_torques(0.0)


def test_update_kinematics():
    """Test that update_kinematics can be called."""
    block = BlockRodSystem([3, 4])
    # Should not raise an exception
    block.update_kinematics(0.0, 1.0)


def test_update_dynamics():
    """Test that update_dynamics can be called."""
    block = BlockRodSystem([3, 4])
    # Should not raise an exception
    block.update_dynamics(0.0, 1.0)


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
