__doc__ = """ Test Boundary conditions for in Elastica implementation"""
import sys

# System imports
import numpy as np
from tests.test_rod.test_rods import MockTestRod
from elastica.boundary_conditions import (
    ConstraintBase,
    FreeBC,
    OneEndFixedBC,
    FixedConstraint,
    HelicalBucklingBC,
)
from numpy.testing import assert_allclose
from elastica.utils import Tolerance
import pytest
from pytest import main

test_built_in_boundary_condition_impls = [
    FreeBC,
    OneEndFixedBC,
    FixedConstraint,
    HelicalBucklingBC,
]


def test_constraint_base():
    test_rod = MockTestRod()
    test_rod.position_collection = np.ones(3) * 3.0
    test_rod.velocity_collection = np.ones(3) * 5.0
    test_rod.director_collection = np.ones(3) * 7.0
    test_rod.omega_collection = np.ones(3) * 11.0

    class TestBC(ConstraintBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def constrain_values(self, rod, time):
            rod.position_collection *= time
            rod.director_collection *= time

        def constrain_rates(self, rod, time):
            rod.velocity_collection *= time
            rod.omega_collection *= time

    testBC = TestBC(_system=test_rod)
    testBC.constrain_values(test_rod, 2)
    testBC.constrain_rates(test_rod, 2)
    assert_allclose(test_rod.position_collection, 6.0, atol=Tolerance.atol())
    assert_allclose(test_rod.director_collection, 14.0, atol=Tolerance.atol())
    assert_allclose(test_rod.velocity_collection, 10.0, atol=Tolerance.atol())
    assert_allclose(test_rod.omega_collection, 22.0, atol=Tolerance.atol())


def test_constraint_base_properties_access():
    test_rod = MockTestRod()

    class TestBC(ConstraintBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Able to access properties in constraint class
            assert self._system == test_rod
            assert self.constrained_position_idx == 11
            assert self.constrained_director_idx == 17

        def constrain_values(self, rod, time):
            assert self._system == test_rod
            assert self.constrained_position_idx == 11
            assert self.constrained_director_idx == 17

        def constrain_rates(self, rod, time):
            assert self._system == test_rod
            assert self.constrained_position_idx == 11
            assert self.constrained_director_idx == 17

    testBC = TestBC(
        constrained_position_idx=11, constrained_director_idx=17, _system=test_rod
    )
    testBC.constrain_values(test_rod, 2)
    testBC.constrain_rates(test_rod, 2)


# tests free rod boundary conditions
def test_free_rod():
    test_rod = MockTestRod()
    free_rod = FreeBC(_system=test_rod)
    test_position_collection = np.random.rand(3, 20)
    test_rod.position_collection = (
        test_position_collection.copy()
    )  # We need copy of the list not a reference to this array
    test_director_collection = np.random.rand(3, 3, 20)
    test_rod.director_collection = (
        test_director_collection.copy()
    )  # We need copy of the list not a reference to this array
    free_rod.constrain_values(test_rod, time=0)
    assert_allclose(
        test_position_collection, test_rod.position_collection, atol=Tolerance.atol()
    )
    assert_allclose(
        test_director_collection, test_rod.director_collection, atol=Tolerance.atol()
    )

    test_velocity_collection = np.random.rand(3, 20)
    test_rod.velocity_collection = (
        test_velocity_collection.copy()
    )  # We need copy of the list not a reference to this array
    test_omega_collection = np.random.rand(3, 20)
    test_rod.omega_collection = (
        test_omega_collection.copy()
    )  # We need copy of the list not a reference to this array
    free_rod.constrain_rates(test_rod, time=0)
    assert_allclose(
        test_velocity_collection, test_rod.velocity_collection, atol=Tolerance.atol()
    )
    assert_allclose(
        test_omega_collection, test_rod.omega_collection, atol=Tolerance.atol()
    )


def test_one_end_fixed_bc():

    test_rod = MockTestRod()
    start_position_collection = np.random.rand(3)
    start_director_collection = np.random.rand(3, 3)
    fixed_rod = OneEndFixedBC(
        start_position_collection, start_director_collection, _system=test_rod
    )
    test_position_collection = np.random.rand(3, 20)
    test_rod.position_collection = (
        test_position_collection.copy()
    )  # We need copy of the list not a reference to this array
    test_director_collection = np.random.rand(3, 3, 20)
    test_rod.director_collection = (
        test_director_collection.copy()
    )  # We need copy of the list not a reference to this array
    fixed_rod.constrain_values(test_rod, time=0)
    test_position_collection[..., 0] = start_position_collection
    test_director_collection[..., 0] = start_director_collection
    assert_allclose(
        test_position_collection, test_rod.position_collection, atol=Tolerance.atol()
    )
    assert_allclose(
        test_director_collection, test_rod.director_collection, atol=Tolerance.atol()
    )

    test_velocity_collection = np.random.rand(3, 20)
    test_rod.velocity_collection = (
        test_velocity_collection.copy()
    )  # We need copy of the list not a reference to this array
    test_omega_collection = np.random.rand(3, 20)
    test_rod.omega_collection = (
        test_omega_collection.copy()
    )  # We need copy of the list not a reference to this array
    fixed_rod.constrain_rates(test_rod, time=0)
    test_velocity_collection[..., 0] = np.array((0, 0, 0))
    test_omega_collection[..., 0] = np.array((0, 0, 0))
    assert_allclose(
        test_velocity_collection, test_rod.velocity_collection, atol=Tolerance.atol()
    )
    assert_allclose(
        test_omega_collection, test_rod.omega_collection, atol=Tolerance.atol()
    )


@pytest.mark.parametrize("seed", [1, 10, 100])
@pytest.mark.parametrize("n_position_constraint", [0, 1, 3, 5])
@pytest.mark.parametrize("n_director_constraint", [0, 2, 6, 9])
def test_fixed_constraint(seed, n_position_constraint, n_director_constraint):
    rng = np.random.default_rng(seed)
    N = 20

    test_rod = MockTestRod()
    start_position_collection = rng.random((n_position_constraint, 3))
    start_director_collection = rng.random((n_director_constraint, 3, 3))
    fixed_rod = FixedConstraint(
        *start_position_collection, *start_director_collection, _system=test_rod
    )
    pos_indices = rng.choice(N, size=n_position_constraint, replace=False)
    dir_indices = rng.choice(N, size=n_director_constraint, replace=False)
    fixed_rod._constrained_position_idx = pos_indices.copy()
    fixed_rod._constrained_director_idx = dir_indices.copy()

    test_position_collection = rng.random((3, N))
    test_rod.position_collection = (
        test_position_collection.copy()
    )  # We need copy of the list not a reference to this array
    test_director_collection = rng.random((3, 3, N))
    test_rod.director_collection = (
        test_director_collection.copy()
    )  # We need copy of the list not a reference to this array
    fixed_rod.constrain_values(test_rod, time=0)
    test_position_collection[..., pos_indices] = start_position_collection.transpose(
        (1, 0)
    )
    test_director_collection[..., dir_indices] = start_director_collection.transpose(
        (1, 2, 0)
    )
    assert_allclose(
        test_position_collection, test_rod.position_collection, atol=Tolerance.atol()
    )
    assert_allclose(
        test_director_collection, test_rod.director_collection, atol=Tolerance.atol()
    )

    test_velocity_collection = rng.random((3, N))
    test_rod.velocity_collection = (
        test_velocity_collection.copy()
    )  # We need copy of the list not a reference to this array
    test_omega_collection = rng.random((3, N))
    test_rod.omega_collection = (
        test_omega_collection.copy()
    )  # We need copy of the list not a reference to this array
    fixed_rod.constrain_rates(test_rod, time=0)
    test_velocity_collection[..., pos_indices] = 0.0
    test_omega_collection[..., dir_indices] = 0.0
    assert_allclose(
        test_velocity_collection, test_rod.velocity_collection, atol=Tolerance.atol()
    )
    assert_allclose(
        test_omega_collection, test_rod.omega_collection, atol=Tolerance.atol()
    )


def test_helical_buckling_bc():

    twisting_time = 500.0
    slack = 3.0
    number_of_rotations = 27.0  # number of 2pi rotations
    start_position_collection = np.array([0.0, 0.0, 0.0])
    start_director_collection = np.identity(3, float)
    end_position_collection = np.array([100.0, 0.0, 0.0])
    end_director_collection = np.identity(3, float)

    test_rod = MockTestRod()

    test_position_collection = np.random.rand(3, 20)
    test_position_collection[..., 0] = start_position_collection
    test_position_collection[..., -1] = end_position_collection
    test_rod.position_collection = (
        test_position_collection.copy()
    )  # We need copy of the list not a reference to this array
    test_director_collection = np.tile(np.identity(3, float), 20).reshape(3, 3, 20)
    test_director_collection[..., 0] = start_director_collection
    test_director_collection[..., -1] = end_director_collection
    test_rod.director_collection = (
        test_director_collection.copy()
    )  # We need copy of the list not a reference to this array

    test_velocity_collection = np.random.rand(3, 20)
    test_rod.velocity_collection = (
        test_velocity_collection.copy()
    )  # We need copy of the list not a reference to this array
    test_omega_collection = np.random.rand(3, 20)
    test_rod.omega_collection = (
        test_omega_collection.copy()
    )  # We need copy of the list not a reference to this array
    position_collection_start = test_rod.position_collection[..., 0]
    position_collection_end = test_rod.position_collection[..., -1]
    director_start = test_rod.director_collection[..., 0]
    director_end = test_rod.director_collection[..., -1]
    helicalbuckling_rod = HelicalBucklingBC(
        position_collection_start,
        position_collection_end,
        director_start,
        director_end,
        twisting_time,
        slack,
        number_of_rotations,
        _system=test_rod,
    )

    # Check Neumann BC
    # time < twisting time
    time = twisting_time - 1.0

    helicalbuckling_rod.constrain_rates(test_rod, time=time)
    test_velocity_collection[..., 0] = np.array([0.003, 0.0, 0.0])
    test_velocity_collection[..., -1] = -np.array([0.003, 0.0, 0.0])
    test_omega_collection[..., 0] = np.array([0.169646, 0.0, 0.0])
    test_omega_collection[..., -1] = -np.array([0.169646, 0.0, 0.0])

    assert_allclose(
        test_velocity_collection, test_rod.velocity_collection, atol=Tolerance.atol()
    )
    assert_allclose(
        test_omega_collection, test_rod.omega_collection, atol=Tolerance.atol()
    )

    # time > twisting time
    time = twisting_time + 1
    helicalbuckling_rod.constrain_rates(test_rod, time=time)
    test_velocity_collection[..., 0] = np.array((0, 0, 0))
    test_velocity_collection[..., -1] = np.array((0, 0, 0))
    test_omega_collection[..., 0] = np.array((0, 0, 0))
    test_omega_collection[..., -1] = np.array((0, 0, 0))
    assert_allclose(
        test_velocity_collection, test_rod.velocity_collection, atol=Tolerance.atol()
    )
    assert_allclose(
        test_omega_collection, test_rod.omega_collection, atol=Tolerance.atol()
    )

    # Check Dirichlet BC

    helicalbuckling_rod.constrain_values(test_rod, time=time)

    test_position_collection[..., 0] = np.array([1.5, 0.0, 0.0])
    test_position_collection[..., -1] = np.array([98.5, 0.0, 0.0])

    test_director_collection[..., 0] = np.array(
        [[1.0, 0.0, 0.0], [0.0, -1.0, -6.85926004e-15], [0.0, 6.85926004e-15, -1.0]]
    )

    test_director_collection[..., -1] = np.array(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 6.85926004e-15], [0.0, -6.85926004e-15, -1.0]]
    )

    assert_allclose(
        test_position_collection, test_rod.position_collection, atol=Tolerance.atol()
    )
    assert_allclose(
        test_director_collection, test_rod.director_collection, atol=Tolerance.atol()
    )


if __name__ == "__main__":
    main([__file__])
