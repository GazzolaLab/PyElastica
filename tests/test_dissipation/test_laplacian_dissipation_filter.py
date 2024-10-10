__doc__ = """ Test implementation of the Laplacian dissipation filter"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from elastica.dissipation import LaplaceDissipationFilter
from elastica.utils import Tolerance
from tests.test_rod.mock_rod import MockTestRod, MockTestRingRod


@pytest.mark.parametrize("filter_order", [-1, 0, 3.2])
def test_laplace_dissipation_filter_init_invalid_filter_order(filter_order):
    test_rod = MockTestRod()
    with pytest.raises(ValueError) as exc_info:
        _ = LaplaceDissipationFilter(
            _system=test_rod,
            filter_order=filter_order,
        )
    assert (
        exc_info.value.args[0]
        == "Invalid filter order! Filter order must be a positive integer."
    )


@pytest.mark.parametrize("filter_order", [2, 3, 4])
def test_laplace_dissipation_filter_init(filter_order):

    test_rod = MockTestRod()
    filter_damper = LaplaceDissipationFilter(
        _system=test_rod,
        filter_order=filter_order,
    )
    assert filter_damper.filter_order == filter_order
    assert_allclose(
        filter_damper.velocity_filter_term, np.zeros((3, test_rod.n_elems + 1))
    )
    assert_allclose(filter_damper.omega_filter_term, np.zeros((3, test_rod.n_elems)))


@pytest.mark.parametrize("filter_order", [2, 3, 4])
def test_laplace_dissipation_filter_for_constant_field(filter_order):
    test_rod = MockTestRod()
    filter_damper = LaplaceDissipationFilter(
        _system=test_rod,
        filter_order=filter_order,
    )
    test_rod.velocity_collection[...] = 2.0
    test_rod.omega_collection[...] = 3.0
    filter_damper.dampen_rates(system=test_rod, time=np.float64(0.0))
    # filter should keep a spatially invariant field unaffected
    post_damping_velocity_collection = 2.0 * np.ones_like(test_rod.velocity_collection)
    post_damping_omega_collection = 3.0 * np.ones_like(test_rod.omega_collection)
    assert_allclose(
        post_damping_velocity_collection,
        test_rod.velocity_collection,
        atol=Tolerance.atol(),
    )
    assert_allclose(
        post_damping_omega_collection, test_rod.omega_collection, atol=Tolerance.atol()
    )


def test_laplace_dissipation_filter_for_flip_flop_field():
    filter_order = 1
    test_rod = MockTestRod()
    filter_damper = LaplaceDissipationFilter(
        _system=test_rod,
        filter_order=filter_order,
    )
    test_rod.velocity_collection[...] = 0.0
    test_rod.velocity_collection[..., 1::2] = 2.0
    test_rod.omega_collection[...] = 0.0
    test_rod.omega_collection[..., 1::2] = 3.0
    pre_damping_velocity_collection = test_rod.velocity_collection.copy()
    pre_damping_omega_collection = test_rod.omega_collection.copy()
    filter_damper.dampen_rates(system=test_rod, time=np.float64(0.0))
    post_damping_velocity_collection = np.zeros_like(test_rod.velocity_collection)
    post_damping_omega_collection = np.zeros_like(test_rod.omega_collection)
    # filter should remove the flip-flop mode th give the average constant mode
    post_damping_velocity_collection[..., 1:-1] = 2.0 / 2
    post_damping_omega_collection[..., 1:-1] = 3.0 / 2
    # end values remain untouched
    post_damping_velocity_collection[..., 0 :: test_rod.n_elems] = (
        pre_damping_velocity_collection[..., 0 :: test_rod.n_elems]
    )
    post_damping_omega_collection[..., 0 :: test_rod.n_elems - 1] = (
        pre_damping_omega_collection[..., 0 :: test_rod.n_elems - 1]
    )
    assert_allclose(
        post_damping_velocity_collection,
        test_rod.velocity_collection,
        atol=Tolerance.atol(),
    )
    assert_allclose(
        post_damping_omega_collection, test_rod.omega_collection, atol=Tolerance.atol()
    )


@pytest.mark.parametrize("filter_order", [-1, 0, 3.2])
def test_laplace_dissipation_filter_init_invalid_filter_order_for_ring_rod(
    filter_order,
):
    test_rod = MockTestRingRod()
    with pytest.raises(ValueError) as exc_info:
        _ = LaplaceDissipationFilter(
            _system=test_rod,
            filter_order=filter_order,
        )
    assert (
        exc_info.value.args[0]
        == "Invalid filter order! Filter order must be a positive integer."
    )


@pytest.mark.parametrize("filter_order", [2, 3, 4])
def test_laplace_dissipation_filter_init_for_ring_rod(filter_order):

    test_rod = MockTestRingRod()
    filter_damper = LaplaceDissipationFilter(
        _system=test_rod,
        filter_order=filter_order,
    )
    assert filter_damper.filter_order == filter_order
    assert_allclose(
        filter_damper.velocity_filter_term[:, 1:-1], np.zeros((3, test_rod.n_elems))
    )
    assert_allclose(
        filter_damper.omega_filter_term[:, 1:-1], np.zeros((3, test_rod.n_elems))
    )


@pytest.mark.parametrize("filter_order", [2, 3, 4])
def test_laplace_dissipation_filter_for_constant_field_for_ring_rod(filter_order):
    test_rod = MockTestRingRod()
    filter_damper = LaplaceDissipationFilter(
        _system=test_rod,
        filter_order=filter_order,
    )
    test_rod.velocity_collection[...] = 2.0
    test_rod.omega_collection[...] = 3.0
    filter_damper.dampen_rates(system=test_rod, time=np.float64(0.0))
    # filter should keep a spatially invariant field unaffected
    post_damping_velocity_collection = 2.0 * np.ones_like(test_rod.velocity_collection)
    post_damping_omega_collection = 3.0 * np.ones_like(test_rod.omega_collection)
    assert_allclose(
        post_damping_velocity_collection,
        test_rod.velocity_collection,
        atol=Tolerance.atol(),
    )
    assert_allclose(
        post_damping_omega_collection, test_rod.omega_collection, atol=Tolerance.atol()
    )


def test_laplace_dissipation_filter_for_flip_flop_field_for_ring_rod():
    filter_order = 1
    test_rod = MockTestRingRod()
    filter_damper = LaplaceDissipationFilter(
        _system=test_rod,
        filter_order=filter_order,
    )
    test_rod.velocity_collection[...] = 0.0
    test_rod.velocity_collection[..., 1::2] = 2.0
    test_rod.omega_collection[...] = 0.0
    test_rod.omega_collection[..., 1::2] = 3.0
    pre_damping_velocity_collection = test_rod.velocity_collection.copy()
    pre_damping_omega_collection = test_rod.omega_collection.copy()
    filter_damper.dampen_rates(system=test_rod, time=np.float64(0.0))
    post_damping_velocity_collection = np.zeros_like(test_rod.velocity_collection)
    post_damping_omega_collection = np.zeros_like(test_rod.omega_collection)
    # filter should remove the flip-flop mode th give the average constant mode
    post_damping_velocity_collection[:] = 2.0 / 2
    post_damping_omega_collection[:] = 3.0 / 2

    assert_allclose(
        post_damping_velocity_collection,
        test_rod.velocity_collection,
        atol=Tolerance.atol(),
    )
    assert_allclose(
        post_damping_omega_collection, test_rod.omega_collection, atol=Tolerance.atol()
    )
