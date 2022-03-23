__doc__ = """ Knot Theory class testing """

import pytest
import numpy as np
from numpy.testing import assert_allclose

from elastica.rod.data_structures import _bootstrap_from_data
from elastica.rod.data_structures import (
    _KinematicState,
    _DynamicState,
)
from elastica.utils import MaxDimension

from test_rods import MockTestRod

from elastica.rod.rod_base import RodBase
from elastica.rod.knot_theory import _compute_additional_segment


@pytest.fixture
def knot_theory():
    from elastica.rod import knot_theory

    return knot_theory


def test_knot_theory_mixin_methods(knot_theory):
    class TestRodWithKnotTheory(MockTestRod, knot_theory.KnotTheory):
        def __init__(self):
            super().__init__()
            self.radius = np.random.randn(MaxDimension.value(), self.n_elem)

    rod = TestRodWithKnotTheory()
    assert hasattr(
        rod, "MIXIN_PROTOCOL"
    ), "Expected to mix-in variables: MIXIN_PROTOCOL"
    assert hasattr(
        rod, "compute_writhe"
    ), "Expected to mix-in functionals into the rod class: compute_writhe"
    assert hasattr(
        rod, "compute_twist"
    ), "Expected to mix-in functionals into the rod class: compute_twist"
    assert hasattr(
        rod, "compute_link"
    ), "Expected to mix-in functionals into the rod class: compute_link"


def test_knot_theory_mixin_methods_with_no_radius(knot_theory):
    class TestRodWithKnotTheoryWithoutRadius(MockTestRod, knot_theory.KnotTheory):
        def __init__(self):
            super().__init__()

    rod = TestRodWithKnotTheoryWithoutRadius()
    with pytest.raises(AttributeError) as e_info:
        rod.compute_writhe()
    with pytest.raises(AttributeError) as e_info:
        rod.compute_link()


@pytest.mark.parametrize("timesteps", [1, 5, 10])
@pytest.mark.parametrize("n_elem", [1, 3, 8])
@pytest.mark.parametrize("segment_length", [1.0, 10.0, 100.0])
def test_knot_theory_compute_additional_segment_next_tangent_case(
    timesteps, n_elem, segment_length
):
    ...


@pytest.mark.parametrize("timesteps", [1, 5, 10])
@pytest.mark.parametrize("n_elem", [1, 3, 8])
@pytest.mark.parametrize("segment_length", [1.0, 10.0, 100.0])
def test_knot_theory_compute_additional_segment_end_to_end_case(
    timesteps, n_elem, segment_length
):
    ...


@pytest.mark.parametrize("timesteps", [1, 5, 10])
@pytest.mark.parametrize("n_elem", [1, 3, 8])
@pytest.mark.parametrize("segment_length", [1.0, 10.0, 100.0])
def test_knot_theory_compute_additional_segment_net_tangent_case(
    timesteps, n_elem, segment_length
):
    ...


@pytest.mark.parametrize("timesteps", [1, 5, 10])
@pytest.mark.parametrize("n_elem", [1, 3, 8])
@pytest.mark.parametrize("segment_length", [1.0, 10.0, 100.0])
def test_knot_theory_compute_additional_segment_none_case(
    timesteps, n_elem, segment_length
):
    center_line = np.random.random([timesteps, 3, n_elem])
    new_center_line, beginning_direction, end_direction = _compute_additional_segment(
        center_line, segment_length, None
    )

    assert_allclose(new_center_line, center_line)
    assert_allclose(beginning_direction, 0.0)
    assert_allclose(end_direction, 0.0)
    assert_allclose(new_center_line.shape, [timesteps, 3, n_elem])
    assert_allclose(beginning_direction.shape[0], timesteps)
    assert_allclose(end_direction.shape[0], timesteps)
