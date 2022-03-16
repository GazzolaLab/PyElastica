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

from test_rod.test_rods import MockTestRod

from elastica.rod.rod_base import RodBase
from elastica.rod.knot_theory import KnotTheory


class TestRodWithKnotTheory(MockTestRod, KnotTheory):
    def __init__(self):
        super().__init__()
        self.radius = np.random.randn(MaxDimension.value(), self.n_elem)


def test_knot_theory_mixin_methods():
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
