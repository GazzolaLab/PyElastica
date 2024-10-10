__doc__ = """ Test implementation of the damper base class"""

import numpy as np
from numpy.testing import assert_allclose

from elastica.dissipation import DamperBase
from elastica.utils import Tolerance
from tests.test_rod.mock_rod import MockTestRod


def test_damper_base():
    test_rod = MockTestRod()
    test_rod.velocity_collection = np.ones(3) * 5.0
    test_rod.omega_collection = np.ones(3) * 11.0

    class TestDamper(DamperBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def dampen_rates(self, rod, time):
            rod.velocity_collection *= time
            rod.omega_collection *= time

    test_damper = TestDamper(_system=test_rod)
    test_damper.dampen_rates(test_rod, np.float64(2.0))
    assert_allclose(test_rod.velocity_collection, 10.0, atol=Tolerance.atol())
    assert_allclose(test_rod.omega_collection, 22.0, atol=Tolerance.atol())


def test_damper_base_properties_access():
    test_rod = MockTestRod()

    class TestDamper(DamperBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Able to access properties in constraint class
            assert self._system == test_rod

        def dampen_rates(self, rod, time):
            assert self._system == test_rod

    test_damper = TestDamper(_system=test_rod)
    test_damper.dampen_rates(test_rod, np.float64(2.0))
