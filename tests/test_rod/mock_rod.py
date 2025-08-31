__doc__ = """Mock Rod classes for testing."""

import numpy as np


from elastica.memory_block.utils import (
    make_block_memory_periodic_boundary_metadata,
)
from elastica.utils import MaxDimension


class MockTestRod:
    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        self.n_elems = 32
        self.ring_rod_flag = False
        self.position_collection = rng.standard_normal(
            (MaxDimension.value(), self.n_elems + 1)
        )
        self.director_collection = rng.standard_normal(
            (MaxDimension.value(), MaxDimension.value(), self.n_elems)
        )
        self.velocity_collection = rng.standard_normal(
            (MaxDimension.value(), self.n_elems + 1)
        )
        self.omega_collection = rng.standard_normal(
            (MaxDimension.value(), self.n_elems)
        )
        self.mass = np.abs(rng.standard_normal(self.n_elems + 1))
        self.external_forces = np.zeros(self.n_elems + 1)


class MockTestRingRod:
    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        self.n_elems = 32
        self.ring_rod_flag = True
        self.position_collection = rng.standard_normal(
            (MaxDimension.value(), self.n_elems)
        )
        self.director_collection = rng.standard_normal(
            (MaxDimension.value(), MaxDimension.value(), self.n_elems)
        )
        self.velocity_collection = rng.standard_normal(
            (MaxDimension.value(), self.n_elems)
        )
        self.omega_collection = rng.standard_normal(
            (MaxDimension.value(), self.n_elems)
        )
        self.mass = np.abs(rng.standard_normal(self.n_elems))
        self.external_forces = np.zeros(self.n_elems)

        n_elems_ring_rods = (np.ones(1) * (self.n_elems - 3)).astype("int64")

        (
            _,
            self.periodic_boundary_nodes_idx,
            self.periodic_boundary_elems_idx,
            self.periodic_boundary_voronoi_idx,
        ) = make_block_memory_periodic_boundary_metadata(n_elems_ring_rods)
