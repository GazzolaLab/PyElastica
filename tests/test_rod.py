__doc__ = """ Rod class for testing module """


import numpy as np
from elastica.utils import MaxDimension


class MockTestRod:
    def __init__(self):
        self.n_elem = 32
        self.position_collection = np.random.randn(MaxDimension.value(), self.n_elem)
        self.director_collection = np.random.randn(
            MaxDimension.value(), MaxDimension.value(), self.n_elem
        )
        self.velocity_collection = np.random.randn(MaxDimension.value(), self.n_elem)
        self.omega_collection = np.random.randn(MaxDimension.value(), self.n_elem)
        self.mass = np.abs(np.random.randn(self.n_elem))
        self.external_forces = np.zeros(self.n_elem)
