__doc__ = """ Rod class for testing module """


import numpy as np
from elastica.utils import MaxDimension


class TestRod:
    def __init__(self):
        bs = 32
        self.position_collection = np.random.randn(MaxDimension.value(), bs)
        self.director_collection = np.random.randn(
            MaxDimension.value(), MaxDimension.value(), bs
        )
        self.velocity_collection = np.random.randn(MaxDimension.value(), bs)
        self.omega_collection = np.random.randn(MaxDimension.value(), bs)
        self.mass = np.abs(np.random.randn(bs))
        self.external_forces = np.zeros(bs)
