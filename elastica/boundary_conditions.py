__doc__ = """ Boundary conditions for rod """

import numpy as np

from ._rod import *
from ._linalg import _batch_matmul, _batch_matvec, _batch_cross


# the base class for rod boundary conditions
# also the free rod class
class FreeRod:
    def __init__(self, rod):
        self.rod = rod

    def dirichlet(self):
        pass

    def neumann(self):
        pass


# start of the rod fixed
class OneEndFixedRod(FreeRod):
    def __init__(self, rod, start_position, start_directors):
        self.rod = rod
        self.start_position = start_position
        self.start_directors = start_directors

    def dirichlet(self):
        self.rod.position[..., 0] = self.start_position
        self.rod.directors[..., 0] = self.start_directors

    def neumann(self):
        self.rod.velocity[..., 0] = 0
        self.rod.omega[..., 0] = 0
