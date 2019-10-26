__doc__ = """ External forcing for rod """

import numpy as np

from ._rod import *
from ._linalg import _batch_matmul, _batch_matvec, _batch_cross


# the base class for rod external forcing
# also the no forcing class
class No_Forces:
    def __init__(self, rod_list):
        self.rod_list = rod_list

    def apply_forces(self):
        pass

    def apply_torques(self):
        pass


# apply gravity on the list of rod
class Gravity_Forces(No_Forces):
    def __init__(self, rod_list, gravity):
        self.rod_list = rod_list
        self.gravity = gravity

    def apply_forces(self):
        for rod in self.rod_list:
            rod.external_forces += np.outer(self.gravity, rod.mass)


# puts constant forces on endpoints
# can be modified for temporal variation
class Endpoint_Forces(No_Forces):
    def __init__(self, rod_list, start_force, end_force):
        self.rod_list = rod_list
        self.start_force = start_force
        self.end_force = end_force

    def apply_forces(self):
        for rod in self.rod_list:
            rod.external_forces[..., 0] += self.start_force
            rod.external_forces[..., -1] += self.end_force
