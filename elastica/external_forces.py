__doc__ = """ External forcing for rod """

import numpy as np




# the base class for rod external forcing
# also the no forcing class
class NoForces:
    def __init__(self, rod):
        self.rod = rod

    def apply_forces(self):
        pass

    def apply_torques(self):
        pass


# apply gravity on the list of rod
class GravityForces(NoForces):
    def __init__(self, rod, gravity):
        self.rod = rod
        self.gravity = gravity

    def apply_forces(self):
        self.rod.external_forces += np.outer(self.gravity, self.rod.mass)


# puts constant forces on endpoints
# can be modified for temporal variation
class EndpointForces(NoForces):
    def __init__(self, rod, start_force, end_force):
        self.rod = rod
        self.start_force = start_force
        self.end_force = end_force

    def apply_forces(self):
        self.rod.external_forces[..., 0] += self.start_force
        self.rod.external_forces[..., -1] += self.end_force
