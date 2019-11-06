__doc__ = """ External forcing for rod """

import numpy as np


class NoForces:
    """
    the base class for rod external forcing
    also the no forcing class
    """

    def __init__(self, rod):
        self.rod = rod

    def apply_forces(self):
        pass

    def apply_torques(self):
        pass


class GravityForces(NoForces):
    """
    apply gravity on the list of rod
    """

    def __init__(self, rod, gravity):
        NoForces.__init__(self, rod)
        self.gravity = gravity

    def apply_forces(self):
        self.rod.external_forces += np.outer(self.gravity, self.rod.mass)


class EndpointForces(NoForces):
    """
    puts constant forces on endpoints
    can be modified for temporal variation
    """

    def __init__(self, rod, start_force, end_force):
        NoForces.__init__(self, rod)
        self.start_force = start_force
        self.end_force = end_force

    def apply_forces(self):
        self.rod.external_forces[..., 0] += self.start_force
        self.rod.external_forces[..., -1] += self.end_force
