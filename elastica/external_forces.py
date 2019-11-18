__doc__ = """ External forcing for rod """

import numpy as np


class NoForces:
    """ Base class for external forcing for Rods

    Can make this an abstract class, but its inconvenient
    for the user to keep on defining apply_forces and
    apply_torques object over and over.
    """

    def __init__(self):
        pass

    def apply_forces(self, system, time: np.float = 0.0):
        """ Apply forces to a system object.

        In NoForces, this routine simply passes.

        Parameters
        ----------
        system : system that is Rod-like
        time : np.float, the time of simulation

        Returns
        -------
        None

        """

        pass

    def apply_torques(self, system, time: np.float = 0.0):
        """ Apply torques to a Rod-like object.

        In NoForces, this routine simply passes.

        Parameters
        ----------
        system : system that is Rod-like
        time : np.float, the time of simulation

        Returns
        -------
        None
        """
        pass


class GravityForces(NoForces):
    """ Applies a constant gravity on the entire rod
    """

    def __init__(self, acc_gravity=np.array([0.0, -9.80665, 0.0])):
        super(GravityForces, self).__init__()
        self.acc_gravity = acc_gravity

    def apply_forces(self, system, time=0.0):
        system.external_forces += np.outer(self.acc_gravity, system.mass)


class EndpointForces(NoForces):
    """ Applies constant forces on endpoints
    """

    def __init__(self, start_force, end_force):
        super(EndpointForces, self).__init__()
        self.start_force = start_force
        self.end_force = end_force

    def apply_forces(self, system, time=0.0):
        system.external_forces[..., 0] += self.start_force
        system.external_forces[..., -1] += self.end_force
