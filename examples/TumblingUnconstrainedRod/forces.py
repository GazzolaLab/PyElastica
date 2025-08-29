import numpy as np
from elastica.external_forces import NoForces
from elastica.typing import SystemType

from tqdm import tqdm


class EndpointforcesWithTimeFactor(NoForces):

    def __init__(self, start_force, end_force, time_factor):

        super(EndpointforcesWithTimeFactor, self).__init__()
        self.start_force = start_force
        self.end_force = end_force
        self.time_factor = time_factor

    def apply_forces(self, system: SystemType, time=0.0):

        factor = self.time_factor(time)

        system.external_forces[..., 0] += self.start_force * factor
        system.external_forces[..., -1] += self.end_force * factor


class EndpointtorqueWithTimeFactor(NoForces):
    def __init__(self, torque, time_factor, direction=np.array([0.0, 0.0, 0.0])):
        super(EndpointtorqueWithTimeFactor, self).__init__()
        self.torque = torque * direction
        self.time_factor = time_factor

    def apply_torques(self, system: SystemType, time: np.float64 = 0.0):
        n_elems = system.n_elems

        factor = self.time_factor(time)

        system.external_torques[..., -1] += self.torque * factor


def lamda_t_function(time):
    if time < 2.5:
        factor = time * (1 / 2.5)
    elif time > 2.5 and time < 5.0:
        factor = -time * (1 / 2.5) + 2
    else:
        factor = 0

    return factor
