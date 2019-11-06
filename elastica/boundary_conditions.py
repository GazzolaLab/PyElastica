__doc__ = """ Boundary conditions for rod """

import numpy as np
from elastica._rotations import _get_rotation_matrix


class FreeRod:
    """
    the base class for rod boundary conditions
    also the free rod class
    """
    def __init__(self, rod):
        self.rod = rod

    def constrain_values(self):
        pass

    def constrain_rates(self):
        pass


class OneEndFixedRod(FreeRod):
    """
    the end of the rod fixed x[-1]
    """
    def __init__(self, rod, start_position, start_directors):
        FreeRod.__init__(self, rod)
        self.start_position = start_position
        self.start_directors = start_directors

    def constrain_values(self):
        self.rod.position[..., 0] = self.start_position
        self.rod.directors[..., 0] = self.start_directors

    def constrain_rates(self):
        self.rod.velocity[..., 0] = 0.0
        self.rod.omega[..., 0] = 0.0


class HelicalBucklingBC(FreeRod):
    """
    boundary condition for helical buckling
    controlled twisting of the ends
    """
    def __init__(self, rod, twisting_time, slack, number_of_rotations):
        FreeRod.__init__(self, rod)
        self.twisting_time = twisting_time

        angel_vel_scalar = (
            2.0 * number_of_rotations * np.pi / self.twisting_time
        ) / 2.0
        shrink_vel_scalar = slack / (self.twisting_time * 2.0)

        direction = (
            self.rod.position[..., -1] - self.rod.position[..., 0]
        ) / np.linalg.norm(self.rod.position[..., -1] - self.rod.position[..., 0])

        self.final_start_position = self.rod.position[..., 0] + slack / 2.0 * direction
        self.final_end_position = self.rod.position[..., -1] - slack / 2.0 * direction

        self.ang_vel = angel_vel_scalar * direction
        self.shrink_vel = shrink_vel_scalar * direction

        theta = number_of_rotations * np.pi

        self.final_start_directors = (
            _get_rotation_matrix(theta, direction.reshape(3, 1)).reshape(3, 3)
            @ self.rod.directors[..., 0]
        )  # rotation_matrix wants vectors 3,1
        self.final_end_directors = (
            _get_rotation_matrix(-theta, direction.reshape(3, 1)).reshape(3, 3)
            @ self.rod.directors[..., -1]
        )  # rotation_matrix wants vectors 3,1

    def constrain_values(self, time):
        if time > self.twisting_time:
            self.rod.position[..., 0] = self.final_start_position
            self.rod.position[..., -1] = self.final_end_position

            self.rod.directors[..., 0] = self.final_start_directors
            self.rod.directors[..., -1] = self.final_end_directors

    def constrain_rates(self, time):
        if time > self.twisting_time:
            self.rod.velocity[..., 0] = 0.0
            self.rod.omega[..., 0] = 0.0

            self.rod.velocity[..., -1] = 0.0
            self.rod.omega[..., -1] = 0.0

        else:
            self.rod.velocity[..., 0] = self.shrink_vel
            self.rod.omega[..., 0] = self.ang_vel

            self.rod.velocity[..., -1] = -self.shrink_vel
            self.rod.omega[..., -1] = -self.ang_vel
