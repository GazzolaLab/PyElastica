__doc__ = """ Boundary conditions for rod in Elastica Numpy implementation"""
__all__ = ["FreeRod", "OneEndFixedRod", "HelicalBucklingBC"]
import numpy as np
from elastica._rotations import _get_rotation_matrix

import numba
from numba import njit


class FreeRod:
    """
    the base class for rod boundary conditions
    also the free rod class
    """

    def __init__(self):
        pass

    def constrain_values(self, rod, time):
        pass

    def constrain_rates(self, rod, time):
        pass


class OneEndFixedRod(FreeRod):
    """
    the end of the rod fixed x[0]
    """

    def __init__(self, fixed_position, fixed_directors):
        FreeRod.__init__(self)
        self.fixed_position = fixed_position
        self.fixed_directors = fixed_directors

    def constrain_values(self, rod, time):
        rod.position_collection[..., 0] = self.fixed_position
        rod.director_collection[..., 0] = self.fixed_directors

    def constrain_rates(self, rod, time):
        rod.velocity_collection[..., 0] = 0.0
        rod.omega_collection[..., 0] = 0.0


class HelicalBucklingBC(FreeRod):
    """
    boundary condition for helical buckling
    controlled twisting of the ends
    """

    def __init__(
        self,
        position_start,
        position_end,
        director_start,
        director_end,
        twisting_time,
        slack,
        number_of_rotations,
    ):
        FreeRod.__init__(self)
        self.twisting_time = twisting_time

        angel_vel_scalar = (
            2.0 * number_of_rotations * np.pi / self.twisting_time
        ) / 2.0
        shrink_vel_scalar = slack / (self.twisting_time * 2.0)

        direction = (position_end - position_start) / np.linalg.norm(
            position_end - position_start
        )

        self.final_start_position = position_start + slack / 2.0 * direction
        self.final_end_position = position_end - slack / 2.0 * direction

        self.ang_vel = angel_vel_scalar * direction
        self.shrink_vel = shrink_vel_scalar * direction

        theta = number_of_rotations * np.pi

        self.final_start_directors = (
            _get_rotation_matrix(theta, direction.reshape(3, 1)).reshape(3, 3)
            @ director_start
        )  # rotation_matrix wants vectors 3,1
        self.final_end_directors = (
            _get_rotation_matrix(-theta, direction.reshape(3, 1)).reshape(3, 3)
            @ director_end
        )  # rotation_matrix wants vectors 3,1

    def constrain_values(self, rod, time):
        if time > self.twisting_time:
            rod.position_collection[..., 0] = self.final_start_position
            rod.position_collection[..., -1] = self.final_end_position

            rod.director_collection[..., 0] = self.final_start_directors
            rod.director_collection[..., -1] = self.final_end_directors

    def constrain_rates(self, rod, time):
        if time > self.twisting_time:
            rod.velocity_collection[..., 0] = 0.0
            rod.omega_collection[..., 0] = 0.0

            rod.velocity_collection[..., -1] = 0.0
            rod.omega_collection[..., -1] = 0.0

        else:
            rod.velocity_collection[..., 0] = self.shrink_vel
            rod.omega_collection[..., 0] = self.ang_vel

            rod.velocity_collection[..., -1] = -self.shrink_vel
            rod.omega_collection[..., -1] = -self.ang_vel
