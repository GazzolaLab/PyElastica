__doc__ = """ Boundary condition module constraining values and rates of the rod. """

import numpy as np
from elastica._rotations import _get_rotation_matrix


class FreeRod:
    """
    This is the base class for rod boundary conditions,
    and it is the free rod class.

    Note
    ----
    Every new rod boundary condition class has to be
    derived from FreeRod class.
    """

    def __init__(self):
        pass

    def constrain_values(self, rod, time):
        pass

    def constrain_rates(self, rod, time):
        pass


class OneEndFixedRod(FreeRod):
    """
    This is the one end fixed rod class. Currently,
    this boundary condition fixes position and directors
    at the first node and first element of the rod.

    Attributes
    ----------
    fixed_positions: numpy.ndarray
        2D (dim, 1) array containing data with 'float' type.
    fixed_directors: numpy.ndarray
        3D (dim, dim, 1) array containing data with 'float' type.
    """

    def __init__(self, fixed_position, fixed_directors):
        """

        Parameters
        ----------
        fixed_position: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
        fixed_directors: numpy.ndarray
            3D (dim, dim, 1) array containing data with 'float' type.
        """
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
    This is the boundary condition class for Helical
    Buckling case in Gazzola et. al. RSOS paper 2018.
    Applied boundary condition is twist and slack on to
    the first and last nodes and elements of the rod.

    Attributes
    ----------
    twisting_time: float
    final_start_position: numpy.ndarray
        2D (dim, 1) array containing data with 'float' type.
    final_end_position: numpy.ndarray
        2D (dim, 1) array containing data with 'float' type.
    ang_vel: numpy.ndarray
        2D (dim, 1) array containing data with 'float' type.
    shrink_vel: numpy.ndarray
        2D (dim, 1) array containing data with 'float' type.
    final_start_directors: numpy.ndarray
        3D (dim, dim, blocksize) array containing data with 'float' type.
    final_end_directors: numpy.ndarray
        3D (dim, dim, blocksize) array containing data with 'float' type.

    Note
    ----
    This is a specific boundary condition for the Helical Buckling case. It
    is suggested that users look at Gazzola et. al. RSOS paper 2018

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
        """

        Parameters
        ----------
        position_start: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
        position_end: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
        director_start: numpy.ndarray
            3D (dim, dim, blocksize) array containing data with 'float' type.
        director_end: numpy.ndarray
            3D (dim, dim, blocksize) array containing data with 'float' type.
        twisting_time: float
        slack: float
        number_of_rotations: float
        """
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
