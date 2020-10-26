__doc__ = """ Numpy implementation module for boundary condition implementations that constrain or
define displacement conditions on the rod"""
__all__ = ["FreeRod", "OneEndFixedRod", "HelicalBucklingBC"]
import numpy as np
from elastica._rotations import _get_rotation_matrix


class FreeRod:
    """
    This is the base class for displacement boundary conditions. It applies no constraints or displacements to the rod.

    Note
    ----
    Every new displacement boundary condition class must be
    derived from FreeRod class.
    """

    def __init__(self):
        """
        Free rod has no input parameters.
        """
        pass

    def constrain_values(self, rod, time):
        """
        Constrain values (position and/or directors) of a rod object.

        In FreeRod class, this routine simply passes.

        Parameters
        ----------
        rod : object
            Rod-like object.
        time : float
            The time of simulation.

        Returns
        -------

        """
        pass

    def constrain_rates(self, rod, time):
        """
        Constrain rates (velocity and/or omega) of a rod object.

        In FreeRod class, this routine simply passes.

        Parameters
        ----------
        rod : object
            Rod-like object.
        time : float
            The time of simulation.

        Returns
        -------

        """
        pass


class OneEndFixedRod(FreeRod):
    """
    This boundary condition class fixes one end of the rod. Currently,
    this boundary condition fixes position and directors
    at the first node and first element of the rod.

        Attributes
        ----------
        fixed_positions : numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
        fixed_directors : numpy.ndarray
            3D (dim, dim, 1) array containing data with 'float' type.
    """

    def __init__(self, fixed_position, fixed_directors):
        """

        Parameters
        ----------
        fixed_position : numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
        fixed_directors : numpy.ndarray
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
    Buckling case in Gazzola et. al. RSoS (2018).
    The applied boundary condition is twist and slack on to
    the first and last nodes and elements of the rod.

        Attributes
        ----------
        twisting_time: float
            Time to complete twist.
        final_start_position: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Position of first node of rod after twist completed.
        final_end_position: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Position of last node of rod after twist completed.
        ang_vel: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Angular velocity of rod during twisting time.
        shrink_vel: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Shrink velocity of rod during twisting time.
        final_start_directors: numpy.ndarray
            3D (dim, dim, blocksize) array containing data with 'float' type.
            Directors of first element of rod after twist completed.
        final_end_directors: numpy.ndarray
            3D (dim, dim, blocksize) array containing data with 'float' type.
            Directors of last element of rod after twist completed.


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

        position_start : numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Initial position of first node.
        position_end : numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Initial position of last node.
        director_start : numpy.ndarray
            3D (dim, dim, blocksize) array containing data with 'float' type.
            Initial director of first element.
        director_end : numpy.ndarray
            3D (dim, dim, blocksize) array containing data with 'float' type.
            Initial director of last element.
        twisting_time : float
            Time to complete twist.
        slack : float
            Slack applied to rod.
        number_of_rotations : float
            Number of rotations applied to rod.
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
