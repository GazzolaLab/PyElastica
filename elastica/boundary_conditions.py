__doc__ = """ Numba implementation module for boundary condition implementations that constrain or
define displacement conditions on the rod"""
__all__ = [
    "ConstraintBase",
    "FreeBC",
    "FreeRod", # Deprecated: remove v0.3.0
    "OneEndFixedBC",
    "OneEndFixedRod", # Deprecated: remove v0.3.0
    "FixedConstraint",
    "HelicalBucklingBC",
]

import warnings
from typing import Optional, Type

import numpy as np

from abc import ABC, abstractmethod

import numba
from numba import njit

from elastica._rotations import _get_rotation_matrix
from elastica.rod import RodBase


class ConstraintBase(ABC):
    """Base class for constraint and boundary condition implementation.

    Note
    ----
    Constraint class must inherit BaseConstraint class.

        Attributes
        -------
        system : RodBase
        node_indices : None or numpy.ndarray
        element_indices : None or numpy.ndarray

    """

    _system: Type[RodBase]
    _position_indices: np.ndarray
    _director_indices: np.ndarray

    def __init__(self, *args, **kwargs):
        """Initialize boundary condition"""
        pass

    @property
    def system(self) -> Type[RodBase]:
        return self._system

    @property
    def position_indices(self) -> Optional[np.ndarray]:
        # TODO: This should be immutable somehow
        return self._position_indices

    @property
    def director_indices(self) -> Optional[np.ndarray]:
        # TODO: This should be immutable somehow
        return self._director_indices

    @abstractmethod
    def constrain_values(self, rod: Type[RodBase], time: float) -> None:
        # TODO: In the future, we can remove rod and use self.system
        """
        Constrain values (position and/or directors) of a rod object.

        Parameters
        ----------
        rod : object
            Rod-like object.
        time : float
            The time of simulation.
        """
        pass

    @abstractmethod
    def constrain_rates(self, rod: Type[RodBase], time: float) -> None:
        # TODO: In the future, we can remove rod and use self.system
        """
        Constrain rates (velocity and/or omega) of a rod object.

        Parameters
        ----------
        rod : object
            Rod-like object.
        time : float
            The time of simulation.

        """
        pass


class FreeBC(ConstraintBase):
    def constrain_values(self, rod: Type[RodBase], time: float) -> None:
        """In FreeBC, this routine simply passes."""
        pass

    def constrain_rates(self, rod: Type[RodBase], time: float) -> None:
        """In FreeBC, this routine simply passes."""
        pass


class FreeRod(FreeBC):
    # Please clear this part beyond version 0.3.0
    """Deprecated 0.2.1: Same implementation as FreeBC"""
    warnings.warn(
        "FreeRod is deprecated and renamed to FreeBC. The deprecated name will be removed in the future.",
        DeprecationWarning,
    )


class OneEndFixedBC(ConstraintBase):
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
        super().__init__()
        self.fixed_position_collection = np.array(fixed_position)
        self.fixed_directors_collection = np.array(fixed_directors)

    def constrain_values(self, rod, time):
        # rod.position_collection[..., 0] = self.fixed_position
        # rod.director_collection[..., 0] = self.fixed_directors
        self.compute_constrain_values(
            rod.position_collection,
            self.fixed_position_collection,
            rod.director_collection,
            self.fixed_directors_collection,
        )

    def constrain_rates(self, rod, time):
        # rod.velocity_collection[..., 0] = 0.0
        # rod.omega_collection[..., 0] = 0.0
        self.compute_constrain_rates(
            rod.velocity_collection,
            rod.omega_collection,
        )

    @staticmethod
    @njit(cache=True)
    def compute_constrain_values(
        position_collection,
        fixed_position_collection,
        director_collection,
        fixed_directors_collection,
    ):
        """
        Computes constrain values in numba njit decorator
        Parameters
        ----------
        position_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        fixed_position : numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
        director_collection : numpy.ndarray
            3D (dim, dim, blocksize) array containing data with `float` type.
        fixed_directors : numpy.ndarray
            3D (dim, dim, 1) array containing data with 'float' type.

        Returns
        -------

        """
        position_collection[..., 0] = fixed_position_collection
        director_collection[..., 0] = fixed_directors_collection

    @staticmethod
    @njit(cache=True)
    def compute_constrain_rates(velocity_collection, omega_collection):
        """
        Compute contrain rates in numba njit decorator
        Parameters
        ----------
        velocity_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        omega_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.

        Returns
        -------

        """
        velocity_collection[..., 0] = 0.0
        omega_collection[..., 0] = 0.0


class OneEndFixedRod(OneEndFixedBC):
    # Please clear this part beyond version 0.3.0
    """Deprecated 0.2.1: Same implementation as OneEndFixedBC"""
    warnings.warn(
        "OneEndFixedRod is deprecated and renamed to OneEndFixedBC. The deprecated name will be removed in the future.",
        DeprecationWarning,
    )


class FixedConstraint(ConstraintBase):
    """
    This boundary condition class fixes the specified node or orientations. 
    Index can be passed as "constrained_position_idx" and "constrained_director_index".
    Constraining position is equivalent to setting 0 translational DOF.
    Constraining director is equivalent to setting 0 rotational DOF.

    Examples
    --------
    TODO
    """

    def __init__(self, *fixed_data):
        """

        Parameters
        ----------
        fixed_data : tuple
            Tuple of position and directors
        """
        super().__init__()
        pos, dir = [], []
        for data in fixed_data:
            if isinstance(data, np.ndarray) and data.shape == (3,):
                pos.append(data)
            elif isinstance(data, np.ndarray) and data.shape == (3,3,):
                dir.append(data)
            else:
                # TODO: This part is prone to error.
                break
        self.fixed_positions = np.array(pos)
        self.fixed_directors = np.array(dir)

    def constrain_values(self, rod, time):
        if self.position_indices.size:
            self.nb_constrain_translational_values(
                rod.position_collection,
                self.fixed_positions,
                self.position_indices,
            )
        if self.director_indices.size:
            self.nb_constraint_rotational_values(
                rod.director_collection,
                self.fixed_directors,
                self.director_indices,
            )

    def constrain_rates(self, rod, time):
        if self.position_indices.size:
            self.nb_constrain_translational_rates(
                rod.velocity_collection,
                self.position_indices,
            )
        if self.director_indices.size:
            self.nb_constrain_rotational_rates(
                rod.omega_collection,
                self.director_indices,
            )

    @staticmethod
    @njit(cache=True)
    def nb_constraint_rotational_values(
        director_collection, fixed_director_collection, indices
    ) -> None:
        """
        Computes constrain values in numba njit decorator
        Parameters
        ----------
        director_collection : numpy.ndarray
            3D (dim, dim, blocksize) array containing data with `float` type.
        fixed_director_collection : numpy.ndarray
            3D (dim, dim, blocksize) array containing data with `float` type.
        indices : numpy.ndarray
            1D array containing the index of constraining nodes

        """
        block_size = indices.size
        for i in range(block_size):
            k = indices[i]
            director_collection[..., k] = fixed_director_collection[i, ...]

    @staticmethod
    @njit(cache=True)
    def nb_constrain_translational_values(
        position_collection, fixed_position_collection, indices
    ) -> None:
        """
        Computes constrain values in numba njit decorator
        Parameters
        ----------
        position_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        fixed_position_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        indices : numpy.ndarray
            1D array containing the index of constraining nodes

        """
        block_size = indices.size
        for i in range(block_size):
            k = indices[i]
            position_collection[0, k] = fixed_position_collection[i, 0]
            position_collection[1, k] = fixed_position_collection[i, 1]
            position_collection[2, k] = fixed_position_collection[i, 2]

    @staticmethod
    @njit(cache=True)
    def nb_constrain_translational_rates(velocity_collection, indices) -> None:
        """
        Compute constrain rates in numba njit decorator
        Parameters
        ----------
        velocity_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        indices : numpy.ndarray
            1D array containing the index of constraining nodes
        """

        block_size = indices.size
        for i in range(block_size):
            k = indices[i]
            velocity_collection[0, k] = 0.0
            velocity_collection[1, k] = 0.0
            velocity_collection[2, k] = 0.0

    @staticmethod
    @njit(cache=True)
    def nb_constrain_rotational_rates(omega_collection, indices) -> None:
        """
        Compute constrain rates in numba njit decorator
        Parameters
        ----------
        omega_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        indices : numpy.ndarray
            1D array containing the index of constraining nodes
        """

        block_size = indices.size
        for i in range(block_size):
            k = indices[i]
            omega_collection[0, k] = 0.0
            omega_collection[1, k] = 0.0
            omega_collection[2, k] = 0.0


class HelicalBucklingBC(ConstraintBase):
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
        super().__init__()
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
