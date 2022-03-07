__doc__ = """ Built-in boundary condition implementationss """
__all__ = [
    "ConstraintBase",
    "FreeBC",
    "FreeRod",  # Deprecated: remove v0.3.0
    "OneEndFixedBC",
    "OneEndFixedRod",  # Deprecated: remove v0.3.0
    "FixedConstraint",
    "HelicalBucklingBC",
]

import warnings
from typing import Optional, Type, Union

import numpy as np

from abc import ABC, abstractmethod

import numba
from numba import njit

from elastica._rotations import _get_rotation_matrix
from elastica.rod import RodBase
from elastica.rigidbody import RigidBodyBase


class ConstraintBase(ABC):
    """Base class for constraint and displacement boundary condition implementation.

    Notes
    -----
    Constraint class must inherit BaseConstraint class.


        Attributes
        ----------
        system : RodBase or RigidBodyBase
        node_indices : None or numpy.ndarray
        element_indices : None or numpy.ndarray

    """

    _system: Union[Type[RodBase], Type[RigidBodyBase]]
    _constrained_position_idx: np.ndarray
    _constrained_director_idx: np.ndarray

    def __init__(self, *args, **kwargs):
        """Initialize boundary condition"""
        try:
            self._system = kwargs["_system"]
            self._constrained_position_idx = np.array(
                kwargs.get("constrained_position_idx", []), dtype=int
            )
            self._constrained_director_idx = np.array(
                kwargs.get("constrained_director_idx", []), dtype=int
            )
        except KeyError:
            raise KeyError(
                "Please use simulator.constrain(...).using(...) syntax to establish constraint."
            )

    @property
    def system(self) -> Union[Type[RodBase], Type[RigidBodyBase]]:
        """get system (rod or rigid body) reference"""
        return self._system

    @property
    def constrained_position_idx(self) -> Optional[np.ndarray]:
        """get position-indices passed to "using" """
        # TODO: This should be immutable somehow
        return self._constrained_position_idx

    @property
    def constrained_director_idx(self) -> Optional[np.ndarray]:
        """get director-indices passed to "using" """
        # TODO: This should be immutable somehow
        return self._constrained_director_idx

    @abstractmethod
    def constrain_values(
        self, rod: Union[Type[RodBase], Type[RigidBodyBase]], time: float
    ) -> None:
        # TODO: In the future, we can remove rod and use self.system
        """
        Constrain values (position and/or directors) of a rod object.

        Parameters
        ----------
        rod : Union[Type[RodBase], Type[RigidBodyBase]]
            Rod or rigid-body object.
        time : float
            The time of simulation.
        """
        pass

    @abstractmethod
    def constrain_rates(
        self, rod: Union[Type[RodBase], Type[RigidBodyBase]], time: float
    ) -> None:
        # TODO: In the future, we can remove rod and use self.system
        """
        Constrain rates (velocity and/or omega) of a rod object.

        Parameters
        ----------
        rod : Union[Type[RodBase], Type[RigidBodyBase]]
            Rod or rigid-body object.
        time : float
            The time of simulation.

        """
        pass


class FreeBC(ConstraintBase):
    """
    Boundary condition template.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def constrain_values(
        self, rod: Union[Type[RodBase], Type[RigidBodyBase]], time: float
    ) -> None:
        """In FreeBC, this routine simply passes."""
        pass

    def constrain_rates(
        self, rod: Union[Type[RodBase], Type[RigidBodyBase]], time: float
    ) -> None:
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

    `Example case (timoshenko) <https://github.com/GazzolaLab/PyElastica/blob/master/examples/TimoshenkoBeamCase/timoshenko.py>`_

    Examples
    --------
    How to fix one ends of the rod:

    >>> simulator.constrain(rod).using(
    ...    OneEndFixedBC,
    ...    constrained_position_idx=(0,),
    ...    constrained_director_idx=(0,)
    ... )
    """

    def __init__(self, fixed_position, fixed_directors, **kwargs):
        """

        Initialization of the constraint. Any parameter passed to 'using' will be available in kwargs.

        Parameters
        ----------
        constrained_position_idx : tuple
            Tuple of position-indices that will be constrained
        constrained_director_idx : tuple
            Tuple of director-indices that will be constrained
        """
        super().__init__(**kwargs)
        self.fixed_position_collection = np.array(fixed_position)
        self.fixed_directors_collection = np.array(fixed_directors)

    def constrain_values(
        self, rod: Union[Type[RodBase], Type[RigidBodyBase]], time: float
    ) -> None:
        # rod.position_collection[..., 0] = self.fixed_position
        # rod.director_collection[..., 0] = self.fixed_directors
        self.compute_constrain_values(
            rod.position_collection,
            self.fixed_position_collection,
            rod.director_collection,
            self.fixed_directors_collection,
        )

    def constrain_rates(
        self, rod: Union[Type[RodBase], Type[RigidBodyBase]], time: float
    ) -> None:
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
    Index can be passed to fix either or both the position or the director.
    Constraining position is equivalent to setting 0 translational DOF.
    Constraining director is equivalent to setting 0 rotational DOF.

    Examples
    --------
    How to fix two ends of the rod:

    >>> simulator.constrain(rod).using(
    ...    FixedConstraint,
    ...    constrained_position_idx=(0,1,-2,-1),
    ...    constrained_director_idx=(0,-1)
    ... )

    How to pin the middle of the rod (10th node), without constraining the rotational DOF.

    >>> simulator.constrain(rod).using(
    ...    FixedConstraint,
    ...    constrained_position_idx=(10)
    ... )
    """

    def __init__(self, *fixed_data, **kwargs):
        """

        Initialization of the constraint. Any parameter passed to 'using' will be available in kwargs.

        Parameters
        ----------
        constrained_position_idx : tuple
            Tuple of position-indices that will be constrained
        constrained_director_idx : tuple
            Tuple of director-indices that will be constrained
        """
        super().__init__(**kwargs)
        pos, dir = [], []
        for data in fixed_data:
            if isinstance(data, np.ndarray) and data.shape == (3,):
                pos.append(data)
            elif isinstance(data, np.ndarray) and data.shape == (
                3,
                3,
            ):
                dir.append(data)
            else:
                # TODO: This part is prone to error.
                break
        self.fixed_positions = np.array(pos)
        self.fixed_directors = np.array(dir)

    def constrain_values(
        self, rod: Union[Type[RodBase], Type[RigidBodyBase]], time: float
    ) -> None:
        if self.constrained_position_idx.size:
            self.nb_constrain_translational_values(
                rod.position_collection,
                self.fixed_positions,
                self.constrained_position_idx,
            )
        if self.constrained_director_idx.size:
            self.nb_constraint_rotational_values(
                rod.director_collection,
                self.fixed_directors,
                self.constrained_director_idx,
            )

    def constrain_rates(
        self, rod: Union[Type[RodBase], Type[RigidBodyBase]], time: float
    ) -> None:
        if self.constrained_position_idx.size:
            self.nb_constrain_translational_rates(
                rod.velocity_collection,
                self.constrained_position_idx,
            )
        if self.constrained_director_idx.size:
            self.nb_constrain_rotational_rates(
                rod.omega_collection,
                self.constrained_director_idx,
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

    `Example case (helical buckling) <https://github.com/GazzolaLab/PyElastica/blob/master/examples/HelicalBucklingCase/helicalbuckling.py>`_

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
            3D (dim, dim, 1) array containing data with 'float' type.
            Directors of first element of rod after twist completed.
        final_end_directors: numpy.ndarray
            3D (dim, dim, 1) array containing data with 'float' type.
            Directors of last element of rod after twist completed.


    """

    def __init__(
        self,
        position_start: np.ndarray,
        position_end: np.ndarray,
        director_start: np.ndarray,
        director_end: np.ndarray,
        twisting_time: float,
        slack: float,
        number_of_rotations: float,
        **kwargs
    ):
        """

        Helical Buckling initializer

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
        super().__init__(**kwargs)
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

    def constrain_values(
        self, rod: Union[Type[RodBase], Type[RigidBodyBase]], time: float
    ) -> None:
        if time > self.twisting_time:
            rod.position_collection[..., 0] = self.final_start_position
            rod.position_collection[..., -1] = self.final_end_position

            rod.director_collection[..., 0] = self.final_start_directors
            rod.director_collection[..., -1] = self.final_end_directors

    def constrain_rates(
        self, rod: Union[Type[RodBase], Type[RigidBodyBase]], time: float
    ) -> None:
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
