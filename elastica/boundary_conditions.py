__doc__ = """ Built-in boundary condition implementationss """

from typing import Any, Optional, TypeVar, Generic

import numpy as np
from numpy.typing import NDArray

from abc import ABC, abstractmethod

from numba import njit

from elastica._linalg import _batch_matvec, _batch_matrix_transpose
from elastica._rotations import _get_rotation_matrix
from elastica.typing import SystemType, RodType, RigidBodyType, ConstrainingIndex


S = TypeVar("S")


class ConstraintBase(ABC, Generic[S]):
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

    _system: S
    _constrained_position_idx: NDArray[np.int32]
    _constrained_director_idx: NDArray[np.int32]

    def __init__(
        self,
        *args: Any,
        constrained_position_idx: ConstrainingIndex = (),
        constrained_director_idx: ConstrainingIndex = (),
        **kwargs: Any,
    ) -> None:
        """Initialize boundary condition"""
        try:
            self._system = kwargs["_system"]
            self._constrained_position_idx = np.array(
                constrained_position_idx, dtype=np.int32
            )
            self._constrained_director_idx = np.array(
                constrained_director_idx, dtype=np.int32
            )
        except KeyError:
            raise KeyError(
                "Please use simulator.constrain(...).using(...) syntax to establish constraint."
            )

    @property
    def system(self) -> S:
        """get system (rod or rigid body) reference"""
        return self._system

    @property
    def constrained_position_idx(self) -> NDArray[np.int32]:
        """get position-indices passed to "using" """
        return self._constrained_position_idx

    @property
    def constrained_director_idx(self) -> NDArray[np.int32]:
        """get director-indices passed to "using" """
        return self._constrained_director_idx

    @abstractmethod
    def constrain_values(self, system: S, time: np.float64) -> None:
        """
        Constrain values (position and/or directors) of a rod object.

        Parameters
        ----------
        system : SystemType
            Rod or rigid-body object.
        time : float
            The time of simulation.
        """
        pass

    @abstractmethod
    def constrain_rates(self, system: S, time: np.float64) -> None:
        """
        Constrain rates (velocity and/or omega) of a rod object.

        Parameters
        ----------
        system : SystemType
            Rod or rigid-body object.
        time : float
            The time of simulation.

        """
        pass


class FreeBC(ConstraintBase):
    """
    Boundary condition template.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def constrain_values(
        self, system: "RodType | RigidBodyType", time: np.float64
    ) -> None:
        """In FreeBC, this routine simply passes."""
        pass

    def constrain_rates(
        self, system: "RodType | RigidBodyType", time: np.float64
    ) -> None:
        """In FreeBC, this routine simply passes."""
        pass


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

    def __init__(
        self,
        fixed_position: tuple[int, ...],
        fixed_directors: tuple[int, ...],
        **kwargs: Any,
    ) -> None:
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
        self, system: "RodType | RigidBodyType", time: np.float64
    ) -> None:
        # system.position_collection[..., 0] = self.fixed_position
        # system.director_collection[..., 0] = self.fixed_directors
        self.compute_constrain_values(
            system.position_collection,
            self.fixed_position_collection,
            system.director_collection,
            self.fixed_directors_collection,
        )

    def constrain_rates(
        self, system: "RodType | RigidBodyType", time: np.float64
    ) -> None:
        # system.velocity_collection[..., 0] = 0.0
        # system.omega_collection[..., 0] = 0.0
        self.compute_constrain_rates(
            system.velocity_collection,
            system.omega_collection,
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def compute_constrain_values(
        position_collection: NDArray[np.float64],
        fixed_position_collection: NDArray[np.float64],
        director_collection: NDArray[np.float64],
        fixed_directors_collection: NDArray[np.float64],
    ) -> None:
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
    @njit(cache=True)  # type: ignore
    def compute_constrain_rates(
        velocity_collection: NDArray[np.float64],
        omega_collection: NDArray[np.float64],
    ) -> None:
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


class GeneralConstraint(ConstraintBase):
    """
    This boundary condition class allows the specified node/link to have a configurable constraint.
    Index can be passed to fix either or both the position or the director.
    Constraining position is equivalent to setting 0 translational DOF.
    Constraining director is equivalent to setting 0 rotational DOF.

    Examples
    --------
    How to fix all translational and rotational dof except allowing twisting around the z-axis in an inertial frame:

    >>> simulator.constrain(system).using(
    ...    GeneralConstraint,
    ...    constrained_position_idx=(0,),
    ...    constrained_director_idx=(0,),
    ...    translational_constraint_selector=np.array([True, True, True]),
    ...    rotational_constraint_selector=np.array([True, True, False]),
    ... )

    How to allow the end of the rod to move in the XY plane and allow all rotational dof:

    >>> simulator.constrain(rod).using(
    ...    GeneralConstraint,
    ...    constrained_position_idx=(-1,),
    ...    translational_constraint_selector=np.array([True, True, False]),
    ... )
    """

    def __init__(
        self,
        *fixed_data: Any,
        translational_constraint_selector: Optional[NDArray[np.bool_]] = None,
        rotational_constraint_selector: Optional[NDArray[np.bool_]] = None,
        **kwargs: Any,
    ) -> None:
        """

        Initialization of the constraint. Any parameter passed to 'using' will be available in kwargs.

        Parameters
        ----------
        constrained_position_idx : tuple
            Tuple of position-indices that will be constrained
        constrained_director_idx : tuple
            Tuple of director-indices that will be constrained
        translational_constraint_selector: Optional[np.ndarray]
            np.array of type bool indicating which translational degrees of freedom (dof) to constrain.
            If entry is True, the corresponding dof will be constrained. If None, we constrain all dofs.
        rotational_constraint_selector: Optional[np.ndarray]
            np.array of type bool indicating which translational degrees of freedom (dof) to constrain.
            If entry is True, the corresponding dof will be constrained.
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

        if len(pos) > 0:
            # transpose from (blocksize, dim) to (dim, blocksize)
            self.fixed_positions = np.array(pos).transpose((1, 0))

        if len(dir) > 0:
            # transpose from (blocksize, dim, dim) to (dim, dim, blocksize)
            self.fixed_directors = np.array(dir).transpose((1, 2, 0))

        if translational_constraint_selector is None:
            translational_constraint_selector = np.array([True, True, True])
        if rotational_constraint_selector is None:
            rotational_constraint_selector = np.array([True, True, True])
        # properly validate the user-provided constraint selectors
        assert (
            isinstance(translational_constraint_selector, np.ndarray)
            and translational_constraint_selector.dtype == bool
            and translational_constraint_selector.shape == (3,)
        ), "Translational constraint selector must be a 1D boolean array of length 3."
        assert (
            isinstance(rotational_constraint_selector, np.ndarray)
            and rotational_constraint_selector.dtype == bool
            and rotational_constraint_selector.shape == (3,)
        ), "Rotational constraint selector must be a 1D boolean array of length 3."
        # cast booleans to int
        self.translational_constraint_selector = (
            translational_constraint_selector.astype(int)
        )
        self.rotational_constraint_selector = rotational_constraint_selector.astype(int)

    def constrain_values(
        self, system: "RodType | RigidBodyType", time: np.float64
    ) -> None:
        if self.constrained_position_idx.size:
            self.nb_constrain_translational_values(
                system.position_collection,
                self.fixed_positions,
                self.constrained_position_idx,
                self.translational_constraint_selector,
            )

    def constrain_rates(
        self, system: "RodType | RigidBodyType", time: np.float64
    ) -> None:
        if self.constrained_position_idx.size:
            self.nb_constrain_translational_rates(
                system.velocity_collection,
                self.constrained_position_idx,
                self.translational_constraint_selector,
            )
        if self.constrained_director_idx.size:
            self.nb_constrain_rotational_rates(
                system.director_collection,
                system.omega_collection,
                self.constrained_director_idx,
                self.rotational_constraint_selector,
            )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def nb_constrain_translational_values(
        position_collection: NDArray[np.float64],
        fixed_position_collection: NDArray[np.float64],
        indices: NDArray[np.int32],
        constraint_selector: NDArray[np.int32],
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
        constraint_selector: numpy.ndarray
            1D array of type int and size (3,) indicating which translational Degrees of Freedom (DoF) to constrain.
            Entries are integers in {0, 1} (e.g. a binary values of either 0 or 1).
            If entry is 1, the concerning DoF will be constrained, otherwise it will be free for translation.
            Selector shall be specified in the inertial frame
        """
        block_size = indices.size
        for i in range(block_size):
            k = indices[i]
            # First term: add the old position values using the inverse constraint selector (e.g. DoF)
            # Second term: add the fixed position values using the constraint selector (e.g. constraint dimensions)
            position_collection[..., k] = (
                1 - constraint_selector
            ) * position_collection[
                ..., k
            ] + constraint_selector * fixed_position_collection[
                ..., i
            ]

    @staticmethod
    @njit(cache=True)  # type: ignore
    def nb_constrain_translational_rates(
        velocity_collection: NDArray[np.float64],
        indices: NDArray[np.int32],
        constraint_selector: NDArray[np.int32],
    ) -> None:
        """
        Compute constrain rates in numba njit decorator

        Parameters
        ----------
        velocity_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        indices : numpy.ndarray
            1D array containing the index of constraining nodes
        constraint_selector: numpy.ndarray
            1D array of type int and size (3,) indicating which translational Degrees of Freedom (DoF) to constrain.
            Entries are integers in {0, 1} (e.g. a binary values of either 0 or 1).
            If entry is 1, the concerning DoF will be constrained, otherwise it will be free for translation.
            Selector shall be specified in the inertial frame
        """

        block_size = indices.size
        for i in range(block_size):
            k = indices[i]
            # set the dofs to 0 where the constraint_selector mask is active
            velocity_collection[..., k] = (
                1 - constraint_selector
            ) * velocity_collection[..., k]

    @staticmethod
    @njit(cache=True)  # type: ignore
    def nb_constrain_rotational_rates(
        director_collection: NDArray[np.float64],
        omega_collection: NDArray[np.float64],
        indices: NDArray[np.int32],
        constraint_selector: NDArray[np.int32],
    ) -> None:
        """
        Compute constrain rates in numba njit decorator

        Parameters
        ----------
        director_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        omega_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        indices : numpy.ndarray
            1D array containing the index of constraining nodes
        constraint_selector: numpy.ndarray
            1D array of type int and size (3,) indicating which rotational Degrees of Freedom (DoF) to constrain.
            Entries are integers in {0, 1} (e.g. a binary values of either 0 or 1).
            If an entry is 1, the rotation around the respective axis will be constrained,
            otherwise the system can freely rotate around the axis.
            The selector shall be specified in the lab frame
        """
        directors = director_collection[..., indices]

        # rotate angular velocities to lab frame
        omega_collection_lab_frame = _batch_matvec(
            _batch_matrix_transpose(directors), omega_collection[..., indices]
        )

        # apply constraint selector to angular velocities in lab frame
        omega_collection_not_constrained = (
            1 - np.expand_dims(constraint_selector, 1)
        ) * omega_collection_lab_frame

        # rotate angular velocities vector back to local frame and apply to omega_collection
        omega_collection[..., indices] = _batch_matvec(
            directors, omega_collection_not_constrained
        )


class FixedConstraint(GeneralConstraint):
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
    ...    constrained_position_idx=(0,-1),
    ...    constrained_director_idx=(0,-1)
    ... )

    How to pin the middle of the rod (10th node), without constraining the rotational DOF.

    >>> simulator.constrain(rod).using(
    ...    FixedConstraint,
    ...    constrained_position_idx=(10,)
    ... )

    See Also
    --------
    GeneralConstraint: Generalized constraint with configurable DOF.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """

        Initialization of the constraint. Any parameter passed to 'using' will be available in kwargs.

        Parameters
        ----------
        constrained_position_idx : tuple
            Tuple of position-indices that will be constrained
        constrained_director_idx : tuple
            Tuple of director-indices that will be constrained
        """
        super().__init__(
            *args,
            translational_constraint_selector=np.array([True, True, True]),
            rotational_constraint_selector=np.array([True, True, True]),
            **kwargs,
        )

    def constrain_values(
        self, system: "RodType | RigidBodyType", time: np.float64
    ) -> None:
        if self.constrained_position_idx.size:
            self.nb_constrain_translational_values(
                system.position_collection,
                self.fixed_positions,
                self.constrained_position_idx,
            )
        if self.constrained_director_idx.size:
            self.nb_constraint_rotational_values(
                system.director_collection,
                self.fixed_directors,
                self.constrained_director_idx,
            )

    def constrain_rates(
        self, system: "RodType | RigidBodyType", time: np.float64
    ) -> None:
        if self.constrained_position_idx.size:
            self.nb_constrain_translational_rates(
                system.velocity_collection,
                self.constrained_position_idx,
            )
        if self.constrained_director_idx.size:
            self.nb_constrain_rotational_rates(
                system.omega_collection,
                self.constrained_director_idx,
            )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def nb_constraint_rotational_values(
        director_collection: NDArray[np.float64],
        fixed_director_collection: NDArray[np.float64],
        indices: NDArray[np.int32],
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
            director_collection[..., k] = fixed_director_collection[..., i]

    @staticmethod
    @njit(cache=True)  # type: ignore
    def nb_constrain_translational_values(
        position_collection: NDArray[np.float64],
        fixed_position_collection: NDArray[np.float64],
        indices: NDArray[np.int32],
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
            position_collection[..., k] = fixed_position_collection[..., i]

    @staticmethod
    @njit(cache=True)  # type: ignore
    def nb_constrain_translational_rates(
        velocity_collection: NDArray[np.float64], indices: NDArray[np.int32]
    ) -> None:
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
    @njit(cache=True)  # type: ignore
    def nb_constrain_rotational_rates(
        omega_collection: NDArray[np.float64], indices: NDArray[np.int32]
    ) -> None:
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
        position_start: NDArray[np.float64],
        position_end: NDArray[np.float64],
        director_start: NDArray[np.float64],
        director_end: NDArray[np.float64],
        twisting_time: float,
        slack: float,
        number_of_rotations: float,
        **kwargs: Any,
    ) -> None:
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
        self.twisting_time = np.float64(twisting_time)

        angel_vel_scalar = np.float64(
            (2.0 * number_of_rotations * np.pi / self.twisting_time) / 2.0
        )
        shrink_vel_scalar = np.float64(slack / (self.twisting_time * 2.0))

        direction = (position_end - position_start) / np.linalg.norm(
            position_end - position_start
        )

        self.final_start_position = position_start + slack / 2.0 * direction
        self.final_end_position = position_end - slack / 2.0 * direction

        self.ang_vel = angel_vel_scalar * direction
        self.shrink_vel = shrink_vel_scalar * direction

        theta = np.float64(number_of_rotations * np.pi)

        self.final_start_directors = (
            _get_rotation_matrix(theta, direction.reshape(3, 1)).reshape(3, 3)
            @ director_start
        )  # rotation_matrix wants vectors 3,1
        self.final_end_directors = (
            _get_rotation_matrix(-theta, direction.reshape(3, 1)).reshape(3, 3)
            @ director_end
        )  # rotation_matrix wants vectors 3,1

    def constrain_values(
        self, system: "RodType | RigidBodyType", time: np.float64
    ) -> None:
        if time > self.twisting_time:
            system.position_collection[..., 0] = self.final_start_position
            system.position_collection[..., -1] = self.final_end_position

            system.director_collection[..., 0] = self.final_start_directors
            system.director_collection[..., -1] = self.final_end_directors

    def constrain_rates(
        self, system: "RodType | RigidBodyType", time: np.float64
    ) -> None:
        if time > self.twisting_time:
            system.velocity_collection[..., 0] = 0.0
            system.omega_collection[..., 0] = 0.0

            system.velocity_collection[..., -1] = 0.0
            system.omega_collection[..., -1] = 0.0

        else:
            system.velocity_collection[..., 0] = self.shrink_vel
            system.omega_collection[..., 0] = self.ang_vel

            system.velocity_collection[..., -1] = -self.shrink_vel
            system.omega_collection[..., -1] = -self.ang_vel
