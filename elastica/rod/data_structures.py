"""
Data structures and Numba-jitted operators for handling rod components
and their integration in a symplectic time-stepping scheme.

This module provides the `_RodSymplecticStepperMixin` for managing
kinematic and dynamic states of rods, and optimized functions for
their in-place updates.
"""

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from numba import njit
from elastica._rotations import _get_rotation_matrix
from elastica._linalg import _batch_matmul

if TYPE_CHECKING:
    from elastica.systems.protocol import SymplecticSystemProtocol
else:
    SymplecticSystemProtocol = "SymplecticSystemProtocol"


class _RodSymplecticStepperMixin:
    """
    Mixin class providing necessary methods for integration of the kinematic and
    dynamic equations of the rod.

    This mixin manages the rod's posture (position and directors), velocity
    (linear and angular), and acceleration states. It provides `update_kinematics`
    and `update_dynamics` methods to apply updates to these states, typically
    called by a symplectic time-stepper.
    """

    n_nodes: int

    # Posture state
    position_collection: NDArray[np.float64]
    director_collection: NDArray[np.float64]
    # Velocity state
    velocity_collection: NDArray[np.float64]
    omega_collection: NDArray[np.float64]
    v_w_collection: NDArray[np.float64]  # Rate collection
    # Acceleration state
    # acceleration_collection: NDArray[np.float64]
    # alpha_collection: NDArray[np.float64]
    dvdt_dwdt_collection: NDArray[np.float64]  # Second derivative collection

    def update_kinematics(
        self,
        time: np.float64,
        prefac: np.float64,
    ) -> None:
        """
        Update kinematic state.

        Typically called after velocity and omega (angular velocity) have been updated.

        Parameters
        ----------
        time : float
            Current time.
        prefac : float
            Integration prefactor.
        """
        overload_operator_kinematic_numba(
            prefac,
            self.position_collection,
            self.director_collection,
            self.velocity_collection,
            self.omega_collection,
        )

    def update_dynamics(
        self,
        time: np.float64,
        prefac: np.float64,
    ) -> None:
        """
        Update dynamic state.

        Typically called after acceleration and alpha (angular acceleration) have been updated.

        Parameters
        ----------
        time : float
            Current time.
        prefac : float
            Integration prefactor.
        """
        overload_operator_dynamic_numba(
            prefac,
            self.v_w_collection,
            self.dvdt_dwdt_collection,
        )


"""
Symplectic stepper operation
"""


@njit(cache=True)  # type: ignore
def overload_operator_kinematic_numba(
    prefac: np.float64,
    position_collection: NDArray[np.float64],
    director_collection: NDArray[np.float64],
    velocity_collection: NDArray[np.float64],
    omega_collection: NDArray[np.float64],
) -> None:
    """Performs in-place update of kinematic states (position and director) using Numba.

    This operator updates the position and director collections of a rod based on
    its velocity and angular velocity. The director update uses Rodrigues' rotation
    formula.

    Parameters
    ----------
    prefac : numpy.float64
        Pre-factor (e.g., time step `dt`) to scale the velocity and angular velocity.
    position_collection : numpy.ndarray
        Position of the rod nodes. Modified in-place.
    director_collection : numpy.ndarray
        Director (orientation) of the rod elements. Modified in-place.
    velocity_collection : numpy.ndarray
        Linear velocity of the rod nodes.
    omega_collection : numpy.ndarray
        Angular velocity of the rod elements.
    """
    # x += v*dt
    blocksize = position_collection.shape[1]
    for i in range(3):
        for k in range(blocksize):
            position_collection[i, k] += prefac * velocity_collection[i, k]
    rotation_matrix = _get_rotation_matrix(prefac, omega_collection)
    director_collection[:] = _batch_matmul(rotation_matrix, director_collection)


@njit(cache=True)  # type: ignore
def overload_operator_dynamic_numba(
    prefac: np.float64,
    rate_collection: NDArray[np.float64],
    second_deriv_array: NDArray[np.float64],
) -> None:
    """Performs in-place update of dynamic states (linear and angular velocities) using Numba.

    This operator updates the rate collection (which stores linear and angular velocities)
    of a rod based on the second derivative array (linear and angular accelerations).

    Parameters
    ----------
    prefac : numpy.float64
        Pre-factor (e.g., time step `dt`) to scale the second derivative terms.
    rate_collection : numpy.ndarray
        Collection of linear and angular velocities of the rod. Modified in-place.
    second_deriv_array : numpy.ndarray
        Collection of linear and angular accelerations (dv/dt, dω/dt) of the rod.
    """
    # Always goes in LHS : that means the update is on the rates alone
    # (v,ω) += dt * (dv/dt, dω/dt)
    # rate_collection[..., : n_kinematic_rates] += second_deriv_aray
    blocksize = second_deriv_array.shape[1]
    for i in range(2):
        for k in range(blocksize):
            rate_collection[i, k] += prefac * second_deriv_array[i, k]
