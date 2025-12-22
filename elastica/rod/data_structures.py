__doc__ = "Data structure wrapper for rod components"

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
    Mixin class providing necessary methods for integration of
    the kinematic and dynamic equations of the rod.
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
    dvdt_dwdt_collection: NDArray[np.float64]  # Second-derivative collection

    def update_kinematics(
        self,
        time: np.float64,
        prefac: np.float64,
    ) -> None:
        overload_operator_kinematic_numba(
            self.n_nodes,
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
        overload_operator_dynamic_numba(
            prefac,
            self.rate_collection,
            self.dvdt_dwdt_collection,
        )


"""
Symplectic stepper operation
"""


@njit(cache=True)  # type: ignore
def overload_operator_kinematic_numba(
    n_nodes: int,
    prefac: np.float64,
    position_collection: NDArray[np.float64],
    director_collection: NDArray[np.float64],
    velocity_collection: NDArray[np.float64],
    omega_collection: NDArray[np.float64],
) -> None:
    """overloaded += operator

    The add for directors is customized to reflect Rodrigues' rotation
    formula.
    Parameters
    ----------
    scaled_deriv_array : np.ndarray containing dt * (v, ω),
    as retured from _DynamicState's `kinematic_rates` method
    Returns
    -------
    self : _KinematicState instance with inplace modified data
    Caveats
    -------
    Takes a np.ndarray and not a _KinematicState object (as one expects).
    This is done for efficiency reasons, see _DynamicState's `kinematic_rates`
    method
    """
    # x += v*dt
    for i in range(3):
        for k in range(n_nodes):
            position_collection[i, k] += prefac * velocity_collection[i, k]
    rotation_matrix = _get_rotation_matrix(1.0, prefac * omega_collection)
    director_collection[:] = _batch_matmul(rotation_matrix, director_collection)


@njit(cache=True)  # type: ignore
def overload_operator_dynamic_numba(
    prefac: np.float64,
    rate_collection: NDArray[np.float64],
    second_deriv_array: NDArray[np.float64],
) -> None:
    """overloaded += operator, updating dynamic_rates
    Parameters
    ----------
    second_deriv_array : np.ndarray containing (dvdt, dωdt),
    as retured from _DynamicState's `dynamic_rates` method
    Returns
    -------
    self : _DynamicState instance with inplace modified data
    Caveats
    -------
    Takes a np.ndarray and not a _DynamicState object (as one expects).
    """
    # Always goes in LHS : that means the update is on the rates alone
    # (v,ω) += dt * (dv/dt, dω/dt)
    # rate_collection[..., : n_kinematic_rates] += second_deriv_aray
    blocksize = second_deriv_array.shape[1]

    for i in range(2):
        for k in range(blocksize):
            rate_collection[i, k] += prefac * second_deriv_array[i, k]
