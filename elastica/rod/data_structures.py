__doc__ = "Data structure wrapper for rod components"

from typing import TYPE_CHECKING, Optional
from typing_extensions import Self
import numpy as np
from numpy.typing import NDArray
from numba import njit
from elastica._rotations import _get_rotation_matrix, _rotate
from elastica._linalg import _batch_matmul

if TYPE_CHECKING:
    from elastica.systems.protocol import SymplecticSystemProtocol
else:
    SymplecticSystemProtocol = "SymplecticSystemProtocol"

class _RodSymplecticStepperMixin:
    def __init__(self: SymplecticSystemProtocol) -> None:
        self.kinematic_states = _KinematicState(
            self.position_collection, self.director_collection
        )
        self.dynamic_states = _DynamicState(
            self.v_w_collection,
            self.dvdt_dwdt_collection,
            self.velocity_collection,
            self.omega_collection,
        )

        # Expose rate returning functions in the interface
        # to be used by the time-stepping algorithm
        # dynamic rates needs to call update_accelerations and henc
        # is another function
        self.kinematic_rates = self.dynamic_states.kinematic_rates

    def dynamic_rates(
        self: SymplecticSystemProtocol,
        time: np.float64,
        prefac: np.float64,
    ) -> NDArray[np.float64]:
        self.update_accelerations(time)
        return self.dynamic_states.dynamic_rates(time, prefac)

"""
Symplectic stepper interface
"""

class _KinematicState:
    """State storing (x,Q) for symplectic steppers.
    Wraps data as state, with overloaded methods for symplectic steppers.
    Allows for separating implementation of stepper from actual
    addition/multiplication/other formulae used.

    Symplectic steppers rely only on in-place modifications to state and so
    only these methods are provided.
    """

    def __init__(
        self,
        position_collection_view: NDArray[np.float64],
        director_collection_view: NDArray[np.float64],
    ) -> None:
        """
        Parameters
        ----------
        position_collection_view : view of positions (or) x
        director_collection_view : view of directors (or) Q
        """
        # super(_KinematicState, self).__init__()

        self.position_collection = position_collection_view
        self.director_collection = director_collection_view


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

    return


class _DynamicState:
    """State storing (v,ω, dv/dt, dω/dt) for symplectic steppers.

    Wraps data as state, with overloaded methods for symplectic steppers.
    Allows for separating implementation of stepper from actual
    addition/multiplication/other formulae used.
    Symplectic steppers rely only on in-place modifications to state and so
    only these methods are provided.
    """

    def __init__(
        self,
        v_w_collection: NDArray[np.float64],
        dvdt_dwdt_collection: NDArray[np.float64],
        velocity_collection: NDArray[np.float64],
        omega_collection: NDArray[np.float64],
    ) -> None:
        """
        Parameters
        ----------
        n_elems : int, number of rod elements
        rate_collection_view : np.ndarray containing (v, ω, dv/dt, dω/dt)
        v_w_collection : numpy.ndarray

        """
        super(_DynamicState, self).__init__()
        # Limit at which (v, w) end
        # Create views for dynamic state
        self.rate_collection = v_w_collection
        self.dvdt_dwdt_collection = dvdt_dwdt_collection
        self.velocity_collection = velocity_collection
        self.omega_collection = omega_collection

    def kinematic_rates(
        self, time: np.float64, prefac: np.float64
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Yields kinematic rates to interact with _KinematicState

        Returns
        -------
        v_and_omega : np.ndarray consisting of (v,ω)
        Caveats
        -------
        Doesn't return a _KinematicState with (dt*v, dt*w) as members,
        as one expects the _Kinematic __add__ operator to interact
        with another _KinematicState. This is done for efficiency purposes.
        """
        # RHS functino call, gives v,w so that
        # Comes from kin_state -> (x,Q) += dt * (v,w) <- First part of dyn_state
        return self.velocity_collection, self.omega_collection

    def dynamic_rates(
        self, time: np.float64, prefac: np.float64
    ) -> NDArray[np.float64]:
        """Yields dynamic rates to add to with _DynamicState
        Returns
        -------
        acc_and_alpha : np.ndarray consisting of (dv/dt,dω/dt)
        Caveats
        -------
        Doesn't return a _DynamicState with (dt*v, dt*w) as members,
        as one expects the _Dynamic __add__ operator to interact
        with another _DynamicState. This is done for efficiency purposes.
        """
        return prefac * self.dvdt_dwdt_collection


@njit(cache=True)  # type: ignore
def overload_operator_dynamic_numba(
    rate_collection: NDArray[np.float64],
    scaled_second_deriv_array: NDArray[np.float64],
) -> None:
    """overloaded += operator, updating dynamic_rates
    Parameters
    ----------
    scaled_second_deriv_array : np.ndarray containing dt * (dvdt, dωdt),
    as retured from _DynamicState's `dynamic_rates` method
    Returns
    -------
    self : _DynamicState instance with inplace modified data
    Caveats
    -------
    Takes a np.ndarray and not a _DynamicState object (as one expects).
    This is done for efficiency reasons, see `dynamic_rates`.
    """
    # Always goes in LHS : that means the update is on the rates alone
    # (v,ω) += dt * (dv/dt, dω/dt) ->  self.dynamic_rates
    # rate_collection[..., : n_kinematic_rates] += scaled_second_deriv_array
    blocksize = scaled_second_deriv_array.shape[1]

    for i in range(2):
        for k in range(blocksize):
            rate_collection[i, k] += scaled_second_deriv_array[i, k]

