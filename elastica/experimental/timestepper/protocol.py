from typing import Protocol

from elastica.typing import StepType
from elastica.systems.protocol import SystemProtocol
from elastica.timestepper.protocol import StepperProtocol
from elastica.rod.data_structures import _State as StateType

import numpy as np
from numpy.typing import NDArray


class ExplicitSystemProtocol(SystemProtocol, Protocol):
    """
    Protocol defining the required interface for explicit time integration.

    TODO: Temporarily made to handle explicit stepper.
    Need to be refactored as the explicit stepper is further developed.
    """

    # Geometry
    n_nodes: int
    n_elems: int

    # State arrays
    position_collection: NDArray[np.float64]
    velocity_collection: NDArray[np.float64]
    acceleration_collection: NDArray[np.float64]
    omega_collection: NDArray[np.float64]
    alpha_collection: NDArray[np.float64]
    director_collection: NDArray[np.float64]

    # Forces/torques
    external_forces: NDArray[np.float64]
    external_torques: NDArray[np.float64]
    internal_forces: NDArray[np.float64]
    internal_torques: NDArray[np.float64]

    def __call__(self, time: np.float64, dt: np.float64) -> np.float64: ...

    @property
    def state(self) -> StateType: ...

    @state.setter
    def state(self, state: StateType) -> None: ...


class MemoryProtocol(Protocol):
    @property
    def initial_state(self) -> bool: ...


class ExplicitStepperProtocol(StepperProtocol, Protocol):
    """symplectic stepper protocol."""

    def get_stages(self) -> list[StepType]: ...

    def get_updates(self) -> list[StepType]: ...


# class _LinearExponentialIntegratorMixin:
#     """
#     Linear Exponential integrator mixin wrapper.
#     """
#
#     def __init__(self):
#         pass
#
#     def _do_stage(self, System, Memory, time, dt):
#         # TODO : Make more general, system should not be calculating what the state
#         # transition matrix directly is, but rather it should just give
#         Memory.linear_operator = System.get_linear_state_transition_operator(time, dt)
#
#     def _do_update(self, System, Memory, time, dt):
#         # FIXME What's the right formula when doing update?
#         # System.linearly_evolving_state = _batch_matmul(
#         #     System.linearly_evolving_state,
#         #     Memory.linear_operator
#         # )
#         System.linearly_evolving_state = np.einsum(
#             "ijk,ljk->ilk", System.linearly_evolving_state, Memory.linear_operator
#         )
#         return time + dt
#
#     def _first_prefactor(self, dt):
#         """Prefactor call to satisfy interface of SymplecticStepper. Should never
#         be used in actual code.
#
#         Parameters
#         ----------
#         dt : the time step of simulation
#
#         Raises
#         ------
#         RuntimeError
#         """
#         raise RuntimeError(
#             "Symplectic prefactor of LinearExponentialIntegrator should not be called!"
#         )
#
#     # Code repeat!
#     # Easy to avoid, but keep for performance.
#     def _do_one_step(self, System, time, prefac):
#         System.linearly_evolving_state = np.einsum(
#             "ijk,ljk->ilk",
#             System.linearly_evolving_state,
#             System.get_linear_state_transition_operator(time, prefac),
#         )
#         return (
#             time  # TODO fix hack that treats time separately here. Shuold be time + dt
#         )
#         # return time + dt
