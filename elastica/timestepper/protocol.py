__doc__ = "Time stepper interface"

from typing import Protocol, Callable, Literal, ClassVar, Type

from elastica.typing import (
    SystemType,
    SteppersOperatorsType,
    OperatorType,
    SystemCollectionType,
)
from .tag import StepperTags

import numpy as np


class StepperProtocol(Protocol):
    """Protocol for all time-steppers"""

    Tag: StepperTags
    steps_and_prefactors: SteppersOperatorsType

    def __init__(self) -> None: ...

    @property
    def n_stages(self) -> int: ...

    def step_methods(self) -> SteppersOperatorsType: ...

    def step(
        self, SystemCollection: SystemCollectionType, time: np.floating, dt: np.floating
    ) -> np.floating: ...

    def step_single_instance(
        self, SystemCollection: SystemType, time: np.floating, dt: np.floating
    ) -> np.floating: ...


class SymplecticStepperProtocol(StepperProtocol, Protocol):
    """symplectic stepper protocol."""

    def get_steps(self) -> list[OperatorType]: ...

    def get_prefactors(self) -> list[OperatorType]: ...


class MemoryProtocol(Protocol):
    @property
    def initial_state(self) -> bool: ...


class ExplicitStepperProtocol(StepperProtocol, Protocol):
    """symplectic stepper protocol."""

    def get_stages(self) -> list[OperatorType]: ...

    def get_updates(self) -> list[OperatorType]: ...


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
