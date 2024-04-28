__doc__ = "Time stepper interface"

from typing import Protocol, Tuple, Callable, Type

import numpy as np


class StepperProtocol(Protocol):
    """Protocol for all time-steppers"""

    def do_step(self, *args, **kwargs) -> float: ...

    @property
    def Tag(self) -> Type: ...


class StatefulStepperProtocol(StepperProtocol):
    """
    Stateful explicit, symplectic stepper wrapper.
    """

    # For stateful steppes, bind memory to self
    def do_step(self, System, time: np.floating, dt: np.floating) -> float:
        """
        Perform one time step of the simulation.
        Return the new time.
        """
        ...

    @property
    def n_stages(self) -> int: ...


class MethodCollectorProtocol(Protocol):
    """
    Protocol for collecting stepper methods.
    """

    def __init__(self, timestepper_instance: StepperProtocol): ...

    def step_methods(self) -> Tuple[Callable]: ...


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
