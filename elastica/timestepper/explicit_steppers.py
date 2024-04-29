__doc__ = """Explicit timesteppers  and concepts"""

from typing import Any, Tuple

import numpy as np
from copy import copy

from elastica.typing import SystemCollectionType, SystemType, SteppersOperatorsType
from .tag import tag, ExplicitStepperTag
from .protocol import StatefulStepperProtocol, MemoryProtocol


"""
Developer Note
--------------
## Motivation for choosing _Mixin classes below

The constraint/problem is that we do not know what
`System` we are integrating apriori. For a single
standalone `System` (which defines a `__call__`
operator and has its own states), we should just
step it like a single system.

Instead if we get a `SystemCollection` made up of
bunch of smaller systems (like Cosserat Rods), we now
need to loop over all these smaller systems and perform
state updates there. Not only that we may also need
to communicate between such smaller systems.

One way to solve this issue is to give the integrator
two methods:

- `do_step`, which does the time-stepping for only a
`System`
- `do_system_step` which does the time-stepping for
a `SystemCollection`

The problem with this approach is that
1. We have more methods than we actually use
(indeed we can only integrate either a `System` or
a `SystemCollection` but not both)
2. From an interface point of view, its ugly and not
graceful (at least IMO).

The second approach is what I have chosen here,
which is to create two mixin classes : one for
integrating `System` and one for integrating
`SystemCollection`. And then depending upon the runtime
type of the object to be integrated, we can dynamically
mixin the required class.

This approach overcomes the disadvantages of the
previous approach (as there's only one `do_step` method
associated with a Stepper at any given point of time),
at the expense of being a tad bit harder to understand
(which this documentation will hopefully fix). In essence,
we "smartly" use a mixin class to define the necessary
`do_step` method, which the `integrate` function then uses.
"""


# TODO: Modify this line and move to elastica/typing.py once system state is defined
StateType = Any


class _SystemInstanceStepper:
    # # noinspection PyUnresolvedReferences
    @staticmethod
    def do_step(
        TimeStepper: StatefulStepperProtocol,
        _stages_and_updates: SteppersOperatorsType,
        System: SystemType,
        Memory: MemoryProtocol,
        time: np.floating,
        dt: np.floating,
    ) -> np.floating:
        for stage, update in _stages_and_updates:
            stage(TimeStepper, System, Memory, time, dt)
            time = update(TimeStepper, System, Memory, time, dt)
        return time


class _SystemCollectionStepper:
    # # noinspection PyUnresolvedReferences
    @staticmethod
    def do_step(
        TimeStepper: StatefulStepperProtocol,
        _stages_and_updates: SteppersOperatorsType,
        SystemCollection: SystemCollectionType,
        MemoryCollection: Tuple[MemoryProtocol, ...],
        time: np.floating,
        dt: np.floating,
    ) -> np.floating:
        for stage, update in _stages_and_updates:
            SystemCollection.synchronize(time)
            for system, memory in zip(SystemCollection[:-1], MemoryCollection[:-1]):
                stage(TimeStepper, system, memory, time, dt)
                _ = update(TimeStepper, system, memory, time, dt)

            stage(TimeStepper, SystemCollection[-1], MemoryCollection[-1], time, dt)
            time = update(
                TimeStepper, SystemCollection[-1], MemoryCollection[-1], time, dt
            )
        return time


class ExplicitStepperMethods:
    """Base class for all explicit steppers
    Can also be used as a mixin with optional cls argument below
    """

    def __init__(self, timestepper_instance: StatefulStepperProtocol):
        take_methods_from = timestepper_instance
        __stages = [
            v
            for (k, v) in take_methods_from.__class__.__dict__.items()
            if k.endswith("stage")
        ]
        __updates = [
            v
            for (k, v) in take_methods_from.__class__.__dict__.items()
            if k.endswith("update")
        ]

        # Tuples are almost immutable
        _n_stages = len(__stages)
        _n_updates = len(__updates)

        assert (
            _n_stages == _n_updates
        ), "Number of stages and updates should be equal to one another"

        self._stages_and_updates = tuple(zip(__stages, __updates))

    def step_methods(self) -> SteppersOperatorsType:
        return self._stages_and_updates

    @property
    def n_stages(self) -> int:
        return len(self._stages_and_updates)


class EulerForwardMemory:
    def __init__(self, initial_state: StateType) -> None:
        self.initial_state = initial_state


@tag(ExplicitStepperTag)
class EulerForward:
    """
    Classical Euler Forward stepper. Stateless, coordinates operations only.
    """

    def _first_stage(
        self,
        System: SystemType,
        Memory: EulerForwardMemory,
        time: np.floating,
        dt: np.floating,
    ) -> None:
        pass

    def _first_update(
        self,
        System: SystemType,
        Memory: EulerForwardMemory,
        time: np.floating,
        dt: np.floating,
    ) -> np.floating:
        System.state += dt * System(time, dt)
        return time + dt


class RungeKutta4Memory:
    """
    Stores all states of Rk within the time-stepper. Works as long as the states
    are all one big numpy array, made possible by carefully using views.

    Convenience wrapper around Stateless that provides memory
    """

    def __init__(
        self,
        initial_state: StateType,
        k_1: StateType,
        k_2: StateType,
        k_3: StateType,
        k_4: StateType,
    ) -> None:
        self.initial_state = initial_state
        self.k_1 = k_1
        self.k_2 = k_2
        self.k_3 = k_3
        self.k_4 = k_4


@tag(ExplicitStepperTag)
class RungeKutta4:
    """
    Stateless runge-kutta4. coordinates operations only, memory needs
    to be externally managed and allocated.
    """

    # These methods should be static, but because we need to enable automatic
    # discovery in ExplicitStepper, these are bound to the RungeKutta4 class
    # For automatic discovery, the order of declaring stages here is very important
    def _first_stage(
        self,
        System: SystemType,
        Memory: RungeKutta4Memory,
        time: np.floating,
        dt: np.floating,
    ) -> None:
        Memory.initial_state = copy(System.state)
        Memory.k_1 = dt * System(time, dt)  # Don't update state yet

    def _first_update(
        self,
        System: SystemType,
        Memory: RungeKutta4Memory,
        time: np.floating,
        dt: np.floating,
    ) -> np.floating:
        # prepare for next stage
        System.state = Memory.initial_state + 0.5 * Memory.k_1
        return time + 0.5 * dt

    def _second_stage(
        self,
        System: SystemType,
        Memory: RungeKutta4Memory,
        time: np.floating,
        dt: np.floating,
    ) -> None:
        Memory.k_2 = dt * System(time, dt)  # Don't update state yet

    def _second_update(
        self,
        System: SystemType,
        Memory: RungeKutta4Memory,
        time: np.floating,
        dt: np.floating,
    ) -> np.floating:
        # prepare for next stage
        System.state = Memory.initial_state + 0.5 * Memory.k_2
        return time

    def _third_stage(
        self,
        System: SystemType,
        Memory: RungeKutta4Memory,
        time: np.floating,
        dt: np.floating,
    ) -> None:
        Memory.k_3 = dt * System(time, dt)  # Don't update state yet

    def _third_update(
        self,
        System: SystemType,
        Memory: RungeKutta4Memory,
        time: np.floating,
        dt: np.floating,
    ) -> np.floating:
        # prepare for next stage
        System.state = Memory.initial_state + Memory.k_3
        return time + 0.5 * dt

    def _fourth_stage(
        self,
        System: SystemType,
        Memory: RungeKutta4Memory,
        time: np.floating,
        dt: np.floating,
    ) -> None:
        Memory.k_4 = dt * System(time, dt)  # Don't update state yet

    def _fourth_update(
        self,
        System: SystemType,
        Memory: RungeKutta4Memory,
        time: np.floating,
        dt: np.floating,
    ) -> np.floating:
        # prepare for next stage
        System.state = (
            Memory.initial_state
            + (Memory.k_1 + 2.0 * Memory.k_2 + 2.0 * Memory.k_3 + Memory.k_4) / 6.0
        )
        return time


# class ExplicitLinearExponentialIntegrator(
#     _LinearExponentialIntegratorMixin, ExplicitStepper
# ):
#     def __init__(self):
#         _LinearExponentialIntegratorMixin.__init__(self)
#         ExplicitStepper.__init__(self, _LinearExponentialIntegratorMixin)
#
#
# class StatefulLinearExponentialIntegrator(_StatefulStepper):
#     def __init__(self):
#         super(StatefulLinearExponentialIntegrator, self).__init__()
#         self.stepper = ExplicitLinearExponentialIntegrator()
#         self.linear_operator = None
