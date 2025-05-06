__doc__ = """Explicit timesteppers  and concepts"""

from typing import Any

import numpy as np
from copy import copy

from elastica.typing import (
    SystemType,
    SystemCollectionType,
    StepType,
    SteppersOperatorsType,
    StateType,
)
from elastica.experimental.timestepper.protocol import (
    ExplicitSystemProtocol,
    ExplicitStepperProtocol,
    MemoryProtocol,
)


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


class EulerForwardMemory:
    def __init__(self, initial_state: StateType) -> None:
        self.initial_state = initial_state


class RungeKutta4Memory:
    """
    Stores all states of Rk within the time-stepper. Works as long as the states
    are all one big numpy array, made possible by carefully using views.

    Convenience wrapper around Stateless that provides memory
    """

    def __init__(
        self,
        initial_state: StateType,
    ) -> None:
        self.initial_state = initial_state
        self.k_1 = initial_state
        self.k_2 = initial_state
        self.k_3 = initial_state
        self.k_4 = initial_state


class ExplicitStepperMixin:
    """Base class for all explicit steppers
    Can also be used as a mixin with optional cls argument below
    """

    def __init__(self: ExplicitStepperProtocol):
        self.steps_and_prefactors = self.step_methods()

    def step_methods(self: ExplicitStepperProtocol) -> SteppersOperatorsType:
        stages = self.get_stages()
        updates = self.get_updates()

        assert len(stages) == len(
            updates
        ), "Number of stages and updates should be equal to one another"
        return tuple(zip(stages, updates))

    @property
    def n_stages(self: ExplicitStepperProtocol) -> int:
        return len(self.steps_and_prefactors)

    def step(
        self: ExplicitStepperProtocol,
        SystemCollection: SystemCollectionType,
        time: np.float64,
        dt: np.float64,
    ) -> np.float64:
        if isinstance(
            self, EulerForward
        ):  # TODO: Cleanup - use depedency injection instead
            Memory = EulerForwardMemory
        elif isinstance(self, RungeKutta4):
            Memory = RungeKutta4Memory  # type: ignore[assignment]
        else:
            raise NotImplementedError(f"Memory class not defined for {self}")
        memory_collection = tuple(
            [Memory(initial_state=system.state) for system in SystemCollection]
        )
        return ExplicitStepperMixin.do_step(self, self.steps_and_prefactors, SystemCollection, memory_collection, time, dt)  # type: ignore[attr-defined]

    @staticmethod
    def do_step(
        TimeStepper: ExplicitStepperProtocol,
        steps_and_prefactors: SteppersOperatorsType,
        SystemCollection: SystemCollectionType,
        MemoryCollection: Any,  # TODO
        time: np.float64,
        dt: np.float64,
    ) -> np.float64:
        for stage, update in steps_and_prefactors:
            SystemCollection.synchronize(time)
            for system, memory in zip(SystemCollection[:-1], MemoryCollection[:-1]):
                stage(system, memory, time, dt)
                _ = update(system, memory, time, dt)

            stage(SystemCollection[-1], MemoryCollection[-1], time, dt)
            time = update(SystemCollection[-1], MemoryCollection[-1], time, dt)
        return time

    def step_single_instance(
        self: ExplicitStepperProtocol,
        System: SystemType,
        Memory: MemoryProtocol,
        time: np.float64,
        dt: np.float64,
    ) -> np.float64:
        for stage, update in self.steps_and_prefactors:
            stage(System, Memory, time, dt)
            time = update(System, Memory, time, dt)
        return time


class EulerForward(ExplicitStepperMixin):
    """
    Classical Euler Forward stepper. Stateless, coordinates operations only.
    """

    def get_stages(self) -> list[StepType]:
        return [self._first_stage]

    def get_updates(self) -> list[StepType]:
        return [self._first_update]

    def _first_stage(
        self,
        System: ExplicitSystemProtocol,
        Memory: EulerForwardMemory,
        time: np.float64,
        dt: np.float64,
    ) -> None:
        pass

    def _first_update(
        self,
        System: ExplicitSystemProtocol,
        Memory: EulerForwardMemory,
        time: np.float64,
        dt: np.float64,
    ) -> np.float64:
        System.state += dt * System(time, dt)  # type: ignore[arg-type]
        return time + dt


class RungeKutta4(ExplicitStepperMixin):
    """
    Stateless runge-kutta4. coordinates operations only, memory needs
    to be externally managed and allocated.
    """

    def get_stages(self) -> list[StepType]:
        return [
            self._first_stage,
            self._second_stage,
            self._third_stage,
            self._fourth_stage,
        ]

    def get_updates(self) -> list[StepType]:
        return [
            self._first_update,
            self._second_update,
            self._third_update,
            self._fourth_update,
        ]

    # These methods should be static, but because we need to enable automatic
    # discovery in ExplicitStepper, these are bound to the RungeKutta4 class
    # For automatic discovery, the order of declaring stages here is very important
    def _first_stage(
        self,
        System: ExplicitSystemProtocol,
        Memory: RungeKutta4Memory,
        time: np.float64,
        dt: np.float64,
    ) -> None:
        Memory.initial_state = copy(System.state)
        Memory.k_1 = dt * System(time, dt)  # type: ignore[operator, assignment]

    def _first_update(
        self,
        System: ExplicitSystemProtocol,
        Memory: RungeKutta4Memory,
        time: np.float64,
        dt: np.float64,
    ) -> np.float64:
        # prepare for next stage
        System.state = Memory.initial_state + 0.5 * Memory.k_1  # type: ignore[operator]
        return time + 0.5 * dt

    def _second_stage(
        self,
        System: ExplicitSystemProtocol,
        Memory: RungeKutta4Memory,
        time: np.float64,
        dt: np.float64,
    ) -> None:
        Memory.k_2 = dt * System(time, dt)  # type: ignore[operator, assignment]

    def _second_update(
        self,
        System: ExplicitSystemProtocol,
        Memory: RungeKutta4Memory,
        time: np.float64,
        dt: np.float64,
    ) -> np.float64:
        # prepare for next stage
        System.state = Memory.initial_state + 0.5 * Memory.k_2  # type: ignore[operator]
        return time

    def _third_stage(
        self,
        System: ExplicitSystemProtocol,
        Memory: RungeKutta4Memory,
        time: np.float64,
        dt: np.float64,
    ) -> None:
        Memory.k_3 = dt * System(time, dt)  # type: ignore[operator, assignment]

    def _third_update(
        self,
        System: ExplicitSystemProtocol,
        Memory: RungeKutta4Memory,
        time: np.float64,
        dt: np.float64,
    ) -> np.float64:
        # prepare for next stage
        System.state = Memory.initial_state + Memory.k_3  # type: ignore[operator]
        return time + 0.5 * dt

    def _fourth_stage(
        self,
        System: ExplicitSystemProtocol,
        Memory: RungeKutta4Memory,
        time: np.float64,
        dt: np.float64,
    ) -> None:
        Memory.k_4 = dt * System(time, dt)  # type: ignore[operator, assignment]

    def _fourth_update(
        self,
        System: ExplicitSystemProtocol,
        Memory: RungeKutta4Memory,
        time: np.float64,
        dt: np.float64,
    ) -> np.float64:
        # prepare for next stage
        System.state = (
            Memory.initial_state
            + (Memory.k_1 + 2.0 * Memory.k_2 + 2.0 * Memory.k_3 + Memory.k_4) / 6.0  # type: ignore[operator]
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
