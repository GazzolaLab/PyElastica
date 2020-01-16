__doc__ = """Explicit timesteppers  and concepts"""
import numpy as np
from copy import copy

from elastica.timestepper._stepper_interface import (
    _TimeStepper,
    _LinearExponentialIntegratorMixin,
    _StatefulStepper,
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


class _SystemInstanceStepperMixin:
    # noinspection PyUnresolvedReferences
    def do_step(self, System, Memory, time: np.float64, dt: np.float64):
        for stage, update in self._stages_and_updates:
            stage(self, System, Memory, time, dt)
            time = update(self, System, Memory, time, dt)
        return time


class _SystemCollectionStepperMixin:
    # noinspection PyUnresolvedReferences
    def do_step(
        self, SystemCollection, MemoryCollection, time: np.float64, dt: np.float64
    ):
        for stage, update in self._stages_and_updates:
            SystemCollection.synchronize(time)
            for system, memory in zip(SystemCollection[:-1], MemoryCollection[:-1]):
                stage(self, system, memory, time, dt)
                _ = update(self, system, memory, time, dt)

            stage(self, SystemCollection[-1], MemoryCollection[-1], time, dt)
            time = update(self, SystemCollection[-1], MemoryCollection[-1], time, dt)
        return time


class ExplicitStepper(_TimeStepper):
    """ Base class for all explicit steppers
    Can also be used as a mixin with optional cls argument below
    """

    def __init__(self, cls=None):
        super(ExplicitStepper, self).__init__()
        take_methods_from = self if cls is None else cls()
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

    @property
    def n_stages(self):
        return len(self._stages_and_updates)


"""
Classical RK4 follows
"""


class RungeKutta4(ExplicitStepper):
    """
    Stateless runge-kutta4. coordinates operations only, memory needs
    to be externally managed and allocated.
    """

    def __init__(self):
        super(RungeKutta4, self).__init__()

    # These methods should be static, but because we need to enable automatic
    # discovery in ExplicitStepper, these are bound to the RungeKutta4 class
    # For automatic discovery, the order of declaring stages here is very important
    def _first_stage(self, System, Memory, time: np.float64, dt: np.float64):
        Memory.initial_state = copy(System.state)
        Memory.k_1 = dt * System(time, dt)  # Don't update state yet

    def _first_update(self, System, Memory, time: np.float64, dt: np.float64):
        # prepare for next stage
        System.state = Memory.initial_state + 0.5 * Memory.k_1
        return time + 0.5 * dt

    def _second_stage(self, System, Memory, time: np.float64, dt: np.float64):
        Memory.k_2 = dt * System(time, dt)  # Don't update state yet

    def _second_update(self, System, Memory, time: np.float64, dt: np.float64):
        # prepare for next stage
        System.state = Memory.initial_state + 0.5 * Memory.k_2
        return time

    def _third_stage(self, System, Memory, time: np.float64, dt: np.float64):
        Memory.k_3 = dt * System(time, dt)  # Don't update state yet

    def _third_update(self, System, Memory, time: np.float64, dt: np.float64):
        # prepare for next stage
        System.state = Memory.initial_state + Memory.k_3
        return time + 0.5 * dt

    def _fourth_stage(self, System, Memory, time: np.float64, dt: np.float64):
        Memory.k_4 = dt * System(time, dt)  # Don't update state yet

    def _fourth_update(self, System, Memory, time: np.float64, dt: np.float64):
        # prepare for next stage
        System.state = (
            Memory.initial_state
            + (Memory.k_1 + 2.0 * Memory.k_2 + 2.0 * Memory.k_3 + Memory.k_4) / 6.0
        )
        return time


class StatefulRungeKutta4(_StatefulStepper):
    """
    Stores all states of Rk within the time-stepper. Works as long as the states
    are all one big numpy array, made possible by carefully using views.

    Convenience wrapper around Stateless that provides memory
    """

    def __init__(self):
        super(StatefulRungeKutta4, self).__init__()
        self.stepper = RungeKutta4()
        self.initial_state = None
        self.k_1 = None
        self.k_2 = None
        self.k_3 = None
        self.k_4 = None


"""
Classical EulerForward
"""


class EulerForward(ExplicitStepper):
    def __init__(self):
        super(EulerForward, self).__init__()

    def _first_stage(self, System, Memory, time, dt):
        pass

    def _first_update(self, System, Memory, time, dt):
        System.state += dt * System(time, dt)
        return time + dt


class StatefulEulerForward(_StatefulStepper):
    def __init__(self):
        super(StatefulEulerForward, self).__init__()
        self.stepper = EulerForward()


class ExplicitLinearExponentialIntegrator(
    _LinearExponentialIntegratorMixin, ExplicitStepper
):
    def __init__(self):
        _LinearExponentialIntegratorMixin.__init__(self)
        ExplicitStepper.__init__(self, _LinearExponentialIntegratorMixin)


class StatefulLinearExponentialIntegrator(_StatefulStepper):
    def __init__(self):
        super(StatefulLinearExponentialIntegrator, self).__init__()
        self.stepper = ExplicitLinearExponentialIntegrator()
        self.linear_operator = None
