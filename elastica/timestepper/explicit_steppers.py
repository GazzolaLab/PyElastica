__doc__ = """Explicit timesteppers  and concepts"""
import numpy as np

from . import TimeStepper, StatefulStepper


class ExplicitStepper(TimeStepper):
    """ Base class for all explicit steppers
    """

    def __init__(self):
        super(ExplicitStepper, self).__init__()
        __stages = [
            v for (k, v) in self.__class__.__dict__.items() if k.endswith("stage")
        ]
        __updates = [
            v for (k, v) in self.__class__.__dict__.items() if k.endswith("update")
        ]

        # Tuples are almost immutable
        _n_stages = len(__stages)
        _n_updates = len(__updates)

        assert (
            _n_stages == _n_updates
        ), "Number of stages and updates should be equal to one another"

        self.__stages_and_updates = tuple(zip(__stages, __updates))

    @property
    def n_stages(self):
        return len(self.__stages_and_updates)

    ### For stateless explicit steppers
    def do_step(self, System, Memory, time: np.float64, dt: np.float64):
        for stage, update in self.__stages_and_updates:
            stage(self, System, Memory, time, dt)
            time = update(self, System, Memory, time, dt)
        return time


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
        Memory.initial_state = System.state.copy()
        # self.initial_state = 1
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


class StatefulRungeKutta4(StatefulStepper):
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


class StatefulEulerForward(StatefulStepper):
    def __init__(self):
        super(StatefulEulerForward, self).__init__()
        self.stepper = EulerForward()


class LinearExponentialIntegrator(ExplicitStepper):
    def __init__(self):
        super(LinearExponentialIntegrator, self).__init__()

    def _do_stage(self, System, Memory, time, dt):
        # TODO : Make more general, system should not be calculating what the state
        # transition matrix directly is, but rather it should just give
        Memory.linear_operator = System.get_linear_state_transition_operator(time, dt)

    def _do_update(self, System, Memory, time, dt):
        # System.linearly_evolving_state = _batch_matmul(
        #     System.linearly_evolving_state,
        #     Memory.linear_operator
        # )
        System.linearly_evolving_state = np.einsum(
            "ijk,ljk->ilk", System.linearly_evolving_state, Memory.linear_operator
        )
        return time + dt


class StatefulLinearExponentialIntegrator(StatefulStepper):
    def __init__(self):
        super(StatefulLinearExponentialIntegrator, self).__init__()
        self.stepper = LinearExponentialIntegrator()
        self.linear_operator = None
