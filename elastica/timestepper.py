__doc__ = """Timestepping utilities to be used with Rod and RigidBody classes"""
import numpy as np
import functools

from ._linalg import _batch_matmul, _batch_matvec, _batch_cross
from .utils import Tolerance

class TimeStepper():
    """ Interface classes for all time-steppers
    """

    def __init__(self):
        pass

    def do_step(self):
        pass

    def _solve_linear_equation(self, time, dt, states, state_transition_matrix):
        return _batch_matmul(states, state_transition_matrix)


class ExplicitStepper(TimeStepper):
    """

    """

    def __init__(self):
        super(ExplicitStepper, self).__init__()
        self.__stages = [v for (k, v) in self.__class__.__dict__.items() if k.endswith('stage')]

        # Tuples are almost immutable
        self._n_stages = (len(self.__stages),)

    @property
    def n_stages(self):
        return self._n_stages[0]

    ### For stateless explicit steppers
    def do_step(self, System, Memory, time : np.float64, dt : np.float64):
        for stage in self.__stages:
            time = stage(self, System, Memory, time, dt)
        return time

class StatefulExplicitStepper():

    def __init__(self):
        pass

    # For stateful steppes, bind memory to self
    def do_step(self, System, time : np.float64, dt : np.float64):
        return self.stepper.do_step(System, self, time, dt)

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
    def _first_stage(self, System, Memory, time : np.float64, dt : np.float64):
        Memory.initial_state = System.state.copy()
        # self.initial_state = 1
        Memory.k_1 = dt * System(time, dt)  # Don't update state yet

        # prepare for next stage
        System.state = Memory.initial_state + 0.5 * Memory.k_1
        return time + 0.5 * dt


    def _second_stage(self, System, Memory, time : np.float64, dt : np.float64):
        Memory.k_2 = dt * System(time, dt)  # Don't update state yet

        # prepare for next stage
        System.state = Memory.initial_state + 0.5 * Memory.k_2
        return time


    def _third_stage(self, System, Memory, time : np.float64, dt : np.float64):
        Memory.k_3 = dt * System(time, dt)  # Don't update state yet

        # prepare for next stage
        time += 0.5 * dt
        System.state = Memory.initial_state + Memory.k_3
        return time


    def _fourth_stage(self, System, Memory, time : np.float64, dt : np.float64):
        k_4 = dt * System(time, dt)  # Don't update state yet

        # prepare for next stage
        System.state = Memory.initial_state + (Memory.k_1 + 2. * Memory.k_2 + 2. * Memory.k_3 + k_4) / 6.0
        return time


class StatefulRungeKutta4(StatefulExplicitStepper):
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

"""
Demonstration of constructing a staged-method 
using EulerForward Timestepper
"""
class EulerForward(ExplicitStepper):
    def __init__(self):
        super(EulerForward, self).__init__()

    def _first_stage(self, System, Memory, time, dt):
        System.state += dt * System(time, dt)
        return time + dt

class StatefulEulerForward(StatefulExplicitStepper):
    def __init__(self):
        super(StatefulEulerForward, self).__init__()
        self.stepper = EulerForward()


# TODO Improve interface of this function to take args and kwargs for ease of use
def integrate(StatefulStepper, System, final_time, n_steps=1000):
    dt = np.float64(final_time / n_steps)
    time = np.float64(0.0)
    while np.abs(final_time - time) > 1e5 * Tolerance.atol():
        time = StatefulStepper.do_step(System, time, dt)
