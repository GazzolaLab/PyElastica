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

    def do_step(self, System, Time, dt):
        _time = self.__stages[0].__call__(self, System, Time, dt)
        for stage in self.__stages[1:]:
            _time = stage(self, System, _time, dt)
        return _time


class StatefulRungeKutta4(ExplicitStepper):
    """
    Stores all states of Rk within the time-stepper. Works as long as the states
    are all one big numpy array, made possible by carefully using views
    """

    def __init__(self):
        super(StatefulRungeKutta4, self).__init__()
        self.initial_state = None
        self.k_1 = None
        self.k_2 = None
        self.k_3 = None
        self.k_4 = None

    def rk4(f, h, y0, t0):
        k1 = f(t0, y0)
        k2 = f(t0 + h / 2, y0 + h / 2 * k1)
        k3 = f(t0 + h / 2, y0 + h / 2 * k2)
        k4 = f(t0 + h, y0 + h * k3)
        return y0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # For automatic discovery, the order of declaring stages here is very important
    def _first_stage(self, system, time, dt):
        self.initial_state = system.state.copy()
        # self.initial_state = 1
        self.k_1 = dt * system(time, dt)  # Don't update state yet

        # prepare for next stage
        system.state = self.initial_state + 0.5 * self.k_1
        return time + 0.5 * dt

    def _second_stage(self, system, time, dt):
        self.k_2 = dt * system(time, dt)  # Don't update state yet

        # prepare for next stage
        system.state = self.initial_state + 0.5 * self.k_2
        return time

    def _third_stage(self, system, time, dt):
        self.k_3 = dt * system(time, dt)  # Don't update state yet

        # prepare for next stage
        time += 0.5 * dt
        system.state = self.initial_state + self.k_3
        return time

    def _fourth_stage(self, system, time, dt):
        self.k_4 = dt * system(time, dt)  # Don't update state yet

        # prepare for next stage
        system.state = self.initial_state + (self.k_1 + 2. * self.k_2 + 2. * self.k_3 + self.k_4) / 6.0
        return time

def integrate(Stepper, System, final_time, n_steps=1000):
    dt = np.float64(final_time / n_steps)
    time = np.float64(0.0)
    while np.abs(final_time - time) > 1e5 * Tolerance.atol():
        time = Stepper.do_step(System, time, dt)
