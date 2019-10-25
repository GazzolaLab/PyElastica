__doc__ = """Timestepping utilities to be used with Rod and RigidBody classes"""
import numpy as np
from ..utils import Tolerance


class TimeStepper:
    """ Interface classes for all time-steppers
    """

    def __init__(self):
        pass

    def do_step(self, *args, **kwargs):
        raise NotImplementedError(
            "TimeStepper hierarchy is not supposed to access the do-step routine of the TimeStepper base class. "
        )


class StatefulStepper:
    def __init__(self):
        pass

    # For stateful steppes, bind memory to self
    def do_step(self, System, time: np.float64, dt: np.float64):
        return self.stepper.do_step(System, self, time, dt)

    @property
    def n_stages(self):
        return self.stepper.n_stages


# TODO Improve interface of this function to take args and kwargs for ease of use
def integrate(StatefulStepper, System, final_time, n_steps=1000):
    dt = np.float64(final_time / n_steps)
    time = np.float64(0.0)
    while np.abs(final_time - time) > 1e5 * Tolerance.atol():
        time = StatefulStepper.do_step(System, time, dt)
