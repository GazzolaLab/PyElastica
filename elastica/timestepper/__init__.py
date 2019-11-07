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


class LinearExponentialIntegratorMixin:
    def __init__(self):
        pass

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

    def _first_prefactor(self, dt):
        """ Prefactor call to satisfy interface of SymplecticStepper. Should never
        be used in actual code.

        Parameters
        ----------
        dt : the time step of simulation

        Raises
        ------
        RuntimeError
        """
        raise RuntimeError(
            "Symplectic prefactor of LinearExponentialIntegrator should not be called!"
        )

    # Code repeat!
    # Easy to avoid, but keep for performance.
    def _do_one_step(self, System, time, prefac):
        System.linearly_evolving_state = np.einsum(
            "ijk,ljk->ilk",
            System.linearly_evolving_state,
            System.get_linear_state_transition_operator(time, prefac),
        )
        return (
            time
        )  # TODO fix hack that treats time separately here. Shuold be time + dt
        # return time + dt


# TODO Improve interface of this function to take args and kwargs for ease of use
def integrate(StatefulStepper, System, final_time, n_steps=1000):
    dt = np.float64(final_time / n_steps)
    time = np.float64(0.0)
    while np.abs(final_time - time) > 1e5 * Tolerance.atol():
        time = StatefulStepper.do_step(System, time, dt)
