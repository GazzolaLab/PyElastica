__doc__ = """ Test all timestepper kernels and interfaces here """

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elastica.timestepper import (
    TimeStepper,
    ExplicitStepper,
    StatefulRungeKutta4,
    integrate,
)
from elastica.utils import Tolerance


class BaseStatefulSystem:
    def __init__(self):
        pass

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        self._state = new_state


class ScalarExponentialDecaySystem(BaseStatefulSystem):
    def __init__(self, exponent=-1, init_val=1):
        super(ScalarExponentialDecaySystem, self).__init__()
        self.exponent = np.float64(exponent)
        self.initial_value = np.float64(init_val)
        self._state = self.initial_value

    def analytical_solution(self, time):
        return self.initial_value * np.exp(self.exponent * time)

    def __call__(self, *args, **kwargs):
        return self.exponent * self._state


class UndampedSimpleHarmonicOscillatorSystem(BaseStatefulSystem):
    def __init__(self, omega=2.0 * np.pi, init_val=np.array([1.0, 0.0])):
        super(UndampedSimpleHarmonicOscillatorSystem, self).__init__()
        self.omega = omega
        self.initial_value = self._state = init_val
        self.A_matrix = np.array([[0.0, 1.0], [-self.omega ** 2, 0.0]])

    def analytical_solution(self, time):
        # http://scipp.ucsc.edu/~haber/ph5B/sho09.pdf
        amplitude = np.sqrt(
            self.initial_value[0] ** 2 + (self.initial_value[1] / self.omega) ** 2
        )
        phase = np.arccos(self.initial_value[0] / amplitude)
        analytical_position = amplitude * np.cos(self.omega * time + phase)
        analytical_velocity = (
            -amplitude * self.omega * np.sin(self.omega * time + phase)
        )
        return np.array([analytical_position, analytical_velocity])

    def __call__(self, *args, **kwargs):
        return self.A_matrix @ self._state


class DampedSimpleHarmonicOscillatorSystem(UndampedSimpleHarmonicOscillatorSystem):
    def __init__(self, damping=0.5, omega=2.0 * np.pi, init_val=np.array([1.0, 0.0])):
        super(DampedSimpleHarmonicOscillatorSystem, self).__init__(omega, init_val)
        self.zeta = 0.5 * damping
        self.modified_omega = np.sqrt(
            self.omega ** 2 - self.zeta ** 2, dtype=np.complex
        )
        self.A_matrix = np.array([[0.0, 1.0], [-self.omega ** 2, -damping]])

    def analytical_solution(self, time):
        # https://en.wikipedia.org/wiki/Harmonic_oscillator#Transient_solution
        amplitude = np.sqrt(
            self.initial_value[0] ** 2
            + (
                (self.initial_value[1] + self.zeta * self.initial_value[0])
                / self.modified_omega
            )
            ** 2
        )
        phase = np.arctan(
            (self.initial_value[1] + self.zeta * self.initial_value[0])
            / self.modified_omega
            / self.initial_value[0]
        )
        analytical_position = (
            amplitude
            * np.exp(-self.zeta * time)
            * np.cos(self.modified_omega * time - phase)
        )
        analytical_velocity = -self.zeta * amplitude * np.exp(
            -self.zeta * time
        ) * np.cos(
            self.modified_omega * time - phase
        ) - self.modified_omega * amplitude * np.exp(
            -self.zeta * time
        ) * np.sin(
            self.modified_omega * time - phase
        )
        return np.array([analytical_position, analytical_velocity])


# Test cases with several simple ODE's with
# changing systems and states to test accuracy
# of our stepper
class TestExplicitSteppers:
    ExplicitSteppers = [StatefulRungeKutta4]

    @pytest.mark.parametrize("stepper", ExplicitSteppers)
    def test_against_scalar_exponential(self, stepper):
        system = ScalarExponentialDecaySystem(-1, 1)
        final_time = 0.1
        n_steps = 1000
        integrate(stepper(), system, final_time=final_time, n_steps=n_steps)

        assert_allclose(
            system.state, system.analytical_solution(final_time), atol=Tolerance.atol()
        )

    @pytest.mark.parametrize("stepper", ExplicitSteppers)
    def test_against_undamped_harmonic_oscillator(self, stepper):
        system = UndampedSimpleHarmonicOscillatorSystem()
        final_time = 4.0 * np.pi
        n_steps = 2000
        integrate(stepper(), system, final_time=final_time, n_steps=n_steps)

        assert_allclose(system.state, system.analytical_solution(final_time), atol=1e-4)

    @pytest.mark.parametrize("stepper", ExplicitSteppers)
    def test_against_damped_harmonic_oscillator(self, stepper):
        system = DampedSimpleHarmonicOscillatorSystem()
        final_time = 4.0 * np.pi
        n_steps = 2000
        integrate(stepper(), system, final_time=final_time, n_steps=n_steps)

        assert_allclose(system.state, system.analytical_solution(final_time), atol=1e-4)
