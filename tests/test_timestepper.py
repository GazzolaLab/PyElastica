import numpy as np
import pytest
from numpy.testing import assert_allclose

from elastica.timestepper import TimeStepper, integrate
from elastica.timestepper.explicit_steppers import (
    StatefulRungeKutta4,
    StatefulEulerForward,
    StatefulLinearExponentialIntegrator,
)
from elastica.timestepper.symplectic_steppers import PositionVerlet, PEFRL
from elastica.timestepper.hybrid_rod_steppers import SymplecticCosseratRodStepper
from elastica.utils import Tolerance
from elastica._rotations import _get_rotation_matrix


class BaseStatefulSystem:
    def __init__(self):
        pass

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        self._state = new_state


class BaseSymplecticSystem:
    def __init__(self):
        pass

    def kinematic_rates(self, *args):
        return self._dyn_state

    @property
    def kinematic_states(self):
        return self._kin_state

    @kinematic_states.setter
    def kinematic_states(self, new_kin_state):
        self._kin_state = new_kin_state

    @property
    def dynamic_states(self):
        return self._dyn_state

    @dynamic_states.setter
    def dynamic_states(self, new_dyn_state):
        self._dyn_state = new_dyn_state


class BaseLinearStatefulSystem:
    def __init__(self):
        pass

    @property
    def linearly_evolving_state(self):
        return self._state

    @linearly_evolving_state.setter
    def linearly_evolving_state(self, new_state):
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


class BaseUndampedSimpleHarmonicOscillatorSystem:
    def __init__(self, omega=2.0 * np.pi, init_val=np.array([1.0, 0.0])):
        super(BaseUndampedSimpleHarmonicOscillatorSystem, self).__init__()
        self.omega = omega
        self.initial_value = init_val.copy()
        self._state = init_val.copy()
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


class UndampedSimpleHarmonicOscillatorSystem(
    BaseUndampedSimpleHarmonicOscillatorSystem, BaseStatefulSystem
):
    def __init__(self, omega=2.0 * np.pi, init_val=np.array([1.0, 0.0])):
        BaseUndampedSimpleHarmonicOscillatorSystem.__init__(
            self, omega=omega, init_val=init_val
        )


class SymplecticUndampedSimpleHarmonicOscillatorSystem(
    BaseUndampedSimpleHarmonicOscillatorSystem, BaseSymplecticSystem
):
    def __init__(self, omega=2.0 * np.pi, init_val=np.array([1.0, 0.0])):
        BaseUndampedSimpleHarmonicOscillatorSystem.__init__(
            self, omega=omega, init_val=init_val
        )
        self._kin_state = self._state[0:1]  # Create a view instead
        self._dyn_state = self._state[1:2]  # Create a view instead

    def dynamic_rates(self, *args, **kwargs):
        return super(SymplecticUndampedSimpleHarmonicOscillatorSystem, self).__call__(
            *args, **kwargs
        )[-1]

    def compute_energy(self, time):
        # http://scipp.ucsc.edu/~haber/ph5B/sho09.pdf
        analytical_state = self.analytical_solution(time)

        def energy(st):
            return self.omega ** 2 * st[0] ** 2 + st[1] ** 2

        anal_energy = energy(analytical_state)
        current_energy = energy(self._state)
        return current_energy, anal_energy


class DampedSimpleHarmonicOscillatorSystem(
    BaseUndampedSimpleHarmonicOscillatorSystem, BaseStatefulSystem
):
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


class MultipleFrameRotationSystem(BaseLinearStatefulSystem):
    def __init__(self, n_frames=128):
        super(MultipleFrameRotationSystem, self).__init__()
        self.initial_value = np.tile(
            np.eye(3, 3).reshape(3, 3, 1), n_frames
        )  # Start aligned initially
        self.omega = np.random.randn(3, n_frames)  # Randomly rotate frames
        # self.omega /= np.norm(self.omega, ord=2, axis=0)
        self._state = self.initial_value.copy()

    def analytical_solution(self, time):
        # http://scipp.ucsc.edu/~haber/ph5B/sho09.pdf
        # return _batch_matmul(self._state, _get_rotation_matrix(time, self.omega))
        return np.einsum(
            "ijk,ljk->ilk", self.initial_value, _get_rotation_matrix(time, self.omega)
        )

    def get_linear_state_transition_operator(self, time, dt):
        return _get_rotation_matrix(dt, self.omega)


class SecondOrderHybridSystem:
    """
    Integrate a simple, non-linear ODE:
        dx/dt = v
        df/dt = -f * ω (f is short for frame, for lack of better notation)
        dv/dt = -v**2
        dω/dt = -ω**2
    Dofs: [x, f, v, ω], with the convention that

    _state in this case are [x, v, ω]
    linear_states are [f]
    _kin_state are [x], taken as a slice
    _dyn_state are [v, ω], taken as a slice
    """

    def __init__(self, init_x=5.0, init_f=3.0, init_v=1.0, init_w=1.0):
        """
        """
        # Contains initial_values for all dofs
        self.initial_value = np.array([init_x, init_f, init_v, init_w])
        self.state = self.initial_value.copy()
        self.kinematic_states = self.state[0:1]  # Create a view instead
        self.dynamic_states = self.state[2:4]  # Create a view instead
        self.linearly_evolving_state = self.state[1].reshape(
            -1, 1, 1
        )  # Requirements of linear_stepper

    def get_linear_state_transition_operator(self, time, prefac):
        return np.array([np.exp(-self.state[3] * prefac)]).reshape(-1, 1, 1)

    def analytical_solution(self, time):
        # http://scipp.ucsc.edu/~haber/ph5B/sho09.pdf
        # return _batch_matmul(self._state, _get_rotation_matrix(time, self.omega))
        v_factor = 1.0 / self.initial_value[2]
        w_factor = 1.0 / self.initial_value[3]
        x = self.initial_value[0] + np.log(1.0 + time / v_factor)
        f = self.initial_value[1] / (1.0 + time / w_factor)
        v = 1.0 / (v_factor + time)
        w = 1.0 / (w_factor + time)
        return np.array([x, f, v, w])

    def kinematic_rates(self, time, prefac):
        return self.dynamic_states[0]  # dx/dt = v

    def dynamic_rates(self, time, prefac):
        return -self.dynamic_states ** 2  # d(v,w)/dt = -(v,w)**2

    def final_solution(self, time):
        if np.allclose(self.linearly_evolving_state[0, 0, 0], self.initial_value[1]):
            val = self.state[1]
        else:
            val = self.linearly_evolving_state[0, 0, 0]
        return np.array([self.state[0], val, self.state[2], self.state[3]])

    def __call__(self, *args, **kwargs):
        return np.array(
            [
                self.state[2],
                -self.state[1] * self.state[3],
                -self.state[2] ** 2,
                -self.state[3] ** 2,
            ]
        )


# Test cases with several simple ODE's with
# changing systems and states to test accuracy
# of our stepper
class TestExplicitSteppers:
    # Added automatic discovery of Stateful explicit integrators
    # ExplicitSteppers = StatefulExplicitStepper.__subclasses__()
    # SymplecticSteppers = SymplecticStepper.__subclasses__()
    ExplicitSteppers = [StatefulRungeKutta4, StatefulEulerForward]
    SymplecticSteppers = [PositionVerlet, PEFRL]

    def test_no_base_access_error(self):
        with pytest.raises(NotImplementedError) as excinfo:
            TimeStepper().do_step()
        assert "not supposed to access" in str(excinfo.value)

    @pytest.mark.parametrize("stepper", ExplicitSteppers + SymplecticSteppers)
    def test_correct_orders(self, stepper):
        assert stepper().n_stages > 0, "Explicit stepper routine has no stages!"

    @pytest.mark.parametrize("stepper", ExplicitSteppers)
    def test_against_scalar_exponential(self, stepper):
        system = ScalarExponentialDecaySystem(-1, 1)
        final_time = 1
        n_steps = 1000
        integrate(stepper(), system, final_time=final_time, n_steps=n_steps)

        assert_allclose(
            system.state,
            system.analytical_solution(final_time),
            rtol=Tolerance.rtol() * 1e3,
            atol=Tolerance.atol(),
        )

    @pytest.mark.parametrize("stepper", ExplicitSteppers[:-1])
    def test_against_undamped_harmonic_oscillator(self, stepper):
        system = UndampedSimpleHarmonicOscillatorSystem()
        final_time = 4.0 * np.pi
        n_steps = 2000
        integrate(stepper(), system, final_time=final_time, n_steps=n_steps)

        assert_allclose(
            system.state,
            system.analytical_solution(final_time),
            rtol=Tolerance.rtol(),
            atol=Tolerance.atol(),
        )

    @pytest.mark.parametrize("stepper", ExplicitSteppers[:-1])
    def test_against_damped_harmonic_oscillator(self, stepper):
        system = DampedSimpleHarmonicOscillatorSystem()
        final_time = 4.0 * np.pi
        n_steps = 2000
        integrate(stepper(), system, final_time=final_time, n_steps=n_steps)

        assert_allclose(
            system.state,
            system.analytical_solution(final_time),
            rtol=Tolerance.rtol(),
            atol=Tolerance.atol(),
        )

    def test_linear_exponential_integrator(self):
        system = MultipleFrameRotationSystem(n_frames=128)
        final_time = np.pi
        n_steps = 1000
        integrate(
            StatefulLinearExponentialIntegrator(),
            system,
            final_time=final_time,
            n_steps=n_steps,
        )

        assert_allclose(
            system.linearly_evolving_state,
            system.analytical_solution(final_time),
            atol=1e-4,
        )

    @pytest.mark.parametrize("stepper", SymplecticSteppers)
    def test_symplectic_against_undamped_harmonic_oscillator(self, stepper):
        system = SymplecticUndampedSimpleHarmonicOscillatorSystem()
        final_time = 4.0 * np.pi
        n_steps = 2000
        integrate(stepper(), system, final_time=final_time, n_steps=n_steps)

        # Symplectic systems conserve energy to a certain extent
        assert_allclose(
            *system.compute_energy(final_time),
            rtol=Tolerance.rtol() * 1e1,
            atol=Tolerance.atol(),
        )

        # assert_allclose(
        #     system._state,
        #     system.analytical_solution(final_time),
        #     rtol=Tolerance.rtol(),
        #     atol=Tolerance.atol(),
        # )

    @pytest.mark.parametrize("symplectic_stepper", SymplecticSteppers)
    def test_hybrid_symplectic_against_analytical_system(self, symplectic_stepper):
        system = SecondOrderHybridSystem()
        final_time = 1.0
        n_steps = 2000
        stepper = SymplecticCosseratRodStepper(symplectic_stepper=symplectic_stepper())
        integrate(stepper, system, final_time=final_time, n_steps=n_steps)

        assert_allclose(
            system.final_solution(final_time),
            system.analytical_solution(final_time),
            rtol=Tolerance.rtol() * 1e2,
            atol=Tolerance.atol(),
        )

    @pytest.mark.parametrize("explicit_stepper", ExplicitSteppers[:-1])
    def test_hybrid_explicit_against_analytical_system(self, explicit_stepper):
        system = SecondOrderHybridSystem()
        final_time = 1.0
        n_steps = 2000
        integrate(explicit_stepper(), system, final_time=final_time, n_steps=n_steps)

        assert_allclose(
            system.final_solution(final_time),
            system.analytical_solution(final_time),
            rtol=Tolerance.rtol() * 1e2,
            atol=Tolerance.atol(),
        )
