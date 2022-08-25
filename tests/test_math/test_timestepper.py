__doc__ = """Testing timesteppers in Elastica Numba implementation """
import numpy as np
import pytest
from numpy.testing import assert_allclose

from elastica.systems.analytical import (
    ScalarExponentialDecaySystem,
    # UndampedSimpleHarmonicOscillatorSystem,
    SymplecticUndampedSimpleHarmonicOscillatorSystem,
    # DampedSimpleHarmonicOscillatorSystem,
    # MultipleFrameRotationSystem,
    # SecondOrderHybridSystem,
    SymplecticUndampedHarmonicOscillatorCollectiveSystem,
    ScalarExponentialDampedHarmonicOscillatorCollectiveSystem,
)
from elastica.timestepper import integrate, extend_stepper_interface
from elastica.timestepper._stepper_interface import _TimeStepper

from elastica.timestepper.explicit_steppers import (
    RungeKutta4,
    ExplicitStepperTag,
    EulerForward,
)

# from elastica.timestepper.explicit_steppers import (
#     StatefulRungeKutta4,
#     StatefulEulerForward,
# )
from elastica.timestepper.symplectic_steppers import (
    PositionVerlet,
    PEFRL,
    SymplecticStepperTag,
)


from elastica.utils import Tolerance


class TestExtendStepperInterface:
    """TODO add documentation"""

    class MockSymplecticStepper:
        Tag = SymplecticStepperTag()

        def _first_prefactor(self):
            pass

        def _first_kinematic_step(self):
            pass

        def _first_dynamic_step(self):
            pass

    class MockExplicitStepper:
        Tag = ExplicitStepperTag()

        def _first_stage(self):
            pass

        def _first_update(self):
            pass

    from elastica.timestepper.symplectic_steppers import (
        _SystemInstanceStepper as symplectic_instance_stepper,
    )
    from elastica.timestepper.symplectic_steppers import (
        _SystemCollectionStepper as symplectic_collection_stepper,
    )

    from elastica.timestepper.explicit_steppers import (
        _SystemInstanceStepper as explicit_instance_stepper,
    )
    from elastica.timestepper.explicit_steppers import (
        _SystemCollectionStepper as explicit_collection_stepper,
    )

    # We cannot call a stepper on a system until both the stepper
    # and system "see" one another (for performance reasons, mostly)
    # So before "seeing" the system, the stepper should not have
    # the interface (interface_cls). It should however have the interface
    # after "seeing" the system, via extend_stepper_interface
    @pytest.mark.parametrize(
        "stepper_and_interface",
        [
            (MockSymplecticStepper, symplectic_instance_stepper),
            (MockExplicitStepper, explicit_instance_stepper),
        ],
    )
    def test_symplectic_stepper_interface_for_simple_systems(
        self, stepper_and_interface
    ):
        system = ScalarExponentialDecaySystem()
        (stepper_cls, interface_cls) = stepper_and_interface
        stepper = stepper_cls()

        stepper_methods = None
        assert stepper_methods is None

        _, stepper_methods = extend_stepper_interface(stepper, system)

        assert stepper_methods

    @pytest.mark.parametrize(
        "stepper_and_interface",
        [
            (MockSymplecticStepper, symplectic_collection_stepper),
            (MockExplicitStepper, explicit_collection_stepper),
        ],
    )
    def test_symplectic_stepper_interface_for_collective_systems(
        self, stepper_and_interface
    ):
        system = SymplecticUndampedHarmonicOscillatorCollectiveSystem()
        (stepper_cls, interface_cls) = stepper_and_interface
        stepper = stepper_cls()

        stepper_methods = None
        assert stepper_methods is None

        _, stepper_methods = extend_stepper_interface(stepper, system)

        assert stepper_methods

    class MockBadStepper:
        Tag = int()  # an arbitrary tag that doesn't mean anything

    @pytest.mark.parametrize(
        "stepper_and_interface", [(MockBadStepper, symplectic_collection_stepper)]
    )
    def test_symplectic_stepper_throws_for_bad_stepper(self, stepper_and_interface):
        system = ScalarExponentialDecaySystem()
        (stepper_cls, interface_cls) = stepper_and_interface
        stepper = stepper_cls()

        assert interface_cls not in stepper.__class__.__bases__

        with pytest.raises(NotImplementedError) as excinfo:
            extend_stepper_interface(stepper, system)
        assert "steppers are supported" in str(excinfo.value)


def test_integrate_throws_an_assert_for_negative_final_time():
    with pytest.raises(AssertionError) as excinfo:
        integrate([], [], -np.random.rand(1))
    assert "time is negative" in str(excinfo.value)


def test_integrate_throws_an_assert_for_negative_total_steps():
    with pytest.raises(AssertionError) as excinfo:
        integrate([], [], np.random.rand(1), -np.random.randint(100, 10000))
    assert "steps is negative" in str(excinfo.value)


# Added automatic discovery of Stateful explicit integrators
# ExplicitSteppers = StatefulExplicitStepper.__subclasses__()
# SymplecticSteppers = SymplecticStepper.__subclasses__()
# StatefulExplicitSteppers = [StatefulRungeKutta4, StatefulEulerForward]
ExplicitSteppers = [EulerForward, RungeKutta4]
SymplecticSteppers = [PositionVerlet, PEFRL]


class TestStepperInterface:
    def test_no_base_access_error(self):
        with pytest.raises(NotImplementedError) as excinfo:
            _TimeStepper().do_step()
        assert "not supposed to access" in str(excinfo.value)

    # @pytest.mark.parametrize("stepper", StatefulExplicitSteppers + SymplecticSteppers)
    # def test_correct_orders(self, stepper):
    #     assert stepper().n_stages > 0, "Explicit stepper routine has no stages!"


"""
class TestExplicitSteppers:
    @pytest.mark.parametrize("stepper", StatefulExplicitSteppers)
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

    @pytest.mark.parametrize("stepper", StatefulExplicitSteppers[:-1])
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

    @pytest.mark.parametrize("stepper", StatefulExplicitSteppers[:-1])
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

    @pytest.mark.parametrize("explicit_stepper", StatefulExplicitSteppers[:-1])
    def test_explicit_against_analytical_system(self, explicit_stepper):
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
"""


class TestSymplecticSteppers:
    @pytest.mark.parametrize("stepper", SymplecticSteppers)
    def test_symplectic_against_undamped_harmonic_oscillator(self, stepper):
        system = SymplecticUndampedSimpleHarmonicOscillatorSystem(
            omega=1.0 * np.pi, init_val=np.array([0.2, 0.8])
        )
        final_time = 4.0 * np.pi
        n_steps = 2000
        time_stepper = stepper()
        integrate(time_stepper, system, final_time=final_time, n_steps=n_steps)

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


"""
    @pytest.mark.xfail
    @pytest.mark.parametrize("symplectic_stepper", SymplecticSteppers)
    def test_hybrid_symplectic_against_analytical_system(self, symplectic_stepper):
        system = SecondOrderHybridSystem()
        final_time = 1.0
        n_steps = 2000
        # stepper = SymplecticCosseratRodStepper(symplectic_stepper=symplectic_stepper())
        stepper = symplectic_stepper()
        integrate(stepper, system, final_time=final_time, n_steps=n_steps)

        assert_allclose(
            system.final_solution(final_time),
            system.analytical_solution(final_time),
            rtol=Tolerance.rtol() * 1e2,
            atol=Tolerance.atol(),
        )
"""


class TestSteppersAgainstCollectiveSystems:
    """Test collection of memory blocks."""

    @pytest.mark.parametrize("symplectic_stepper", SymplecticSteppers)
    def test_symplectic_steppers(self, symplectic_stepper):
        collective_system = SymplecticUndampedHarmonicOscillatorCollectiveSystem()
        final_time = 1.0
        n_steps = 2000
        stepper = symplectic_stepper()
        integrate(stepper, collective_system, final_time=final_time, n_steps=n_steps)

        for system in collective_system:
            assert_allclose(
                *system.compute_energy(final_time),
                rtol=Tolerance.rtol() * 1e1,
                atol=Tolerance.atol(),
            )

    @pytest.mark.parametrize("explicit_stepper", ExplicitSteppers)
    def test_explicit_steppers(self, explicit_stepper):
        collective_system = ScalarExponentialDampedHarmonicOscillatorCollectiveSystem()
        final_time = 1.0
        if explicit_stepper == EulerForward:
            # Euler requires very small time-steps and in order not to slow down test,
            # we are scaling the difference between analytical and numerical solution.
            n_steps = 25000
            scale = 1e3
        else:
            n_steps = 500
            scale = 1

        stepper = explicit_stepper()

        dt = np.float64(float(final_time) / n_steps)
        time = np.float64(0.0)
        tol = Tolerance.atol()

        # Before stepping, let's extend the interface of the stepper
        # while providing memory slots
        from elastica.systems import make_memory_for_explicit_stepper

        memory_collection = make_memory_for_explicit_stepper(stepper, collective_system)
        from elastica.timestepper import extend_stepper_interface

        do_step, stagets_and_updates = extend_stepper_interface(
            stepper, collective_system
        )

        while np.abs(final_time - time) > 1e5 * tol:
            time = do_step(
                stepper,
                stagets_and_updates,
                collective_system,
                memory_collection,
                time,
                dt,
            )

        for system in collective_system:
            assert_allclose(
                system.state,
                system.analytical_solution(final_time),
                rtol=Tolerance.rtol() * scale,
                atol=Tolerance.atol() * scale,
            )

    # @pytest.mark.parametrize("symplectic_stepper", SymplecticSteppers)
    # def test_symplectic_against_collective_system(self, symplectic_stepper):


class TestSteppersAgainstRodLikeSystems:
    """The rods compose specific data-structures that
    act as an interface to timesteppers (see `rod/data_structures.py`)
    """

    # TODO : Figure out a way of integrating rods with explicit timesteppers
    # @pytest.mark.xfail
    # @pytest.mark.parametrize("explicit_stepper", StatefulExplicitSteppers[:-1])
    # def test_explicit_against_ellipse_motion(self, explicit_stepper):
    #     from elastica._systems._analytical import (
    #         SimpleSystemWithPositionsDirectors,
    #     )
    #
    #     rod_like_system = SimpleSystemWithPositionsDirectors(
    #         np.array([0.0, 0.0, 0.0]), np.random.randn(3, 3, 1)
    #     )
    #     final_time = 1.0
    #     n_steps = 500
    #     stepper = explicit_stepper()
    #
    #     integrate(stepper, rod_like_system, final_time=final_time, n_steps=n_steps)
    #
    #     assert_allclose(
    #         rod_like_system.position_collection,
    #         rod_like_system.analytical_solution("Positions", final_time),
    #         rtol=Tolerance.rtol() * 1e1,
    #         atol=Tolerance.atol(),
    #     )

    @pytest.mark.parametrize("symplectic_stepper", SymplecticSteppers)
    def test_symplectics_against_ellipse_motion(self, symplectic_stepper):
        from elastica.systems.analytical import (
            make_simple_system_with_positions_directors,
        )

        random_start_position = np.random.randn(3, 1)
        random_end_position = np.random.randn(3, 1)
        random_directors, _ = np.linalg.qr(np.random.randn(3, 3))
        random_directors = random_directors.reshape(3, 3, 1)

        rod_like_system = make_simple_system_with_positions_directors(
            random_start_position, random_end_position, random_directors
        )
        final_time = 1.0
        n_steps = 1000
        stepper = symplectic_stepper()

        integrate(stepper, rod_like_system, final_time=final_time, n_steps=n_steps)

        assert_allclose(
            rod_like_system.position_collection,
            rod_like_system.analytical_solution("Positions", final_time),
            rtol=Tolerance.rtol() * 1e1,
            atol=Tolerance.atol(),
        )

        assert_allclose(
            rod_like_system.velocity_collection,
            rod_like_system.analytical_solution("Velocity", final_time),
            rtol=Tolerance.rtol() * 1e1,
            atol=Tolerance.atol(),
        )

        # Reshaping done in the director collection to prevent numba from
        # complaining about returning multiple types
        assert_allclose(
            rod_like_system.director_collection.reshape(-1, 1),
            rod_like_system.analytical_solution("Directors", final_time),
            rtol=Tolerance.rtol() * 1e1,
            atol=Tolerance.atol(),
        )
