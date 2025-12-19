__doc__ = """Testing timesteppers in Elastica Numba implementation """
import numpy as np
import pytest
from numpy.testing import assert_allclose

from elastica.timestepper import integrate, extend_stepper_interface
from elastica.timestepper.symplectic_steppers import (
    PositionVerlet,
    PEFRL,
    SymplecticStepperMixin,
)
from elastica.utils import Tolerance

from tests.analytical import (
    ScalarExponentialDecaySystem,
    # UndampedSimpleHarmonicOscillatorSystem,
    SymplecticUndampedSimpleHarmonicOscillatorSystem,
    # DampedSimpleHarmonicOscillatorSystem,
    # MultipleFrameRotationSystem,
    # SecondOrderHybridSystem,
    SymplecticUndampedHarmonicOscillatorCollectiveSystem,
    ScalarExponentialDampedHarmonicOscillatorCollectiveSystem,
    make_simple_system_with_positions_directors,
)


class TestExtendStepperInterface:
    """TODO add documentation"""

    class MockSymplecticStepper(SymplecticStepperMixin):

        def get_steps(self):
            return [self._kinematic_step, self._dynamic_step, self._kinematic_step]

        def get_prefactors(self):
            return [self._prefactor, self._prefactor]

        def _prefactor(self):
            pass

        def _kinematic_step(self):
            pass

        def _dynamic_step(self):
            pass

    # We cannot call a stepper on a system until both the stepper
    # and system "see" one another (for performance reasons, mostly)
    # So before "seeing" the system, the stepper should not have
    # the interface (interface_cls). It should however have the interface
    # after "seeing" the system, via extend_stepper_interface
    @pytest.mark.parametrize(
        "stepper_module",
        [
            MockSymplecticStepper,
        ],
    )
    def test_symplectic_stepper_interface_for_simple_systems(self, stepper_module):
        system = ScalarExponentialDecaySystem()
        stepper = stepper_module()

        stepper_methods = None
        _, stepper_methods = extend_stepper_interface(stepper, system)

        assert stepper_methods

    @pytest.mark.parametrize(
        "stepper_module",
        [MockSymplecticStepper],
    )
    def test_symplectic_stepper_interface_for_collective_systems(self, stepper_module):
        system = SymplecticUndampedHarmonicOscillatorCollectiveSystem()
        stepper = stepper_module()

        stepper_methods = None
        _, stepper_methods = extend_stepper_interface(stepper, system)

        assert stepper_methods == stepper.steps_and_prefactors

    class MockBadStepper:
        pass

    @pytest.mark.parametrize("stepper_module", [MockBadStepper])
    def test_symplectic_stepper_throws_for_bad_stepper(self, stepper_module):
        system = SymplecticUndampedHarmonicOscillatorCollectiveSystem()
        stepper = stepper_module()

        with pytest.raises(NotImplementedError) as excinfo:
            extend_stepper_interface(stepper, system)
        assert "stepper is not supported" in str(excinfo.value)


def test_integrate_throws_an_assert_for_negative_final_time(rng):
    with pytest.raises(AssertionError) as excinfo:
        integrate([], [], -rng.random(1))
    assert "time is negative" in str(excinfo.value)


def test_integrate_throws_an_assert_for_negative_total_steps(rng):
    with pytest.raises(AssertionError) as excinfo:
        integrate([], [], rng.random(1), -rng.randint(100, 10000))
    assert "steps is negative" in str(excinfo.value)


# SymplecticSteppers = SymplecticStepper.__subclasses__()
SymplecticSteppers = [PositionVerlet, PEFRL]


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


class TestSteppersAgainstRodLikeSystems:
    """The rods compose specific data-structures that
    act as an interface to timesteppers (see `rod/data_structures.py`)
    """

    # TODO : Figure out a way of integrating rods with explicit timesteppers
    # @pytest.mark.xfail
    # @pytest.mark.parametrize("explicit_stepper", StatefulExplicitSteppers[:-1])
    # def test_explicit_against_ellipse_motion(self, explicit_stepper, rng):
    #     from elastica._systems._analytical import (
    #         SimpleSystemWithPositionsDirectors,
    #     )
    #
    #     rod_like_system = SimpleSystemWithPositionsDirectors(
    #         np.array([0.0, 0.0, 0.0]), rng.normal((3, 3, 1))
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
    def test_symplectics_against_ellipse_motion(self, symplectic_stepper, rng):

        random_start_position = rng.standard_normal((3, 1))
        random_end_position = rng.standard_normal((3, 1))
        random_directors, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        random_directors = random_directors.reshape(3, 3, 1)

        rod_like_system = make_simple_system_with_positions_directors(
            random_start_position, random_end_position, random_directors
        )
        final_time = 1.0
        n_steps = 1000
        dt = final_time / n_steps

        stepper = symplectic_stepper()

        time = 0.0
        for _ in range(n_steps):
            time = stepper.step_single_instance(rod_like_system, time, dt)

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
