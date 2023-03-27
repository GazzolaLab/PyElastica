__doc__ = """ Test rigid body data structures """
import numpy as np
import pytest
from numpy.testing import assert_allclose
from elastica.utils import Tolerance
from elastica.rigidbody.data_structures import _RigidRodSymplecticStepperMixin
from elastica._rotations import _rotate
from elastica.timestepper import (
    RungeKutta4,
    EulerForward,
    PEFRL,
    PositionVerlet,
    integrate,
)


def make_simple_system_with_positions_directors(start_position, start_director):
    return SimpleSystemWithPositionsDirectors(start_position, start_director)


class SimpleSystemWithPositionsDirectors(_RigidRodSymplecticStepperMixin):
    def __init__(self, start_position, start_director):
        self.a = 0.5
        self.b = 1
        self.c = 2
        self.n_elems = 1
        self.n_nodes = self.n_elems
        self.init_pos = start_position.reshape(3, self.n_elems)
        # final_pos = init_pos + start_director[2, : , 0].reshape(3, self.n_elems) * self.a
        # self.final_pos = end_position.reshape(3, self.n_elems)
        all_positions = self.init_pos.copy()
        velocities = 1.0 / np.pi + 0.0 * all_positions
        accelerations = 0.0 * all_positions
        omegas = 0.0 * self.init_pos
        # For omega, don't start with exactly 0.0 as we divide by magnitude
        # at the start of the get_rotate_matrix routine
        self.omega_value = 1.0 * np.pi
        omegas[2, ...] = self.omega_value
        angular_accelerations = 0.0 * omegas
        self._vector_states = np.hstack(
            (all_positions, velocities, omegas, accelerations, angular_accelerations)
        )
        self.init_dir = start_director.copy()
        self._matrix_states = start_director.copy()

        self.v_w_collection = np.zeros((2, 3, 1))
        self.dvdt_dwdt_collection = np.zeros((2, 3, 1))
        self.velocity_collection = np.ndarray.view(self.v_w_collection[0, :, :])
        self.omega_collection = np.ndarray.view(self.v_w_collection[1, :, :])
        self.acceleration_collection = np.ndarray.view(
            self.dvdt_dwdt_collection[0, :, :]
        )
        self.alpha_collection = np.ndarray.view(self.dvdt_dwdt_collection[1, :, :])

        self.position_collection = self.init_pos
        self.velocity_collection[:] = velocities
        self.omega_collection[:] = omegas
        self.acceleration_collection[:] = accelerations
        self.alpha_collection[:] = angular_accelerations
        self.director_collection = start_director
        self.external_forces = np.zeros((3, 1))
        self.external_torques = np.zeros((3, 1))

        # Givees position, director etc.
        super(SimpleSystemWithPositionsDirectors, self).__init__()

    def _compute_internal_forces_and_torques(self, time):
        pass

    def update_accelerations(self, time):
        np.copyto(self.acceleration_collection, -np.sin(np.pi * time))
        np.copyto(self.alpha_collection[2, ...], 0.1 * np.pi)

    def analytical_solution(self, type, time):
        if type == "Positions":
            analytical_solution = (
                np.hstack((self.init_pos)) + np.sin(np.pi * time) / np.pi ** 2
            )
        elif type == "Velocity":
            analytical_solution = (
                0.0 * np.hstack((self.init_pos)) + np.cos(np.pi * time) / np.pi
            )
        elif type == "Directors":
            final_angle = self.omega_value * time + 0.5 * 0.1 * np.pi * time ** 2
            axis = np.array([0.0, 0.0, 1.0]).reshape(3, 1)  # There is only one director
            # Reshaping done to prevent numba equivalent to complain
            # While we can prevent it here, its' done to make the front end testing scripts "look"
            # nicer and cleaner
            analytical_solution = _rotate(self.init_dir, final_angle, axis).reshape(
                -1, 1
            )
        return analytical_solution


ExplicitSteppers = [EulerForward, RungeKutta4]
SymplecticSteppers = [PositionVerlet, PEFRL]


class TestSteppersAgainstRigidBodyLikeSystems:
    """
    The rigid body compose specific data-structures that
    act as an interface to timesteppers (see `rigid_body/data_structures.py`)
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
    def test_symplectics_against_ellipse_motion_with_numba(self, symplectic_stepper):

        random_start_position = np.random.randn(3, 1)
        random_directors, _ = np.linalg.qr(np.random.randn(3, 3))
        random_directors = random_directors.reshape(3, 3, 1)

        rod_like_system = make_simple_system_with_positions_directors(
            random_start_position, random_directors
        )
        final_time = 1.0
        n_steps = 1000
        stepper = symplectic_stepper()

        integrate(stepper, rod_like_system, final_time=final_time, n_steps=n_steps)

        assert_allclose(
            rod_like_system.position_collection.reshape(3),
            rod_like_system.analytical_solution("Positions", final_time),
            rtol=Tolerance.rtol() * 1e1,
            atol=Tolerance.atol(),
        )

        assert_allclose(
            rod_like_system.velocity_collection.reshape(3),
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


# TODO: We are dropping Numpy code so remove the below code
#     def test_symplectics_against_ellipse_motion_with_numpy_PositionVerlet(
#         self, monkeypatch
#     ):
#         monkeypatch.setenv("IMPORT_TEST_NUMPY", "True", prepend=False)
#         # After changing the import flag reload the modules.
#         importlib.reload(elastica)
#         importlib.reload(elastica.timestepper.symplectic_steppers)
#         # importlib.reload(elastica.timestepper.integrate)
#         importlib.reload(elastica.timestepper)
#         from elastica.timestepper.symplectic_steppers import PositionVerlet
#         from elastica.timestepper import integrate
#
#         random_start_position = np.random.randn(3, 1)
#         random_directors, _ = np.linalg.qr(np.random.randn(3, 3))
#         random_directors = random_directors.reshape(3, 3, 1)
#
#         rod_like_system = make_simple_system_with_positions_directors(
#             random_start_position, random_directors
#         )
#         final_time = 1.0
#         n_steps = 1000
#         stepper = PositionVerlet()
#
#         integrate(stepper, rod_like_system, final_time=final_time, n_steps=n_steps)
#
#         assert_allclose(
#             rod_like_system.position_collection.reshape(3),
#             rod_like_system.analytical_solution("Positions", final_time),
#             rtol=Tolerance.rtol() * 1e1,
#             atol=Tolerance.atol(),
#         )
#
#         assert_allclose(
#             rod_like_system.velocity_collection.reshape(3),
#             rod_like_system.analytical_solution("Velocity", final_time),
#             rtol=Tolerance.rtol() * 1e1,
#             atol=Tolerance.atol(),
#         )
#
#         # Reshaping done in the director collection to prevent numba from
#         # complaining about returning multiple types
#         assert_allclose(
#             rod_like_system.director_collection.reshape(-1, 1),
#             rod_like_system.analytical_solution("Directors", final_time),
#             rtol=Tolerance.rtol() * 1e1,
#             atol=Tolerance.atol(),
#         )
#
#         # Remove the import flag
#         monkeypatch.delenv("IMPORT_TEST_NUMPY")
#         # Reload the elastica after changing flag
#         importlib.reload(elastica)
#         importlib.reload(elastica.timestepper.symplectic_steppers)
#         importlib.reload(elastica.timestepper)
#         from elastica.timestepper.symplectic_steppers import PositionVerlet
#         from elastica.timestepper import integrate
#
#     def test_symplectics_against_ellipse_motion_with_numpy_PEFRL(self, monkeypatch):
#         monkeypatch.setenv("IMPORT_TEST_NUMPY", "True", prepend=False)
#         # After changing the import flag reload the modules.
#         importlib.reload(elastica)
#         importlib.reload(elastica.timestepper.symplectic_steppers)
#         # importlib.reload(elastica.timestepper.integrate)
#         importlib.reload(elastica.timestepper)
#         from elastica.timestepper.symplectic_steppers import PEFRL
#         from elastica.timestepper import integrate
#
#         random_start_position = np.random.randn(3, 1)
#         random_directors, _ = np.linalg.qr(np.random.randn(3, 3))
#         random_directors = random_directors.reshape(3, 3, 1)
#
#         rod_like_system = make_simple_system_with_positions_directors(
#             random_start_position, random_directors
#         )
#         final_time = 1.0
#         n_steps = 1000
#         stepper = PEFRL()
#
#         integrate(stepper, rod_like_system, final_time=final_time, n_steps=n_steps)
#
#         assert_allclose(
#             rod_like_system.position_collection.reshape(3),
#             rod_like_system.analytical_solution("Positions", final_time),
#             rtol=Tolerance.rtol() * 1e1,
#             atol=Tolerance.atol(),
#         )
#
#         assert_allclose(
#             rod_like_system.velocity_collection.reshape(3),
#             rod_like_system.analytical_solution("Velocity", final_time),
#             rtol=Tolerance.rtol() * 1e1,
#             atol=Tolerance.atol(),
#         )
#
#         # Reshaping done in the director collection to prevent numba from
#         # complaining about returning multiple types
#         assert_allclose(
#             rod_like_system.director_collection.reshape(-1, 1),
#             rod_like_system.analytical_solution("Directors", final_time),
#             rtol=Tolerance.rtol() * 1e1,
#             atol=Tolerance.atol(),
#         )
#
#         # Remove the import flag
#         monkeypatch.delenv("IMPORT_TEST_NUMPY")
#         # Reload the elastica after changing flag
#         importlib.reload(elastica)
#         importlib.reload(elastica.timestepper.symplectic_steppers)
#         importlib.reload(elastica.timestepper)
#         from elastica.timestepper.symplectic_steppers import PEFRL
#         from elastica.timestepper import integrate
