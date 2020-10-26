__doc__ = "Analytically integrable systems, used primarily for testing time-steppers of Elastica Numba implementation"

import numpy as np
import numba
from numba import jitclass, float64, uint64
from elastica._elastica_numba._rotations import _get_rotation_matrix, _rotate
from elastica._elastica_numba._linalg import _batch_matmul
from elastica._elastica_numba._rod._data_structures import _RodSymplecticStepperMixin
from elastica._elastica_numba._rod._data_structures import _bootstrap_from_data
from elastica.rod.data_structures import _KinematicState, _DynamicState


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
        return self._dyn_state.rate_collection

    @property
    def kinematic_states(self):
        return self._kin_state

    @kinematic_states.setter
    def kinematic_states(self, new_kin_state):
        self._kin_state.position_collection = new_kin_state.position_collection
        self._kin_state, director_collection = new_kin_state.director_collection
        self._kin_state.n_nodes = new_kin_state.n_nodes

    @property
    def dynamic_states(self):
        return self._dyn_state

    @dynamic_states.setter
    def dynamic_states(self, new_dyn_state):
        self._dyn_state.dynamic_states.rate_collection = new_dyn_state.rate_collection
        self._dyn_state.n_kinematic_rates = new_dyn_state.n_kinematic_rates


class TestKinematicState:
    def __init__(self, state):
        self.position_collection = np.zeros((3, state.shape[0]))
        self.director_collection = np.zeros((3, 3, state.shape[0]))
        # Expand rate vector in order to be consistent with time-stepper implementation
        for k in range(state.shape[0]):
            self.position_collection[:, k] = state[k]
            self.director_collection[:, :, k] = np.identity(3)
        self.n_nodes = state.shape[0]


class TestDynamicState:
    def __init__(self, state):
        blocksize = state.shape[0]
        self.rate_collection = np.zeros((3, blocksize + 1))
        # Expand rate vector in order to be consistent with time-stepper implementation
        for k in range(blocksize + 1):
            self.rate_collection[:, k] = state
        self.n_kinematic_rates = blocksize


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

    def __call__(self, time):
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
        self._kin_state = TestKinematicState(self._state[0:1])  # Create a view instead
        self._dyn_state = TestDynamicState(self._state[1:2])  # Create a view instead

    def dynamic_rates(self, time, *args, **kwargs):
        temp = super(SymplecticUndampedSimpleHarmonicOscillatorSystem, self).__call__(
            *args, **kwargs
        )[-1]
        # Expand rate vector in order to be consistent with time-stepper implementation
        blocksize = self._dyn_state.n_kinematic_rates
        rate = np.zeros((3, blocksize))
        for k in range(blocksize):
            rate[:, k] = temp

        return rate

    def compute_energy(self, time):
        # http://scipp.ucsc.edu/~haber/ph5B/sho09.pdf
        analytical_state = self.analytical_solution(time)

        def energy(st):
            return self.omega ** 2 * st[0] ** 2 + st[1] ** 2

        anal_energy = energy(analytical_state)
        current_energy = energy(self._state)
        return current_energy, anal_energy

    def update_internal_forces_and_torques(self, time):
        pass


# ks_spec = [("kin_state_view", float64[:])]
#
# @jitclass(ks_spec)
# class KinematicStateForAnalyticalSystems:
#     def __init__(self, kin_state_view):
#         self.kin_state_view = kin_state_view
#
#     def iadd(self, scaled_other_deriv):
#         self.kin_state_view += scaled_other_deriv
#         return self
#
# ds_spec = [("dyn_state_view", float64[:])]
#
# @jitclass(ds_spec)
# class DynamicStateForAnalyticalSystems:
#     def __init__(self, dyn_state_view):
#         self.dyn_state_view = dyn_state_view
#
#     def iadd(self, scaled_other_deriv):
#         self.dyn_state_view += scaled_other_deriv
#         return self
#
# # Spec for the base
# busho_spec = [
#     ("omega", float64),
#     ("initial_value", float64[:]),
#     ("_state", float64[:]),
#     ("A_matrix", float64[:, :]),
# ]
#
# susho_spec = [
#     (
#         "kinematic_states",
#         KinematicStateForAnalyticalSystems
#         if not hasattr(KinematicStateForAnalyticalSystems, "class_type")
#         else KinematicStateForAnalyticalSystems.class_type.instance_type,
#     ),
#     (
#         "dynamic_states",
#         DynamicStateForAnalyticalSystems
#         if not hasattr(DynamicStateForAnalyticalSystems, "class_type")
#         else DynamicStateForAnalyticalSystems.class_type.instance_type,
#     ),
# ]
#
# @jitclass(busho_spec + susho_spec)
# class SymplecticUndampedSimpleHarmonicOscillatorSystem(
#     BaseUndampedSimpleHarmonicOscillatorSystem
# ):
#     __init__BUSHO = BaseUndampedSimpleHarmonicOscillatorSystem.__init__
#
#     def __init__(self, omega=2.0 * np.pi, init_val=np.array([1.0, 0.0])):
#         self.__init__BUSHO(omega=omega, init_val=init_val)
#         self.kinematic_states = KinematicStateForAnalyticalSystems(
#             self._state[0:1]
#         )  # Create a view instead
#         self.dynamic_states = DynamicStateForAnalyticalSystems(
#             self._state[1:2]
#         )  # Create a view instead
#
#     def kinematic_rates(self, time):
#         return self._state[1:2]
#
#     def dynamic_rates(self, time):
#         return np.dot(self.A_matrix, self._state)[1]
#         # return BaseUndampedSimpleHarmonicOscillatorSystem.__call__(time)[-1]
#
#     def compute_energy(self, time):
#         # http://scipp.ucsc.edu/~haber/ph5B/sho09.pdf
#         analytical_state = self.analytical_solution(time)
#
#         def energy(st):
#             return self.omega ** 2 * st[0] ** 2 + st[1] ** 2
#
#         anal_energy = energy(analytical_state)
#         current_energy = energy(self._state)
#         return current_energy, anal_energy
#
#     def update_internal_forces_and_torques(self, time):
#         pass


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
        """"""
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


class CollectiveSystem:
    def __init__(self):
        self.systems = []

    def __getitem__(self, idx):
        return self.systems[idx]

    def __len__(self):
        return len(self.systems)

    def __iter__(self):
        return self.systems.__iter__()

    def synchronize(self, time):
        pass

    def constrain_values(self, time):
        pass

    def constrain_rates(self, time):
        pass

    def apply_callbacks(self, time, current_step: int):
        pass


class SymplecticUndampedHarmonicOscillatorCollectiveSystem(CollectiveSystem):
    def __init__(self):
        super(SymplecticUndampedHarmonicOscillatorCollectiveSystem, self).__init__()
        self.systems.append(
            SymplecticUndampedSimpleHarmonicOscillatorSystem(
                omega=2.0 * np.pi, init_val=np.array([1.0, 0.0])
            )
        )
        self.systems.append(
            SymplecticUndampedSimpleHarmonicOscillatorSystem(
                omega=1.0 * np.pi, init_val=np.array([0.0, 0.5])
            )
        )


class ScalarExponentialDampedHarmonicOscillatorCollectiveSystem(CollectiveSystem):
    def __init__(self):
        super(
            ScalarExponentialDampedHarmonicOscillatorCollectiveSystem, self
        ).__init__()
        self.systems.append(ScalarExponentialDecaySystem())
        self.systems.append(DampedSimpleHarmonicOscillatorSystem())


def make_simple_system_with_positions_directors(
    start_position, end_position, start_director
):
    return SimpleSystemWithPositionsDirectors(
        start_position, end_position, start_director
    )


class SimpleSystemWithPositionsDirectors(_RodSymplecticStepperMixin):
    def __init__(self, start_position, end_position, start_director):
        self.a = 0.5
        self.b = 1
        self.c = 2
        self.n_elems = 1
        self.init_pos = start_position.reshape(3, self.n_elems)
        # final_pos = init_pos + start_director[2, : , 0].reshape(3, self.n_elems) * self.a
        self.final_pos = end_position.reshape(3, self.n_elems)
        all_positions = np.hstack((self.init_pos, self.final_pos))
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

        # Givees position, director etc.
        super(SimpleSystemWithPositionsDirectors, self).__init__()

    def update_accelerations(self, time):
        np.copyto(self.acceleration_collection, -np.sin(np.pi * time))
        np.copyto(self.alpha_collection[2, ...], 0.1 * np.pi)

    def analytical_solution(self, type, time):
        if type == "Positions":
            analytical_solution = (
                np.hstack((self.init_pos, self.final_pos))
                + np.sin(np.pi * time) / np.pi ** 2
            )
        elif type == "Velocity":
            analytical_solution = (
                0.0 * np.hstack((self.init_pos, self.final_pos))
                + np.cos(np.pi * time) / np.pi
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


# mixin_spec = [
#     ("position_collection", float64[:, :]),
#     ("velocity_collection", float64[:, :]),
#     ("acceleration_collection", float64[:, :]),
#     ("director_collection", float64[:, :, :]),
#     ("omega_collection", float64[:, :]),
#     ("alpha_collection", float64[:, :]),
#     (
#         "kinematic_states",
#         _KinematicState.class_type.instance_type
#         if hasattr(_KinematicState, "class_type")
#         else _KinematicState,
#     ),
#     (
#         "dynamic_states",
#         _DynamicState.class_type.instance_type
#         if hasattr(_DynamicState, "class_type")
#         else _DynamicState,
#     ),
# ]
# sspd_spec = [
#     ("a", float64),
#     ("b", float64),
#     ("c", float64),
#     ("n_elems", uint64),
#     ("init_pos", float64[:, :]),
#     ("final_pos", float64[:, :]),
#     ("omega_value", float64),
#     ("_vector_states", float64[:, :]),
#     ("_matrix_states", float64[:, :, :]),
#     ("init_dir", float64[:, :, :]),
# ]
#
# def make_simple_system_with_positions_directors(
#     start_position, end_position, start_director
# ):
#     a = 0.5
#     b = 1
#     c = 2
#     n_elems = 1
#     init_pos = start_position.reshape(3, n_elems)
#     # final_pos = init_pos + start_director[2, : , 0].reshape(3, self.n_elems) * self.a
#     final_pos = end_position.reshape(3, n_elems)
#     all_positions = np.hstack((init_pos, final_pos))
#     velocities = 1.0 / np.pi + 0.0 * all_positions
#     accelerations = 0.0 * all_positions
#     omegas = 0.0 * init_pos
#     # For omega, don't start with exactly 0.0 as we divide by magnitude
#     # at the start of the get_rotate_matrix routine
#     omega_value = 1.0 * np.pi
#     omegas[2, ...] = omega_value
#     angular_accelerations = 0.0 * omegas
#     _vector_states = np.hstack(
#         (all_positions, velocities, omegas, accelerations, angular_accelerations,)
#     )
#     init_dir = start_director.copy()
#     _matrix_states = start_director.copy()
#
#     (
#         kinematic_states,
#         dynamic_states,
#         position_collection,
#         director_collection,
#         velocity_collection,
#         omega_collection,
#         acceleration_collection,
#         alpha_collection,  # angular acceleration
#     ) = _bootstrap_from_data("symplectic", n_elems, _vector_states, _matrix_states)
#
#     return SimpleSystemWithPositionsDirectors(
#         a,
#         b,
#         c,
#         n_elems,
#         init_pos,
#         final_pos,
#         omega_value,
#         _vector_states,
#         init_dir,
#         _matrix_states,
#         kinematic_states,
#         dynamic_states,
#         position_collection,
#         director_collection,
#         velocity_collection,
#         omega_collection,
#         acceleration_collection,
#         alpha_collection,
#     )
#
# @jitclass(mixin_spec + sspd_spec)
# class SimpleSystemWithPositionsDirectors:
#     def __init__(
#         self,
#         a,
#         b,
#         c,
#         n_elems,
#         init_pos,
#         final_pos,
#         omega_value,
#         _vector_states,
#         init_dir,
#         _matrix_states,
#         kinematic_states,
#         dynamic_states,
#         position_collection,
#         director_collection,
#         velocity_collection,
#         omega_collection,
#         acceleration_collection,
#         alpha_collection,
#     ):
#         self.a = a
#         self.b = b
#         self.c = c
#         self.n_elems = n_elems
#         self.init_pos = init_pos
#         # final_pos = init_pos + start_director[2, : , 0].reshape(3, self.n_elems) * self.a
#         self.final_pos = final_pos
#         # For omega, don't start with exactly 0.0 as we divide by magnitude
#         # at the start of the get_rotate_matrix routine
#         self.omega_value = omega_value
#         self._vector_states = _vector_states
#         self.init_dir = init_dir
#         self._matrix_states = _matrix_states
#
#         self.kinematic_states = kinematic_states
#         self.dynamic_states = dynamic_states
#         self.position_collection = position_collection
#         self.director_collection = director_collection
#         self.velocity_collection = velocity_collection
#         self.omega_collection = omega_collection
#         self.acceleration_collection = acceleration_collection
#         self.alpha_collection = alpha_collection
#
#     def kinematic_rates(self, time):
#         return self.dynamic_states.kinematic_rates(time)
#
#     def dynamic_rates(self, time):
#         self.update_accelerations(time)
#
#         """
#         The following commented block of code is a test to ensure that
#         the time-integrator always updates the view of the
#         collection variables, and not an independent variable
#         (aka no copy is made). It exists only for legacy
#         purposes and will be either refactored or removed once
#         testing is done.
#         """
#         # def shmem(x):
#         #     if np.shares_memory(
#         #             self.dynamic_states.rate_collection, x
#         #     ) : print("Shares memory")
#         #     else :
#         #         print("Explicit states does not share memory")
#         # shmem(self.velocity_collection)
#         # shmem(self.acceleration_collection)
#         # shmem(self.omega_collection)
#         # shmem(self.alpha_collection)
#         return self.dynamic_states.dynamic_rates(time)
#
#     def update_accelerations(self, time):
#         self.acceleration_collection[...] = -np.sin(np.pi * time)
#         self.alpha_collection[2, ...] = 0.1 * np.pi
#
#     def analytical_solution(self, type, time):
#         if type == "Positions":
#             analytical_solution = (
#                 np.hstack((self.init_pos, self.final_pos))
#                 + np.sin(np.pi * time) / np.pi ** 2
#             )
#         elif type == "Velocity":
#             analytical_solution = (
#                 0.0 * np.hstack((self.init_pos, self.final_pos))
#                 + np.cos(np.pi * time) / np.pi
#             )
#         elif type == "Directors":
#             final_angle = self.omega_value * time + 0.5 * 0.1 * np.pi * time ** 2
#             axis = np.array([0.0, 0.0, 1.0]).reshape(
#                 3, 1
#             )  # There is only one director
#             rot_mat = _get_rotation_matrix(final_angle, axis)
#             # Reshaping done to prevent numba from complaining about how we cannot merge
#             # array2D and array3D as return types
#             analytical_solution = _batch_matmul(rot_mat, self.init_dir).reshape(
#                 -1, 1
#             )
#             # analytical_solution = _rotate(self.init_dir, final_angle, axis)
#         return analytical_solution
