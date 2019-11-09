__doc__ = "Data structure wrapper for rod components"

import numpy as np


class _RodExplicitStepperMixin:
    def __init__(self):
        (
            self.state,
            self.__deriv_state,
            self.position_collection,
            self.director_collection,
            self.velocity_collection,
            self.omega_collection,
            self.acceleration_collection,
            self.alpha_collection,
        ) = _bootstrap_from_data(
            "explicit", self.n_elems, self._vector_states, self._matrix_states
        )

    def __call__(self, *args, **kwargs):
        self.update_accelerations()  # Internal, external
        return self.__deriv_state


class _RodSymplecticStepperMixin:
    def __init__(self):
        (
            self.kinematic_states,
            self.dynamic_states,
            self.position_collection,
            self.director_collection,
            self.velocity_collection,
            self.omega_collection,
            self.acceleration_collection,
            self.alpha_collection,
        ) = _bootstrap_from_data(
            "symplectic", self.n_elems, self._vector_states, self._matrix_states
        )
        self.kinematic_rates = self.dynamic_states.kinematic_rates
        self.dynamic_rates = self.dynamic_states.dynamic_rates


def _bootstrap_from_data(stepper_type: str, n_elems: int, vector_states, matrix_states):
    n_nodes = n_elems + 1
    position = np.ndarray.view(vector_states[..., :n_nodes])
    directors = np.ndarray.view(matrix_states)
    v_w_dvdt_dwdt = np.ndarray.view(vector_states[..., n_nodes:])
    output = ()
    if stepper_type == "explicit":
        v_w_states = np.ndarray.view(vector_states[..., n_nodes : 3 * n_nodes - 1])
        output += (
            _State(n_elems, position, directors, v_w_states),
            _DerivativeState(n_elems, v_w_dvdt_dwdt),
        )
    elif stepper_type == "symplectic":
        output += (
            _KinematicState(n_elems, position, directors),
            _DynamicState(n_elems, v_w_dvdt_dwdt),
        )
    else:
        return

    n_velocity_end = n_nodes + n_nodes
    velocity = np.ndarray.view(vector_states[..., n_nodes:n_velocity_end])

    n_omega_end = n_velocity_end + n_elems
    omega = np.ndarray.view(vector_states[..., n_velocity_end:n_omega_end])

    n_acceleration_end = n_omega_end + n_nodes
    acceleration = np.ndarray.view(vector_states[..., n_omega_end:n_acceleration_end])

    n_alpha_end = n_acceleration_end + n_elems
    alpha = np.ndarray.view(vector_states[..., n_acceleration_end:n_alpha_end])

    return output + (position, directors, velocity, omega, acceleration, alpha)


"""
Explicit stepper interface
"""


class _State:
    """
    Wraps state. Explicit stepper.
    """

    # TODO : args, kwargs instead of hardcoding types
    def __init__(
        self,
        n_elems,
        position_collection_view,
        director_collection_view,
        kinematic_rate_collection_view,
    ):
        super(_State, self).__init__()
        self.n_nodes = n_elems + 1
        self.n_kinematic_rates = self.n_nodes + n_elems
        self.position_collection = position_collection_view  # x
        self.kinematic_rate_collection = kinematic_rate_collection_view  # v, w
        self.director_collection = director_collection_view  # Q

    def __iadd__(self, scaled_deriv_array):
        # scaled_deriv_array : dt * (v, w, dv_dt, dw_dt)
        # v
        self.position_collection += scaled_deriv_array[..., : self.n_nodes]
        # w
        self.director_collection += scaled_deriv_array[
            ..., self.n_nodes : self.n_kinematic_rates
        ]
        # dv_dt, dw_dt
        self.kinematic_rate_collection += scaled_deriv_array[
            ..., self.n_kinematic_rates :
        ]
        return self

    def __add__(self, scaled_derivative_state):
        # scaled_derivative_state : (v, w, dv_dt, dw_dt)
        # v
        position_collection = (
            self.position_collection + scaled_derivative_state[..., : self.n_nodes]
        )
        # w
        director_collection = (
            self.director_collection
            + scaled_derivative_state[..., self.n_nodes : self.n_kinematic_rates]
        )
        # dv_dt, dw_dt
        kinematic_rate_collection = (
            self.kinematic_rate_collection
            + scaled_derivative_state[..., self.n_kinematic_rates :]
        )
        return _State(
            self.n_nodes,
            position_collection,
            director_collection,
            kinematic_rate_collection,
        )


class _DerivativeState:
    def __init__(self, n_elems, rate_collection_view):
        super(_DerivativeState, self).__init__()
        # x, v, w, dvdt, dwdt -> v, w, dvdt, dwdt
        # self.vector_states = np.ndarray.view(rate_collection_view[..., n_nodes: ])
        self.rate_collection = rate_collection_view

    def __rmul__(self, scalar):
        """ normal mul at start"""
        # We can return a State here with (v*dt, w*dt, dvdt*dt, dwdt*dt) as members
        # but it's less efficient
        return scalar * self.rate_collection

    def __mul__(self, scalar):
        return self.__rmul__(scalar)


"""
Symplectic stepper interface
"""


class _KinematicState:
    def __init__(self, n_elems, position_collection_view, director_collection_view):
        super(_KinematicState, self).__init__()
        self.n_nodes = n_elems + 1
        # self.x = np.ndarray.view(vector_states[..., :n_nodes])
        # self.Q = np.ndarray.view(matrix_states)
        self.position_collection = position_collection_view  # x
        self.director_collection = director_collection_view  # Q

    def __iadd__(self, scaled_deriv_array):
        # scaled_derivative_state : dt * (v, w)
        self.position_collection += scaled_deriv_array[..., : self.n_nodes]
        self.director_collection += scaled_deriv_array[..., self.n_nodes :]
        return self


class _DynamicState:
    def __init__(self, n_elems, rate_collection_view):
        super(_DynamicState, self).__init__()
        # self.dynamic_states_limit = 2 * n_nodes - 1
        self.n_kinematic_rates = 2 * n_elems + 1
        # self.other_states = np.ndarray.view(vector_states[..., n_nodes:])
        self.rate_collection = rate_collection_view  # v, w, dvdt, dwdt

    def __iadd__(self, scaled_second_deriv_array):
        # Always goes in LHS : that means the update is on the rates alone
        # (v,w) += dt * (dv/dt, dw/dt) ->  self.dynamic_rates
        self.other_states[..., : self.n_kinematic_rates] += scaled_second_deriv_array
        return self

    def kinematic_rates(self):
        # RHS functino call, gives v,w so that
        #   Comes from kin_state -> (x,Q) += dt * (v,w) <- First part of dyn_state
        # We can return a _KinematicState with (dt*v, dt*w) as members, but its
        # less efficient
        return self.rate_collection[..., : self.n_kinematic_rates]

    def dynamic_rates(self):
        # We can return a _DynamicState with (dt*dvdt, dt*dwdt) as members, but its
        # less efficient
        return self.rate_collection[..., self.n_kinematic_rates :]
