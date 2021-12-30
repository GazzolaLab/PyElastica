__doc__ = "Data structure wrapper for rod components"
__all__ = [
    "_RodSymplecticStepperMixin",
    "_bootstrap_from_data",
    "_State",
    "_DerivativeState",
    "_KinematicState",
    "_DynamicState",
]
import numpy as np
import numba
from numba import njit
from elastica._rotations import _get_rotation_matrix, _rotate
from elastica._linalg import _batch_matmul


# FIXME : Explicit Stepper doesn't work as States lose the
# views they initially had when working with a timestepper.
"""
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
            self.alpha_collection,  # angular acceleration
        ) = _bootstrap_from_data(
            "explicit", self.n_elems, self._vector_states, self._matrix_states
        )

    # def __setattr__(self, name, value):
    #     np.copy(self.__dict__[name], value)

    def __call__(self, time, *args, **kwargs):
        self.update_accelerations(time)  # Internal, external

        # print("KRC", self.state.kinematic_rate_collection)
        # print("DEr", self.__deriv_state.rate_collection)
        if np.shares_memory(
            self.state.kinematic_rate_collection,
            self.velocity_collection
            # self.__deriv_state.rate_collection
        ):
            print("Shares memory")
        else:
            print("Explicit states does not share memory")
        return self.__deriv_state
"""


class _RodSymplecticStepperMixin:
    def __init__(self):
        self.kinematic_states = _KinematicState(
            self.position_collection, self.director_collection
        )
        self.dynamic_states = _DynamicState(
            self.v_w_collection,
            self.dvdt_dwdt_collection,
            self.velocity_collection,
            self.omega_collection,
        )

        # Expose rate returning functions in the interface
        # to be used by the time-stepping algorithm
        # dynamic rates needs to call update_accelerations and henc
        # is another function
        self.kinematic_rates = self.dynamic_states.kinematic_rates

    def update_internal_forces_and_torques(self, time, *args, **kwargs):
        self.compute_internal_forces_and_torques(time)

    def dynamic_rates(self, time, prefac, *args, **kwargs):
        self.update_accelerations(time)
        return self.dynamic_states.dynamic_rates(time, prefac, *args, **kwargs)

    def reset_external_forces_and_torques(self, time, *args, **kwargs):
        self.zeroed_out_external_forces_and_torques(time)


def _bootstrap_from_data(stepper_type: str, n_elems: int, vector_states, matrix_states):
    """Returns states wrapping numpy arrays based on the time-stepping algorithm

    Convenience method that takes in rod internal (raw np.ndarray) data, create views
    (references) from it, and outputs State classes that are used in the time-stepping
    algorithm. This means that modifying the state modifies the internal data!

    Parameters
    ----------
    stepper_type : str (likely to change in future), representing stepper type
    Allowed parameters are ['explicit', 'symplectic']
    n_elems : int, number of rod elements
    vector_states : np.ndarray of shape (dim, *) with the following structure
        `vector_states` = [`position`,`velocity`,`omega`,`acceleration`,`angular acceleration`]
        `n_nodes = n_elems + 1`
        `position = 0 -> n_nodes , size = n_nodes`
        `velocity = n_nodes -> 2 * n_nodes, size = n_nodes`
        `omega = 2 * n_nodes -> 2 * n_nodes + nelem, size = nelem`
        `acceleration = 2 * n_nodes + nelem -> 3 * n_nodes + nelem, size = n_nodes`
        `angular acceleration = 3 * n_nodes + nelem -> 3 * n_nodes + 2 * nelem, size = n_elems`
    matrix_states : np.ndarray of shape (dim, dim, n_elems) containing the directors

    Returns
    -------
    output : tuple of len 8 containing
    (state, derivative_state, position, directors, velocity, omega, acceleration, alpha)
    derivative_state carries rate information

    """
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
        # TODO: Consider removing.
        # output += (
        #     _KinematicState(n_elems, position, directors),
        #     _DynamicState(n_elems, v_w_dvdt_dwdt),
        # )
        raise NotImplementedError
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
    """State for explicit steppers.

    Wraps data as state, with overloaded methods for explicit steppers
    (steppers that integrate all states in one-step/stage).
    Allows for separating implementation of stepper from actual
    addition/multiplication/other formulae used.
    """

    # TODO : args, kwargs instead of hardcoding types
    def __init__(
        self,
        n_elems: int,
        position_collection_view,
        director_collection_view,
        kinematic_rate_collection_view,
    ):
        """
        Parameters
        ----------
        n_elems : int, number of rod elements
        position_collection_view : view of positions (or) x
        director_collection_view : view of directors (or) Q
        kinematic_rate_collection_view : view of velocity and omega (or) (v,ω)
        """
        super(_State, self).__init__()
        self.n_nodes = n_elems + 1
        self.n_kinematic_rates = self.n_nodes + n_elems  # start of (v,ω) in (x,Q,v,ω)
        self.position_collection = position_collection_view
        self.director_collection = director_collection_view
        self.kinematic_rate_collection = kinematic_rate_collection_view

    def __iadd__(self, scaled_deriv_array):
        """overloaded += operator

        The add for directors is customized to reflect Rodrigues' rotation
        formula.

        Parameters
        ----------
        scaled_deriv_array : np.ndarray containing dt * (v, ω, dv/dt, dω/dt)
        ,as returned from _DerivativeState's __mul__ method

        Returns
        -------
        self : _State with inplace modified data

        """
        # x += v*dt
        self.position_collection += scaled_deriv_array[..., : self.n_nodes]
        # TODO : Verify the math in this note
        r"""
        Developer Note
        --------------
        Here the overloaded `+=` operator is exploited to perform
        matrix multiplication for the directors, which is counter-
        intutive at first. While this provides a stable interface
        to interact the rod states with the timesteppers and the
        rest of the world, the reasons behind including it here also has
        a depper mathematical significance.

        Firstly, position lies in the vector space corresponding to R^{3}
        and update is done this space (with the + and * operators defined
        as usual), hence the `+=` operator (or `__iadd__`) is reflected
        as `+=` operator in the position update (line 163 above).

        For directors rather, which lie in a restricteed R^{3} \otimes
        R^{3} tensorial space, the space with Q^T.Q = Q.Q^T = I, the +
        operator can be thought of as an equivalent `*=` update for a
        'exponential' multiplication with a rotation matrix (e^{At}).
        . This does not correspond to the position update. However, if
        we view this in a logarithmic space the `*=` becomse the '+='
        operator once again! After performing this `+=` operation, we
        bring it back into its original space using the exponential
        operator. So we are still indirectly doing the '+='
        update.

        To avoid all this hassle with the operators and spaces, we simply define
        '+=' or '__iadd__' in the case of directors as an equivalent
        '*=' (matrix multiply) with the RHS below.
        """
        # TODO Q *= exp(w*dt) , whats' the formua again?
        # TODO the scale factor 1.0 does not seem to be necessary, although
        # we perform more work in the present framework (muliply dt to entire vector, then take
        # norm) rather than vector norm then multiple by dt (1/3 operation costs)
        # TODO optimize (somehow) extra copy away : if we don't make a copy
        # its even more slower, maybe due to aliasing effects
        np.einsum(
            "ijk,jlk->ilk",
            _get_rotation_matrix(
                1.0, scaled_deriv_array[..., self.n_nodes : self.n_kinematic_rates]
            ),
            self.director_collection.copy(),
            out=self.director_collection,
        )
        # (v,ω) += (dv/dt, dω/dt)*dt
        self.kinematic_rate_collection += scaled_deriv_array[
            ..., self.n_kinematic_rates :
        ]
        return self

    def __add__(self, scaled_derivative_state):
        """overloaded + operator, useful in state.k1 = state + dt * deriv_state

        The add for directors is customized to reflect Rodrigues' rotation
        formula.

        Parameters
        ----------
        scaled_derivative_state : np.ndarray with dt * (v, ω, dv/dt, dω/dt)
        ,as returned from _DerivativeState's __mul__ method

        Returns
        -------
        state : new _State object with modified data (copied)

        Caveats
        -------
        Note that the argument is not a `other` _State object but is rather
        assumed to be a `np.ndarray` from calling _DerivativeState's __mul__
        method. This reflects the most common use-case in time-steppers

        """
        # x += v*dt
        position_collection = (
            self.position_collection + scaled_derivative_state[..., : self.n_nodes]
        )
        # Devs : see `_State.__iadd__` for reasons why we do matmul here
        director_collection = _rotate(
            self.director_collection,
            1.0,
            scaled_derivative_state[..., self.n_nodes : self.n_kinematic_rates],
        )
        # (v,ω) += (dv/dt, dω/dt)*dt
        kinematic_rate_collection = (
            self.kinematic_rate_collection
            + scaled_derivative_state[..., self.n_kinematic_rates :]
        )
        return _State(
            self.n_nodes - 1,
            position_collection,
            director_collection,
            kinematic_rate_collection,
        )


class _DerivativeState:
    """TimeDerivative of States for explicit steppers.

    Wraps time-derivative data as state, with overloaded methods for
    explicit steppers (steppers that integrate all states in one-step/stage).
    Allows for separating implementation of stepper from actual addition
    /multiplication used.
    """

    def __init__(self, _unused_n_elems: int, rate_collection_view):
        """
        Parameters
        ----------
        _unused_n_elems : int, number of elements (unused, kept for
        compatibility with `_bootstrap_from_data`)
        rate_collection_view : np.ndarray containing (v, ω, dv/dt, dω/dt)
        """
        super(_DerivativeState, self).__init__()
        self.rate_collection = rate_collection_view

    def __rmul__(self, scalar):
        """overloaded scalar * self,

        Parameters
        ----------
        scalar : float, typically dt (the time-step)

        Returns
        -------
        output : np.ndarray containing (v*dt, ω*dt, dv/dt*dt, dω/dt*dt)

        Caveats
        -------
        Returns a np.ndarray and not a State object (as one expects).
        Returning a State here with (v*dt, ω*dt, dv/dt*dt, dω/dt*dt) as members
        is possible but it's less efficient, especially because this is hot
        piece of code
        """
        """
        Developer Note
        --------------

        Q : Why do we need to overload operators here?

        The Derivative class naturally doesn't have a `mul` overloaded
        operator. That means if this method is not present,
        doing something like
        ```
        ds = _DerivativeState(...)
        new_state = 2 * ds
        ```
        will throw an error. Note that you can do something like
        ```
        ds = _DerivativeState(...)
        new_state = 2 * ds.rate_collection
        ```
        but this is hacky, as we are exposing the members outside,
        in the calling scope (defeats encapsulation and hiding).
        The point of having this class is that it works
        well with the time-stepper (where we only use `+` and `*`
        operations on the State/DerivativeState like above,
        i.e. `state = dt * derivative_state`  and not something like
        `state = dt * derivative_state.rate_collection`).
        It also provides an interface for anything outside
        the `Rod` system as a whole.
        """
        return scalar * self.rate_collection

    def __mul__(self, scalar):
        """overloaded self * scalar

        TODO Check if this pattern (forwarding to __mul__) has
        any disdvantages apart from extra function call penalty

        Parameters
        ----------
        scalar : float, typically dt (the time-step)

        Returns
        -------
        output : np.ndarray containing (v*dt, ω*dt, dv/dt*dt, dω/dt*dt)

        """
        return self.__rmul__(scalar)


"""
Symplectic stepper interface
"""


class _KinematicState:
    """State storing (x,Q) for symplectic steppers.
    Wraps data as state, with overloaded methods for symplectic steppers.
    Allows for separating implementation of stepper from actual
    addition/multiplication/other formulae used.

    Symplectic steppers rely only on in-place modifications to state and so
    only these methods are provided.
    """

    def __init__(self, position_collection_view, director_collection_view):
        """
        Parameters
        ----------
        position_collection_view : view of positions (or) x
        director_collection_view : view of directors (or) Q
        """
        # super(_KinematicState, self).__init__()

        self.position_collection = position_collection_view
        self.director_collection = director_collection_view


@njit(cache=True)
def overload_operator_kinematic_numba(
    n_nodes,
    prefac,
    position_collection,
    director_collection,
    velocity_collection,
    omega_collection,
):
    """overloaded += operator

    The add for directors is customized to reflect Rodrigues' rotation
    formula.
    Parameters
    ----------
    scaled_deriv_array : np.ndarray containing dt * (v, ω),
    as retured from _DynamicState's `kinematic_rates` method
    Returns
    -------
    self : _KinematicState instance with inplace modified data
    Caveats
    -------
    Takes a np.ndarray and not a _KinematicState object (as one expects).
    This is done for efficiency reasons, see _DynamicState's `kinematic_rates`
    method
    """
    # x += v*dt
    for i in range(3):
        for k in range(n_nodes):
            position_collection[i, k] += prefac * velocity_collection[i, k]
    rotation_matrix = _get_rotation_matrix(1.0, prefac * omega_collection)
    director_collection[:] = _batch_matmul(rotation_matrix, director_collection)

    return


class _DynamicState:
    """State storing (v,ω, dv/dt, dω/dt) for symplectic steppers.

    Wraps data as state, with overloaded methods for symplectic steppers.
    Allows for separating implementation of stepper from actual
    addition/multiplication/other formulae used.
    Symplectic steppers rely only on in-place modifications to state and so
    only these methods are provided.
    """

    def __init__(
        self,
        v_w_collection,
        dvdt_dwdt_collection,
        velocity_collection,
        omega_collection,
    ):
        """
        Parameters
        ----------
        n_elems : int, number of rod elements
        rate_collection_view : np.ndarray containing (v, ω, dv/dt, dω/dt)
        v_w_collection : numpy.ndarray

        """
        super(_DynamicState, self).__init__()
        # Limit at which (v, w) end
        # Create views for dynamic state
        self.rate_collection = v_w_collection
        self.dvdt_dwdt_collection = dvdt_dwdt_collection
        self.velocity_collection = velocity_collection
        self.omega_collection = omega_collection

    def kinematic_rates(self, time, *args, **kwargs):
        """Yields kinematic rates to interact with _KinematicState

        Returns
        -------
        v_and_omega : np.ndarray consisting of (v,ω)
        Caveats
        -------
        Doesn't return a _KinematicState with (dt*v, dt*w) as members,
        as one expects the _Kinematic __add__ operator to interact
        with another _KinematicState. This is done for efficiency purposes.
        """
        # RHS functino call, gives v,w so that
        # Comes from kin_state -> (x,Q) += dt * (v,w) <- First part of dyn_state
        return self.velocity_collection, self.omega_collection

    def dynamic_rates(self, time, prefac, *args, **kwargs):
        """Yields dynamic rates to add to with _DynamicState
        Returns
        -------
        acc_and_alpha : np.ndarray consisting of (dv/dt,dω/dt)
        Caveats
        -------
        Doesn't return a _DynamicState with (dt*v, dt*w) as members,
        as one expects the _Dynamic __add__ operator to interact
        with another _DynamicState. This is done for efficiency purposes.
        """
        return prefac * self.dvdt_dwdt_collection


@njit(cache=True)
def overload_operator_dynamic_numba(rate_collection, scaled_second_deriv_array):
    """overloaded += operator, updating dynamic_rates
    Parameters
    ----------
    scaled_second_deriv_array : np.ndarray containing dt * (dvdt, dωdt),
    as retured from _DynamicState's `dynamic_rates` method
    Returns
    -------
    self : _DynamicState instance with inplace modified data
    Caveats
    -------
    Takes a np.ndarray and not a _DynamicState object (as one expects).
    This is done for efficiency reasons, see `dynamic_rates`.
    """
    # Always goes in LHS : that means the update is on the rates alone
    # (v,ω) += dt * (dv/dt, dω/dt) ->  self.dynamic_rates
    # rate_collection[..., : n_kinematic_rates] += scaled_second_deriv_array
    blocksize = scaled_second_deriv_array.shape[1]

    for i in range(2):
        for k in range(blocksize):
            rate_collection[i, k] += scaled_second_deriv_array[i, k]

    return
