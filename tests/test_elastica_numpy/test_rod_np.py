__doc__ = """ Rod class for testing module in Elastica Numpy implementation"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from elastica._elastica_numpy._rod._data_structures import _bootstrap_from_data
from elastica.utils import MaxDimension


class TestRod:
    def __init__(self):
        bs = 32
        self.position_collection = np.random.randn(MaxDimension.value(), bs)
        self.director_collection = np.random.randn(
            MaxDimension.value(), MaxDimension.value(), bs
        )
        self.velocity_collection = np.random.randn(MaxDimension.value(), bs)
        self.omega_collection = np.random.randn(MaxDimension.value(), bs)
        self.mass = np.abs(np.random.randn(bs))
        self.external_forces = np.zeros(bs)


# Choosing 15 and 31 as nelems to reflect common expected
# use case of blocksize = 2*(k), k = int
# https://docs.pytest.org/en/latest/fixture.html
@pytest.fixture(scope="module", params=[15, 31])
def load_data_for_bootstrapping_state(request):
    """ Yield states for bootstrapping """
    n_elem = request.param
    n_nodes = n_elem + 1
    dim = 3

    position_collection = np.random.randn(dim, n_nodes)
    velocity_collection = np.random.randn(dim, n_nodes)
    acceleration_collection = np.random.randn(dim, n_nodes)
    omega_collection = np.random.randn(dim, n_elem)
    alpha_collection = np.random.randn(dim, n_elem)
    director_collection = np.random.randn(dim, dim, n_elem)

    vector_states = np.hstack(
        (
            position_collection,
            velocity_collection,
            omega_collection,
            acceleration_collection,
            alpha_collection,
        )
    )

    yield n_elem, vector_states, director_collection


@pytest.mark.parametrize("stepper_type", ["explicit", "symplectic"])
def test_bootstrapping_integrity(load_data_for_bootstrapping_state, stepper_type):
    (n_elem, vectors, directors) = load_data_for_bootstrapping_state
    all_states = _bootstrap_from_data(stepper_type, *load_data_for_bootstrapping_state)

    assert np.shares_memory(
        all_states[2], vectors
    ), "Integrity of bootstrapping from vector states compromised"
    assert np.shares_memory(
        all_states[3], directors
    ), "Integrity of bootstrapping from matrix states compromised"
    for state in all_states[4:]:
        assert np.shares_memory(
            state, vectors
        ), "Integrity of bootstrapping from vector states compromised"


def assert_instance(obj, cls):
    assert isinstance(obj, cls), "object is not a {} type".format(cls.__name__)


def test_bootstrapping_types_for_explicit_steppers(load_data_for_bootstrapping_state):
    all_states = _bootstrap_from_data("explicit", *load_data_for_bootstrapping_state)
    from elastica._elastica_numpy._rod._data_structures import _State, _DerivativeState

    assert_instance(all_states[0], _State)
    assert_instance(all_states[1], _DerivativeState)

    test_states = all_states[0]
    test_derivatives = all_states[1]

    assert np.shares_memory(
        test_states.kinematic_rate_collection, test_derivatives.rate_collection
    ), "Explicit states does not share memory"


def test_bootstrapping_types_for_symplectic_steppers(load_data_for_bootstrapping_state):
    all_states = _bootstrap_from_data("symplectic", *load_data_for_bootstrapping_state)
    from elastica._elastica_numpy._rod._data_structures import (
        _KinematicState,
        _DynamicState,
    )

    assert_instance(all_states[0], _KinematicState)
    assert_instance(all_states[1], _DynamicState)


# TODO Add realistic example with states used in a real time-stepper to solve some ODE
class LoadStates:
    """Mixin class for testing explicit and symplectic
    stepper behaviors that manipulate state objects
    """

    Vectors = None
    Directors = None
    States = {
        "Position": None,
        "Directors": None,
        "Velocity": None,
        "Omega": None,
        "Acceleration": None,
        "Alpha": None,
    }

    @pytest.fixture
    def load_states(self, load_data_for_bootstrapping_state):
        (n_elem, vectors, directors) = load_data_for_bootstrapping_state
        self.Vectors = vectors.copy()
        self.Directors = directors.copy()
        # Stepper Type found in base-classes TestExplicitStepperStateBehavior
        # and TestSymplecticStepperStateBehavior below.
        all_states = _bootstrap_from_data(
            self.StepperType, *load_data_for_bootstrapping_state
        )

        # Create copies of states (position, velocity etc) into
        # the states dictionary. We then do the SAME manipulation
        # of self.States and all_states to check their correctness
        for src_state, tgt_key in zip(all_states[2:], self.States):
            self.States[tgt_key] = src_state.copy()

        # all_states[0,1] here depends on the StepperType used above
        # if TestExplicitStepperStateBehavior.StepperType, then [_State, _DerivativeState]
        # if TestSymplecticStepperStateBehavior.StepperType, then [_KinematicState, _DynamicState]
        return all_states[0], all_states[1]


class TestExplicitStepperStateBehavior(LoadStates):
    # TODO : update tests after including Rodrigues rotation properly
    StepperType = "explicit"

    @pytest.mark.parametrize("mul_type", ["Pre multiply", "Post multiply"])
    def test_derivative_rmul(self, load_states, mul_type):
        _, derivative = load_states
        dt = np.random.randn()
        if mul_type == "Pre multiply":
            test_derivative = dt * derivative
        elif mul_type == "Post multiply":
            test_derivative = derivative * dt
        else:
            raise RuntimeError("Shouldn't be here")
        assert_instance(test_derivative, np.ndarray)
        correct_derivative = dt * self.Vectors
        assert np.all(np.in1d(test_derivative.ravel(), correct_derivative.ravel()))

    def test_state_add(self, load_states):
        state, derivative = load_states

        def func(x, y):
            return x + 1.0 * y

        test_state = func(state, derivative)

        from elastica._elastica_numpy._rod._data_structures import (
            _State,
            _DerivativeState,
        )

        assert_instance(test_state, _State)

        assert test_state.n_nodes == state.n_nodes, "Nodes unequal"
        assert_allclose(
            test_state.position_collection,
            func(self.States["Position"], self.States["Velocity"]),
        )
        # FIXME How to test directors again?
        assert np.all(
            np.in1d(
                func(self.States["Velocity"], self.States["Acceleration"]).ravel(),
                test_state.kinematic_rate_collection.ravel(),
            )
        )
        assert np.all(
            np.in1d(
                func(self.States["Omega"], self.States["Alpha"]).ravel(),
                test_state.kinematic_rate_collection.ravel(),
            )
        )

    def test_state_iadd(self, load_states):
        state, derivative = load_states

        scalar = 2.0

        def inplace_func(x, y):
            x.__iadd__(scalar * y)

        def func(x, y):
            return x + scalar * y

        inplace_func(state, derivative)

        assert_allclose(
            state.position_collection,
            func(self.States["Position"], self.States["Velocity"]),
        )
        assert np.all(
            np.in1d(
                func(self.States["Velocity"], self.States["Acceleration"]).ravel(),
                state.kinematic_rate_collection.ravel(),
            )
        )
        assert np.all(
            np.in1d(
                func(self.States["Omega"], self.States["Alpha"]).ravel(),
                state.kinematic_rate_collection.ravel(),
            )
        )


class TestSymplecticStepperStateBehavior(LoadStates):
    # TODO : update tests after including Rodrigues rotation properly
    StepperType = "symplectic"

    def test_dynamic_state_returns_correct_kinematic_rates(self, load_states):
        kin_state, dyn_state = load_states
        # 0.0 in the function parameter is time=0.0
        # numba complains if this is included
        assert np.all(
            np.in1d(
                self.States["Velocity"].ravel(), dyn_state.kinematic_rates(0.0).ravel()
            )
        )
        assert np.all(
            np.in1d(
                self.States["Omega"].ravel(), dyn_state.kinematic_rates(0.0).ravel()
            )
        )

    def test_dynamic_state_returns_correct_dynamic_rates(self, load_states):
        kin_state, dyn_state = load_states
        # 0.0 in the function parameter is time=0.0
        # numba complains if this is included
        assert np.all(
            np.in1d(
                self.States["Acceleration"].ravel(),
                dyn_state.dynamic_rates(0.0).ravel(),
            )
        )
        assert np.all(
            np.in1d(self.States["Alpha"].ravel(), dyn_state.dynamic_rates(0.0).ravel())
        )

    def test_dynamic_state_iadd(self, load_states):
        _, dyn_state = load_states
        scalar = 2.0

        def inplace_func(x, y):
            x.__iadd__(scalar * y.dynamic_rates(time=0.0))

        def func(x, y):
            return x + scalar * y

        inplace_func(dyn_state, dyn_state)

        temp = func(self.States["Velocity"], self.States["Acceleration"])
        assert np.all(np.in1d(temp.ravel(), dyn_state.rate_collection.ravel()))

        temp = func(self.States["Omega"], self.States["Alpha"])
        assert np.all(np.in1d(temp.ravel(), dyn_state.rate_collection.ravel()))

    def test_kinematic_state_iadd(self, load_states):
        kin_state, dyn_state = load_states
        scalar = 2.0

        def inplace_func(x, y):
            x.__iadd__(scalar * y.kinematic_rates(time=0.0))

        def func(x, y):
            return x + scalar * y

        inplace_func(kin_state, dyn_state)

        assert_allclose(
            kin_state.position_collection,
            func(self.States["Position"], self.States["Velocity"]),
        )
        # FIXME How to test directors?
