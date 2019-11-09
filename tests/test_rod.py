__doc__ = """ Rod class for testing module """

import pytest
import numpy as np
from numpy.testing import assert_allclose


class TestRod:
    def __init__(self):
        self.position = 0
        self.directors = 0
        self.velocity = 0
        self.omega = 0
        self.mass = 0
        self.external_forces = 0


from elastica.rod.data_structures import _bootstrap_from_data


@pytest.fixture(scope="module", params=[15, 31])
def load_data_for_bootstrapping_state(request):
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
    from elastica.rod.data_structures import _State, _DerivativeState

    assert_instance(all_states[0], _State)
    assert_instance(all_states[1], _DerivativeState)


def test_bootstrapping_types_for_symplectic_steppers(load_data_for_bootstrapping_state):
    all_states = _bootstrap_from_data("symplectic", *load_data_for_bootstrapping_state)
    from elastica.rod.data_structures import _KinematicState, _DynamicState

    assert_instance(all_states[0], _KinematicState)
    assert_instance(all_states[1], _DynamicState)


class LoadStates:
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
        all_states = _bootstrap_from_data(
            self.StepperType, *load_data_for_bootstrapping_state
        )

        for src_state, tgt_key in zip(all_states[2:], self.States):
            self.States[tgt_key] = src_state.copy()

        return all_states[0], all_states[1]


class TestExplicitStepperStateBehavior(LoadStates):
    StepperType = "explicit"

    def test_derivative_rmul(self, load_states):
        _, derivative = load_states
        dt = np.random.randn()
        test_derivative = dt * derivative
        assert_instance(test_derivative, np.ndarray)
        correct_derivative = dt * self.Vectors
        assert np.all(np.in1d(test_derivative, correct_derivative))

    def test_state_add(self, load_states):
        state, derivative = load_states
        func = lambda x, y: x + 1.0 * y
        test_state = func(state, derivative)

        from elastica.rod.data_structures import _State, _DerivativeState

        assert_instance(test_state, _State)

        assert_allclose(
            test_state.position_collection,
            func(self.States["Position"], self.States["Velocity"]),
        )
        assert np.all(
            np.in1d(
                func(self.States["Velocity"], self.States["Acceleration"]),
                test_state.kinematic_rate_collection,
            )
        )
        assert np.all(
            np.in1d(
                func(self.States["Omega"], self.States["Alpha"]),
                test_state.kinematic_rate_collection,
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
                func(self.States["Velocity"], self.States["Acceleration"]),
                state.kinematic_rate_collection,
            )
        )
        assert np.all(
            np.in1d(
                func(self.States["Omega"], self.States["Alpha"]),
                state.kinematic_rate_collection,
            )
        )
