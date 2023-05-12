__doc__ = """ Rod class for testing module """

import pytest
import numpy as np

# from elastica.utils import MaxDimension
from numpy.testing import assert_allclose

from elastica.rod.data_structures import _bootstrap_from_data
from elastica.rod.data_structures import (
    _KinematicState,
    _DynamicState,
)
from elastica.memory_block.memory_block_rod_base import (
    make_block_memory_periodic_boundary_metadata,
)
from elastica.utils import MaxDimension


class MockTestRod:
    def __init__(self):
        self.n_elems = 32
        self.ring_rod_flag = False
        self.position_collection = np.random.randn(
            MaxDimension.value(), self.n_elems + 1
        )
        self.director_collection = np.random.randn(
            MaxDimension.value(), MaxDimension.value(), self.n_elems
        )
        self.velocity_collection = np.random.randn(
            MaxDimension.value(), self.n_elems + 1
        )
        self.omega_collection = np.random.randn(MaxDimension.value(), self.n_elems)
        self.mass = np.abs(np.random.randn(self.n_elems + 1))
        self.external_forces = np.zeros(self.n_elems + 1)


class MockTestRingRod:
    def __init__(self):
        self.n_elems = 32
        self.ring_rod_flag = True
        self.position_collection = np.random.randn(MaxDimension.value(), self.n_elems)
        self.director_collection = np.random.randn(
            MaxDimension.value(), MaxDimension.value(), self.n_elems
        )
        self.velocity_collection = np.random.randn(MaxDimension.value(), self.n_elems)
        self.omega_collection = np.random.randn(MaxDimension.value(), self.n_elems)
        self.mass = np.abs(np.random.randn(self.n_elems))
        self.external_forces = np.zeros(self.n_elems)

        n_elems_ring_rods = (np.ones(1) * (self.n_elems - 3)).astype("int64")

        (
            _,
            self.periodic_boundary_nodes_idx,
            self.periodic_boundary_elems_idx,
            self.periodic_boundary_voronoi_idx,
        ) = make_block_memory_periodic_boundary_metadata(n_elems_ring_rods)


# Choosing 15 and 31 as nelems to reflect common expected
# use case of blocksize = 2*(k), k = int
# https://docs.pytest.org/en/latest/fixture.html
@pytest.fixture(scope="module", params=[15, 31])
def load_data_for_bootstrapping_state(request):
    """Yield states for bootstrapping"""
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


@pytest.mark.parametrize("stepper_type", ["explicit"])
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

    test_states = all_states[0]
    test_derivatives = all_states[1]

    assert np.shares_memory(
        test_states.kinematic_rate_collection, test_derivatives.rate_collection
    ), "Explicit states does not share memory"


@pytest.mark.xfail
def test_bootstrapping_types_for_symplectic_steppers(load_data_for_bootstrapping_state):
    """For block structure we drop the boot strap from data function. Thus this test fails."""
    all_states = _bootstrap_from_data("symplectic", *load_data_for_bootstrapping_state)
    from elastica.rod.data_structures import (
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

        from elastica.rod.data_structures import _State

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


# Choosing 15 and 31 as nelems to reflect common expected
# use case of blocksize = 2*(k), k = int
# https://docs.pytest.org/en/latest/fixture.html
@pytest.fixture(scope="module", params=[15, 31])
def load_data_for_symplectic_stepper(request):
    """Creates data for symplectic stepper classes _KinematicState and _DynamicStata"""
    n_elem = request.param
    n_nodes = n_elem + 1
    dim = 3

    position_collection = np.random.randn(dim, n_nodes)
    velocity_collection = np.random.randn(dim, n_nodes)
    acceleration_collection = np.random.randn(dim, n_nodes)
    omega_collection = np.random.randn(dim, n_elem)
    alpha_collection = np.random.randn(dim, n_elem)
    director_collection = np.random.randn(dim, dim, n_elem)

    v_w_collection = np.zeros((2, dim * n_nodes))
    v_w_collection[0] = velocity_collection.reshape(dim * n_nodes)
    # Stack extra zeros to make dimensions match
    v_w_collection[1] = np.hstack(
        (omega_collection.reshape(dim * n_elem), np.zeros((3)))
    )

    dvdt_dwdt_collection = np.zeros((2, dim * n_nodes))
    dvdt_dwdt_collection[0] = acceleration_collection.reshape(dim * n_nodes)
    # Stack extra zeros to make dimensions match
    dvdt_dwdt_collection[1] = np.hstack(
        (alpha_collection.reshape(dim * n_elem), np.zeros((3)))
    )

    yield n_elem, position_collection, director_collection, v_w_collection, dvdt_dwdt_collection, velocity_collection, omega_collection


class LoadStatesForSymplecticStepper:
    """Mixin class for testing  symplectic
    stepper behaviors that manipulate state objects.
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
    def load_states(self, load_data_for_symplectic_stepper):
        dim = 3
        (
            n_elem,
            position_collection,
            director_collection,
            v_w_collection,
            dvdt_dwdt_collection,
            velocity_collection,
            omega_collection,
        ) = load_data_for_symplectic_stepper

        kinematic_states = _KinematicState(position_collection, director_collection)
        dynamic_states = _DynamicState(
            v_w_collection, dvdt_dwdt_collection, velocity_collection, omega_collection
        )

        self.States["Position"] = position_collection.copy()
        self.States["Directors"] = director_collection.copy()
        self.States["Velocity"] = velocity_collection.copy()
        self.States["Omega"] = omega_collection.copy()
        self.States["Acceleration"] = (
            dvdt_dwdt_collection[0].reshape(dim, n_elem + 1).copy()
        )
        self.States["Alpha"] = (
            dvdt_dwdt_collection[1, 0 : dim * n_elem].reshape(dim, n_elem).copy()
        )

        return kinematic_states, dynamic_states, n_elem


class TestSymplecticStepperStateBehavior(LoadStatesForSymplecticStepper):
    """This test case is changed for the block structure implementation and it is only testing
    symplectic steppers.
    """

    # TODO : update tests after including Rodrigues rotation properly
    StepperType = "symplectic"

    def test_dynamic_state_returns_correct_kinematic_rates(self, load_states):
        kin_state, dyn_state, _ = load_states
        # 0.0 in the function parameter is time=0.0
        # numba complains if this is included
        assert np.all(
            np.in1d(self.States["Velocity"].ravel(), dyn_state.kinematic_rates(0.0)[0])
        )
        assert np.all(
            np.in1d(self.States["Omega"].ravel(), dyn_state.kinematic_rates(0.0)[1])
        )

    def test_dynamic_state_returns_correct_dynamic_rates(self, load_states):
        kin_state, dyn_state, _ = load_states
        # 0.0 in the function parameter is time=0.0
        # numba complains if this is included
        assert np.all(
            np.in1d(
                self.States["Acceleration"].ravel(),
                dyn_state.dynamic_rates(0.0, 1.0).ravel(),
            )
        )
        assert np.all(
            np.in1d(
                self.States["Alpha"].ravel(), dyn_state.dynamic_rates(0.0, 1.0).ravel()
            )
        )

    def test_dynamic_state_iadd(self, load_states):
        from elastica.rod.data_structures import (
            overload_operator_dynamic_numba,
        )

        _, dyn_state, _ = load_states
        scalar = 2.0

        def inplace_func(x, y):
            # x.iadd(scalar * y.dynamic_rates(0.0)) # USE THIS WITH JITCLASS
            overload_operator_dynamic_numba(
                x.rate_collection, y.dynamic_rates(0.0, scalar)
            )

        def func(x, y):
            return x + scalar * y

        inplace_func(dyn_state, dyn_state)

        temp = func(self.States["Velocity"], self.States["Acceleration"])
        assert np.all(np.in1d(temp.ravel(), dyn_state.rate_collection.ravel()))

        temp = func(self.States["Omega"], self.States["Alpha"])
        assert np.all(np.in1d(temp.ravel(), dyn_state.rate_collection.ravel()))

    def test_kinematic_state_iadd(self, load_states):
        from elastica.rod.data_structures import (
            overload_operator_kinematic_numba,
        )

        kin_state, dyn_state, n_elem = load_states
        scalar = 2.0

        def inplace_func(x, y):
            # x.iadd(scalar * y.kinematic_rates(0.0)) # USE THIS WITH JITCLASS
            overload_operator_kinematic_numba(
                n_elem + 1,
                scalar,
                x.position_collection,
                x.director_collection,
                y.velocity_collection,
                y.omega_collection,
            )

        def func(x, y):
            return x + scalar * y

        inplace_func(kin_state, dyn_state)

        assert_allclose(
            kin_state.position_collection,
            func(self.States["Position"], self.States["Velocity"]),
        )
        # FIXME How to test directors?
