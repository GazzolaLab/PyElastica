__doc__ = """ Test modules for constraints """
import numpy as np
from numpy.testing import assert_allclose
import pytest

from elastica.modules import Constraints
from elastica.modules.constraints import _Constraint


class TestConstraint:
    @pytest.fixture(scope="function")
    def load_constraint(self, request):
        return _Constraint(100)  # This is the id for some reason

    @pytest.mark.parametrize("illegal_constraint", [int, list])
    def test_using_with_illegal_constraint_throws_assertion_error(
        self, load_constraint, illegal_constraint
    ):
        with pytest.raises(AssertionError) as excinfo:
            load_constraint.using(illegal_constraint)
        assert "not a valid constraint" in str(excinfo.value)

    from elastica.boundary_conditions import FreeBC, OneEndFixedBC, HelicalBucklingBC
    from elastica.boundary_conditions import FreeBC as TestBC

    @pytest.mark.parametrize(
        "legal_constraint", [FreeBC, OneEndFixedBC, HelicalBucklingBC]
    )
    def test_using_with_legal_constraint(self, load_constraint, legal_constraint):
        constraint = load_constraint
        constraint.using(legal_constraint, 3, 4.0, "5", k=1, l_var="2", j=3.0)

        assert constraint._bc_cls == legal_constraint
        assert constraint._args == (3, 4.0, "5")
        assert constraint._kwargs == {"k": 1, "l_var": "2", "j": 3.0}

    def test_id(self, load_constraint):
        # This is purely for coverage purposes, no actual test
        # since its a simple return
        assert load_constraint.id() == 100

    def test_call_without_setting_constraint_throws_runtime_error(
        self, load_constraint
    ):
        constraint = load_constraint

        with pytest.raises(RuntimeError) as excinfo:
            constraint(None)  # None is the rod/system parameter
        assert "No boundary condition" in str(excinfo.value)

    def test_call_without_position_director_kwargs(self, load_constraint):
        def mock_init(self, *args, **kwargs):
            self.dummy_one = args[0]
            self.k = kwargs.get("k")

        # in place class
        MockBC = type("MockBC", (self.TestBC, object), {"__init__": mock_init})

        constraint = load_constraint
        constraint.using(MockBC, 3.9, 4.0, "5", k=1, l_var="2", j=3.0)

        # Actual test is here, this should not throw
        mock_bc = constraint(None)  # None is Fake rod

        # More tests reinforcing the first
        assert mock_bc.dummy_one == 3.9
        assert mock_bc.k == 1

    class MockRod:
        def __init__(self):
            self.position_collection = np.random.randn(3, 8)
            self.director_collection = np.random.randn(3, 3, 7)

    @pytest.mark.parametrize("position_indices", [(0,), (1, 2), (0, 3, 6)])
    def test_call_with_positions_kwargs(self, load_constraint, position_indices):
        def mock_init(self, *args, **kwargs):
            self.start_pos = args
            self.k = kwargs.get("k")

        # in place class
        MockBC = type("MockBC", (self.TestBC, object), {"__init__": mock_init})

        constraint = load_constraint
        constraint.using(
            MockBC,
            3.9,
            4.0,
            constrained_position_idx=position_indices,
            k=1,
            l_var="2",
            j=3.0,
        )

        # Actual test is here, this should not throw
        mock_rod = self.MockRod()
        mock_bc = constraint(mock_rod)

        # More tests reinforcing the first
        for pos_idx_in_rod, pos_idx_in_bc in zip(position_indices, range(3)):
            assert_allclose(
                mock_rod.position_collection[..., pos_idx_in_rod],
                mock_bc.start_pos[pos_idx_in_bc],
            )
        assert mock_bc.k == 1

    @pytest.mark.parametrize("director_indices", [(4,), (0, 3), (0, 1, 5)])
    def test_call_with_directors_kwargs(self, load_constraint, director_indices):
        def mock_init(self, *args, **kwargs):
            self.start_pos = args
            self.k = kwargs.get("k")

        # in place class
        MockBC = type("MockBC", (self.TestBC, object), {"__init__": mock_init})

        constraint = load_constraint
        constraint.using(
            MockBC,
            3.9,
            4.0,
            constrained_director_idx=director_indices,
            k=1,
            l_var="2",
            j=3.0,
        )

        # Actual test is here, this should not throw
        mock_rod = self.MockRod()
        mock_bc = constraint(mock_rod)

        # More tests reinforcing the first
        for dir_idx_in_rod, dir_idx_in_bc in zip(director_indices, range(3)):
            assert_allclose(
                mock_rod.director_collection[..., dir_idx_in_rod],
                mock_bc.start_pos[dir_idx_in_bc],
            )
        assert mock_bc.k == 1

    @pytest.mark.parametrize("dof_indices", [(4,), (0, 3), (0, 1, 5)])
    def test_call_with_positions_and_directors_kwargs(
        self, load_constraint, dof_indices
    ):
        def mock_init(self, *args, **kwargs):
            self.start_pos = args
            self.k = kwargs.get("k")

        # in place class
        MockBC = type("MockBC", (self.TestBC, object), {"__init__": mock_init})

        constraint = load_constraint
        constraint.using(
            MockBC,
            3.9,
            4.0,
            constrained_position_idx=dof_indices,
            constrained_director_idx=dof_indices,
            k=1,
            l_var="2",
            j=3.0,
        )

        # Actual test is here, this should not throw
        mock_rod = self.MockRod()
        mock_bc = constraint(mock_rod)

        # More tests reinforcing the first
        pos_dir_offset = len(dof_indices)
        for dof_idx_in_rod, dof_idx_in_bc in zip(dof_indices, range(3)):
            assert_allclose(
                mock_rod.position_collection[..., dof_idx_in_rod],
                mock_bc.start_pos[dof_idx_in_bc],
            )
            assert_allclose(
                mock_rod.director_collection[..., dof_idx_in_rod],
                mock_bc.start_pos[dof_idx_in_bc + pos_dir_offset],
            )
        assert mock_bc.k == 1

    @pytest.mark.parametrize("dof_indices", [(4,), (0, 3)])
    def test_call_improper_bc_throws_type_error(self, load_constraint, dof_indices):
        # Example of bad initiailization function
        # Init should always have rod-based arguments first
        # For example
        # def __init__(self, pos_one, pos_two, dir_one, *args)
        def mock_init(self, nu, **kwargs):
            self.nu = nu
            self.k = kwargs.get("k")

        # in place class
        MockBC = type("MockBC", (self.TestBC, object), {"__init__": mock_init})

        constraint = load_constraint
        constraint.using(
            MockBC,
            4.0,
            constrained_position_idx=dof_indices,
            constrained_director_idx=dof_indices,
            k=1,
            l_var="2",
            j=3.0,
        )  # The user thinks 4.0 goes to nu, but we don't accept it

        mock_rod = self.MockRod()
        # Actual test is here, this should not throw
        with pytest.raises(TypeError) as excinfo:
            _ = constraint(mock_rod)
        assert "Unable to construct" in str(excinfo.value)


class TestConstraintsMixin:
    from elastica.modules import BaseSystemCollection

    class SystemCollectionWithConstraintsMixedin(BaseSystemCollection, Constraints):
        pass

    # TODO fix link after new PR
    from elastica.rod import RodBase

    class MockRod(RodBase):
        def __init__(self, *args, **kwargs):
            pass

    @pytest.fixture(scope="function", params=[2, 10])
    def load_system_with_constraints(self, request):
        n_sys = request.param
        sys_coll_with_constraints = self.SystemCollectionWithConstraintsMixedin()
        for i_sys in range(n_sys):
            sys_coll_with_constraints.append(self.MockRod(2, 3, 4, 5))
        return sys_coll_with_constraints

    """ The following calls test _get_sys_idx_if_valid from BaseSystem indirectly,
    and are here because of legacy reasons. I have not removed them because there
    are Connections require testing against multiple indices, which is still use
    ful to cross-verify against.

    START
    """

    def test_constrain_with_illegal_index_throws(self, load_system_with_constraints):
        scwc = load_system_with_constraints

        with pytest.raises(AssertionError) as excinfo:
            scwc.constrain(100)
        assert "exceeds number of" in str(excinfo.value)

        with pytest.raises(AssertionError) as excinfo:
            scwc.constrain(np.int_(100))
        assert "exceeds number of" in str(excinfo.value)

    def test_constrain_with_unregistered_system_throws(
        self, load_system_with_constraints
    ):
        scwc = load_system_with_constraints

        # Don't register this rod
        mock_rod = self.MockRod(2, 3, 4, 5)

        with pytest.raises(ValueError) as excinfo:
            scwc.constrain(mock_rod)
        assert "was not found, did you" in str(excinfo.value)

    def test_constrain_with_illegal_system_throws(self, load_system_with_constraints):
        scwc = load_system_with_constraints

        # Not a rod, but a list!
        mock_rod = [1, 2, 3, 5]

        with pytest.raises(TypeError) as excinfo:
            scwc.constrain(mock_rod)
        assert "not a system" in str(excinfo.value)

    """
    END of testing BaseSystem calls
    """

    def test_constrain_registers_and_returns_Constraint(
        self, load_system_with_constraints
    ):
        scwc = load_system_with_constraints

        mock_rod = self.MockRod(2, 3, 4, 5)
        scwc.append(mock_rod)

        _mock_constraint = scwc.constrain(mock_rod)
        assert _mock_constraint in scwc._constraints
        assert _mock_constraint.__class__ == _Constraint

    from elastica.boundary_conditions import ConstraintBase

    @pytest.fixture
    def load_rod_with_constraints(self, load_system_with_constraints):
        scwc = load_system_with_constraints

        mock_rod = self.MockRod(2, 3, 4, 5)
        scwc.append(mock_rod)

        # in place class
        class MockBC(self.ConstraintBase):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def constrain_values(self, *args, **kwargs) -> None:
                pass

            def constrain_rates(self, *args, **kwargs) -> None:
                pass

        # MockBC = type("MockBC", (self.TestBC, object), {"__init__": mock_init})

        # Constrain any and all systems
        scwc.constrain(1).using(MockBC, 2, 42)  # index based constraint
        scwc.constrain(0).using(MockBC, 1, 2)  # index based constraint
        scwc.constrain(mock_rod).using(MockBC, 2, 3)  # system based constraint

        return scwc, MockBC

    def test_constrain_finalize_correctness(self, load_rod_with_constraints):
        scwc, bc_cls = load_rod_with_constraints

        scwc._finalize_constraints()

        for (x, y) in scwc._constraints:
            assert type(x) is int
            assert type(y) is bc_cls

    def test_constraint_properties(self, load_rod_with_constraints):
        scwc, _ = load_rod_with_constraints
        scwc._finalize_constraints()

        for i in [0, 1, -1]:
            x, y = scwc._constraints[i]
            mock_rod = scwc._systems[i]
            # Test system
            assert type(x) is int
            assert type(y.system) is type(mock_rod)
            assert y.system is mock_rod, f"{len(scwc._systems)}"
            # Test node indices
            assert y.constrained_position_idx.size == 0
            # Test element indices. TODO: maybe add more generalized test
            assert y.constrained_director_idx.size == 0

    @pytest.mark.xfail
    def test_constrain_finalize_sorted(self, load_rod_with_constraints):
        scwc, bc_cls = load_rod_with_constraints

        scwc._finalize_constraints()

        # this is allowed to fail (not critical)
        num = -np.inf
        for (x, _) in scwc._constraints:
            assert num < x
            num = x

    def test_constrain_call_on_systems(self):
        # TODO Finish after the architecture is complete
        pass
