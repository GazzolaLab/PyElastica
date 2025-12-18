__doc__ = """ Test modules for base systems """

import pytest
import warnings
import numpy as np

from elastica.modules import (
    BaseSystemCollection,
    Constraints,
    Forcing,
    Connections,
    CallBacks,
)


class TestBaseSystemCollection:
    @pytest.mark.parametrize("illegal_type", [int, list, tuple])
    def test_check_type_with_illegal_type_throws(self, illegal_type):
        bsc = BaseSystemCollection()

        with pytest.raises(TypeError) as excinfo:
            # None is the rod/system parameter
            bsc._check_type(illegal_type())
        assert "not a system" in str(excinfo.value)

    @pytest.fixture(scope="class")
    def load_collection(self):
        rng = np.random.default_rng(42)  # Fixed seed for test reproducibility

        bsc = BaseSystemCollection()
        bsc.extend_allowed_types((int, float, str))
        bsc.append_allowed_types(np.ndarray)
        # Bypass check, but its fine for testing
        bsc.append(3)
        bsc.append(5.0)
        bsc.append("a")
        bsc.append(rng.standard_normal((3, 5)))
        return bsc

    def test_len(self, load_collection):
        assert len(load_collection) == 4

    def test_getitem(self, load_collection):
        assert load_collection[0] == 3
        assert load_collection[2] == "a"

    @pytest.mark.xfail
    def test_getitem_with_faulty_index_fails(self, load_collection):
        # Fails and exception is raised
        load_collection[100]

    @pytest.fixture(scope="function")
    def mock_rod(self):
        from elastica.rod import RodBase

        MockRod = type("MockRod", (RodBase,), {})
        return MockRod()

    def test_setitem(self, load_collection, mock_rod):
        # If this fails, an exception is raised
        # and pytest automatically fails
        load_collection[3] = mock_rod

    @pytest.mark.xfail
    def test_setitem_with_faulty_index_fails(self, load_collection, mock_rod):
        # If this fails, an exception is raised
        # and pytest automatically fails
        load_collection[200] = mock_rod

    def test_insert(self, load_collection, mock_rod):
        load_collection.insert(10, mock_rod)

    def test_str(self, load_collection):
        assert str(load_collection)[1] == "3"

    def test_extend_allowed_types(self, load_collection):
        bsc = load_collection

        from elastica.rod import RodBase
        from elastica.rigidbody import RigidBodyBase
        from elastica.systems.protocol import StaticSystemBase, SystemProtocol

        # Types are extended in the fixture
        assert int in bsc.allowed_sys_types
        assert float in bsc.allowed_sys_types
        assert str in bsc.allowed_sys_types
        assert np.ndarray in bsc.allowed_sys_types
        assert StaticSystemBase in bsc.allowed_sys_types  # Minimal requirement

    def test_extend_correctness(self, load_collection):
        """
        The last test adds types to the load_collection
        object, as the scope is class. Here we test the
        type_check attribute again and see if
        no exception is raised
        """
        bsc = load_collection
        bsc._check_type(2)  # an int object
        bsc._check_type(3.0)  # a float object
        bsc._check_type("whats the point of doing a PhD?")  # a str object

    def test_override_allowed_types(self, load_collection, mock_rod):
        bsc = load_collection
        bsc._override_allowed_types((int, float, str))

        # First check that adding a rod object throws an
        # error as we have replaced rods now it
        with pytest.raises(TypeError) as excinfo:
            # None is the rod/system parameter
            bsc._check_type(mock_rod)
        assert "not a system" in str(excinfo.value)

    def test_override_correctness(self, load_collection):
        bsc = load_collection
        # then see if int, float and str are okay.
        bsc._check_type(2)  # an int object
        bsc._check_type(3.0)  # a float object
        bsc._check_type("whats the point of doing a PhD?")  # a str object

    def test_invalid_idx_in_get_sys_index_throws(self, load_collection):
        from elastica.rod import RodBase

        bsc = load_collection
        bsc._override_allowed_types((RodBase,))
        with pytest.raises(AssertionError) as excinfo:
            bsc.get_system_index(100)
        assert "exceeds number of" in str(excinfo.value)

        with pytest.raises(AssertionError) as excinfo:
            load_collection.get_system_index(np.int32(100))
        assert "exceeds number of" in str(excinfo.value)

    def test_unregistered_system_in_get_sys_index_throws(
        self, load_collection, mock_rod
    ):
        # Don't register this rod
        my_mock_rod = mock_rod

        with pytest.raises(ValueError) as excinfo:
            load_collection.get_system_index(my_mock_rod)
        assert "was not found, did you" in str(excinfo.value)

    def test_get_sys_index_returns_correct_idx(self, load_collection):
        assert load_collection.get_system_index(1) == 1

    def test_duplicate_system_warning(self):
        """Test that adding the same system instance twice emits a warning."""
        bsc = BaseSystemCollection()
        bsc.extend_allowed_types((int,))

        # Add a system
        test_system = 42
        bsc.append(test_system)

        # Try to add the same system again - should emit warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bsc._check_type(test_system)

            # Verify warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "already in the system collection" in str(w[0].message)
            assert "not recommended" in str(w[0].message)

    def test_duplicate_system_warning_with_rod(self, mock_rod):
        """Test that adding the same rod instance twice emits a warning."""
        bsc = BaseSystemCollection()
        from elastica.rod import RodBase

        bsc.extend_allowed_types((RodBase,))

        # Add a rod
        bsc.append(mock_rod)

        # Try to add the same rod again - should emit warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bsc.append(mock_rod)

            # Verify warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "already in the system collection" in str(w[0].message)
            assert "not recommended" in str(w[0].message)

    @pytest.mark.xfail
    def test_delitem(self, load_collection):
        del load_collection[0]
        assert load_collection[0] == 3

    def test_requisite_modules_error(self):
        """Test that RuntimeError is raised when system requires modules not present."""

        class Collection(BaseSystemCollection):
            pass

        bsc = Collection()

        # Create a mock system class that requires Constraints module
        class SystemWithRequisiteModules:
            REQUISITE_MODULES = [int]  # Require int module

        system = SystemWithRequisiteModules()
        bsc.append_allowed_types(
            SystemWithRequisiteModules,
        )

        # Should raise RuntimeError because BaseSystemCollection doesn't have Constraints
        # The type check passes (SystemWithRequisiteModules is in allowed_sys_types),
        # but REQUISITE_MODULES check fails
        with pytest.raises(RuntimeError) as excinfo:
            bsc._check_type(system)
        assert "requires the following modules" in str(excinfo.value)
        assert "int" in str(excinfo.value)

    def test_requisite_modules_success(self):
        """Test that system with REQUISITE_MODULES passes when modules are present."""

        # Create a simulator with Constraints module
        class SimulatorInt(BaseSystemCollection, int):
            pass

        bsc = SimulatorInt()

        # Create a mock system class that requires Constraints module
        class SystemWithRequisiteModules:
            REQUISITE_MODULES = [int]

        system = SystemWithRequisiteModules()
        bsc.append_allowed_types(
            SystemWithRequisiteModules,
        )

        # Should pass because BaseSystemCollection has Constraints
        assert bsc._check_type(system) is True

    def test_enable_block_supports_new_system_type(self):
        """Test enable_block_supports when system_type is not in any block_supports (else clause)."""
        from elastica.rod.cosserat_rod import CosseratRod

        class CustomBlock:
            pass

        class DerivedRod(CosseratRod):
            def __init__(self):
                pass

        derived_rod = DerivedRod()

        bsc = BaseSystemCollection()

        # Initially, CustomRod should not be in block_supports
        found = False
        for block_type in bsc._block_supports.values():
            if derived_rod in block_type:
                found = True
                break
        assert not found, "CustomRod should not be in block_supports initially"

        # Enable block support for CustomRod (else clause - creates new entry)
        bsc.enable_block_supports(derived_rod, CustomBlock)
        assert derived_rod in bsc._block_supports[CustomBlock]

    def test_enable_block_supports_existing_system_type(self):
        """Test enable_block_supports when system_type is already in block_supports (if branch)."""
        from elastica.memory_block.memory_block_rod import MemoryBlockCosseratRod
        from elastica.rod.cosserat_rod import CosseratRod

        class CustomBlock:
            pass

        bsc = BaseSystemCollection()

        # CosseratRod should already be in block_supports (set in __init__)
        assert CosseratRod in bsc._block_supports[MemoryBlockCosseratRod]

        # Get the initial count
        bsc.enable_block_supports(CosseratRod, MemoryBlockCosseratRod)
        assert bsc._block_supports[MemoryBlockCosseratRod].count(CosseratRod) == 1

        # Switch block support
        bsc.enable_block_supports(CosseratRod, CustomBlock)
        assert bsc._block_supports[MemoryBlockCosseratRod].count(CosseratRod) == 0
        assert bsc._block_supports[CustomBlock].count(CosseratRod) == 1

        # Create no duplicates
        bsc.enable_block_supports(CosseratRod, CustomBlock)
        assert bsc._block_supports[MemoryBlockCosseratRod].count(CosseratRod) == 0
        assert bsc._block_supports[CustomBlock].count(CosseratRod) == 1

        # Switch block support back
        bsc.enable_block_supports(CosseratRod, MemoryBlockCosseratRod)
        assert bsc._block_supports[MemoryBlockCosseratRod].count(CosseratRod) == 1
        assert bsc._block_supports[CustomBlock].count(CosseratRod) == 0


class GenericSimulatorClass(
    BaseSystemCollection, Constraints, Forcing, Connections, CallBacks
):
    pass


class TestBaseSystemWithFeaturesUsingCosseratRod:
    @pytest.fixture(scope="function")
    def load_collection(self):
        sc = GenericSimulatorClass()
        from elastica.rod.cosserat_rod import CosseratRod

        # rod = RodBase()
        rod = CosseratRod.straight_rod(
            n_elements=10,
            start=np.zeros((3)),
            direction=np.array([0, 1, 0.0]),
            normal=np.array([1, 0, 0.0]),
            base_length=1,
            base_radius=1,
            density=1,
            youngs_modulus=1,
        )
        # Bypass check, but its fine for testing
        sc.append(rod)

        return sc, rod

    from elastica.boundary_conditions import FreeBC

    @pytest.mark.parametrize("legal_constraint", [FreeBC])
    def test_constraint(self, load_collection, legal_constraint):
        simulator_class, rod = load_collection
        simulator_class.constrain(rod).using(legal_constraint)
        simulator_class.finalize()
        # After finalize check if the created constrain object is instance of the class we have given.
        assert isinstance(
            simulator_class._feature_group_constrain_values._operator_collection[-1][
                -1
            ].func.__self__,
            legal_constraint,
        )
        assert isinstance(
            simulator_class._feature_group_constrain_rates._operator_collection[-1][
                -1
            ].func.__self__,
            legal_constraint,
        )

        # TODO: this is a dummy test for constrain values and rates find a better way to test them
        simulator_class.constrain_values(time=0)
        simulator_class.constrain_rates(time=0)

    from elastica.external_forces import NoForces

    @pytest.mark.parametrize("legal_forces", [NoForces])
    def test_forcing(self, load_collection, legal_forces):
        simulator_class, rod = load_collection
        simulator_class.add_forcing_to(rod).using(legal_forces)
        simulator_class.finalize()
        # After finalize check if the created forcing object is instance of the class we have given.
        assert isinstance(
            simulator_class._feature_group_synchronize._operator_collection[-1][
                -1
            ].func.__self__,
            legal_forces,
        )
        assert isinstance(
            simulator_class._feature_group_synchronize._operator_collection[-1][
                -2
            ].func.__self__,
            legal_forces,
        )

        # TODO: this is a dummy test for synchronize find a better way to test them
        simulator_class.synchronize(time=0)

    from elastica.callback_functions import CallBackBaseClass

    @pytest.mark.parametrize("legal_callback", [CallBackBaseClass])
    def test_callback(self, mocker, load_collection, legal_callback):
        simulator_class, rod = load_collection

        spy = mocker.spy(legal_callback, "make_callback")

        simulator_class.collect_diagnostics(rod).using(legal_callback)
        simulator_class.finalize()
        # After finalize check if the created callback object is instance of the class we have given.
        assert isinstance(
            simulator_class._feature_group_callback._operator_collection[-1][
                -1
            ].func.__self__,
            legal_callback,
        )

        simulator_class.apply_callbacks(time=0, current_step=0)

        assert (
            spy.call_count == 2
        )  # Callback should be called twice: once during the finalize and once during the apply_callbacks
        assert spy.call_args[1]["system"] == rod
        assert spy.call_args[1]["time"] == np.float64(0.0)
        assert spy.call_args[1]["current_step"] == 0

    @pytest.mark.parametrize("legal_callback", [CallBackBaseClass])
    def test_callback_in_data_structure(self, mocker, load_collection, legal_callback):
        simulator_class, rod = load_collection

        spy = mocker.spy(legal_callback, "make_callback")

        simulator_class.collect_diagnostics((rod, rod)).using(legal_callback)
        simulator_class.finalize()
        # After finalize check if the created callback object is instance of the class we have given.
        assert isinstance(
            simulator_class._feature_group_callback._operator_collection[-1][
                -1
            ].func.__self__,
            legal_callback,
        )

        simulator_class.apply_callbacks(time=0, current_step=0)

        assert (
            spy.call_count == 2
        )  # Callback should be called twice: once during the finalize and once during the apply_callbacks
        assert spy.call_args[1]["system"] == (rod, rod)
        assert spy.call_args[1]["time"] == np.float64(0.0)
        assert spy.call_args[1]["current_step"] == 0

    @pytest.mark.parametrize("legal_callback", [CallBackBaseClass])
    def test_callback_in_ellipsis(self, mocker, load_collection, legal_callback):
        simulator_class, rod = load_collection
        simulator_class.extend_allowed_types((int,))

        simulator_class.append(rod)

        spy = mocker.spy(legal_callback, "make_callback")

        simulator_class.collect_diagnostics(...).using(legal_callback)
        simulator_class.finalize()
        # After finalize check if the created callback object is instance of the class we have given.
        assert isinstance(
            simulator_class._feature_group_callback._operator_collection[-1][
                -1
            ].func.__self__,
            legal_callback,
        )

        simulator_class.apply_callbacks(time=0, current_step=0)
        simulator_class.apply_callbacks(time=1, current_step=1)

        assert (
            spy.call_count == 3
        )  # Callback should be called twice: once during the finalize and once during the apply_callbacks
        assert spy.call_args_list[1][1]["system"][0] == rod
        assert spy.call_args_list[1][1]["system"][1] == rod
        assert spy.call_args_list[1][1]["time"] == 0
        assert spy.call_args_list[1][1]["current_step"] == 0
        assert spy.call_args_list[2][1]["time"] == 1
        assert spy.call_args_list[2][1]["current_step"] == 1
