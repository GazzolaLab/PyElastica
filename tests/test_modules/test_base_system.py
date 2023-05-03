__doc__ = """ Test modules for base systems """

import pytest
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
        bsc = BaseSystemCollection()
        # Bypass check, but its fine for testing
        bsc._systems.append(3)
        bsc._systems.append(5.0)
        bsc._systems.append("a")
        bsc._systems.append(np.random.randn(3, 5))
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
        bsc.extend_allowed_types((int, float, str))

        from elastica.rod import RodBase
        from elastica.rigidbody import RigidBodyBase

        assert bsc.allowed_sys_types == (RodBase, RigidBodyBase, int, float, str)

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
        bsc.override_allowed_types((int, float, str))

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
        bsc.override_allowed_types((RodBase,))
        with pytest.raises(AssertionError) as excinfo:
            bsc._get_sys_idx_if_valid(100)
        assert "exceeds number of" in str(excinfo.value)

        with pytest.raises(AssertionError) as excinfo:
            load_collection._get_sys_idx_if_valid(np.int_(100))
        assert "exceeds number of" in str(excinfo.value)

    def test_unregistered_system_in_get_sys_index_throws(
        self, load_collection, mock_rod
    ):
        # Don't register this rod
        my_mock_rod = mock_rod

        with pytest.raises(ValueError) as excinfo:
            load_collection._get_sys_idx_if_valid(my_mock_rod)
        assert "was not found, did you" in str(excinfo.value)

    def test_get_sys_index_returns_correct_idx(self, load_collection):
        assert load_collection._get_sys_idx_if_valid(1) == 1

    @pytest.mark.xfail
    def test_delitem(self, load_collection):
        del load_collection[0]
        assert load_collection[0] == 3


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
        sc._systems.append(rod)

        return sc, rod

    from elastica.boundary_conditions import FreeBC

    @pytest.mark.parametrize("legal_constraint", [FreeBC])
    def test_constraint(self, load_collection, legal_constraint):
        simulator_class, rod = load_collection
        simulator_class.constrain(rod).using(legal_constraint)
        simulator_class.finalize()
        # After finalize check if the created constrain object is instance of the class we have given.
        assert isinstance(simulator_class._constraints[-1][-1], legal_constraint)

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
        assert isinstance(simulator_class._ext_forces_torques[-1][-1], legal_forces)

        # TODO: this is a dummy test for synchronize find a better way to test them
        simulator_class.synchronize(time=0)

    from elastica.callback_functions import CallBackBaseClass

    @pytest.mark.parametrize("legal_callback", [CallBackBaseClass])
    def test_callback(self, load_collection, legal_callback):
        simulator_class, rod = load_collection
        simulator_class.collect_diagnostics(rod).using(legal_callback)
        simulator_class.finalize()
        # After finalize check if the created callback object is instance of the class we have given.
        assert isinstance(simulator_class._callback_list[-1][-1], legal_callback)

        # TODO: this is a dummy test for apply_callbacks find a better way to test them
        simulator_class.apply_callbacks(time=0, current_step=0)
