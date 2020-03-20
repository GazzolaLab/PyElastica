"""
test_base_system
----------------
"""

import pytest
import numpy as np

from elastica.wrappers import BaseSystemCollection


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

    # TODO : Check synchronize calls
