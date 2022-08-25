__doc__ = """ Test modules for callback """
import numpy as np
import pytest

from elastica.modules import CallBacks
from elastica.modules.callbacks import _CallBack


class TestCallBacks:
    @pytest.fixture(scope="function")
    def load_callback(self, request):
        return _CallBack(100)  # This is the id for some reason

    @pytest.mark.parametrize("illegal_callback", [int, list])
    def test_using_with_illegal_callback_throws_assertion_error(
        self, load_callback, illegal_callback
    ):
        with pytest.raises(AssertionError) as excinfo:
            load_callback.using(illegal_callback)
        assert "not a valid call back" in str(excinfo.value)

    from elastica.callback_functions import CallBackBaseClass, MyCallBack

    @pytest.mark.parametrize("legal_callback", [CallBackBaseClass, MyCallBack])
    def test_using_with_legal_constraint(self, load_callback, legal_callback):
        callback = load_callback
        callback.using(legal_callback, 3, 4.0, "5", k=1, l_var="2", j=3.0)

        assert callback._callback_cls == legal_callback
        assert callback._args == (3, 4.0, "5")
        assert callback._kwargs == {"k": 1, "l_var": "2", "j": 3.0}

    def test_id(self, load_callback):
        # This is purely for coverage purposes, no actual test
        # since its a simple return
        assert load_callback.id() == 100

    def test_call_improper_args_throws(self, load_callback):
        # Example of bad initiailization function
        # This needs at least four args which the user might
        # forget to pass later on
        def mock_init(self, *args, **kwargs):
            self.nu = args[3]  # Need at least four args
            self.k = kwargs.get("k")

        # in place class
        MockCallBack = type(
            "MockCallBack", (self.CallBackBaseClass, object), {"__init__": mock_init}
        )

        # The user thinks 4.0 goes to nu, but we don't accept it because of error in
        # construction og a Forcing class
        callback = load_callback
        callback.using(MockCallBack, 4.0, k=1, l_var="2", j=3.0)

        # Actual test is here, this should not throw
        with pytest.raises(TypeError) as excinfo:
            _ = callback()
        assert "Unable to construct" in str(excinfo.value)


class TestCallBacksMixin:
    from elastica.modules import BaseSystemCollection

    class SystemCollectionWithCallBacksMixedin(BaseSystemCollection, CallBacks):
        pass

    # TODO fix link after new PR
    from elastica.rod import RodBase

    class MockRod(RodBase):
        def __init__(self, *args, **kwargs):
            pass

    @pytest.fixture(scope="function", params=[2, 10])
    def load_system_with_callbacks(self, request):
        n_sys = request.param
        sys_coll_with_callbacks = self.SystemCollectionWithCallBacksMixedin()
        for i_sys in range(n_sys):
            sys_coll_with_callbacks.append(self.MockRod(2, 3, 4, 5))
        return sys_coll_with_callbacks

    """ The following calls test _get_sys_idx_if_valid from BaseSystem indirectly,
    and are here because of legacy reasons. I have not removed them because there
    are Callbacks require testing against multiple indices, which is still use
    ful to cross-verify against.

    START
    """

    def test_callback_with_illegal_index_throws(self, load_system_with_callbacks):
        scwc = load_system_with_callbacks

        with pytest.raises(AssertionError) as excinfo:
            scwc.collect_diagnostics(100)
        assert "exceeds number of" in str(excinfo.value)

        with pytest.raises(AssertionError) as excinfo:
            scwc.collect_diagnostics(np.int_(100))
        assert "exceeds number of" in str(excinfo.value)

    def test_callback_with_unregistered_system_throws(self, load_system_with_callbacks):
        scwc = load_system_with_callbacks

        # Don't register this rod
        mock_rod = self.MockRod(2, 3, 4, 5)

        with pytest.raises(ValueError) as excinfo:
            scwc.collect_diagnostics(mock_rod)
        assert "was not found, did you" in str(excinfo.value)

    def test_callback_with_illegal_system_throws(self, load_system_with_callbacks):
        scwc = load_system_with_callbacks

        # Not a rod, but a list!
        mock_rod = [1, 2, 3, 5]

        with pytest.raises(TypeError) as excinfo:
            scwc.collect_diagnostics(mock_rod)
        assert "not a system" in str(excinfo.value)

    """
    END of testing BaseSystem calls
    """

    def test_callback_registers_and_returns_Callback(self, load_system_with_callbacks):
        scwc = load_system_with_callbacks

        mock_rod = self.MockRod(2, 3, 4, 5)
        scwc.append(mock_rod)

        _mock_callback = scwc.collect_diagnostics(mock_rod)
        assert _mock_callback in scwc._callback_list
        assert _mock_callback.__class__ == _CallBack

    from elastica.callback_functions import CallBackBaseClass

    @pytest.fixture
    def load_rod_with_callbacks(self, load_system_with_callbacks):
        scwc = load_system_with_callbacks

        mock_rod = self.MockRod(2, 3, 4, 5)
        scwc.append(mock_rod)

        def mock_init(self, *args, **kwargs):
            pass

        # in place class
        MockCallBack = type(
            "MockCallBack", (self.CallBackBaseClass, object), {"__init__": mock_init}
        )

        # Constrain any and all systems
        scwc.collect_diagnostics(1).using(MockCallBack, 2, 42)  # index based constraint
        scwc.collect_diagnostics(0).using(MockCallBack, 1, 2)  # index based constraint
        scwc.collect_diagnostics(mock_rod).using(
            MockCallBack, 2, 3
        )  # system based constraint

        return scwc, MockCallBack

    def test_callback_finalize_correctness(self, load_rod_with_callbacks):
        scwc, callback_cls = load_rod_with_callbacks

        scwc._finalize_callback()

        for (x, y) in scwc._callback_list:
            assert type(x) is int
            assert type(y) is callback_cls

    @pytest.mark.xfail
    def test_callback_finalize_sorted(self, load_rod_with_callbacks):
        scwc, callback_cls = load_rod_with_callbacks

        scwc._finalize_callback()

        # this is allowed to fail (not critical)
        num = -np.inf
        for (x, _) in scwc._callback_list:
            assert num < x
            num = x
