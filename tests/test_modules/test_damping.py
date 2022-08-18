__doc__ = """ Test modules for damping """
import numpy as np
import pytest

from elastica.modules import Damping
from elastica.modules.damping import _Damper


class TestDamper:
    @pytest.fixture(scope="function")
    def load_damper(self):
        return _Damper(100)  # This is the id for some reason

    @pytest.mark.parametrize("illegal_damper", [int, list])
    def test_using_with_illegal_damper_throws_assertion_error(
        self, load_damper, illegal_damper
    ):
        with pytest.raises(AssertionError) as excinfo:
            load_damper.using(illegal_damper)
        assert "not a valid damper" in str(excinfo.value)

    from elastica.dissipation import AnalyticalLinearDamper as TestDamper

    @pytest.mark.parametrize("legal_damper", [TestDamper])
    def test_using_with_legal_damper(self, load_damper, legal_damper):
        damper = load_damper
        damper.using(legal_damper, 3, 4.0, "5", k=1, l_var="2", j=3.0)

        assert damper._damper_cls == legal_damper
        assert damper._args == (3, 4.0, "5")
        assert damper._kwargs == {"k": 1, "l_var": "2", "j": 3.0}

    def test_id(self, load_damper):
        # This is purely for coverage purposes, no actual test
        # since its a simple return
        assert load_damper.id() == 100

    def test_call_without_setting_damper_throws_runtime_error(self, load_damper):
        damper = load_damper

        with pytest.raises(RuntimeError) as excinfo:
            damper(None)  # None is the rod/system parameter
        assert "No damper" in str(excinfo.value)

    def test_call_with_args_and_kwargs(self, load_damper):
        def mock_init(self, *args, **kwargs):
            self.dummy_one = args[0]
            self.k = kwargs.get("k")

        # in place class
        MockDamper = type(
            "MockDamper", (self.TestDamper, object), {"__init__": mock_init}
        )

        damper = load_damper
        damper.using(MockDamper, 3.9, 4.0, "5", k=1, l_var="2", j=3.0)

        # Actual test is here, this should not throw
        mock_damper = damper(None)  # None is Fake rod

        # More tests reinforcing the first
        assert mock_damper.dummy_one == 3.9
        assert mock_damper.k == 1

    class MockRod:
        def __init__(self):
            self.mass = np.random.randn(3, 8)

    def test_call_improper_bc_throws_type_error(self, load_damper):
        # Example of bad initiailization function
        damper = load_damper
        damper.using(
            self.TestDamper,
            dissipation_constant=2,
            time_step="2",
        )  # Passing string as time-step, which is wrong

        mock_rod = self.MockRod()
        # Actual test is here, this should not throw
        with pytest.raises(TypeError) as excinfo:
            _ = damper(mock_rod)
        assert "Unable to construct" in str(excinfo.value)


class TestDampingMixin:
    from elastica.modules import BaseSystemCollection

    class SystemCollectionWithDampingMixedin(BaseSystemCollection, Damping):
        pass

    # TODO fix link after new PR
    from elastica.rod import RodBase

    class MockRod(RodBase):
        def __init__(self, *args, **kwargs):
            pass

    @pytest.fixture(scope="function", params=[2, 10])
    def load_system_with_dampers(self, request):
        n_sys = request.param
        sys_coll_with_dampers = self.SystemCollectionWithDampingMixedin()
        for i_sys in range(n_sys):
            sys_coll_with_dampers.append(self.MockRod(2, 3, 4, 5))
        return sys_coll_with_dampers

    """ The following calls test _get_sys_idx_if_valid from BaseSystem indirectly,
    and are here because of legacy reasons. I have not removed them because there
    are Connections require testing against multiple indices, which is still use
    ful to cross-verify against.

    START
    """

    def test_dampen_with_illegal_index_throws(self, load_system_with_dampers):
        scwd = load_system_with_dampers

        with pytest.raises(AssertionError) as excinfo:
            scwd.dampen(100)
        assert "exceeds number of" in str(excinfo.value)

        with pytest.raises(AssertionError) as excinfo:
            scwd.dampen(np.int_(100))
        assert "exceeds number of" in str(excinfo.value)

    def test_dampen_with_unregistered_system_throws(self, load_system_with_dampers):
        scwd = load_system_with_dampers

        # Don't register this rod
        mock_rod = self.MockRod(2, 3, 4, 5)

        with pytest.raises(ValueError) as excinfo:
            scwd.dampen(mock_rod)
        assert "was not found, did you" in str(excinfo.value)

    def test_dampen_with_illegal_system_throws(self, load_system_with_dampers):
        scwd = load_system_with_dampers

        # Not a rod, but a list!
        mock_rod = [1, 2, 3, 5]

        with pytest.raises(TypeError) as excinfo:
            scwd.dampen(mock_rod)
        assert "not a system" in str(excinfo.value)

    """
    END of testing BaseSystem calls
    """

    def test_dampen_registers_and_returns_Damper(self, load_system_with_dampers):
        scwd = load_system_with_dampers

        mock_rod = self.MockRod(2, 3, 4, 5)
        scwd.append(mock_rod)

        _mock_damper = scwd.dampen(mock_rod)
        assert _mock_damper in scwd._dampers
        assert _mock_damper.__class__ == _Damper

    from elastica.dissipation import DamperBase

    @pytest.fixture
    def load_rod_with_dampers(self, load_system_with_dampers):
        scwd = load_system_with_dampers

        mock_rod = self.MockRod(2, 3, 4, 5)
        scwd.append(mock_rod)

        # in place class
        class MockDamper(self.DamperBase):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def dampen_rates(self, *args, **kwargs) -> None:
                pass

        # Constrain any and all systems
        scwd.dampen(1).using(MockDamper, 2, 42)  # index based damper
        scwd.dampen(0).using(MockDamper, 1, 2)  # index based damper
        scwd.dampen(mock_rod).using(MockDamper, 2, 3)  # system based damper

        return scwd, MockDamper

    def test_dampen_finalize_correctness(self, load_rod_with_dampers):
        scwd, damper_cls = load_rod_with_dampers

        scwd._finalize_dampers()

        for (x, y) in scwd._dampers:
            assert type(x) is int
            assert type(y) is damper_cls

    def test_damper_properties(self, load_rod_with_dampers):
        scwd, _ = load_rod_with_dampers
        scwd._finalize_dampers()

        for i in [0, 1, -1]:
            x, y = scwd._dampers[i]
            mock_rod = scwd._systems[i]
            # Test system
            assert type(x) is int
            assert type(y.system) is type(mock_rod)
            assert y.system is mock_rod, f"{len(scwd._systems)}"

    @pytest.mark.xfail
    def test_dampers_finalize_sorted(self, load_rod_with_dampers):
        scwd, damper_cls = load_rod_with_dampers

        scwd._finalize_dampers()

        # this is allowed to fail (not critical)
        num = -np.inf
        for (x, _) in scwd._dampers:
            assert num < x
            num = x
