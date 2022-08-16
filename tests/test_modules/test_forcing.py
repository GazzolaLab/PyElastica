__doc__ = """ Test modules for forcings """
import numpy as np
import pytest

from elastica.modules import Forcing
from elastica.modules.forcing import _ExtForceTorque


class TestExtForceTorque:
    @pytest.fixture(scope="function")
    def load_forcing(self):
        return _ExtForceTorque(100)  # This is the id for some reason

    @pytest.mark.parametrize("illegal_forcing", [int, list])
    def test_using_with_illegal_forcing_throws_assertion_error(
        self, load_forcing, illegal_forcing
    ):
        with pytest.raises(AssertionError) as excinfo:
            load_forcing.using(illegal_forcing)
        assert "not a valid forcing" in str(excinfo.value)

    from elastica.external_forces import NoForces, EndpointForces, GravityForces

    @pytest.mark.parametrize("legal_forcing", [NoForces, EndpointForces, GravityForces])
    def test_using_with_legal_forcing(self, load_forcing, legal_forcing):
        forcing = load_forcing
        forcing.using(legal_forcing, 3, 4.0, "5", k=1, l_var="2", j=3.0)

        assert forcing._forcing_cls == legal_forcing
        assert forcing._args == (3, 4.0, "5")
        assert forcing._kwargs == {"k": 1, "l_var": "2", "j": 3.0}

    def test_id(self, load_forcing):
        # This is purely for coverage purposes, no actual test
        # since its a simple return
        assert load_forcing.id() == 100

    def test_call_without_setting_forcing_throws_runtime_error(self, load_forcing):
        forcing = load_forcing

        with pytest.raises(RuntimeError) as excinfo:
            forcing(None)  # None is the rod/system parameter
        assert "No forcing" in str(excinfo.value)

    def test_call_improper_args_throws(self, load_forcing):
        # Example of bad initiailization function
        # This needs at least four args which the user might
        # forget to pass later on
        def mock_init(self, *args, **kwargs):
            self.nu = args[3]  # Need at least four args
            self.k = kwargs.get("k")

        # in place class
        MockForcing = type(
            "MockForcing", (self.NoForces, object), {"__init__": mock_init}
        )

        # The user thinks 4.0 goes to nu, but we don't accept it because of error in
        # construction og a Forcing class
        forcing = load_forcing
        forcing.using(MockForcing, 4.0, k=1, l_var="2", j=3.0)

        # Actual test is here, this should not throw
        with pytest.raises(TypeError) as excinfo:
            _ = forcing()
        assert "Unable to construct" in str(excinfo.value)


class TestForcingMixin:
    from elastica.modules import BaseSystemCollection

    class SystemCollectionWithForcingMixedin(BaseSystemCollection, Forcing):
        pass

    # TODO fix link after new PR
    from elastica.rod import RodBase

    class MockRod(RodBase):
        def __init__(self, *args, **kwargs):
            pass

    @pytest.fixture(scope="function", params=[2, 10])
    def load_system_with_forcings(self, request):
        n_sys = request.param
        sys_coll_with_forcings = self.SystemCollectionWithForcingMixedin()
        for i_sys in range(n_sys):
            sys_coll_with_forcings.append(self.MockRod(2, 3, 4, 5))
        return sys_coll_with_forcings

    """ The following calls test _get_sys_idx_if_valid from BaseSystem indirectly,
    and are here because of legacy reasons. I have not removed them because there
    are Connections require testing against multiple indices, which is still use
    ful to cross-verify against.

    START
    """

    def test_constrain_with_illegal_index_throws(self, load_system_with_forcings):
        scwf = load_system_with_forcings

        with pytest.raises(AssertionError) as excinfo:
            scwf.add_forcing_to(100)
        assert "exceeds number of" in str(excinfo.value)

        with pytest.raises(AssertionError) as excinfo:
            scwf.add_forcing_to(np.int_(100))
        assert "exceeds number of" in str(excinfo.value)

    def test_constrain_with_unregistered_system_throws(self, load_system_with_forcings):
        scwf = load_system_with_forcings

        # Don't register this rod
        mock_rod = self.MockRod(2, 3, 4, 5)

        with pytest.raises(ValueError) as excinfo:
            scwf.add_forcing_to(mock_rod)
        assert "was not found, did you" in str(excinfo.value)

    def test_constrain_with_illegal_system_throws(self, load_system_with_forcings):
        scwf = load_system_with_forcings

        # Not a rod, but a list!
        mock_rod = [1, 2, 3, 5]

        with pytest.raises(TypeError) as excinfo:
            scwf.add_forcing_to(mock_rod)
        assert "not a system" in str(excinfo.value)

    """
    END of testing BaseSystem calls
    """

    def test_constrain_registers_and_returns_ExtForceTorque(
        self, load_system_with_forcings
    ):
        scwf = load_system_with_forcings

        mock_rod = self.MockRod(2, 3, 4, 5)
        scwf.append(mock_rod)

        _mock_forcing = scwf.add_forcing_to(mock_rod)
        assert _mock_forcing in scwf._ext_forces_torques
        assert _mock_forcing.__class__ == _ExtForceTorque

    from elastica.external_forces import NoForces

    @pytest.fixture
    def load_rod_with_forcings(self, load_system_with_forcings):
        scwf = load_system_with_forcings

        mock_rod = self.MockRod(2, 3, 4, 5)
        scwf.append(mock_rod)

        def mock_init(self, *args, **kwargs):
            pass

        # in place class
        MockForcing = type(
            "MockForcing", (self.NoForces, object), {"__init__": mock_init}
        )

        # Constrain any and all systems
        scwf.add_forcing_to(1).using(MockForcing, 2, 42)  # index based forcing
        scwf.add_forcing_to(0).using(MockForcing, 1, 2)  # index based forcing
        scwf.add_forcing_to(mock_rod).using(MockForcing, 2, 3)  # system based forcing

        return scwf, MockForcing

    def test_friction_plane_forcing_class_sorting(self, load_system_with_forcings):

        scwf = load_system_with_forcings

        mock_rod = self.MockRod(2, 3, 4, 5)
        scwf.append(mock_rod)

        from elastica.interaction import AnisotropicFrictionalPlane

        # Add friction plane
        scwf.add_forcing_to(1).using(
            AnisotropicFrictionalPlane,
            k=0,
            nu=0,
            plane_origin=np.zeros((3,)),
            plane_normal=np.zeros((3,)),
            slip_velocity_tol=0,
            static_mu_array=[0, 0, 0],
            kinetic_mu_array=[0, 0, 0],
        )
        # Add another forcing class

        def mock_init(self, *args, **kwargs):
            pass

        MockForcing = type(
            "MockForcing", (self.NoForces, object), {"__init__": mock_init}
        )
        scwf.add_forcing_to(1).using(MockForcing, 2, 42)  # index based forcing

        scwf._finalize_forcing()

        # Now check if the Anisotropic friction is the last forcing class
        assert isinstance(scwf._ext_forces_torques[-1][-1], AnisotropicFrictionalPlane)

    def test_constrain_finalize_correctness(self, load_rod_with_forcings):
        scwf, forcing_cls = load_rod_with_forcings

        scwf._finalize_forcing()

        for (x, y) in scwf._ext_forces_torques:
            assert type(x) is int
            assert type(y) is forcing_cls

    @pytest.mark.xfail
    def test_constrain_finalize_sorted(self, load_rod_with_forcings):
        scwf, forcing_cls = load_rod_with_forcings

        scwf._finalize_forcing()

        # this is allowed to fail (not critical)
        num = -np.inf
        for (x, _) in scwf._ext_forces_torques:
            assert num < x
            num = x

    def test_constrain_call_on_systems(self):
        # TODO Finish after the architecture is complete
        pass
