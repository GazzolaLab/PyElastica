__doc__ = """ Test modules for connections """
import numpy as np
import pytest

from elastica.modules import Connections
from elastica.modules.connections import _Connect
from numpy.testing import assert_allclose
from elastica.utils import Tolerance


class TestConnect:
    @pytest.fixture(scope="function")
    def load_connect(self, request):
        # connect 15th and 23rd rod each having 100 Dofs
        return _Connect(15, 23, 100, 100)

    # idx between -100 and 99 passes,
    # test combinations for first and second rod
    @pytest.mark.parametrize(
        "illegal_idx",
        [
            (120, 120),
            (120, 50),
            (50, 120),
            (-120, -120),
            (-120, -50),
            (-50, -120),
            (-120, 50),
            (50, -120),
            (np.array([50]), np.array([-120])),
            (np.array([50, 120]), np.array([-120, -50])),
            ([-50, 120], [-120, 50]),
            # test for edge cases
            (-102, 99),
            (99, -102),
            (-101, 101),
            (101, -101),
        ],
    )
    def test_set_index_with_illegal_idx_throws(self, load_connect, illegal_idx):
        with pytest.raises(AssertionError) as excinfo:
            load_connect.set_index(*illegal_idx)
        assert "Connection index of" in str(excinfo.value)

    # idx between -100 and 99 passes,
    # test combinations for first and second rod
    @pytest.mark.parametrize(
        "legal_idx",
        [
            # edge cases that should not throw
            (-101, -101),
            (-101, 100),
            (100, -101),
            (100, 100),
        ],
    )
    def test_set_index_with_legal_idx_does_not_throw(self, load_connect, legal_idx):
        try:
            load_connect.set_index(*legal_idx)
        except AssertionError:
            pytest.fail("Unexpected AssertionError ..")

    # Test different idx types input by the user
    @pytest.mark.parametrize(
        "different_type_idx",
        [(10, 10.0), (11, np.array([5])), ((10, 5), 1), ([4, 6], 1)],
    )
    def test_set_index_with_different_type_idx_throws(
        self, load_connect, different_type_idx
    ):
        with pytest.raises(AssertionError) as excinfo:
            load_connect.set_index(*different_type_idx)
        assert "Type of first_connect_idx" in str(excinfo)

    # Test illegal idx types input by the user
    @pytest.mark.parametrize(
        "illegal_type_idx",
        [(10.0, 5.0), (np.array([5.0])[0], np.array([7.0])[0]), (str("3"), str("5"))],
    )
    def test_set_index_with_illegal_type_idx_throws(
        self, load_connect, illegal_type_idx
    ):
        with pytest.raises(AssertionError) as excinfo:
            load_connect.set_index(*illegal_type_idx)
        assert "Connection index type is not supported" in str(excinfo)

    # Test illegal idx types for first rod index.
    @pytest.mark.parametrize(
        "illegal_type_first_idx", [(np.array([5.0]), np.array([6])), ([2.0], [5])]
    )
    def test_set_index_with_illegal_type_first_idx_throws(
        self, load_connect, illegal_type_first_idx
    ):
        with pytest.raises(AssertionError) as excinfo:
            load_connect.set_index(*illegal_type_first_idx)
        assert "Connection index of first rod" in str(excinfo)

    # Test illegal idx types for second rod index.
    @pytest.mark.parametrize(
        "illegal_type_second_idx", [(np.array([5]), np.array([6.0])), ([2], [5.0])]
    )
    def test_set_index_with_illegal_type_second_idx_throws(
        self, load_connect, illegal_type_second_idx
    ):
        with pytest.raises(AssertionError) as excinfo:
            load_connect.set_index(*illegal_type_second_idx)
        assert "Connection index of second rod is not integer" in str(excinfo)

    # Below test is to increase code coverage. If we pass nothing or idx=None, then do nothing.
    def test_set_index_no_input(self, load_connect):
        load_connect.set_index(first_idx=(), second_idx=())

    @pytest.mark.parametrize(
        "legal_idx", [(80, 80), (0, 50), (50, 0), (-20, -20), (-20, 50), (-50, -20)]
    )
    def test_set_index_with_legal_idx(self, load_connect, legal_idx):
        connect = load_connect
        connect.set_index(*legal_idx)

        assert connect.first_sys_connection_idx == legal_idx[0]
        assert connect.second_sys_connection_idx == legal_idx[1]

    @pytest.mark.parametrize("illegal_connect", [int, list])
    def test_using_with_illegal_connect_throws_assertion_error(
        self, load_connect, illegal_connect
    ):
        with pytest.raises(AssertionError) as excinfo:
            load_connect.using(illegal_connect)
        assert "not a valid joint" in str(excinfo.value)

    from elastica.joint import FreeJoint, FixedJoint, HingeJoint

    @pytest.mark.parametrize("legal_connect", [FreeJoint, HingeJoint, FixedJoint])
    def test_using_with_legal_connect(self, load_connect, legal_connect):
        connect = load_connect
        connect.using(legal_connect, 3, 4.0, "5", k=1, l_var="2", j=3.0)

        assert connect._connect_cls == legal_connect
        assert connect._args == (3, 4.0, "5")
        assert connect._kwargs == {"k": 1, "l_var": "2", "j": 3.0}

    def test_id(self, load_connect):
        connect = load_connect
        connect.set_index(20, -20)
        # This is purely for coverage purposes, no actual test
        # since its a simple return
        assert connect.id() == (15, 23, 20, -20)

    def test_call_without_setting_connect_throws_runtime_error(self, load_connect):
        connect = load_connect

        with pytest.raises(RuntimeError) as excinfo:
            connect.instantiate()
        assert "No connections provided" in str(excinfo.value)

    def test_call_improper_args_throws(self, load_connect):
        # Example of bad initiailization function
        # This needs at least four args which the user might
        # forget to pass later on
        def mock_init(self, *args, **kwargs):
            self.nu = args[3]  # Need at least four args
            self.k = kwargs.get("k")

        # in place class
        MockConnect = type(
            "MockConnect", (self.FreeJoint, object), {"__init__": mock_init}
        )

        # The user thinks 4.0 goes to nu, but we don't accept it because of error in
        # construction og a Connect class
        connect = load_connect
        connect.using(MockConnect, 4.0, k=1, l_var="2", j=3.0)

        # Actual test is here, this should not throw
        with pytest.raises(TypeError) as excinfo:
            _ = connect.instantiate()
        assert (
            r"Unable to construct connection class.\nDid you provide all necessary joint properties?"
            == str(excinfo.value)
        )


class TestConnectionsMixin:
    from elastica.modules import BaseSystemCollection

    class SystemCollectionWithConnectionsMixin(BaseSystemCollection, Connections):
        pass

    # TODO fix link after new PR
    from elastica.rod import RodBase

    class MockRod(RodBase):
        def __init__(self, *args, **kwargs):
            self.n_elems = 3  # arbitrary number

    @pytest.fixture(scope="function", params=[2, 10])
    def load_system_with_connects(self, request):
        n_sys = request.param
        sys_coll_with_connects = self.SystemCollectionWithConnectionsMixin()
        for i_sys in range(n_sys):
            sys_coll_with_connects.append(self.MockRod(2, 3, 4, 5))
        return sys_coll_with_connects

    """ The following calls test get_system_index from BaseSystem indirectly,
    and are here because of legacy reasons. I have not removed them because there
    are Connections require testing against multiple indices, which is still use
    ful to cross-verify against.

    START
    """

    @pytest.mark.parametrize(
        "sys_idx",
        [
            (12, 3),
            (3, 12),
            (-12, 3),
            (-3, 12),
            (12, -3),
            (-12, -3),
            (3, -12),
            (-3, -12),
        ],
    )
    def test_connect_with_illegal_index_throws(
        self, load_system_with_connects, sys_idx
    ):
        system_collection_with_connections = load_system_with_connects

        with pytest.raises(AssertionError) as excinfo:
            system_collection_with_connections.connect(*sys_idx)
        assert "exceeds number of" in str(excinfo.value)

        with pytest.raises(AssertionError) as excinfo:
            system_collection_with_connections.connect(*[np.int32(x) for x in sys_idx])
        assert "exceeds number of" in str(excinfo.value)

    def test_connect_with_unregistered_system_throws(self, load_system_with_connects):
        system_collection_with_connections = load_system_with_connects

        # Register this rod
        mock_rod_registered = self.MockRod(5, 5, 5, 5)
        system_collection_with_connections.append(mock_rod_registered)
        # Don't register this rod
        mock_rod = self.MockRod(2, 3, 4, 5)

        with pytest.raises(ValueError) as excinfo:
            system_collection_with_connections.connect(mock_rod, mock_rod_registered)
        assert "was not found, did you" in str(excinfo.value)

        # Switch arguments
        with pytest.raises(ValueError) as excinfo:
            system_collection_with_connections.connect(mock_rod_registered, mock_rod)
        assert "was not found, did you" in str(excinfo.value)

    def test_connect_with_illegal_system_throws(self, load_system_with_connects):
        system_collection_with_connections = load_system_with_connects

        # Register this rod
        mock_rod_registered = self.MockRod(5, 5, 5, 5)
        system_collection_with_connections.append(mock_rod_registered)

        # Not a rod, but a list!
        mock_rod = [1, 2, 3, 5]

        with pytest.raises(TypeError) as excinfo:
            system_collection_with_connections.connect(mock_rod, mock_rod_registered)
        assert "not a sys" in str(excinfo.value)

        # Switch arguments
        with pytest.raises(TypeError) as excinfo:
            system_collection_with_connections.connect(mock_rod_registered, mock_rod)
        assert "not a sys" in str(excinfo.value)

    """
    END of testing BaseSystem calls
    """

    def test_connect_registers_and_returns_Connect(self, load_system_with_connects):
        system_collection_with_connections = load_system_with_connects

        mock_rod_one = self.MockRod(2, 3, 4, 5)
        system_collection_with_connections.append(mock_rod_one)

        mock_rod_two = self.MockRod(4, 5)
        system_collection_with_connections.append(mock_rod_two)

        _mock_connect = system_collection_with_connections.connect(
            mock_rod_one, mock_rod_two
        )
        assert _mock_connect in system_collection_with_connections._connections
        assert _mock_connect.__class__ == _Connect
        # check sane defaults provided for connection indices
        assert _mock_connect.id()[2] == ()
        assert _mock_connect.id()[3] == ()

    from elastica.joint import FreeJoint

    @pytest.fixture
    def load_rod_with_connects(self, load_system_with_connects):
        system_collection_with_connections = load_system_with_connects

        mock_rod_one = self.MockRod(2, 3, 4, 5)
        system_collection_with_connections.append(mock_rod_one)
        mock_rod_two = self.MockRod(5.0, 5.0)
        system_collection_with_connections.append(mock_rod_two)

        def mock_init(self, *args, **kwargs):
            pass

        # in place class
        MockConnect = type(
            "MockConnect", (self.FreeJoint, object), {"__init__": mock_init}
        )

        # Constrain any and all systems
        # system_collection_with_connections.connect(0, 1).using(
        #    MockConnect, 2, 42
        # )  # index based connect
        system_collection_with_connections.connect(mock_rod_one, mock_rod_two).using(
            MockConnect, 2, 3
        )  # system based connect
        # system_collection_with_connections.connect(0, mock_rod_one).using(
        #    MockConnect, 1, 2
        # )  # index/system based connect

        return system_collection_with_connections, MockConnect

    def test_connect_finalize_correctness(self, load_rod_with_connects):
        system_collection_with_connections, connect_cls = load_rod_with_connects
        connect = system_collection_with_connections._connections[0]
        assert connect._connect_cls == connect_cls

        system_collection_with_connections._finalize_connections()
        assert (
            system_collection_with_connections._feature_group_synchronize._operator_ids[
                0
            ]
            == id(connect)
        )

        assert not hasattr(system_collection_with_connections, "_connections")

    @pytest.fixture
    def load_rod_with_connects_and_indices(self, load_system_with_connects):
        system_collection_with_connections_and_indices = load_system_with_connects

        mock_rod_one = self.MockRod(1.0, 2.0, 3.0, 4.0)
        mock_rod_one.position_collection = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        mock_rod_one.velocity_collection = np.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )
        mock_rod_one.external_forces = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        system_collection_with_connections_and_indices.append(mock_rod_one)
        mock_rod_two = self.MockRod(1.0, 1.0)
        mock_rod_two.position_collection = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
        )
        mock_rod_two.velocity_collection = np.array(
            [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]
        )
        mock_rod_two.external_forces = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        system_collection_with_connections_and_indices.append(mock_rod_two)

        def mock_init(self, *args, **kwargs):
            self.k = 1.0
            self.nu = 0.1

        # in place class
        MockConnect = type(
            "MockConnect", (self.FreeJoint, object), {"__init__": mock_init}
        )

        # Constrain any and all systems
        system_collection_with_connections_and_indices.connect(
            mock_rod_one, mock_rod_two, 0, 0
        ).using(
            MockConnect, 2, 42
        )  # with connection indices
        return system_collection_with_connections_and_indices, MockConnect

    def test_connect_call_on_systems(self, load_rod_with_connects_and_indices):
        (
            system_collection_with_connections_and_indices,
            connect_cls,
        ) = load_rod_with_connects_and_indices
        mock_connections = [
            c for c in system_collection_with_connections_and_indices._connections
        ]

        system_collection_with_connections_and_indices._finalize_connections()
        system_collection_with_connections_and_indices.synchronize(0)

        for connection in mock_connections:
            fidx, sidx, fconnect, sconnect = connection.id()
            connect = connection.instantiate()

            end_distance_vector = (
                system_collection_with_connections_and_indices[
                    sidx
                ].position_collection[..., sconnect]
                - system_collection_with_connections_and_indices[
                    fidx
                ].position_collection[..., fconnect]
            )
            elastic_force = connect.k * end_distance_vector

            relative_velocity = (
                system_collection_with_connections_and_indices[
                    sidx
                ].velocity_collection[..., sconnect]
                - system_collection_with_connections_and_indices[
                    fidx
                ].velocity_collection[..., fconnect]
            )
            damping_force = connect.nu * relative_velocity

            contact_force = elastic_force + damping_force

            assert_allclose(
                system_collection_with_connections_and_indices[fidx].external_forces[
                    ..., fconnect
                ],
                contact_force,
                atol=Tolerance.atol(),
            )
            assert_allclose(
                system_collection_with_connections_and_indices[sidx].external_forces[
                    ..., sconnect
                ],
                -1 * contact_force,
                atol=Tolerance.atol(),
            )
