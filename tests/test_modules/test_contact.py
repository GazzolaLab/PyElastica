__doc__ = """ Test modules for contact """
import numpy as np
import pytest
from elastica.modules import Contact
from elastica.modules.contact import _Contact
from numpy.testing import assert_allclose
from elastica.utils import Tolerance


class TestContact:
    @pytest.fixture(scope="function")
    def load_contact(self, request):
        # contact between 15th and 23rd rod
        return _Contact(15, 23)

    @pytest.mark.parametrize("illegal_contact", [int, list])
    def test_using_with_illegal_contact_throws_assertion_error(
        self, load_contact, illegal_contact
    ):
        with pytest.raises(AssertionError) as excinfo:
            load_contact.using(illegal_contact)
        assert "{} is not a valid contact class. Did you forget to derive from NoContact?".format(
            illegal_contact
        ) == str(
            excinfo.value
        )

    from elastica.contact_forces import NoContact, RodRodContact, RodSelfContact

    @pytest.mark.parametrize(
        "legal_contact", [NoContact, RodRodContact, RodSelfContact]
    )
    def test_using_with_legal_contact(self, load_contact, legal_contact):
        contact = load_contact
        contact.using(legal_contact, 3, 4.0, "5", k=1, l_var="2", j=3.0)

        assert contact._contact_cls == legal_contact
        assert contact._args == (3, 4.0, "5")
        assert contact._kwargs == {"k": 1, "l_var": "2", "j": 3.0}

    def test_id(self, load_contact):
        contact = load_contact
        # This is purely for coverage purposes, no actual test
        # since its a simple return
        assert contact.id() == (15, 23)

    def test_call_without_setting_contact_throws_runtime_error(self, load_contact):
        contact = load_contact

        with pytest.raises(RuntimeError) as excinfo:
            contact.instantiate()
        assert "No contacts provided to to establish contact between rod-like object id {0} and {1}, but a Contact was intended as per code. Did you forget to call the `using` method?".format(
            *contact.id()
        ) == str(
            excinfo.value
        )

    def test_call_improper_args_throws(self, load_contact):
        # Example of bad initiailization function
        # This needs at least four args which the user might
        # forget to pass later on
        def mock_init(self, *args, **kwargs):
            self.nu = args[3]  # Need at least four args
            self.k = kwargs.get("k")

        # in place class
        MockContact = type(
            "MockContact", (self.NoContact, object), {"__init__": mock_init}
        )

        # The user thinks 4.0 goes to nu, but we don't accept it because of error in
        # construction og a Contact class
        contact = load_contact
        contact.using(MockContact, 4.0, k=1, l_var="2", j=3.0)

        # Actual test is here, this should not throw
        with pytest.raises(TypeError) as excinfo:
            _ = contact.instantiate()
        assert (
            r"Unable to construct contact class.\nDid you provide all necessary contact properties?"
            == str(excinfo.value)
        )


class TestContactMixin:
    from elastica.modules import BaseSystemCollection

    class SystemCollectionWithContactMixin(BaseSystemCollection, Contact):
        pass

    from elastica.rod import RodBase
    from elastica.rigidbody import RigidBodyBase
    from elastica.surface import SurfaceBase

    class MockRod(RodBase):
        def __init__(self, *args, **kwargs):
            self.n_elems = 2
            self.position_collection = np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0]])
            self.radius = np.array([1, 1])
            self.lengths = np.array([1, 1])
            self.tangents = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
            self.velocity_collection = np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )
            self.internal_forces = np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )
            self.external_forces = np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )

    class MockRigidBody(RigidBodyBase):
        def __init__(self, *args, **kwargs):
            self.n_elems = 1

    class MockSurface(SurfaceBase):
        def __init__(self, *args, **kwargs):
            self.n_facets = 1

    @pytest.fixture(scope="function", params=[2, 10])
    def load_system_with_contacts(self, request):
        n_sys = request.param
        sys_coll_with_contacts = self.SystemCollectionWithContactMixin()
        for i_sys in range(n_sys):
            sys_coll_with_contacts.append(self.MockRod(2, 3, 4, 5))
        return sys_coll_with_contacts

    """ The following calls test get_system_index from BaseSystem indirectly,
    and are here because of legacy reasons. I have not removed them because there
    are Contacts require testing against multiple indices, which is still use
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
    def test_contact_with_illegal_index_throws(
        self, load_system_with_contacts, sys_idx
    ):
        system_collection_with_contacts = load_system_with_contacts

        with pytest.raises(AssertionError) as excinfo:
            system_collection_with_contacts.detect_contact_between(*sys_idx)
        assert "exceeds number of" in str(excinfo.value)

        with pytest.raises(AssertionError) as excinfo:
            system_collection_with_contacts.detect_contact_between(
                *[np.int32(x) for x in sys_idx]
            )
        assert "exceeds number of" in str(excinfo.value)

    def test_contact_with_unregistered_system_throws(self, load_system_with_contacts):
        system_collection_with_contacts = load_system_with_contacts

        # Register this rod
        mock_rod_registered = self.MockRod(5, 5, 5, 5)
        system_collection_with_contacts.append(mock_rod_registered)
        # Don't register this rod
        mock_rod = self.MockRod(2, 3, 4, 5)

        with pytest.raises(ValueError) as excinfo:
            system_collection_with_contacts.detect_contact_between(
                mock_rod, mock_rod_registered
            )
        assert "was not found, did you" in str(excinfo.value)

        # Switch arguments
        with pytest.raises(ValueError) as excinfo:
            system_collection_with_contacts.detect_contact_between(
                mock_rod_registered, mock_rod
            )
        assert "was not found, did you" in str(excinfo.value)

    def test_contact_with_illegal_system_throws(self, load_system_with_contacts):
        system_collection_with_contacts = load_system_with_contacts

        # Register this rod
        mock_rod_registered = self.MockRod(5, 5, 5, 5)
        system_collection_with_contacts.append(mock_rod_registered)

        # Not a rod, but a list!
        mock_rod = [1, 2, 3, 5]

        with pytest.raises(TypeError) as excinfo:
            system_collection_with_contacts.detect_contact_between(
                mock_rod, mock_rod_registered
            )
        assert "not a sys" in str(excinfo.value)

        # Switch arguments
        with pytest.raises(TypeError) as excinfo:
            system_collection_with_contacts.detect_contact_between(
                mock_rod_registered, mock_rod
            )
        assert "not a sys" in str(excinfo.value)

    """
    END of testing BaseSystem calls
    """

    def test_contact_registers_and_returns_Contact(self, load_system_with_contacts):
        system_collection_with_contacts = load_system_with_contacts

        mock_rod_one = self.MockRod(2, 3, 4, 5)
        system_collection_with_contacts.append(mock_rod_one)

        mock_rod_two = self.MockRod(4, 5)
        system_collection_with_contacts.append(mock_rod_two)

        _mock_contact = system_collection_with_contacts.detect_contact_between(
            mock_rod_one, mock_rod_two
        )
        assert _mock_contact in system_collection_with_contacts._contacts
        assert _mock_contact.__class__ == _Contact

    from elastica.contact_forces import NoContact

    @pytest.fixture
    def load_rod_with_contacts(self, load_system_with_contacts):
        system_collection_with_contacts = load_system_with_contacts

        mock_rod_one = self.MockRod(2, 3, 4, 5)
        system_collection_with_contacts.append(mock_rod_one)
        mock_rod_two = self.MockRod(5.0, 5.0)
        system_collection_with_contacts.append(mock_rod_two)

        def mock_init(self, *args, **kwargs):
            pass

        # in place class
        class MockContact(self.NoContact):
            def __init__(self, *args, **kwargs):
                pass

            @property
            def _allowed_system_one(self):
                return [TestContactMixin.MockRod]

            @property
            def _allowed_system_two(self):
                return [TestContactMixin.MockRod]

        # Constrain any and all systems
        system_collection_with_contacts.detect_contact_between(0, 1).using(
            MockContact
        )  # index based contact
        system_collection_with_contacts.detect_contact_between(
            mock_rod_one, mock_rod_two
        ).using(
            MockContact
        )  # system based contact
        system_collection_with_contacts.detect_contact_between(0, mock_rod_one).using(
            MockContact
        )  # index/system based contact
        return system_collection_with_contacts, MockContact

    def test_contact_finalize_correctness(self, load_rod_with_contacts):
        system_collection_with_contacts, contact_cls = load_rod_with_contacts
        contact = system_collection_with_contacts._contacts[0].instantiate()
        fidx, sidx = system_collection_with_contacts._contacts[0].id()

        system_collection_with_contacts._finalize_contact()

        assert not hasattr(system_collection_with_contacts, "_contacts")
        assert type(fidx) is int
        assert type(sidx) is int
        assert type(contact) is contact_cls

    @pytest.fixture
    def load_contact_objects_with_incorrect_order(self, load_system_with_contacts):
        system_collection_with_contacts = load_system_with_contacts

        mock_rod = self.MockRod(2, 3, 4, 5)
        system_collection_with_contacts.append(mock_rod)
        mock_rigid_body = self.MockRigidBody(5.0, 5.0)
        system_collection_with_contacts.append(mock_rigid_body)

        def mock_init(self, *args, **kwargs):
            pass

        # in place class
        class MockContact(self.NoContact):
            def __init__(self, *args, **kwargs):
                pass

            @property
            def _allowed_system_one(self):
                return [TestContactMixin.MockRod]

            @property
            def _allowed_system_two(self):
                return [TestContactMixin.MockRigidBody]

        # incorrect order contact
        system_collection_with_contacts.detect_contact_between(
            mock_rigid_body, mock_rod
        ).using(
            MockContact
        )  # rigid body before rod

        return system_collection_with_contacts, MockContact

    def test_contact_check_order(self, load_contact_objects_with_incorrect_order):
        (
            system_collection_with_contacts,
            contact_cls,
        ) = load_contact_objects_with_incorrect_order

        with pytest.raises(TypeError) as excinfo:
            system_collection_with_contacts._finalize_contact()
        assert (
            "System provided (MockRigidBody) must be derived from ['MockRod']"
            in str(excinfo.value)
        )

    @pytest.fixture
    def load_system_with_rods_in_contact(self, load_system_with_contacts):
        system_collection_with_rods_in_contact = load_system_with_contacts

        mock_rod_one = self.MockRod(1.0, 2.0, 3.0, 4.0)
        system_collection_with_rods_in_contact.append(mock_rod_one)
        mock_rod_two = self.MockRod(1.0, 1.0)
        "Move second rod above first rod to make contact in parallel"
        mock_rod_two.position_collection = np.array(
            [[1, 2, 3], [0.5, 0.5, 0.5], [0, 0, 0]]
        )
        system_collection_with_rods_in_contact.append(mock_rod_two)

        # in place class
        from elastica.contact_forces import RodRodContact

        # Constrain any and all systems
        system_collection_with_rods_in_contact.detect_contact_between(
            mock_rod_one, mock_rod_two
        ).using(
            RodRodContact,
            k=1.0,
            nu=0.1,
        )
        return system_collection_with_rods_in_contact

    def test_contact_call_on_systems(self, load_system_with_rods_in_contact):
        from elastica.contact_forces import _calculate_contact_forces_rod_rod

        system_collection_with_rods_in_contact = load_system_with_rods_in_contact
        mock_contacts = [c for c in system_collection_with_rods_in_contact._contacts]

        system_collection_with_rods_in_contact._finalize_contact()
        system_collection_with_rods_in_contact.synchronize(time=0)

        for _contact in mock_contacts:
            fidx, sidx = _contact.id()
            contact = _contact.instantiate()

            system_one = system_collection_with_rods_in_contact[fidx]
            system_two = system_collection_with_rods_in_contact[sidx]
            external_forces_system_one = np.zeros_like(system_one.external_forces)
            external_forces_system_two = np.zeros_like(system_two.external_forces)

            _calculate_contact_forces_rod_rod(
                system_one.position_collection[
                    ..., :-1
                ],  # Discount last node, we want element start position
                system_one.radius,
                system_one.lengths,
                system_one.tangents,
                system_one.velocity_collection,
                system_one.internal_forces,
                external_forces_system_one,
                system_two.position_collection[
                    ..., :-1
                ],  # Discount last node, we want element start position
                system_two.radius,
                system_two.lengths,
                system_two.tangents,
                system_two.velocity_collection,
                system_two.internal_forces,
                external_forces_system_two,
                contact.k,
                contact.nu,
            )

            assert_allclose(
                system_collection_with_rods_in_contact[fidx].external_forces,
                external_forces_system_one,
                atol=Tolerance.atol(),
            )
            assert_allclose(
                system_collection_with_rods_in_contact[sidx].external_forces,
                external_forces_system_two,
                atol=Tolerance.atol(),
            )
