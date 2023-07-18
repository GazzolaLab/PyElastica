__doc__ = """ Test modules for contact """
import numpy as np
import pytest

from elastica.modules import Contact
from elastica.modules.contact import _Contact


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
        assert "not a valid contact class" in str(excinfo.value)

    from elastica.contact_forces import NoContact

    # TODO Add other legal contact later
    @pytest.mark.parametrize("legal_contact", [NoContact])
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
            contact()
        assert "No contacts provided" in str(excinfo.value)

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
            _ = contact()
        assert "Unable to construct" in str(excinfo.value)


class TestContactMixin:
    from elastica.modules import BaseSystemCollection

    class SystemCollectionWithContactMixedin(BaseSystemCollection, Contact):
        pass

    from elastica.rod import RodBase

    class MockRod(RodBase):
        def __init__(self, *args, **kwargs):
            self.n_elems = 3  # arbitrary number

        # Contacts assume that this promise is met
        def __len__(self):
            return 2  # a random number

    @pytest.fixture(scope="function", params=[2, 10])
    def load_system_with_contacts(self, request):
        n_sys = request.param
        sys_coll_with_contacts = self.SystemCollectionWithContactMixedin()
        for i_sys in range(n_sys):
            sys_coll_with_contacts.append(self.MockRod(2, 3, 4, 5))
        return sys_coll_with_contacts

    """ The following calls test _get_sys_idx_if_valid from BaseSystem indirectly,
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
        scwc = load_system_with_contacts

        with pytest.raises(AssertionError) as excinfo:
            scwc.add_contact_to(*sys_idx)
        assert "exceeds number of" in str(excinfo.value)

        with pytest.raises(AssertionError) as excinfo:
            scwc.add_contact_to(*[np.int_(x) for x in sys_idx])
        assert "exceeds number of" in str(excinfo.value)

    def test_contact_with_unregistered_system_throws(self, load_system_with_contacts):
        scwc = load_system_with_contacts

        # Register this rod
        mock_rod_registered = self.MockRod(5, 5, 5, 5)
        scwc.append(mock_rod_registered)
        # Don't register this rod
        mock_rod = self.MockRod(2, 3, 4, 5)

        with pytest.raises(ValueError) as excinfo:
            scwc.add_contact_to(mock_rod, mock_rod_registered)
        assert "was not found, did you" in str(excinfo.value)

        # Switch arguments
        with pytest.raises(ValueError) as excinfo:
            scwc.add_contact_to(mock_rod_registered, mock_rod)
        assert "was not found, did you" in str(excinfo.value)

    def test_contact_with_illegal_system_throws(self, load_system_with_contacts):
        scwc = load_system_with_contacts

        # Register this rod
        mock_rod_registered = self.MockRod(5, 5, 5, 5)
        scwc.append(mock_rod_registered)

        # Not a rod, but a list!
        mock_rod = [1, 2, 3, 5]

        with pytest.raises(TypeError) as excinfo:
            scwc.add_contact_to(mock_rod, mock_rod_registered)
        assert "not a sys" in str(excinfo.value)

        # Switch arguments
        with pytest.raises(TypeError) as excinfo:
            scwc.add_contact_to(mock_rod_registered, mock_rod)
        assert "not a sys" in str(excinfo.value)

    """
    END of testing BaseSystem calls
    """

    def test_contact_registers_and_returns_Contact(self, load_system_with_contacts):
        scwc = load_system_with_contacts

        mock_rod_one = self.MockRod(2, 3, 4, 5)
        scwc.append(mock_rod_one)

        mock_rod_two = self.MockRod(4, 5)
        scwc.append(mock_rod_two)

        _mock_contact = scwc.add_contact_to(mock_rod_one, mock_rod_two)
        assert _mock_contact in scwc._contacts
        assert _mock_contact.__class__ == _Contact

    from elastica.contact_forces import NoContact

    @pytest.fixture
    def load_rod_with_contacts(self, load_system_with_contacts):
        scwc = load_system_with_contacts

        mock_rod_one = self.MockRod(2, 3, 4, 5)
        scwc.append(mock_rod_one)
        mock_rod_two = self.MockRod(5.0, 5.0)
        scwc.append(mock_rod_two)

        def mock_init(self, *args, **kwargs):
            pass

        # in place class
        MockContact = type(
            "MockContact", (self.NoContact, object), {"__init__": mock_init}
        )

        # Constrain any and all systems
        scwc.add_contact_to(0, 1).using(MockContact)  # index based contact
        scwc.add_contact_to(mock_rod_one, mock_rod_two).using(
            MockContact
        )  # system based contact
        scwc.add_contact_to(0, mock_rod_one).using(
            MockContact
        )  # index/system based contact

        return scwc, MockContact

    def test_contact_finalize_correctness(self, load_rod_with_contacts):
        scwc, contact_cls = load_rod_with_contacts

        scwc._finalize_contact()

        for (fidx, sidx, contact) in scwc._contacts:
            assert type(fidx) is int
            assert type(sidx) is int
            assert type(contact) is contact_cls

    def test_contact_call_on_systems(self):
        # TODO Finish after the architecture is complete
        pass
