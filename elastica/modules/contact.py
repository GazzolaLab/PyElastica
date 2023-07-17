__doc__ = """
Contact
-------

Provides the contact interface to apply contact forces between objects
(rods, rigid bodies, surfaces).
"""


class Contact:
    """
    The Contact class is a module for applying contact between rod-like objects . To apply contact between rod-like objects,
    the simulator class must be derived from the Contact class.

        Attributes
        ----------
        _contacts: list
            List of contact classes defined for rod-like objects.
    """

    def __init__(self):
        self._contacts = []
        super(Contact, self).__init__()
        self._feature_group_synchronize.append(self._call_contacts)
        self._feature_group_finalize.append(self._finalize_contact)

    def add_contact_to(self, first_system, second_system):
        """
        This method adds contact between two objects using the selected contact class.
        You need to input the two objects that are to be connected.

        Parameters
        ----------
        first_rod : object
            Rod-like object
        second_rod : object
            Rod-like object or surface object

        Returns
        -------

        """
        sys_idx = [None] * 2
        for i_sys, sys in enumerate((first_system, second_system)):
            sys_idx[i_sys] = self._get_sys_idx_if_valid(sys)

        # Create _Constraint object, cache it and return to user
        _contact = _Contact(*sys_idx)
        self._contacts.append(_contact)

        return _contact

    def _finalize_contact(self):

        # dev : the first indices stores the
        # (first_rod_idx, second_rod_idx)
        # to apply the contacts to
        # Technically we can use another array but it its one more book-keeping
        # step. Being lazy, I put them both in the same array
        self._contacts[:] = [(*contact.id(), contact()) for contact in self._contacts]

        for (
            first_sys_idx,
            second_sys_idx,
            contact,
        ) in self._contacts:
            contact._generate_contact_function(
                self._systems[first_sys_idx],
                self._systems[second_sys_idx],
            )

    def _call_contacts(self, time, *args, **kwargs):
        for (
            first_sys_idx,
            second_sys_idx,
            contact,
        ) in self._contacts:
            contact.apply_contact(
                self._systems[first_sys_idx],
                self._systems[second_sys_idx],
                *args,
                **kwargs,
            )


class _Contact:
    """
    Contact module private class

    Attributes
    ----------
    _first_sys_idx: int
    _second_sys_idx: int
    _contact_cls: list
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.
    """

    def __init__(
        self,
        first_sys_idx: int,
        second_sys_idx: int,
    ):
        """

        Parameters
        ----------
        first_sys_idx
        second_sys_idx
        """
        self.first_sys_idx = first_sys_idx
        self.second_sys_idx = second_sys_idx
        self._contact_cls = None
        self._args = ()
        self._kwargs = {}

    def using(self, contact_cls, *args, **kwargs):
        """
        This method is a module to set which contact class is used to apply contact
        between user defined rod-like objects.

        Parameters
        ----------
        contact_cls: object
            User defined callback class.
        *args
            Variable length argument list
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------

        """
        from elastica.contact_forces import NoContact

        assert issubclass(
            contact_cls, NoContact
        ), "{} is not a valid contact class. Did you forget to derive from NoContact?".format(
            contact_cls
        )
        self._contact_cls = contact_cls
        self._args = args
        self._kwargs = kwargs
        return self

    def id(self):
        return (
            self.first_sys_idx,
            self.second_sys_idx,
        )

    def __call__(self, *args, **kwargs):
        if not self._contact_cls:
            raise RuntimeError(
                "No contacts provided to link rod-like object id {0}"
                " and {1}, but a Contact"
                "was intended as per code. Did you forget to"
                "call the `using` method?".format(*self.id())
            )

        try:
            return self._contact_cls(*self._args, **self._kwargs)
        except (TypeError, IndexError):
            raise TypeError(
                r"Unable to construct contact class.\n"
                r"Did you provide all necessary contact properties?"
            )
