__doc__ = """
Contact
-------

Provides the contact interface to apply contact forces between objects
(rods, rigid bodies, surfaces).
"""
from typing import Type, Any
from typing_extensions import Self

import functools
from elastica.typing import (
    SystemIdxType,
    OperatorType,
    StaticSystemType,
    SystemType,
)
from .protocol import ContactedSystemCollectionProtocol, ModuleProtocol

import logging

import numpy as np

from elastica.contact_forces import NoContact

logger = logging.getLogger(__name__)


def warnings() -> None:
    logger.warning("Contact features should be instantiated lastly.")


class Contact:
    """
    The Contact class is a module for applying contact between rod-like objects . To apply contact between rod-like objects,
    the simulator class must be derived from the Contact class.

        Attributes
        ----------
        _contacts: list
            List of contact classes defined for rod-like objects.
    """

    def __init__(self: ContactedSystemCollectionProtocol) -> None:
        self._contacts: list[ModuleProtocol] = []
        super(Contact, self).__init__()
        self._feature_group_finalize.append(self._finalize_contact)

    def detect_contact_between(
        self: ContactedSystemCollectionProtocol,
        first_system: SystemType,
        second_system: "SystemType | StaticSystemType",
    ) -> ModuleProtocol:
        """
        This method adds contact detection between two objects using the selected contact class.
        You need to input the two objects that are to be connected.

        Parameters
        ----------
        first_system : SystemType
        second_system : SystemType | StaticSystemType

        Returns
        -------

        """
        sys_idx_first = self.get_system_index(first_system)
        sys_idx_second = self.get_system_index(second_system)

        # Create _Contact object, cache it and return to user
        _contact = _Contact(sys_idx_first, sys_idx_second)
        self._contacts.append(_contact)
        self._feature_group_synchronize.append_id(_contact)

        return _contact

    def _finalize_contact(self: ContactedSystemCollectionProtocol) -> None:

        # dev : the first indices stores the
        # (first_rod_idx, second_rod_idx)
        # to apply the contacts to

        for contact in self._contacts:
            first_sys_idx, second_sys_idx = contact.id()
            contact_instance = contact.instantiate()

            contact_instance._check_systems_validity(
                self[first_sys_idx],
                self[second_sys_idx],
            )

            func = functools.partial(
                contact_instance.apply_contact,
                system_one=self[first_sys_idx],
                system_two=self[second_sys_idx],
            )

            self._feature_group_synchronize.add_operators(contact, [func])

            if not self._feature_group_synchronize.is_last(contact):
                warnings()

        self._contacts = []
        del self._contacts


class _Contact:
    """
    Contact module private class

    Attributes
    ----------
    _first_sys_idx: SystemIdxType
    _second_sys_idx: SystemIdxType
    _contact_cls: Type[NoContact]
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.
    """

    def __init__(
        self,
        first_sys_idx: SystemIdxType,
        second_sys_idx: SystemIdxType,
    ) -> None:
        """

        Parameters
        ----------
        first_sys_idx
        second_sys_idx
        """
        self.first_sys_idx = first_sys_idx
        self.second_sys_idx = second_sys_idx
        self._contact_cls: Type[NoContact]
        self._args: Any
        self._kwargs: Any

    def using(self, cls: Type[NoContact], *args: Any, **kwargs: Any) -> Self:
        """
        This method is a module to set which contact class is used to apply contact
        between user defined rod-like objects.

        Parameters
        ----------
        cls: Type[NoContact]
            User defined contact class.
        *args
            Variable length argument list
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------

        """
        assert issubclass(
            cls, NoContact
        ), "{} is not a valid contact class. Did you forget to derive from NoContact?".format(
            cls
        )
        self._contact_cls = cls
        self._args = args
        self._kwargs = kwargs
        return self

    def id(self) -> Any:
        return (
            self.first_sys_idx,
            self.second_sys_idx,
        )

    def instantiate(self) -> NoContact:
        if not hasattr(self, "_contact_cls"):
            raise RuntimeError(
                "No contacts provided to to establish contact between rod-like object id {0}"
                " and {1}, but a Contact"
                " was intended as per code. Did you forget to"
                " call the `using` method?".format(*self.id())
            )

        try:
            return self._contact_cls(*self._args, **self._kwargs)
        except (TypeError, IndexError):
            raise TypeError(
                r"Unable to construct contact class.\n"
                r"Did you provide all necessary contact properties?"
            )
