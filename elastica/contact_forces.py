__doc__ = """ Numba implementation module containing contact between rods and rigid bodies and other rods rigid bodies or surfaces."""

from elastica.typing import SystemType, AllowedContactType


class NoContact:
    """
    This is the base class for contact applied between rod-like objects and allowed contact objects.

    Notes
    -----
    Every new contact class must be derived
    from NoContact class.

    """

    def __init__(self):
        """

        Parameters
        ----------

        """

    def _generate_contact_function(self, system_one, system_two):
        self._apply_contact = NotImplementedError

    def apply_contact(
        self,
        system_one: SystemType,
        system_two: AllowedContactType,
    ):
        """
        Apply contact forces and torques between SystemType object and AllowedContactType object.

        In NoContact class, this routine simply passes.

        Parameters
        ----------
        system_one : SystemType
            Rod or rigid-body object
        system_two : AllowedContactType
            Rod, rigid-body or surface object
        Returns
        -------

        """
        self._apply_contact(system_one, system_two)

        return
