__doc__ = """ Numba implementation module containing contact between rods and rigid bodies and other rods rigid bodies or surfaces."""

from elastica.typing import SystemType, AllowedContactType
from elastica.rod import RodBase


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

    def _order_check(
        self,
        system_one,
        system_two,
    ):
        """
        This checks the contact order between a SystemType object and an AllowedContactType object, the order should follow: Rod, Rigid body, Surface.
        In NoContact class, this just checks if system_two is a rod then system_one must be a rod.


        Parameters
        ----------
        system_one
            SystemType
        system_two
            AllowedContactType
        Returns
        -------

        """
        if issubclass(system_two.__class__, RodBase):
            if not issubclass(system_one.__class__, RodBase):
                raise TypeError(
                    "Systems provided to the contact class have incorrect order. First system is {0} and second system is {1} If the first system is a rod, the second system can be a rod, rigid body or surface. If the first system is a rigid body, the second system can be a rigid body or surface.".format(
                        system_one.__class__, system_two.__class__
                    )
                )

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
            Rod, rigid-body, or surface object
        Returns
        -------

        """
        pass
