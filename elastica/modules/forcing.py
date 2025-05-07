__doc__ = """
Forcing
-------

Provides the forcing interface to apply forces and torques to rod-like objects
(external point force, muscle torques, etc).
"""
import logging
import functools
from typing import Any, Type, List
from typing_extensions import Self

import numpy as np

from elastica.external_forces import NoForces
from elastica.typing import SystemType, SystemIdxType
from .protocol import ForcedSystemCollectionProtocol, ModuleProtocol

logger = logging.getLogger(__name__)


class Forcing:
    """
    The Forcing class is a module for applying boundary conditions that
    consist of applied external forces. To apply forcing on rod-like objects,
    the simulator class must be derived from the Forcing class.

        Attributes
        ----------
        _ext_forces_torques: list
            List of forcing class defined for rod-like objects.
    """

    def __init__(self: ForcedSystemCollectionProtocol) -> None:
        self._ext_forces_torques: List[ModuleProtocol] = []
        super().__init__()
        self._feature_group_finalize.append(self._finalize_forcing)

    def add_forcing_to(
        self: ForcedSystemCollectionProtocol, system: SystemType
    ) -> ModuleProtocol:
        """
        This method applies external forces and torques on the relevant
        user-defined system or rod-like object. You must input the system
        or rod-like object that you want to apply external forces and torques on.

        Parameters
        ----------
        system: object
            System is a rod-like object.

        Returns
        -------

        """
        sys_idx = self.get_system_index(system)

        # Create _Constraint object, cache it and return to user
        _ext_force_torque = _ExtForceTorque(sys_idx)
        self._ext_forces_torques.append(_ext_force_torque)
        self._feature_group_synchronize.append_id(_ext_force_torque)

        return _ext_force_torque

    def _finalize_forcing(self: ForcedSystemCollectionProtocol) -> None:
        # From stored _ExtForceTorque objects, and instantiate a Force
        # inplace : https://stackoverflow.com/a/1208792

        # dev : the first index stores the rod index to apply the boundary condition
        # to.
        for external_force_and_torque in self._ext_forces_torques:
            sys_id = external_force_and_torque.id()
            forcing_instance = external_force_and_torque.instantiate()

            apply_forces = functools.partial(
                forcing_instance.apply_forces, system=self[sys_id]
            )
            apply_torques = functools.partial(
                forcing_instance.apply_torques, system=self[sys_id]
            )

            self._feature_group_synchronize.add_operators(
                external_force_and_torque, [apply_forces, apply_torques]
            )

        self._ext_forces_torques = []
        del self._ext_forces_torques


class _ExtForceTorque:
    """
    Forcing module private class

    Attributes
    ----------
    _sys_idx: int
    _forcing_cls: Type[NoForces]
    *args: Any
        Variable length argument list.
    **kwargs: Any
        Arbitrary keyword arguments.
    """

    def __init__(self, sys_idx: SystemIdxType) -> None:
        """
        Parameters
        ----------
        sys_idx: int
        """
        self._sys_idx = sys_idx
        self._forcing_cls: Type[NoForces]
        self._args: Any
        self._kwargs: Any

    def using(self, cls: Type[NoForces], *args: Any, **kwargs: Any) -> Self:
        """
        This method sets which forcing class is used to apply forcing
        to user defined rod-like objects.

        Parameters
        ----------
        cls: Type[Any]
            User defined forcing class.
        *args: Any
            Variable length argument list.
        **kwargs: Any
            Arbitrary keyword arguments.

        Returns
        -------

        """
        assert issubclass(
            cls, NoForces
        ), "{} is not a valid forcing. Did you forget to derive from NoForces?".format(
            cls
        )
        self._forcing_cls = cls
        self._args = args
        self._kwargs = kwargs
        return self

    def id(self) -> SystemIdxType:
        return self._sys_idx

    def instantiate(self) -> NoForces:
        """Constructs a constraint after checks"""
        if not hasattr(self, "_forcing_cls"):
            raise RuntimeError(
                "No forcing provided to act on rod id {0}"
                "but a force was registered. Did you forget to call"
                "the `using` method".format(self.id())
            )

        try:
            return self._forcing_cls(*self._args, **self._kwargs)
        except (TypeError, IndexError):
            raise TypeError(
                r"Unable to construct forcing class.\n"
                r"Did you provide all necessary force properties?"
            )
