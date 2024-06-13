__doc__ = """
Damping
-------

(added in version 0.3.0)

Provides the damper interface to apply damping
on the rods. (see `dissipation.py`).

"""

from typing import Any, Type, List
from typing_extensions import Self

import numpy as np

from elastica.dissipation import DamperBase
from elastica.typing import SystemType, SystemIdxType
from .protocol import SystemCollectionProtocol, ModuleProtocol


class Damping:
    """
    The Damping class is a module for applying damping
    on rod-like objects, the simulator class must be derived from
    Damping class.

        Attributes
        ----------
        _dampers: list
            List of damper classes defined for rod-like objects.
    """

    def __init__(self: SystemCollectionProtocol) -> None:
        self._damping_list: List[ModuleProtocol] = []
        super().__init__()
        self._feature_group_constrain_rates.append(self._dampen_rates)
        self._feature_group_finalize.append(self._finalize_dampers)

    def dampen(self: SystemCollectionProtocol, system: SystemType) -> ModuleProtocol:
        """
        This method applies damping on relevant user-defined
        system or rod-like object. You must input the system or rod-like
        object that you want to apply damping on.

        Parameters
        ----------
        system: object
            System is a rod-like object.

        Returns
        -------

        """
        sys_idx = self._get_sys_idx_if_valid(system)

        # Create _Damper object, cache it and return to user
        _damper: ModuleProtocol = _Damper(sys_idx)
        self._damping_list.append(_damper)

        return _damper

    def _finalize_dampers(self: SystemCollectionProtocol) -> None:
        # From stored _Damping objects, instantiate the dissipation/damping
        # inplace : https://stackoverflow.com/a/1208792

        self._damping_operators = [
            (damper.id(), damper.instantiate(self._systems[damper.id()]))
            for damper in self._damping_list
        ]

        # Sort from lowest id to highest id for potentially better memory access
        # _dampers contains list of tuples. First element of tuple is rod number and
        # following elements are the type of damping.
        # Thus using lambda we iterate over the list of tuples and use rod number (x[0])
        # to sort dampers.
        self._damping_operators.sort(key=lambda x: x[0])

    def _dampen_rates(self: SystemCollectionProtocol, time: np.floating) -> None:
        for sys_id, damper in self._damping_operators:
            damper.dampen_rates(self._systems[sys_id], time)


class _Damper:
    """
    Damper module private class

    Attributes
    ----------
    _sys_idx: int
    _damper_cls: list
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.
    """

    def __init__(self, sys_idx: SystemIdxType) -> None:
        """

        Parameters
        ----------
        sys_idx: int

        """
        self._sys_idx = sys_idx
        self._damper_cls: Type[DamperBase]
        self._args: Any
        self._kwargs: Any

    def using(self, cls: Type[DamperBase], *args: Any, **kwargs: Any) -> Self:
        """
        This method is a module to set which damper class is used to
        enforce damping from user defined rod-like objects.

        Parameters
        ----------
        cls : Type[DamperBase]
            User defined damper class.
        *args: Any
            Variable length argument list.
        **kwargs: Any
            Arbitrary keyword arguments.

        Returns
        -------

        """
        assert issubclass(
            cls, DamperBase
        ), "{} is not a valid damper. Damper must be driven from DamperBase.".format(
            cls
        )
        self._damper_cls = cls
        self._args = args
        self._kwargs = kwargs
        return self

    def id(self) -> SystemIdxType:
        return self._sys_idx

    def instantiate(self, rod: SystemType) -> DamperBase:
        """Constructs a Damper class object after checks

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        if not hasattr(self, "_damper_cls"):
            raise RuntimeError(
                "No damper provided to dampen rod id {0} at {1},"
                "but damping was intended. Did you"
                "forget to call the `using` method?".format(self.id(), rod)
            )

        try:
            damper = self._damper_cls(*self._args, _system=rod, **self._kwargs)
            return damper
        except (TypeError, IndexError):
            raise TypeError("Unable to construct damping class.\n")
