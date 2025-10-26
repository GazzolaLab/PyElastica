from __future__ import annotations

__doc__ = """
CallBacks
-----------

Provides the callBack interface to collect data over time (see `callback_functions.py`).
"""
from types import EllipsisType
from typing import Type, Any, TypeAlias, cast
from elastica.typing import (
    SystemType,
    SystemIdxType,
    OperatorFinalizeType,
    SystemProtocol,
)
from .protocol import ModuleProtocol

import functools

import numpy as np

from elastica.callback_functions import CallBackBaseClass
from .protocol import SystemCollectionWithCallbackProtocol


SystemIdxDSType: TypeAlias = """
(
    SystemIdxType
    | tuple[SystemIdxType, ...]
    | list[SystemIdxType]
    | dict[Any, SystemIdxType]
)
"""

SystemDSType: TypeAlias = """
(
    SystemType | tuple[SystemType, ...] | list[SystemType] | dict[Any, SystemType]
)
"""


class CallBacks:
    """
    CallBacks class is a module for calling callback functions, set by the user. If the user
    wants to collect data from the simulation, the simulator class has to be derived
    from the CallBacks class.

        Attributes
        ----------
        _callback_list: list
            List of call back classes defined for rod-like objects.
    """

    def __init__(self: SystemCollectionWithCallbackProtocol) -> None:
        self._callback_list: list[ModuleProtocol] = []
        super(CallBacks, self).__init__()
        self._feature_group_finalize.append(self._finalize_callback)

    def collect_diagnostics(
        self: SystemCollectionWithCallbackProtocol,
        system: SystemDSType | EllipsisType,
    ) -> ModuleProtocol:
        """
        This method calls user-defined call-back classes for a
        user-defined system or rod-like object. You need to input the
        system or rod-like object that you want to collect data from.

        Parameters
        ----------
        system: object
            System is a rod-like object.

        Returns
        -------

        """
        sys_idx: SystemIdxDSType
        if system is Ellipsis:
            sys_idx = tuple([self.get_system_index(sys) for sys in self])
        elif isinstance(system, list):
            sys_idx = [self.get_system_index(sys) for sys in system]
        elif isinstance(system, dict):
            sys_idx = {key: self.get_system_index(sys) for key, sys in system.items()}
        elif isinstance(system, tuple):
            sys_idx = tuple([self.get_system_index(sys) for sys in system])
        else:
            # Single entity
            sys_idx = self.get_system_index(system)

        # Create _Constraint object, cache it and return to user
        _callback: ModuleProtocol = _CallBack(sys_idx)
        self._callback_list.append(_callback)
        self._feature_group_callback.append_id(_callback)
        self._feature_group_on_close.append_id(_callback)

        return _callback

    def _finalize_callback(self: SystemCollectionWithCallbackProtocol) -> None:
        # dev : the first index stores the rod index to collect data.
        for callback in self._callback_list:
            sys_id = callback.id()
            callback_instance = callback.instantiate()

            system: SystemDSType
            if isinstance(sys_id, (tuple, list)):
                _T = type(sys_id)
                system = _T([self[sys_id_] for sys_id_ in sys_id])
            elif isinstance(sys_id, dict):
                sys_id = cast(dict[Any, SystemIdxType], sys_id)
                system = {key: self[sys_id_] for key, sys_id_ in sys_id.items()}
            else:
                system = self[sys_id]

            callback_operator = functools.partial(
                callback_instance.make_callback, system=system
            )
            self._feature_group_callback.add_operators(callback, [callback_operator])
            self._feature_group_on_close.add_operators(
                callback, [callback_instance.on_close]
            )

        self._callback_list.clear()
        del self._callback_list


class _CallBack:
    """
    CallBack module private class

        Attributes
        ----------
        _sys_idx: rod object index
        _callback_cls: list
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
    """

    def __init__(self, sys_idx: SystemIdxDSType):
        """

        Parameters
        ----------
        sys_idx: int
            rod object index
        """
        self._sys_idx: SystemIdxDSType = sys_idx
        self._callback_cls: Type[CallBackBaseClass]
        self._args: Any
        self._kwargs: Any

    def using(
        self,
        cls: Type[CallBackBaseClass],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        This method is a module to set which callback class is used to collect data
        from user defined rod-like object.

        Parameters
        ----------
        cls: object
            User defined callback class.

        Returns
        -------

        """
        assert issubclass(
            cls, CallBackBaseClass
        ), "{} is not a valid call back. Did you forget to derive from CallBackClass?".format(
            cls
        )
        self._callback_cls = cls
        self._args = args
        self._kwargs = kwargs

    def id(self) -> SystemIdxDSType:
        return self._sys_idx

    def instantiate(self) -> CallBackBaseClass:
        """Constructs a callback functions after checks"""
        if not hasattr(self, "_callback_cls"):
            raise RuntimeError(
                "No callback provided to act on rod id {0}"
                "but a callback was registered. Did you forget to call"
                "the `using` method".format(self.id())
            )

        try:
            return self._callback_cls(*self._args, **self._kwargs)
        except (TypeError, IndexError):
            raise TypeError(
                r"Unable to construct callback class.\n"
                r"Did you provide all necessary callback properties?"
            )
