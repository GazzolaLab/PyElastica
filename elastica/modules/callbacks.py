__doc__ = """
CallBacks
-----------

Provides the callBack interface to collect data over time (see `callback_functions.py`).
"""
from typing import Type, Any
from typing_extensions import Self  # 3.11: from typing import Self
from elastica.typing import SystemType, SystemIdxType, OperatorFinalizeType
from .protocol import ModuleProtocol

import functools

import numpy as np

from elastica.callback_functions import CallBackBaseClass
from .protocol import SystemCollectionWithCallbackProtocol


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
        self: SystemCollectionWithCallbackProtocol, system: SystemType
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
        sys_idx: SystemIdxType = self.get_system_index(system)

        # Create _Constraint object, cache it and return to user
        _callback: ModuleProtocol = _CallBack(sys_idx)
        self._callback_list.append(_callback)
        self._feature_group_callback.append_id(_callback)

        return _callback

    def _finalize_callback(self: SystemCollectionWithCallbackProtocol) -> None:
        # dev : the first index stores the rod index to collect data.
        for callback in self._callback_list:
            sys_id = callback.id()
            callback_instance = callback.instantiate()

            callback_operator = functools.partial(
                callback_instance.make_callback, system=self[sys_id]
            )
            self._feature_group_callback.add_operators(callback, [callback_operator])

        self._callback_list.clear()
        del self._callback_list

        # First callback execution
        self.apply_callbacks(time=np.float64(0.0), current_step=0)


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

    def __init__(self, sys_idx: SystemIdxType):
        """

        Parameters
        ----------
        sys_idx: int
            rod object index
        """
        self._sys_idx: SystemIdxType = sys_idx
        self._callback_cls: Type[CallBackBaseClass]
        self._args: Any
        self._kwargs: Any

    def using(
        self,
        cls: Type[CallBackBaseClass],
        *args: Any,
        **kwargs: Any,
    ) -> Self:
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
        return self

    def id(self) -> SystemIdxType:
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
