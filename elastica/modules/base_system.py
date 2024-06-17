__doc__ = """
Base System
-----------

Basic coordinating for multiple, smaller systems that have an independently integrable
interface (i.e. works with symplectic or explicit routines `timestepper.py`.)
"""
from typing import Type, Generator, Iterable, Any, overload
from typing import final
from elastica.typing import (
    StaticSystemType,
    SystemType,
    SystemIdxType,
    OperatorType,
    OperatorCallbackType,
    OperatorFinalizeType,
)

import numpy as np

from collections.abc import MutableSequence

from elastica.rod.rod_base import RodBase
from elastica.rigidbody.rigid_body import RigidBodyBase
from elastica.surface.surface_base import SurfaceBase

from .memory_block import construct_memory_block_structures
from .operator_group import OperatorGroupFIFO


class BaseSystemCollection(MutableSequence):
    """
    Base System for simulator classes. Every simulation class written by the user
    must be derived from the BaseSystemCollection class; otherwise the simulation will
    proceed.

        Attributes
        ----------
        allowed_sys_types: tuple
            Tuple of allowed type rod-like objects. Here use a base class for objects, i.e. RodBase.
        _systems: list
            List of rod-like objects.

    Note
    ----
    We can directly subclass a list for the
    most part, but this is a bad idea, as List is non abstract
    https://stackoverflow.com/q/3945940
    """

    def __init__(self) -> None:
        # Collection of functions. Each group is executed as a collection at the different steps.
        # Each component (Forcing, Connection, etc.) registers the executable (callable) function
        # in the group that that needs to be executed. These should be initialized before mixin.
        self._feature_group_synchronize: Iterable[OperatorType] = OperatorGroupFIFO()
        self._feature_group_constrain_values: list[OperatorType] = []
        self._feature_group_constrain_rates: list[OperatorType] = []
        self._feature_group_callback: list[OperatorCallbackType] = []
        self._feature_group_finalize: list[OperatorFinalizeType] = []
        # We need to initialize our mixin classes
        super().__init__()

        # List of system types/bases that are allowed
        self.allowed_sys_types: tuple[Type, ...] = (
            RodBase,
            RigidBodyBase,
            SurfaceBase,
        )

        # List of systems to be integrated
        self._systems: list[StaticSystemType] = []
        self.__final_systems: list[SystemType] = []

        # Flag Finalize: Finalizing twice will cause an error,
        # but the error message is very misleading
        self._finalize_flag: bool = False

    @final
    def _check_type(self, sys_to_be_added: Any) -> bool:
        if not isinstance(sys_to_be_added, self.allowed_sys_types):
            raise TypeError(
                "{0}\n"
                "is not a system passing validity\n"
                "checks, that can be added into BaseSystem. If you are sure that\n"
                "{0}\n"
                "satisfies all criteria for being a system, please add\n"
                "it using BaseSystem.extend_allowed_types.\n"
                "The allowed types are\n"
                "{1}".format(sys_to_be_added.__class__, self.allowed_sys_types)
            )
        if not all(
            isinstance(self, req)
            for req in getattr(sys_to_be_added, "REQUISITE_MODULES", [])
        ):
            raise RuntimeError(
                f"The system {sys_to_be_added.__class__} requires the following modules:\n"
                f"{sys_to_be_added.REQUISITE_MODULES}\n"
            )
        return True

    def __len__(self) -> int:
        return len(self._systems)

    @overload
    def __getitem__(self, idx: int, /) -> SystemType: ...

    @overload
    def __getitem__(self, idx: slice, /) -> list[SystemType]: ...

    def __getitem__(self, idx, /):  # type: ignore
        return self._systems[idx]

    def __delitem__(self, idx, /):  # type: ignore
        del self._systems[idx]

    def __setitem__(self, idx, system, /):  # type: ignore
        self._check_type(system)
        self._systems[idx] = system

    def insert(self, idx, system) -> None:  # type: ignore
        self._check_type(system)
        self._systems.insert(idx, system)

    def __str__(self) -> str:
        """To be readable"""
        return str(self._systems)

    @final
    def extend_allowed_types(
        self, additional_types: tuple[Type[SystemType], ...]
    ) -> None:
        self.allowed_sys_types += additional_types

    @final
    def override_allowed_types(
        self, allowed_types: tuple[Type[SystemType], ...]
    ) -> None:
        self.allowed_sys_types = allowed_types

    @final
    def _get_sys_idx_if_valid(
        self, sys_to_be_added: "SystemType | StaticSystemType"
    ) -> SystemIdxType:
        n_systems = len(self)  # Total number of systems from mixed-in class

        sys_idx: SystemIdxType
        if isinstance(sys_to_be_added, (int, np.int_)):
            # 1. If they are indices themselves, check range
            assert (
                -n_systems <= sys_to_be_added < n_systems
            ), "Rod index {} exceeds number of registered rodtems".format(
                sys_to_be_added
            )
            sys_idx = int(sys_to_be_added)
        elif self._check_type(sys_to_be_added):
            # 2. If they are rod objects (most likely), lookup indices
            # index might have some problems : https://stackoverflow.com/a/176921
            try:
                sys_idx = self._systems.index(sys_to_be_added)
            except ValueError:
                raise ValueError(
                    "Rod {} was not found, did you append it to the system?".format(
                        sys_to_be_added
                    )
                )

        return sys_idx

    @final
    def systems(self) -> Generator[SystemType, None, None]:
        # assert self._finalize_flag, "The simulator is not finalized."
        for block in self.__final_systems:
            yield block

    @final
    def finalize(self) -> None:
        """
        This method finalizes the simulator class. When it is called, it is assumed that the user has appended
        all rod-like objects to the simulator as well as all boundary conditions, callbacks, etc.,
        acting on these rod-like objects. After the finalize method called,
        the user cannot add new features to the simulator class.
        """

        assert not self._finalize_flag, "The finalize cannot be called twice."
        self._finalize_flag = True

        # construct memory block
        self.__final_systems = construct_memory_block_structures(self._systems)
        # TODO: try to remove the _systems list for memory optimization
        # self._systems.clear()
        # del self._systems

        # Recurrent call finalize functions for all components.
        for finalize in self._feature_group_finalize:
            finalize()

        # Clear the finalize feature group, just for the safety.
        self._feature_group_finalize.clear()
        del self._feature_group_finalize

    @final
    def synchronize(self, time: np.floating) -> None:
        # Collection call _feature_group_synchronize
        for func in self._feature_group_synchronize:
            func(time=time)

    @final
    def constrain_values(self, time: np.floating) -> None:
        # Collection call _feature_group_constrain_values
        for func in self._feature_group_constrain_values:
            func(time=time)

    @final
    def constrain_rates(self, time: np.floating) -> None:
        # Collection call _feature_group_constrain_rates
        for func in self._feature_group_constrain_rates:
            func(time=time)

    @final
    def apply_callbacks(self, time: np.floating, current_step: int) -> None:
        # Collection call _feature_group_callback
        for func in self._feature_group_callback:
            func(time=time, current_step=current_step)
