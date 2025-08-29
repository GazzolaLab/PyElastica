__doc__ = """
Base System
-----------

Basic coordinating for multiple, smaller systems that have an independently integrable
interface (i.e. works with symplectic or explicit routines `timestepper.py`.)
"""
from typing import TYPE_CHECKING, Type, Generator, Any, overload
from typing import final
from elastica.typing import (
    SystemType,
    StaticSystemType,
    BlockSystemType,
    SystemIdxType,
    OperatorType,
    OperatorCallbackType,
    OperatorFinalizeType,
)

import numpy as np
from itertools import chain

from collections.abc import MutableSequence

from elastica.rod.rod_base import RodBase
from elastica.rigidbody.rigid_body import RigidBodyBase
from elastica.surface.surface_base import SurfaceBase

from .memory_block import construct_memory_block_structures
from .operator_group import OperatorGroupFIFO
from .protocol import ModuleProtocol


class BaseSystemCollection(MutableSequence):
    """
    Base System for simulator classes. Every simulation class written by the user
    must be derived from the BaseSystemCollection class; otherwise the simulation will
    proceed.

        Attributes
        ----------
        allowed_sys_types: tuple[Type]
            Tuple of allowed type rod-like objects. Here use a base class for objects, i.e. RodBase.
        systems: Callable
            Returns all system objects. Once finalize, block objects are also included.
        blocks: Callable
            Returns block objects. Should be called after finalize.

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
        self._feature_group_synchronize: OperatorGroupFIFO[
            OperatorType, ModuleProtocol
        ] = OperatorGroupFIFO()
        self._feature_group_constrain_values: OperatorGroupFIFO[
            OperatorType, ModuleProtocol
        ] = OperatorGroupFIFO()
        self._feature_group_constrain_rates: OperatorGroupFIFO[
            OperatorType, ModuleProtocol
        ] = OperatorGroupFIFO()
        self._feature_group_damping: OperatorGroupFIFO[OperatorType, ModuleProtocol] = (
            OperatorGroupFIFO()
        )
        self._feature_group_callback: OperatorGroupFIFO[
            OperatorCallbackType, ModuleProtocol
        ] = OperatorGroupFIFO()
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
        self.__systems: list[StaticSystemType] = []
        self.__final_blocks: list[BlockSystemType] = []

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
        return len(self.__systems)

    @overload  # type: ignore
    def __getitem__(self, idx: slice, /) -> list[SystemType]: ...  # type: ignore

    @overload  # type: ignore
    def __getitem__(self, idx: int, /) -> SystemType: ...  # type: ignore

    def __getitem__(self, idx, /):  # type: ignore
        return self.__systems[idx]

    def __delitem__(self, idx, /):  # type: ignore
        del self.__systems[idx]

    def __setitem__(self, idx, system, /):  # type: ignore
        self._check_type(system)
        self.__systems[idx] = system

    def insert(self, idx, system) -> None:  # type: ignore
        self._check_type(system)
        self.__systems.insert(idx, system)

    def __str__(self) -> str:
        """To be readable"""
        return str(self.__systems)

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
    def get_system_index(
        self, system: "SystemType | StaticSystemType"
    ) -> SystemIdxType:
        """
        Get the index of the system object in the system list.
        System list is private, so this is the only way to get the index of the system object.

        Example
        -------
        >>> system_collection: SystemCollectionProtocol
        >>> system: SystemType
        ...
        >>> system_idx = system_collection.get_system_index(system)  # save idx
        ...
        >>> system = system_collection[system_idx]  # just need idx to retrieve

        Parameters
        ----------
        system: SystemType
            System object to be found in the system list.
        """
        n_systems = len(self)  # Total number of systems from mixed-in class

        sys_idx: SystemIdxType
        if isinstance(
            system, (int, np.integer)
        ):  # np.integer includes both int32 and int64
            # 1. If they are indices themselves, check range
            # This is only used for testing purposes
            assert (
                -n_systems <= system < n_systems
            ), "System index {} exceeds number of registered rodtems".format(system)
            sys_idx = int(system)
        elif self._check_type(system):
            # 2. If they are system object (most likely), lookup indices
            # index might have some problems : https://stackoverflow.com/a/176921
            try:
                sys_idx = self.__systems.index(system)
            except ValueError:
                raise ValueError(
                    "System {} was not found, did you append it to the system?".format(
                        system
                    )
                )

        return sys_idx

    @final
    def systems(self) -> Generator[StaticSystemType, None, None]:
        """
        Iterate over all systems in the system collection.
        If the system collection is finalized, block objects are also included.
        """
        for system in self.__systems:
            yield system

    @final
    def block_systems(self) -> Generator[BlockSystemType, None, None]:
        """
        Iterate over all block systems in the system collection.
        """
        for block in self.__final_blocks:
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

        # Construct memory block
        self.__final_blocks = construct_memory_block_structures(self.__systems)
        # FIXME: We need this to make ring-rod working.
        # But probably need to be refactored
        self.__systems.extend(self.__final_blocks)

        # Recurrent call finalize functions for all components.
        for finalize in self._feature_group_finalize:
            finalize()

        # Clear the finalize feature group, just for the safety.
        self._feature_group_finalize.clear()
        del self._feature_group_finalize

    @final
    def synchronize(self, time: np.float64) -> None:
        """
        Call synchronize functions for all features.
        Features are registered in _feature_group_synchronize.
        """
        for func in self._feature_group_synchronize:
            func(time=time)

    @final
    def constrain_values(self, time: np.float64) -> None:
        """
        Call constrain values functions for all features.
        Features are registered in _feature_group_constrain_values.
        """
        for func in self._feature_group_constrain_values:
            func(time=time)

    @final
    def constrain_rates(self, time: np.float64) -> None:
        """
        Call constrain rates functions for all features.
        Features are registered in _feature_group_constrain_rates.
        """
        for func in chain(
            self._feature_group_constrain_rates, self._feature_group_damping
        ):
            func(time=time)

    @final
    def apply_callbacks(self, time: np.float64, current_step: int) -> None:
        """
        Call callback functions for all features.
        Features are registered in _feature_group_callback.
        """
        for func in self._feature_group_callback:
            func(time=time, current_step=current_step)


if TYPE_CHECKING:
    from .protocol import SystemCollectionProtocol
    from .constraints import Constraints
    from .forcing import Forcing
    from .connections import Connections
    from .contact import Contact
    from .damping import Damping
    from .callbacks import CallBacks

    class BaseFeature(BaseSystemCollection):
        pass

    class PartialFeatureA(
        BaseSystemCollection, Constraints, Forcing, Damping, CallBacks
    ):
        pass

    class PartialFeatureB(BaseSystemCollection, Contact, Connections):
        pass

    class FullFeature(
        BaseSystemCollection,
        Constraints,
        Contact,
        Connections,
        Forcing,
        Damping,
        CallBacks,
    ):
        pass

    _: SystemCollectionProtocol = FullFeature()
    _: SystemCollectionProtocol = PartialFeatureA()  # type: ignore[no-redef]
    _: SystemCollectionProtocol = PartialFeatureB()  # type: ignore[no-redef]
    _: SystemCollectionProtocol = BaseFeature()  # type: ignore[no-redef]
