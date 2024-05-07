__doc__ = """
Base System
-----------

Basic coordinating for multiple, smaller systems that have an independently integrable
interface (i.e. works with symplectic or explicit routines `timestepper.py`.)
"""
from typing import Iterable, Callable, AnyStr, Type, Generator
from typing import final
from elastica.typing import (
    SystemType,
    SystemIdxType,
    SynchronizeOperator,
    ConstrainValuesOperator,
    ConstrainRatesOperator,
    CallbackOperator,
    FinalizeOperator,
)

import numpy as np

from collections.abc import MutableSequence

from elastica.rod import RodBase
from elastica.rigidbody import RigidBodyBase
from elastica.surface import SurfaceBase
from elastica.modules.memory_block import construct_memory_block_structures
from elastica._synchronize_periodic_boundary import _ConstrainPeriodicBoundaries


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

    def __init__(self):
        # Collection of functions. Each group is executed as a collection at the different steps.
        # Each component (Forcing, Connection, etc.) registers the executable (callable) function
        # in the group that that needs to be executed. These should be initialized before mixin.
        self._feature_group_synchronize: list[SynchronizeOperator] = []
        self._feature_group_constrain_values: list[ConstrainValuesOperator] = []
        self._feature_group_constrain_rates: list[ConstrainRatesOperator] = []
        self._feature_group_callback: list[CallbackOperator] = []
        self._feature_group_finalize: list[FinalizeOperator] = []

        # We need to initialize our mixin classes
        super(BaseSystemCollection, self).__init__()

        # List of system types/bases that are allowed
        self.allowed_sys_types: tuple[Type[SystemType]] = (
            RodBase,
            RigidBodyBase,
            SurfaceBase,
        )

        # List of systems to be integrated
        self._systems: list[SystemType] = []
        self._memory_blocks: list[SystemType] = []

        # Flag Finalize: Finalizing twice will cause an error,
        # but the error message is very misleading
        self._finalize_flag: bool = False

    @final
    def _check_type(self, sys_to_be_added: AnyStr):
        if not issubclass(sys_to_be_added.__class__, self.allowed_sys_types):
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
        return True

    def __len__(self) -> int:
        return len(self._systems)

    def __getitem__(self, idx: int) -> SystemType:
        return self._systems[idx]

    def __delitem__(self, idx: int) -> None:
        del self._systems[idx]

    def __setitem__(self, idx: int, system: SystemType) -> None:
        self._check_type(system)
        self._systems[idx] = system

    def insert(self, idx: int, system: SystemType) -> None:
        self._check_type(system)
        self._systems.insert(idx, system)

    def __str__(self) -> str:
        """To be readable"""
        return str(self._systems)

    @final
    def extend_allowed_types(self, additional_types) -> None:
        self.allowed_sys_types += additional_types

    @final
    def override_allowed_types(self, allowed_types) -> None:
        self.allowed_sys_types = allowed_types

    @final
    def _get_sys_idx_if_valid(self, sys_to_be_added: SystemType) -> SystemIdxType:
        n_systems = len(self)  # Total number of systems from mixed-in class

        if isinstance(sys_to_be_added, (int, np.int_)):
            # 1. If they are indices themselves, check range
            assert (
                -n_systems <= sys_to_be_added < n_systems
            ), "Rod index {} exceeds number of registered rodtems".format(
                sys_to_be_added
            )
            sys_idx = sys_to_be_added
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
    def blocks(self) -> Generator[SystemType, None, None]:
        assert self._finalize_flag, "The simulator is not finalized."
        for block in self._memory_blocks:
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
        self._memory_blocks = construct_memory_block_structures(self._systems)

        """
        In case memory block have ring rod, then periodic boundaries have to be synched. In order to synchronize
        periodic boundaries, a new constrain for memory block rod added called as _ConstrainPeriodicBoundaries. This
        constrain will synchronize the only periodic boundaries of position, director, velocity and omega variables.
        """
        for block in self._memory_blocks:
            # append the memory block to the simulation as a system. Memory block is the final system in the simulation.
            self.append(block)
            if hasattr(block, "ring_rod_flag"):
                # Apply the constrain to synchronize the periodic boundaries of the memory rod. Find the memory block
                # sys idx among other systems added and then apply boundary conditions.
                memory_block_idx = self._get_sys_idx_if_valid(block)
                self.constrain(self._systems[memory_block_idx]).using(
                    _ConstrainPeriodicBoundaries,
                )

        # Recurrent call finalize functions for all components.
        for finalize in self._feature_group_finalize:
            finalize()

        # Clear the finalize feature group, just for the safety.
        self._feature_group_finalize.clear()
        self._feature_group_finalize = None

        # Toggle the finalize_flag
        # sort _feature_group_synchronize so that _call_contacts is at the end
        _call_contacts_index = []
        for idx, feature in enumerate(self._feature_group_synchronize):
            if feature.__name__ == "_call_contacts":
                _call_contacts_index.append(idx)

        # Move to the _call_contacts to the end of the _feature_group_synchronize list.
        for index in _call_contacts_index:
            self._feature_group_synchronize.append(
                self._feature_group_synchronize.pop(index)
            )

    @final
    def synchronize(self, time: np.floating) -> None:
        # Collection call _feature_group_synchronize
        for feature in self._feature_group_synchronize:
            feature(time)

    @final
    def constrain_values(self, time: np.floating) -> None:
        # Collection call _feature_group_constrain_values
        for feature in self._feature_group_constrain_values:
            feature(time)

    @final
    def constrain_rates(self, time: np.floating) -> None:
        # Collection call _feature_group_constrain_rates
        for feature in self._feature_group_constrain_rates:
            feature(time)

    @final
    def apply_callbacks(self, time: np.floating, current_step: int) -> None:
        # Collection call _feature_group_callback
        for feature in self._feature_group_callback:
            feature(time, current_step)
