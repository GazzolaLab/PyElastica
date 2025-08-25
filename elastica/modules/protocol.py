from typing import Protocol, Generator, TypeVar, Any, Type, overload, Iterator
from typing import TYPE_CHECKING
from typing_extensions import Self  # python 3.11: from typing import Self

from elastica.typing import (
    SystemIdxType,
    OperatorType,
    OperatorCallbackType,
    OperatorFinalizeType,
    StaticSystemType,
    SystemType,
    RodType,
    RigidBodyType,
    BlockSystemType,
    ConnectionIndex,
)
from elastica.joint import FreeJoint
from elastica.callback_functions import CallBackBaseClass
from elastica.boundary_conditions import ConstraintBase
from elastica.dissipation import DamperBase

import numpy as np

if TYPE_CHECKING:
    from .operator_group import OperatorGroupFIFO


class MixinProtocol(Protocol):
    # def finalize(self) -> None: ...
    ...


M = TypeVar("M", bound=MixinProtocol)


class ModuleProtocol(Protocol[M]):
    def using(self, cls: Type[M], *args: Any, **kwargs: Any) -> Self: ...

    def instantiate(self, *args: Any, **kwargs: Any) -> M: ...

    def id(self) -> Any: ...


class SystemCollectionProtocol(Protocol):
    def __len__(self) -> int: ...

    def systems(self) -> Generator[StaticSystemType, None, None]: ...

    def block_systems(self) -> Generator[BlockSystemType, None, None]: ...

    @overload
    def __getitem__(self, i: slice) -> list[SystemType]: ...
    @overload
    def __getitem__(self, i: int) -> SystemType: ...
    def __getitem__(self, i: slice | int) -> "list[SystemType] | SystemType": ...

    def __delitem__(self, i: slice | int) -> None: ...
    def __setitem__(self, i: slice | int, value: SystemType) -> None: ...
    def insert(self, i: int, value: SystemType) -> None: ...
    def __iter__(self) -> Iterator[SystemType]: ...

    def get_system_index(
        self, sys_to_be_added: "SystemType | StaticSystemType"
    ) -> SystemIdxType: ...

    # Operator Group
    _feature_group_synchronize: "OperatorGroupFIFO[OperatorType, ModuleProtocol]"
    _feature_group_constrain_values: "OperatorGroupFIFO[OperatorType, ModuleProtocol]"
    _feature_group_constrain_rates: "OperatorGroupFIFO[OperatorType, ModuleProtocol]"
    _feature_group_damping: "OperatorGroupFIFO[OperatorType, ModuleProtocol]"
    _feature_group_callback: "OperatorGroupFIFO[OperatorCallbackType, ModuleProtocol]"

    def synchronize(self, time: np.float64) -> None: ...
    def constrain_values(self, time: np.float64) -> None: ...
    def constrain_rates(self, time: np.float64) -> None: ...
    def apply_callbacks(self, time: np.float64, current_step: int) -> None: ...

    # Finalize Operations
    _feature_group_finalize: list[OperatorFinalizeType]

    def finalize(self) -> None: ...


# Mixin Protocols (Used to type Self)
class ConnectedSystemCollectionProtocol(SystemCollectionProtocol, Protocol):
    # Connection API
    _connections: list[ModuleProtocol]

    def _finalize_connections(self) -> None: ...

    def connect(
        self,
        first_rod: "RodType | RigidBodyType",
        second_rod: "RodType | RigidBodyType",
        first_connect_idx: ConnectionIndex,
        second_connect_idx: ConnectionIndex,
    ) -> ModuleProtocol: ...


class ForcedSystemCollectionProtocol(SystemCollectionProtocol, Protocol):
    # Forcing API
    _ext_forces_torques: list[ModuleProtocol]

    def _finalize_forcing(self) -> None: ...

    def add_forcing_to(self, system: SystemType) -> ModuleProtocol: ...


class ContactedSystemCollectionProtocol(SystemCollectionProtocol, Protocol):
    # Contact API
    _contacts: list[ModuleProtocol]

    def _finalize_contact(self) -> None: ...

    def detect_contact_between(
        self, first_system: SystemType, second_system: SystemType
    ) -> ModuleProtocol: ...


class ConstrainedSystemCollectionProtocol(SystemCollectionProtocol, Protocol):
    # Constraints API
    _constraints_list: list[ModuleProtocol]

    def _finalize_constraints(self) -> None: ...

    def constrain(self, system: "RodType | RigidBodyType") -> ModuleProtocol: ...


class SystemCollectionWithCallbackProtocol(SystemCollectionProtocol, Protocol):
    # CallBack API
    _callback_list: list[ModuleProtocol]

    def _finalize_callback(self) -> None: ...

    def collect_diagnostics(self, system: SystemType) -> ModuleProtocol: ...


class DampenedSystemCollectionProtocol(SystemCollectionProtocol, Protocol):
    # Damping API
    _damping_list: list[ModuleProtocol]

    def _finalize_dampers(self) -> None: ...

    def dampen(self, system: RodType) -> ModuleProtocol: ...
