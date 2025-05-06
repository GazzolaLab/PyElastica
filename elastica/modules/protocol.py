from typing import Protocol, Generator, TypeVar, Any, Type, overload
from typing import TYPE_CHECKING
from typing_extensions import Self  # python 3.11: from typing import Self

from abc import abstractmethod

from elastica.typing import (
    SystemIdxType,
    OperatorType,
    OperatorCallbackType,
    OperatorFinalizeType,
    StaticSystemType,
    SystemType,
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

    @property
    def _feature_group_synchronize(
        self,
    ) -> "OperatorGroupFIFO[OperatorType, ModuleProtocol]": ...

    def synchronize(self, time: np.float64) -> None: ...

    @property
    def _feature_group_constrain_values(
        self,
    ) -> "OperatorGroupFIFO[OperatorType, ModuleProtocol]": ...

    def constrain_values(self, time: np.float64) -> None: ...

    @property
    def _feature_group_constrain_rates(
        self,
    ) -> "OperatorGroupFIFO[OperatorType, ModuleProtocol]": ...

    def constrain_rates(self, time: np.float64) -> None: ...

    @property
    def _feature_group_callback(
        self,
    ) -> "OperatorGroupFIFO[OperatorCallbackType, ModuleProtocol]": ...

    def apply_callbacks(self, time: np.float64, current_step: int) -> None: ...

    @property
    def _feature_group_finalize(self) -> list[OperatorFinalizeType]: ...

    def get_system_index(
        self, sys_to_be_added: "SystemType | StaticSystemType"
    ) -> SystemIdxType: ...

    # Connection API
    _finalize_connections: OperatorFinalizeType
    _connections: list[ModuleProtocol]

    @abstractmethod
    def connect(
        self,
        first_rod: SystemType,
        second_rod: SystemType,
        first_connect_idx: ConnectionIndex,
        second_connect_idx: ConnectionIndex,
    ) -> ModuleProtocol:
        raise NotImplementedError

    # CallBack API
    _finalize_callback: OperatorFinalizeType
    _callback_list: list[ModuleProtocol]

    @abstractmethod
    def collect_diagnostics(self, system: SystemType) -> ModuleProtocol:
        raise NotImplementedError

    # Constraints API
    _constraints_list: list[ModuleProtocol]
    _finalize_constraints: OperatorFinalizeType

    @abstractmethod
    def constrain(self, system: SystemType) -> ModuleProtocol:
        raise NotImplementedError

    # Forcing API
    _ext_forces_torques: list[ModuleProtocol]
    _finalize_forcing: OperatorFinalizeType

    @abstractmethod
    def add_forcing_to(self, system: SystemType) -> ModuleProtocol:
        raise NotImplementedError

    # Contact API
    _contacts: list[ModuleProtocol]
    _finalize_contact: OperatorFinalizeType

    @abstractmethod
    def detect_contact_between(
        self, first_system: SystemType, second_system: SystemType
    ) -> ModuleProtocol:
        raise NotImplementedError

    # Damping API
    _damping_list: list[ModuleProtocol]
    _finalize_dampers: OperatorFinalizeType

    @abstractmethod
    def dampen(self, system: SystemType) -> ModuleProtocol:
        raise NotImplementedError
