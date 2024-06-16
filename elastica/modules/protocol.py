from typing import Protocol, Generator, TypeVar, Any, Type
from typing_extensions import Self  # 3.11: from typing import Self

from abc import abstractmethod
from collections.abc import MutableSequence

from elastica.typing import (
    SystemIdxType,
    OperatorType,
    OperatorCallbackType,
    OperatorFinalizeType,
    SystemType,
    ConnectionIndex,
)
from elastica.joint import FreeJoint
from elastica.callback_functions import CallBackBaseClass
from elastica.boundary_conditions import ConstraintBase
from elastica.dissipation import DamperBase

import numpy as np

from .operator_group import OperatorGroupFIFO


M = TypeVar("M", bound="ModuleProtocol")


class ModuleProtocol(Protocol[M]):
    def using(self, cls: Type[M], *args: Any, **kwargs: Any) -> Self: ...

    def instantiate(self, *args: Any, **kwargs: Any) -> M: ...

    def id(self) -> Any: ...


class SystemCollectionProtocol(MutableSequence, Protocol):
    _systems: list[SystemType]

    @property
    def _feature_group_synchronize(self) -> OperatorGroupFIFO: ...

    def synchronize(self, time: np.floating) -> None: ...

    @property
    def _feature_group_constrain_values(self) -> list[OperatorType]: ...

    def constrain_values(self, time: np.floating) -> None: ...

    @property
    def _feature_group_constrain_rates(self) -> list[OperatorType]: ...

    def constrain_rates(self, time: np.floating) -> None: ...

    @property
    def _feature_group_callback(self) -> list[OperatorCallbackType]: ...

    def apply_callbacks(self, time: np.floating, current_step: int) -> None: ...

    @property
    def _feature_group_finalize(self) -> list[OperatorFinalizeType]: ...

    def blocks(self) -> Generator[SystemType, None, None]: ...

    def _get_sys_idx_if_valid(self, sys_to_be_added: SystemType) -> SystemIdxType: ...

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
    _callback_operators: list[tuple[int, CallBackBaseClass]]

    @abstractmethod
    def collect_diagnostics(self, system: SystemType) -> ModuleProtocol:
        raise NotImplementedError

    @abstractmethod
    def _callback_execution(
        self, time: np.floating, current_step: int, *args: Any, **kwargs: Any
    ) -> None:
        raise NotImplementedError

    # Constraints API
    _constraints_list: list[ModuleProtocol]
    _constraints_operators: list[tuple[int, ConstraintBase]]
    _finalize_constraints: OperatorFinalizeType

    @abstractmethod
    def constrain(self, system: SystemType) -> ModuleProtocol:
        raise NotImplementedError

    @abstractmethod
    def _constrain_values(self, time: np.floating) -> None:
        raise NotImplementedError

    @abstractmethod
    def _constrain_rates(self, time: np.floating) -> None:
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
    _damping_operators: list[tuple[int, DamperBase]]
    _finalize_dampers: OperatorFinalizeType

    @abstractmethod
    def dampen(self, system: SystemType) -> ModuleProtocol:
        raise NotImplementedError

    @abstractmethod
    def _dampen_rates(self, time: np.floating) -> None:
        raise NotImplementedError
