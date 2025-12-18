from typing import Protocol, Generator, Any, Type, Callable, overload
from typing import TYPE_CHECKING

from elastica.typing import (
    SystemIdxType,
    OperatorType,
    OperatorCallbackType,
    OperatorFinalizeType,
    StaticSystemType,
    SystemType,
    BlockSystemType,
)

import numpy as np

if TYPE_CHECKING:
    from .operator_group import OperatorGroupFIFO


class ModuleProtocol(Protocol):
    """Protocol for module handles (e.g., _Connect, _Constraint, _Damper, etc.)."""

    def using(self, cls: Type[Any], *args: Any, **kwargs: Any) -> None: ...

    def instantiate(self, *args: Any, **kwargs: Any) -> Any: ...

    def id(self) -> Any: ...


class SystemCollectionProtocol(Protocol):
    """
    Protocol for system collections.

    This protocol defines the interface for system collections including
    container operations, lifecycle methods, and internal feature groups
    used for operator registration.
    """

    # Container access
    @overload
    def __getitem__(self, i: slice) -> list[SystemType]: ...
    @overload
    def __getitem__(self, i: int) -> SystemType: ...
    def __getitem__(self, i: slice | int) -> "list[SystemType] | SystemType": ...

    def systems(self) -> Generator[StaticSystemType, None, None]: ...

    def final_systems(self) -> Generator[SystemType, None, None]: ...

    def get_system_index(
        self, sys_to_be_added: "SystemType | StaticSystemType"
    ) -> SystemIdxType: ...

    # Lifecycle methods
    def synchronize(self, time: np.float64) -> None: ...
    def constrain_values(self, time: np.float64) -> None: ...
    def constrain_rates(self, time: np.float64) -> None: ...
    def apply_callbacks(self, time: np.float64, current_step: int) -> None: ...

    # Internal feature groups for operator registration
    _feature_group_synchronize: "OperatorGroupFIFO[OperatorType, ModuleProtocol]"
    _feature_group_constrain_values: "OperatorGroupFIFO[OperatorType, ModuleProtocol]"
    _feature_group_constrain_rates: "OperatorGroupFIFO[OperatorType, ModuleProtocol]"
    _feature_group_damping: "OperatorGroupFIFO[OperatorType, ModuleProtocol]"
    _feature_group_callback: "OperatorGroupFIFO[OperatorCallbackType, ModuleProtocol]"
    _feature_group_finalize: list[OperatorFinalizeType]
    _feature_group_on_close: "OperatorGroupFIFO[Callable, ModuleProtocol]"
