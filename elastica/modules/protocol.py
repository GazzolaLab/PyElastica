from typing import Protocol, Generator, TypeVar, Any
from typing_extensions import Self  # 3.11: from typing import Self
from elastica.typing import (
    SystemIdxType,
    OperatorType,
    OperatorCallbackType,
    OperatorFinalizeType,
    SystemType,
)

from .operator_group import OperatorGroupFIFO


class SystemCollectionProtocol(Protocol):
    _systems: list[SystemType]

    @property
    def _feature_group_synchronize(self) -> OperatorGroupFIFO: ...

    @property
    def _feature_group_constrain_values(self) -> list[OperatorType]: ...

    @property
    def _feature_group_constrain_rates(self) -> list[OperatorType]: ...

    @property
    def _feature_group_callback(self) -> list[OperatorCallbackType]: ...

    @property
    def _feature_group_finalize(self) -> list[OperatorFinalizeType]: ...

    def blocks(self) -> Generator[SystemType, None, None]: ...

    def _get_sys_idx_if_valid(self, sys_to_be_added: SystemType) -> SystemIdxType: ...


M = TypeVar("M", bound="ModuleProtocol")


class ModuleProtocol(Protocol[M]):
    def using(self, callback_cls, *args, **kwargs) -> Self: ...

    def instantiate(self) -> M: ...

    def id(self) -> Any: ...
