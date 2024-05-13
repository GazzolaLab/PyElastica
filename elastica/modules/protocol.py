from typing import Protocol, Type, Generator, Iterable
from typing_extensions import Self  # 3.11: from typing import Self

from elastica.typing import (
    SystemIdxType,
    OperatorType,
    OperatorCallbackType,
    OperatorFinalizeType,
    SystemType,
)

pass


class SystemCollectionProtocol(Protocol):
    _systems: list[SystemType]

    @property
    def _feature_group_synchronize(self) -> Iterable[OperatorType]: ...

    @property
    def _feature_group_constrain_values(self) -> Iterable[OperatorType]: ...

    @property
    def _feature_group_constrain_rates(self) -> Iterable[OperatorType]: ...

    @property
    def _feature_group_callback(self) -> Iterable[OperatorCallbackType]: ...

    @property
    def _feature_group_finalize(self) -> Iterable[OperatorFinalizeType]: ...

    def blocks(self) -> Generator[SystemType, None, None]: ...

    def _get_sys_idx_if_valid(self, sys_to_be_added: SystemType) -> SystemIdxType: ...


class ModuleProtocol(Protocol):
    def using(self, callback_cls, *args, **kwargs) -> Self: ...

    def instantiate(self) -> Type: ...

    def id(self) -> SystemIdxType: ...
