from __future__ import annotations

__doc__ = """
This module contains aliases of type-hints for elastica.

"""

from typing import TYPE_CHECKING
from typing import Callable, Any, TypeAlias, Protocol

import numpy as np


if TYPE_CHECKING:
    # Used for type hinting without circular imports
    # NEVER BACK-IMPORT ANY ELASTICA MODULES HERE
    from .rod.rod_base import RodBase
    from .rigidbody.rigid_body_base import RigidBodyBase
    from .modules.base_system import BaseSystemCollection

    from .modules.protocol import SystemCollectionProtocol
    from .systems.protocol import (
        StaticSystemProtocol,
        SystemProtocol,
        SymplecticSystemProtocol,
    )
    from .timestepper.protocol import (
        StepperProtocol,
        SymplecticStepperProtocol,
    )
    from .memory_block.protocol import BlockSystemProtocol

else:
    RodBase = "RodBase"
    RigidBodyType = "RigidBodyBase"
    BaseSystemCollection = "BaseSystemCollection"

    SystemCollectionProtocol = "SystemCollectionProtocol"
    SystemProtocol = "SystemProtocol"
    StaticSystemProtocol = "StaticSystemProtocol"
    SymplecticSystemProtocol = "SymplecticSystemProtocol"
    StepperProtocol = "StepperProtocol"
    SymplecticStepperProtocol = "SymplecticStepperProtocol"
    BlockSystemProtocol = "BlockSystemProtocol"


StaticSystemType: TypeAlias = "StaticSystemProtocol"
SystemType: TypeAlias = "SystemProtocol"
SystemIdxType: TypeAlias = int
BlockSystemType: TypeAlias = "BlockSystemProtocol"

StepType: TypeAlias = Callable[..., Any]
SteppersOperatorsType: TypeAlias = tuple[tuple[StepType, ...], ...]

RodType: TypeAlias = "RodBase"
RigidBodyType: TypeAlias = "RigidBodyBase"

SystemCollectionType: TypeAlias = "SystemCollectionProtocol"

# Indexing types
ConstrainingIndex: TypeAlias = tuple[int, ...]
ConnectionIndex: TypeAlias = (
    int | np.int32 | list[int] | tuple[int, ...] | np.typing.NDArray[np.int32]
)


# Operators in elastica.modules
class OperatorType(Protocol):
    def __call__(self, time: np.float64) -> None: ...


class OperatorCallbackType(Protocol):
    def __call__(self, time: np.float64, current_step: int) -> None: ...


OperatorFinalizeType: TypeAlias = Callable[[], None]
