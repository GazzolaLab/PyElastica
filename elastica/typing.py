__doc__ = """
This module contains aliases of type-hints for elastica.

"""

from typing import TYPE_CHECKING
from typing import Callable, Any, TypeAlias, Protocol

import numpy as np


if TYPE_CHECKING:
    # Used for type hinting without circular imports
    # NEVER BACK-IMPORT ANY ELASTICA MODULES HERE
    from .rod.protocol import CosseratRodProtocol
    from .rigidbody.protocol import RigidBodyProtocol
    from .surface.surface_base import SurfaceBase
    from .modules.base_system import BaseSystemCollection

    from .modules.protocol import SystemCollectionProtocol
    from .rod.data_structures import _State as State
    from .systems.protocol import (
        SystemProtocol,
        StaticSystemProtocol,
        SymplecticSystemProtocol,
    )
    from .timestepper.protocol import (
        StepperProtocol,
        SymplecticStepperProtocol,
    )
    from .memory_block.protocol import BlockSystemProtocol

else:
    CosseratRodProtocol = "CosseratRodProtocol"
    RigidBodyProtocol = "RigidBodyProtocol"
    SurfaceBase = "SurfaceBase"
    BaseSystemCollection = "BaseSystemCollection"

    SystemCollectionProtocol = "SystemCollectionProtocol"
    State = "State"
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


# Mostly used in explicit stepper: for symplectic, use kinetic and dynamic state
StateType: TypeAlias = "State"

# TODO: Maybe can be more specific. Up for discussion.
StepType: TypeAlias = Callable[..., Any]
SteppersOperatorsType: TypeAlias = tuple[tuple[StepType, ...], ...]


RodType: TypeAlias = "CosseratRodProtocol"
RigidBodyType: TypeAlias = "RigidBodyProtocol"
SurfaceType: TypeAlias = "SurfaceBase"

SystemCollectionType: TypeAlias = "SystemCollectionProtocol"

# Indexing types
# TODO: Maybe just use slice??
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
