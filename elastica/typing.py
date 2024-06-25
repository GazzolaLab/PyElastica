__doc__ = """
This module contains aliases of type-hints for elastica.

"""

from typing import TYPE_CHECKING
from typing import Callable, Any, ParamSpec, TypeAlias

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
        ExplicitSystemProtocol,
    )
    from .timestepper.protocol import (
        StepperProtocol,
        SymplecticStepperProtocol,
        MemoryProtocol,
    )
    from .memory_block.protocol import BlockSystemProtocol

    from .mesh.protocol import MeshProtocol


StaticSystemType: TypeAlias = "StaticSystemProtocol"
SystemType: TypeAlias = "SystemProtocol"
SystemIdxType: TypeAlias = int
BlockSystemType: TypeAlias = "BlockSystemProtocol"


# Mostly used in explicit stepper: for symplectic, use kinetic and dynamic state
StateType: TypeAlias = "State"

# TODO: Maybe can be more specific. Up for discussion.
OperatorType: TypeAlias = Callable[..., Any]
SteppersOperatorsType: TypeAlias = tuple[tuple[OperatorType, ...], ...]


RodType: TypeAlias = "CosseratRodProtocol"
RigidBodyType: TypeAlias = "RigidBodyProtocol"
SurfaceType: TypeAlias = "SurfaceBase"

SystemCollectionType: TypeAlias = "SystemCollectionProtocol"

# Indexing types
# TODO: Maybe just use slice??
ConstrainingIndex: TypeAlias = list[int] | tuple[int, ...] | np.typing.NDArray
ConnectionIndex: TypeAlias = (
    int | np.int_ | list[int] | tuple[int] | np.typing.NDArray | None
)

# Operators in elastica.modules
OperatorParam = ParamSpec("OperatorParam")
OperatorCallbackType: TypeAlias = Callable[..., None]
OperatorFinalizeType: TypeAlias = Callable[..., None]

MeshType: TypeAlias = "MeshProtocol"
