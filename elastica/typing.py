__doc__ = """
This module contains aliases of type-hints for elastica.

"""

from typing import TYPE_CHECKING
from typing import Type, Callable, Any, ParamSpec
from typing import TypeAlias

import numpy as np


if TYPE_CHECKING:
    # Used for type hinting without circular imports
    # NEVER BACK-IMPORT ANY ELASTICA MODULES HERE
    from .rod.protocol import CosseratRodProtocol
    from .rigidbody import RigidBodyBase
    from .surface import SurfaceBase
    from .modules.base_system import BaseSystemCollection

    from .modules.protocol import SystemCollectionProtocol
    from .rod.data_structures import _State as State
    from .systems.protocol import SymplecticSystemProtocol, ExplicitSystemProtocol
    from .timestepper.protocol import (
        StepperProtocol,
        SymplecticStepperProtocol,
        MemoryProtocol,
    )
    from memory_block.protocol import (
        BlockCosseratRodProtocol,
    )  # , BlockRigidBodyProtocol

    # Modules Base Classes
    from .boundary_conditions import FreeBC
    from .callback_functions import CallBackBaseClass
    from .contact_forces import NoContact
    from .dissipation import DamperBase
    from .external_forces import NoForces
    from .joint import FreeJoint


SystemType: TypeAlias = "SymplecticSystemProtocol"  # | ExplicitSystemProtocol
SystemIdxType: TypeAlias = int

# ModuleObjectTypes: TypeAlias = (
#     NoForces | NoContact | FreeJoint | FreeBC | DamperBase | CallBackBaseClass
# )

# TODO: Modify this line and move to elastica/typing.py once system state is defined
# Mostly used in explicit stepper: for symplectic, use kinetic and dynamic state
StateType: TypeAlias = "State"

# TODO: Maybe can be more specific. Up for discussion.
OperatorType: TypeAlias = Callable[..., Any]
SteppersOperatorsType: TypeAlias = tuple[tuple[OperatorType, ...], ...]


RodType: TypeAlias = "CosseratRodProtocol"
RigidBodyType: TypeAlias = "RigidBodyProtocol"

SystemCollectionType: TypeAlias = "SystemCollectionProtocol"
AllowedContactType: TypeAlias = (
    SystemType | Type["SurfaceBase"]
)  # FIXME: SurfaceBase needs to be treated differently
BlockType: TypeAlias = "BlockCosseratRodProtocol"  # | "BlockRigidBodyProtocol"

# Indexing types
# TODO: Maybe just use slice??
ConstrainingIndex: TypeAlias = list[int] | tuple[int] | np.typing.NDArray | None
ConnectionIndex: TypeAlias = (
    int | np.int_ | list[int] | tuple[int] | np.typing.NDArray | None
)

# Operators in elastica.modules
OperatorParam = ParamSpec("OperatorParam")
OperatorType: TypeAlias = Callable[OperatorParam, None]
OperatorCallbackType: TypeAlias = Callable[..., None]
OperatorFinalizeType: TypeAlias = Callable[..., None]
