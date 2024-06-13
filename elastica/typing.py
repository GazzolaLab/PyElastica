__doc__ = """
This module contains aliases of type-hints for elastica.

"""

from typing import TYPE_CHECKING
from typing import Type, Callable, Any, ParamSpec
from typing import TypeAlias


if TYPE_CHECKING:
    # Used for type hinting without circular imports
    # NEVER BACK-IMPORT ANY ELASTICA MODULES HERE
    from .rod import RodBase
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

    # Modules Base Classes
    from .boundary_conditions import FreeBC
    from .callback_functions import CallBackBaseClass
    from .contact_forces import NoContact
    from .dissipation import DamperBase
    from .external_forces import NoForces
    from .joint import FreeJoint
else:
    RodBase = None
    RigidBodyBase = None
    SurfaceBase = None

    SystemCollectionProtocol = None

    State = "State"
    SymplecticSystemProtocol = None
    ExplicitSystemProtocol = None

    StepperProtocol = None
    SymplecticStepperProtocol = None
    MemoryProtocol = None

    # Modules Base Classes
    FreeBC = None
    CallBackBaseClass = None
    NoContact = None
    DamperBase = None
    NoForces = None
    FreeJoint = None


SystemType: TypeAlias = SymplecticSystemProtocol  # | ExplicitSystemProtocol
SystemIdxType: TypeAlias = int

# ModuleObjectTypes: TypeAlias = (
#     NoForces | NoContact | FreeJoint | FreeBC | DamperBase | CallBackBaseClass
# )

# TODO: Modify this line and move to elastica/typing.py once system state is defined
# Mostly used in explicit stepper: for symplectic, use kinetic and dynamic state
StateType: TypeAlias = State

OperatorType: TypeAlias = Callable[
    ..., Any
]  # TODO: Maybe can be more specific. Up for discussion.
SteppersOperatorsType: TypeAlias = tuple[tuple[OperatorType, ...], ...]

RodType: TypeAlias = Type[RodBase]
SystemCollectionType: TypeAlias = SystemCollectionProtocol
AllowedContactType: TypeAlias = SystemType | Type[SurfaceBase]

# Operators in elastica.modules
OperatorParam = ParamSpec("OperatorParam")
OperatorType: TypeAlias = Callable[OperatorParam, None]
OperatorCallbackType: TypeAlias = Callable[..., None]
OperatorFinalizeType: TypeAlias = Callable[..., None]
