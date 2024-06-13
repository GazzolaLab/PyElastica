__doc__ = """
This module contains aliases of type-hints for elastica.

"""

from typing import TYPE_CHECKING
from typing import Type, Union, Callable, Any, ParamSpec
from typing import TypeAlias


if TYPE_CHECKING:
    # Used for type hinting without circular imports
    # NEVER BACK-IMPORT ANY ELASTICA MODULES HERE
    from .rod import RodBase
    from .rigidbody import RigidBodyBase
    from .surface import SurfaceBase
    from .modules.base_system import BaseSystemCollection

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
    BaseSystemCollection = None

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


SystemType: TypeAlias = Union[SymplecticSystemProtocol, ExplicitSystemProtocol]
SystemIdxType: TypeAlias = int

# TODO: Modify this line and move to elastica/typing.py once system state is defined
# Mostly used in explicit stepper: for symplectic, use kinetic and dynamic state
StateType: TypeAlias = State

# NoOpt stepper
# Symplectic stepper
# StepOperatorType = Callable[
#    [SymplecticStepperProtocol, SymplecticSystemProtocol, np.floating, np.floating], None
# ]
# PrefactorOperatorType = Callable[
#    [SymplecticStepperProtocol, np.floating], np.floating
# ]
OperatorType: TypeAlias = Callable[
    ..., Any
]  # TODO: Maybe can be more specific. Up for discussion.
SteppersOperatorsType: TypeAlias = tuple[tuple[OperatorType, ...], ...]
# tuple[Union[PrefactorOperatorType, StepOperatorType, NoOpType, np.floating], ...], ...
# Explicit stepper
# ExplicitStageOperatorType = Callable[
#    [
#        SymplecticStepperProtocol,
#        ExplicitSystemProtocol,
#        MemoryProtocol,
#        np.floating,
#        np.floating,
#    ],
#    None,
# ]
# ExplicitUpdateOperatorType = Callable[
#    [
#        SymplecticStepperProtocol,
#        ExplicitSystemProtocol,
#        MemoryProtocol,
#        np.floating,
#        np.floating,
#    ],
#    np.floating,
# ]

RodType: TypeAlias = Type[RodBase]
SystemCollectionType: TypeAlias = BaseSystemCollection
AllowedContactType: TypeAlias = Union[SystemType, Type[SurfaceBase]]

# Operators in elastica.modules
CallbackParam = ParamSpec("CallbackParam")
OperatorType: TypeAlias = Callable[[float], None]
OperatorCallbackType: TypeAlias = Callable[CallbackParam, None]
OperatorFinalizeType: TypeAlias = Callable
