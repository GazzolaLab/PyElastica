__doc__ = """
This module contains aliases of type-hints for elastica.

"""

from typing import TYPE_CHECKING
from typing import Type, Union, Callable, Any, AnyStr
from typing import TypeAlias


if TYPE_CHECKING:
    # Used for type hinting without circular imports
    # NEVER BACK-IMPORT ANY ELASTICA MODULES HERE
    from .rod import RodBase
    from .rigidbody import RigidBodyBase
    from .surface import SurfaceBase
    from .modules import BaseSystemCollection

    from .rod.data_structures import _State as State
    from .systems.protocol import SymplecticSystemProtocol, ExplicitSystemProtocol
    from .timestepper.protocol import StatefulStepperProtocol, MemoryProtocol
else:
    RodBase = None
    RigidBodyBase = None
    SurfaceBase = None
    BaseSystemCollection = None

    State = "State"
    SymplecticSystemProtocol = None
    ExplicitSystemProtocol = None
    StatefulStepperProtocol = None
    MemoryProtocol = None


SystemType: TypeAlias = Union[SymplecticSystemProtocol, ExplicitSystemProtocol]

# TODO: Modify this line and move to elastica/typing.py once system state is defined
# Mostly used in explicit stepper: for symplectic, use kinetic and dynamic state
StateType: TypeAlias = State

# NoOpt stepper
# Symplectic stepper
# StepOperatorType = Callable[
#    [StatefulStepperProtocol, SymplecticSystemProtocol, np.floating, np.floating], None
# ]
# PrefactorOperatorType = Callable[
#    [StatefulStepperProtocol, np.floating], np.floating
# ]
OperatorType: TypeAlias = Callable[
    Any, Any
]  # TODO: Maybe can be more specific. Up for discussion.
SteppersOperatorsType: TypeAlias = tuple[tuple[OperatorType, ...], ...]
# tuple[Union[PrefactorOperatorType, StepOperatorType, NoOpType, np.floating], ...], ...
# Explicit stepper
# ExplicitStageOperatorType = Callable[
#    [
#        StatefulStepperProtocol,
#        ExplicitSystemProtocol,
#        MemoryProtocol,
#        np.floating,
#        np.floating,
#    ],
#    None,
# ]
# ExplicitUpdateOperatorType = Callable[
#    [
#        StatefulStepperProtocol,
#        ExplicitSystemProtocol,
#        MemoryProtocol,
#        np.floating,
#        np.floating,
#    ],
#    np.floating,
# ]
ExplicitOperatorsType: TypeAlias = tuple[tuple[OperatorType, ...], ...]

RodType: TypeAlias = Type[RodBase]
SystemCollectionType: TypeAlias = BaseSystemCollection
AllowedContactType: TypeAlias = Union[SystemType, Type[SurfaceBase]]

# Operators in elastica.modules
SynchronizeOperator: TypeAlias = Callable[[float], None]
ConstrainValuesOperator: TypeAlias = Callable[[float], None]
ConstrainRatesOperator: TypeAlias = Callable[[float], None]
CallbackOperator: TypeAlias = Callable[[float, int, AnyStr], None]
FinalizeOperator: TypeAlias = Callable[[], None]
