from typing import TYPE_CHECKING
from typing import Type, Union, Callable
from typing import Protocol

from .systems.protocol import SystemProtocol

import numpy as np

if TYPE_CHECKING:
    # Used for type hinting without circular imports
    from .rod import RodBase
    from .rigidbody import RigidBodyBase
    from .surface import SurfaceBase
    from .modules import BaseSystemCollection
else:
    RodBase = None
    RigidBodyBase = None
    SurfaceBase = None
    BaseSystemCollection = None


SystemType = SystemProtocol

StepOperatorType = Callable[[SystemType, ...], None]
PrefactorOperatorType = Callable[[np.floating], np.floating]
SteppersOperatorsType = tuple[
    tuple[Union[PrefactorOperatorType, StepOperatorType], ...], ...
]

RodType = Type[RodBase]
SystemCollectionType = BaseSystemCollection
AllowedContactType = Union[SystemType, Type[SurfaceBase]]
