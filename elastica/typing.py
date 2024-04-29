from elastica.rod import RodBase
from elastica.rigidbody import RigidBodyBase
from elastica.surface import SurfaceBase
from elastica.module import BaseSystemCollection

from .system.protocol import SystemProtocol

import numpy as np

from typing import Type, Union, Callable, tuple
from typing import Protocol

SystemType = SystemProtocol

StepOperatorType = Callable[[SystemType, ...], None]
PrefactorOperatorType = Callable[[np.floating], np.floating]
SteppersOperatorsType = tuple[
    tuple[Union[PrefactorOperatorType, StepOperatorType], ...], ...
]

RodType = Type[RodBase]
SystemCollectionType = Type[BaseSystemCollection]
AllowedContactType = Union[SystemType, Type[SurfaceBase]]
