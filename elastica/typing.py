from elastica.rod import RodBase
from elastica.rigidbody import RigidBodyBase
from elastica.surface import SurfaceBase

from typing import Type, Union, TypeAlias, Callable

RodType = Type[RodBase]
SystemType = Union[RodType, Type[RigidBodyBase]]
AllowedContactType = Union[SystemType, Type[SurfaceBase]]

OperatorType: TypeAlias = Callable[[float], None]
OperatorCallbackType: TypeAlias = Callable[[float, int], None]
OperatorFinalizeType: TypeAlias = Callable
