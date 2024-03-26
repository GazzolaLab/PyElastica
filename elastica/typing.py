from elastica.rod import RodBase
from elastica.rigidbody import RigidBodyBase
from elastica.surface import SurfaceBase

from typing import Type, Union

RodType = Type[RodBase]
SystemType = Union[RodType, Type[RigidBodyBase]]
AllowedContactType = Union[SystemType, Type[SurfaceBase]]
