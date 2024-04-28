from elastica.rod import RodBase
from elastica.rigidbody import RigidBodyBase
from elastica.surface import SurfaceBase
from elastica.module import BaseSystemCollection

from typing import Type, Union

RodType = Type[RodBase]
SystemCollectionType = Type[BaseSystemCollection]
SystemType = Union[RodType, Type[RigidBodyBase], SystemCollectionType]
AllowedContactType = Union[SystemType, Type[SurfaceBase]]
