from elastica.rod import RodBase, RodProtocol
from elastica.rigidbody import RigidBodyBase
from elastica.surface import SurfaceBase
from elastica.module import BaseSystemCollection

from typing import Type, Union

RodType = RodProtocol
SystemCollectionType = Type[BaseSystemCollection]
SystemType = Union[RodProtocol, Type[RigidBodyBase], SystemCollectionType]
AllowedContactType = Union[SystemType, Type[SurfaceBase]]
