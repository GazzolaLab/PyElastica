from elastica.rod import RodBase
from elastica.rigidbody import RigidBodyBase
from elastica.surface import SurfaceBase
from elastica.module import BaseSystemCollection

from .system.protocol import SystemProtocol

from typing import Type, Union
from typing import Protocol

SystemType = SystemProtocol

RodType = Type[RodBase]
SystemCollectionType = Type[BaseSystemCollection]
AllowedContactType = Union[SystemType, Type[SurfaceBase]]
