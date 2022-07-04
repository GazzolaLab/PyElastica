from elastica.rod import RodBase
from elastica.rigidbody import RigidBodyBase

from typing import Type, Union

RodType = Type[RodBase]
SystemType = Union[Type[RodBase], Type[RigidBodyBase]]