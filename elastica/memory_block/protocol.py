from typing import Protocol
from elastica.rod.protocol import CosseratRodProtocol
from elastica.rigidbody.protocol import RigidBodyProtocol
from elastica.systems.protocol import SystemProtocol


class BlockProtocol(Protocol):
    @property
    def n_systems(self) -> int:
        """Number of systems in the block."""


class BlockSystemProtocol(SystemProtocol, BlockProtocol, Protocol):
    pass


class BlockRodProtocol(BlockProtocol, CosseratRodProtocol, Protocol):
    pass


class BlockRigidBodyProtocol(BlockProtocol, RigidBodyProtocol, Protocol):
    pass
