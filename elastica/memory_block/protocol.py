from typing import Protocol
from elastica.systems.protocol import SystemProtocol


class BlockProtocol(Protocol):
    @property
    def n_systems(self) -> int:
        """Number of systems in the block."""


class BlockSystemProtocol(SystemProtocol, BlockProtocol, Protocol):
    pass
