from typing import Protocol, runtime_checkable
from elastica.typing import StaticSystemType, SystemIdxType
from elastica.systems.protocol import SystemProtocol


@runtime_checkable
class BlockSystemProtocol(SystemProtocol, Protocol):
    """
    Protocol for block systems.
    Block systems are systems that are used to store the data of multiple systems.
    """

    def __init__(
        self, systems: list[StaticSystemType], system_idx_list: list[SystemIdxType]
    ) -> None:
        """
        Block initializer takes the list of systems and the list of system indices.
        """

    @property
    def n_systems(self) -> int:
        """Number of systems in the block."""
