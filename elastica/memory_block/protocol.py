from typing import Protocol
from elastica.typing import SystemType

import numpy as np

from elastica.rod.protocol import CosseratRodProtocol
from elastica.rigid_body.protocol import RigidBodyProtocol
from elastica.systems.protocol import SymplecticSystemProtocol


class BlockSystemProtocol(SystemType, Protocol):
    @property
    def n_bodies(self) -> int:
        """Number of systems in the block."""


class BlockCosseratRodProtocol(CosseratRodProtocol, SymplecticSystemProtocol, Protocol):
    pass


class BlockRigidBodyProtocol(RigidBodyProtocol, SymplecticSystemProtocol, Protocol):
    pass
