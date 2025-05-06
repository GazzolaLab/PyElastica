from typing import Protocol
from elastica.systems.protocol import SystemProtocol

import numpy as np

from elastica.rod.protocol import CosseratRodProtocol
from elastica.rigidbody.protocol import RigidBodyProtocol
from elastica.systems.protocol import SymplecticSystemProtocol


class BlockSystemProtocol(SystemProtocol, Protocol):
    @property
    def n_systems(self) -> int:
        """Number of systems in the block."""
