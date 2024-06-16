from typing import Protocol

from elastica.rod.protocol import CosseratRodProtocol
from elastica.systems.protocol import SymplecticSystemProtocol


class BlockCosseratRodProtocol(CosseratRodProtocol, SymplecticSystemProtocol, Protocol):
    pass


# class BlockRigidBodyProtocol(RigidBodyProtocol, SymplecticSystemProtocol, Protocol):
#     pass
