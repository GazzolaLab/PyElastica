__doc__ = """ Rigid body abstract base class """

from abc import ABC, abstractmethod


class RigidBodyBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def update_accelerations(self):
        pass
