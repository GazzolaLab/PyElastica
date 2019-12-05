import numpy as np

from elastica._linalg import _batch_matvec


class RodBase:
    """
    Base class for all rods
    # TODO : What needs to be ported here?

    # The interface class, as seen from global scope
    # Can be made common to all entities in the code
    """

    def __init__(self):
        pass
        # self.position_collection = NotImplemented
        # self.omega_collection = NotImplemented
        # self.acceleration_collection = NotImplemented
        # self.alpha_collection = NotImplemented
        # self.external_forces = NotImplemented
        # self.external_torques = NotImplemented
