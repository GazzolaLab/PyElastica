import numpy as np

from elastica._linalg import _batch_matvec


# TODO : What needs to be ported here?
class RodBase:
    """
    Base class for all rods.

    Note
    ----
    All new rod classes must be derived from this RodBase class.


    """

    def __init__(self):
        """
        RodBase does not take any arguments.
        """
        pass
        # self.position_collection = NotImplemented
        # self.omega_collection = NotImplemented
        # self.acceleration_collection = NotImplemented
        # self.alpha_collection = NotImplemented
        # self.external_forces = NotImplemented
        # self.external_torques = NotImplemented
