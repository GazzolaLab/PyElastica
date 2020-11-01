__doc__ = """Rod classes and its data structures """

from elastica.rod.data_structures import *


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
