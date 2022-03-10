__doc__ = """Base class for rods"""


class RodBase:
    """
    Base class for all rods.

    Notes
    -----
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
