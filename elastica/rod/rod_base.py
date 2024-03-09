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
        self.position_collection: int
        self.omega_collection: int
        self.acceleration_collection: int
        self.alpha_collection: int
        self.external_forces: int
        self.external_torques: int
