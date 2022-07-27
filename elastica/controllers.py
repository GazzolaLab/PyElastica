from elastica.typing import SystemType
import numpy as np
from typing import Dict


class ControllerBase:
    """
    This is the base class for controllers acting on one or multiple systems.

    Notes
    -----
    Every new controller class must be derived
    from the ControllerBase class.

    """

    def __init__(self):
        """
        ControllerBase class does not need any input parameters.
        """
        pass

    def apply_forces(self, systems: Dict[str, SystemType], time: np.float64 = 0.0):
        """Apply forces to a system object.

        In ControllerBase class, this routine simply passes.

        Parameters
        ----------
        systems : Dict[str, SystemType]
             Dictionary of system objects.
        time : float
            The time of simulation.

        Returns
        -------


        """

        pass

    def apply_torques(self, systems: Dict[str, SystemType], time: np.float64 = 0.0):
        """Apply torques to a system object.

        In ControllerBase class, this routine simply passes.

        Parameters
        ----------
        systems : Dict[str, SystemType]
            Dictionary of system objects.
        time : float
            The time of simulation.

        Returns
        -------

        """
        pass
