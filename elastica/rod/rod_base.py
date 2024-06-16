__doc__ = """Base class for rods"""

from typing import Any
import numpy as np
from numpy.typing import NDArray


class RodBase:
    """
    Base class for all rods.

    Notes
    -----
    All new rod classes must be derived from this RodBase class.

    """

    REQUISITE_MODULES: list[Any] = []

    def __init__(self) -> None:
        """
        RodBase does not take any arguments.
        """
        self.position_collection: NDArray[np.floating]
        self.velocity_collection: NDArray[np.floating]
        self.acceleration_collection: NDArray[np.floating]
        self.director_collection: NDArray[np.floating]
        self.omega_collection: NDArray[np.floating]
        self.alpha_collection: NDArray[np.floating]
        self.external_forces: NDArray[np.floating]
        self.external_torques: NDArray[np.floating]
