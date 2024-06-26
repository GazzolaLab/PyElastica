__doc__ = """Base class for rods"""

from typing import Type
import numpy as np
from numpy.typing import NDArray


class RodBase:
    """
    Base class for all rods.

    Notes
    -----
    All new rod classes must be derived from this RodBase class.

    """

    REQUISITE_MODULES: list[Type] = []

    def __init__(self) -> None:
        """
        RodBase does not take any arguments.
        """
        self.position_collection: NDArray[np.float64]
        self.velocity_collection: NDArray[np.float64]
        self.acceleration_collection: NDArray[np.float64]
        self.director_collection: NDArray[np.float64]
        self.omega_collection: NDArray[np.float64]
        self.alpha_collection: NDArray[np.float64]
        self.external_forces: NDArray[np.float64]
        self.external_torques: NDArray[np.float64]

        self.ghost_voronoi_idx: NDArray[np.int32]
        self.ghost_elems_idx: NDArray[np.int32]

        self.periodic_boundary_nodes_idx: NDArray[np.int32]
        self.periodic_boundary_elems_idx: NDArray[np.int32]
        self.periodic_boundary_voronoi_idx: NDArray[np.int32]
