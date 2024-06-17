__doc__ = """Base class for surfaces"""
from typing import Type

import numpy as np
from numpy.typing import NDArray


class SurfaceBase:
    """
    Base class for all surfaces.

    Notes
    -----
    All new surface classes must be derived from this SurfaceBase class.

    """

    REQUISITE_MODULES: list[Type] = []

    def __init__(self) -> None:
        """
        SurfaceBase does not take any arguments.
        """
        self.normal: NDArray[np.floating]  # (3,)
        self.origin: NDArray[np.floating]  # (3, 1)
