__doc__ = """Module containing plane surface implementation for contact interactions."""
from typing import Type

import numpy as np
from numpy.typing import NDArray
from elastica.utils import Tolerance


class Plane:
    """
    Plane static system. Static system does not change by the timestepping.

    Attributes
    ----------
    normal : numpy.ndarray
        1D (3,) array containing the normal vector of the plane.
    origin : numpy.ndarray
        2D (3, 1) array containing the origin of the plane.
    """

    REQUISITE_MODULES: list[Type] = []

    def __init__(
        self, plane_origin: NDArray[np.float64], plane_normal: NDArray[np.float64]
    ):
        """
        Plane initializer.

        Parameters
        ----------
        plane_origin: numpy.ndarray
            1D (3,) or 2D (3, 1) array containing data with 'float' type.
            Origin of the plane.
        plane_normal: numpy.ndarray
            1D (3,) or 2D (3, 1) array containing data with 'float' type.
            The normal vector of the plane, must be normalized.
        """

        assert np.allclose(
            np.linalg.norm(plane_normal),
            1,
            atol=float(Tolerance.atol()),
        ), "plane normal is not a unit vector"
        self.normal = np.asarray(plane_normal).reshape(3)
        self.origin = np.asarray(plane_origin).reshape(3, 1)
