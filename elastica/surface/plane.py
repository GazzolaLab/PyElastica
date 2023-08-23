__doc__ = """plane surface class"""

from elastica.surface.surface_base import SurfaceBase
import numpy as np
from numpy.testing import assert_allclose
from elastica.utils import Tolerance


class Plane(SurfaceBase):
    def __init__(self, plane_origin: np.ndarray, plane_normal: np.ndarray):
        """
        Plane surface initializer.

        Parameters
        ----------
        plane_origin: numpy.ndarray
        2D (dim, 1) array containing data with 'float' type.
        Origin of the plane.
        plane_normal: numpy.ndarray
        2D (dim, 1) array containing data with 'float' type.
        The normal vector of the plane, must be normalized.
        """

        assert_allclose(
            np.linalg.norm(plane_normal),
            1,
            atol=Tolerance.atol(),
            err_msg="plane normal is not a unit vector",
        )
        self.normal = plane_normal.reshape(3)
        self.origin = plane_origin.reshape(3, 1)
