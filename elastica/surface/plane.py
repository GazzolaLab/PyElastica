__doc__ = """"""

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
        plane_origin: np.ndarray
            Origin of the plane.
            Expect (3,1)-shaped array.
        plane_normal: np.ndarray
            The normal vector of the plane, must be normalized.
            Expect (3,1)-shaped array.
        """

        assert_allclose(
            np.linalg.norm(plane_normal),
            1,
            atol=Tolerance.atol(),
            err_msg="plane normal is not a unit vector",
        )
        self.normal = np.asarray(plane_normal).reshape(3)
        self.origin = np.asarray(plane_origin).reshape(3, 1)
