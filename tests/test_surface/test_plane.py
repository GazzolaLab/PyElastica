__doc__ = """Tests for plane surface class"""
import numpy as np
from numpy.testing import assert_allclose
from elastica.utils import Tolerance
from elastica.surface import Plane
import pytest


# tests Initialisation of plane
def test_plane_initialization():
    """
    This test case is for testing initialization of rigid sphere and it checks the
    validity of the members of sphere class.

    Returns
    -------

    """
    # setting up test params
    plane_origin = np.random.rand(3).reshape(3, 1)
    plane_normal_direction = np.random.rand(3).reshape(3)
    plane_normal = plane_normal_direction / np.linalg.norm(plane_normal_direction)

    test_plane = Plane(plane_origin, plane_normal)
    # checking plane origin
    assert_allclose(
        test_plane.origin,
        plane_origin,
        atol=Tolerance.atol(),
    )

    # checking plane normal
    assert_allclose(test_plane.normal, plane_normal, atol=Tolerance.atol())

    # check unnormalized error message
    invalid_plane_normal = np.zeros(
        3,
    )
    with pytest.raises(AssertionError) as excinfo:
        test_plane = Plane(plane_origin, invalid_plane_normal)
    assert "plane normal is not a unit vector" in str(excinfo.value)
