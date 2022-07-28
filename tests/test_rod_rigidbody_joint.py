__doc__ = """ Joint between rods and rigid bodies test module """

# System imports
import elastica
from elastica.experimental.connection_contact_joint.rod_rigidbody_connection import (
    compute_position_of_point,
    compute_velocity_of_point,
)
from elastica.joint import (
    FreeJoint,
    HingeJoint,
    FixedJoint,
)
from numpy.testing import assert_allclose
from elastica._rotations import _inv_rotate
from elastica.rigidbody import Cylinder, Sphere, RigidBodyBase
from elastica.rod.cosserat_rod import CosseratRod
from elastica.utils import Tolerance
import importlib
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from typing import *

# seed random number generator
np.random.seed(0)


@pytest.mark.parametrize("start", [np.array([0.0, 0.0, 0.0]), np.random.rand(3)])
@pytest.mark.parametrize(
    "euler_angles", [np.array([0.0, 0.0, 0.0]), 2 * np.pi * np.random.rand(3)]
)
@pytest.mark.parametrize(
    "point",
    [
        np.zeros((3,)),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.random.rand(3),
    ],
)
def test_compute_point_position_and_velocity(start, euler_angles, point):
    """
    This test is testing the `compute_position_of_point` and `compute_velocity_of_point` methods of the `RigidBody`
    class.

    Returns
    -------

    """
    base_length = 10
    base_radius = np.random.uniform(1, 10)
    density = np.random.uniform(1, 10)

    director = Rotation.from_euler("xyz", euler_angles).as_matrix().T

    test_cylinder = Cylinder(
        start, director[2, :], director[0, :], base_length, base_radius, density
    )

    # add linear velocity
    test_cylinder.velocity_collection[:, 0] = np.random.randn(3)
    # add angular velocity
    test_cylinder.omega_collection[:, 0] = np.random.randn(3)

    # position of point in inertial frame
    computed_position = compute_position_of_point(
        system=test_cylinder, point=point, index=0
    )
    target_position = test_cylinder.position_collection[:, 0] + np.dot(
        test_cylinder.director_collection[:, :, 0].T, point
    )

    # assert that computation of velocity is correct
    assert_allclose(target_position, computed_position, atol=Tolerance.atol())

    position = computed_position

    # rotate point into the inertial frame
    point_inertial_frame = np.dot(test_cylinder.director_collection[..., 0].T, point)
    # rotate angular velocity to inertial frame
    omega_inertial_frame = np.dot(
        test_cylinder.director_collection[..., 0].T,
        test_cylinder.omega_collection[..., 0],
    )
    # apply the euler differentiation rule
    target_velocity = test_cylinder.velocity_collection[..., 0] + np.cross(
        omega_inertial_frame, point_inertial_frame
    )

    computed_velocity = compute_velocity_of_point(
        system=test_cylinder, point=point, index=0
    )

    # assert that computation of velocity is correct
    assert_allclose(target_velocity, computed_velocity, atol=Tolerance.atol())
