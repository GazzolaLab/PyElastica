__doc__ = """ Joints for generic system types test module """

# System imports
from elastica.experimental.connection_contact_joint.generic_system_type_connection import (
    GenericSystemTypeFreeJoint,
    compute_position_of_point,
    compute_velocity_of_point,
)
from numpy.testing import assert_allclose
from elastica.rigidbody import Cylinder, Sphere, RigidBodyBase
from elastica.rod.cosserat_rod import CosseratRod
from elastica.utils import Tolerance
import numpy as np
import pytest
from typing import *

# seed random number generator
rng = np.random.default_rng(0)


def init_system(system_class, origin: np.array = np.array([0.0, 0.0, 0.0])):
    # Some rod properties. We need them for constructor, they are not used.
    n = 4  # Number of elements
    normal = np.array([1.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    base_length = 1
    base_radius = 0.2
    density = 1
    # Youngs Modulus [Pa]
    youngs_modulus = 1e6
    # poisson ratio
    poisson_ratio = 0.5
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

    if system_class == CosseratRod:
        system = CosseratRod.straight_rod(
            n,
            origin,
            direction,
            normal,
            base_length,
            base_radius,
            density,
            youngs_modulus=youngs_modulus,
            shear_modulus=shear_modulus,
        )
    elif system_class == Cylinder:
        system = Cylinder(
            start=(origin - base_length / 2 * direction),
            direction=direction,
            normal=normal,
            base_length=base_length,
            base_radius=base_radius,
            density=density,
        )
    elif system_class == Sphere:
        system = Sphere(origin, base_radius, density)
    else:
        raise ValueError

    return system


@pytest.mark.parametrize("system_one_class", [CosseratRod, Sphere, Cylinder])
@pytest.mark.parametrize("system_two_class", [CosseratRod, Sphere, Cylinder])
@pytest.mark.parametrize(
    "point_system_one",
    [
        np.zeros((3,)),
        np.array([0.0, 0.0, 1.0]),
        rng.random(3),
        rng.random(3),
    ],
)
@pytest.mark.parametrize(
    "point_system_two",
    [
        np.zeros((3,)),
        np.array([0.0, 0.0, -1.0]),
        rng.random(3),
        rng.random(3),
    ],
)
def test_generic_free_joint(
    system_one_class: Union[Type[CosseratRod], Type[RigidBodyBase]],
    system_two_class: Union[Type[CosseratRod], Type[RigidBodyBase]],
    point_system_one: np.ndarray,
    point_system_two: np.ndarray,
):

    # Origin of the systems
    origin1 = np.array([0.0, 0.0, 0.0])
    origin2 = np.array([1.0, 0.0, 0.0])

    system_one = init_system(system_one_class, origin1)
    system_two = init_system(system_one_class, origin2)

    # Stiffness between points
    k = 1e8

    # Damping between two points
    nu = 1

    # System indices
    system_one_index = -1
    system_two_index = 0

    # System velocity
    v1 = np.array([-1, 0, 0]).reshape(3, 1)
    v2 = v1 * -1
    system_one.velocity_collection = np.tile(
        v1, (1, system_one.velocity_collection.shape[1] + 1)
    )
    system_two.velocity_collection = np.tile(
        v2, (1, system_two.velocity_collection.shape[1] + 1)
    )

    # System angular velocity
    omega1 = np.array([0.0, 1.0, 0.0]).reshape(3, 1)
    omega2 = -omega1
    system_one.omega_collection = np.tile(
        omega1, (1, system_one.omega_collection.shape[1] + 1)
    )
    system_two.omega_collection = np.tile(
        omega2, (1, system_two.omega_collection.shape[1] + 1)
    )

    # Verify positions
    position_system_one = compute_position_of_point(
        system=system_one, point=point_system_one, index=system_one_index
    )
    position_system_two = compute_position_of_point(
        system=system_two, point=point_system_two, index=system_two_index
    )

    # this is possible because neither system is rotated yet and still in the initial position
    # and because the rotation between inertial frame and local frame of the bodies is zero at the start.
    # The second point can be achieved by choosing the "normal" and "direction" vectors smartly.
    assert_allclose(
        position_system_one,
        system_one.position_collection[:, system_one_index] + point_system_one,
    )
    assert_allclose(
        position_system_two,
        system_two.position_collection[:, system_two_index] + point_system_two,
    )

    # Compute the distance between the connection points
    distance = position_system_two - position_system_one
    end_distance = np.linalg.norm(distance)
    if end_distance <= Tolerance.atol():
        end_distance = 1.0

    # Verify velocities calculated for system one
    velocity_system_one = compute_velocity_of_point(
        system=system_one, point=point_system_one, index=system_one_index
    )
    if np.array_equal(point_system_one, np.zeros((3, 1))):
        # for the joint on the node / CoM,
        # the velocity of the joint needs to be the same as the velocity of the node of system one
        assert_allclose(
            velocity_system_one,
            system_one.velocity_collection[..., system_one_index],
            atol=Tolerance.atol(),
        )
    elif np.array_equal(point_system_one, np.array([0.0, 0.0, 1.0])):
        # analytical computation of velocity for special case of point_system_two == np.array([0.0, 0.0, 1.0])
        assert_allclose(
            velocity_system_one,
            system_one.velocity_collection[..., system_one_index]
            + np.array([1.0, 0.0, 0.0]),
            atol=Tolerance.atol(),
        )

    # Verify velocities calculated for system two
    velocity_system_two = compute_velocity_of_point(
        system=system_two, point=point_system_two, index=system_two_index
    )
    if np.array_equal(point_system_two, np.zeros((3, 1))):
        # for the joint on the node / CoM,
        # the velocity of the joint needs to be the same as the velocity of the node of system two
        assert_allclose(
            velocity_system_two,
            system_two.velocity_collection[..., system_two_index],
            atol=Tolerance.atol(),
        )
    elif np.array_equal(point_system_two, np.array([0.0, 0.0, -1.0])):
        # analytical computation of velocity for special case of point_system_two == np.array([0.0, 0.0, -1.0])
        assert_allclose(
            velocity_system_two,
            system_two.velocity_collection[..., system_two_index]
            + np.array([1.0, 0.0, 0.0]),
            atol=Tolerance.atol(),
        )

    # Compute the relative velocity
    relative_vel = velocity_system_two - velocity_system_one

    # Compute the free joint forces
    elastic_force = k * distance
    damping_force = nu * relative_vel
    external_force = elastic_force + damping_force

    external_force_system_one = external_force
    external_force_system_two = -1 * external_force

    frjt = GenericSystemTypeFreeJoint(
        k=k,
        nu=nu,
        point_system_one=point_system_one,
        point_system_two=point_system_two,
    )
    frjt.apply_forces(system_one, system_one_index, system_two, system_two_index)
    frjt.apply_torques(system_one, system_one_index, system_two, system_two_index)

    assert_allclose(
        system_one.external_forces[..., system_one_index],
        external_force_system_one,
        atol=Tolerance.atol(),
    )
    assert_allclose(
        system_two.external_forces[..., system_two_index],
        external_force_system_two,
        atol=Tolerance.atol(),
    )

    # these checks are only possible because the system is not rotated
    # if system is rotated, first distance between CoM and joint connection point would need to be computed
    # and torque would need to be rotated into local frame at the end.

    # first check torque on system one
    if np.array_equal(point_system_one, np.zeros((3, 1))):
        # for the joint connected at the node / CoM there shouldn't be any torque generated
        assert_allclose(
            system_one.external_torques[..., system_one_index],
            np.zeros((3,)),
            atol=Tolerance.atol(),
        )
    elif np.array_equal(point_system_one, np.array([0.0, 0.0, 1.0])):
        assert_allclose(
            system_one.external_torques[..., system_one_index],
            np.array(
                [-external_force_system_one[1], external_force_system_one[0], 0.0]
            ),
            atol=Tolerance.atol(),
        )
    else:
        assert_allclose(
            system_one.external_torques[..., system_one_index],
            np.cross(point_system_one, external_force_system_one),
            atol=Tolerance.atol(),
        )

    if np.array_equal(point_system_two, np.zeros((3, 1))):
        # for the joint connected at the node / CoM there shouldn't be any torque generated
        assert_allclose(
            system_two.external_torques[..., system_two_index],
            np.zeros((3,)),
            atol=Tolerance.atol(),
        )
    elif np.array_equal(point_system_two, np.array([0.0, 0.0, 1.0])):
        assert_allclose(
            system_two.external_torques[..., system_two_index],
            np.array(
                [-external_force_system_two[1], external_force_system_two[0], 0.0]
            ),
            atol=Tolerance.atol(),
        )
    else:
        assert_allclose(
            system_two.external_torques[..., system_two_index],
            np.cross(point_system_two, external_force_system_two),
            atol=Tolerance.atol(),
        )
