__doc__ = """ Module containing joint classes to connect multiple rods together. """
__all__ = ["FreeJoint", "HingeJoint", "FixedJoint", "ExternalContact"]

import numpy as np
import numba
from elastica.utils import Tolerance, MaxDimension
from math import sqrt


class FreeJoint:
    """
    This free joint class is the base class for all joints. Free or spherical
    joints constrains the relative movement between two nodes (chosen by the user)
    by applying restoring forces. For implementation details, refer to Zhang et al. Nature Communications (2019).

        Attributes
        ----------
        k: float
            Stiffness coefficient of the joint.
        nu: float
            Damping coefficient of the joint.

    Note
    ----
    Every new joint class must be derived from the FreeJoint class.


    """

    # pass the k and nu for the forces
    # also the necessary rods for the joint
    # indices should be 0 or -1, we will provide wrappers for users later
    def __init__(self, k, nu):
        """

        Parameters
        ----------
        k: float
           Stiffness coefficient of the joint.
        nu: float
           Damping coefficient of the joint.

        """
        self.k = k
        self.nu = nu

    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        """
        Apply joint force to the connected rod objects.

        Parameters
        ----------
        rod_one : object
            Rod-like object
        index_one : int
            Index of first rod for joint.
        rod_two : object
            Rod-like object
        index_two : int
            Index of second rod for joint.

        Returns
        -------

        """
        end_distance_vector = (
            rod_two.position_collection[..., index_two]
            - rod_one.position_collection[..., index_one]
        )
        # Calculate norm of end_distance_vector
        # this implementation timed: 2.48 µs ± 126 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        end_distance = np.sqrt(np.dot(end_distance_vector, end_distance_vector))

        # Below if check is not efficient find something else
        # We are checking if end of rod1 and start of rod2 are at the same point in space
        # If they are at the same point in space, it is a zero vector.
        if end_distance <= Tolerance.atol():
            normalized_end_distance_vector = np.array([0.0, 0.0, 0.0])
        else:
            normalized_end_distance_vector = end_distance_vector / end_distance

        elastic_force = self.k * end_distance_vector

        relative_velocity = (
            rod_two.velocity_collection[..., index_two]
            - rod_one.velocity_collection[..., index_one]
        )
        normal_relative_velocity = (
            np.dot(relative_velocity, normalized_end_distance_vector)
            * normalized_end_distance_vector
        )
        damping_force = -self.nu * normal_relative_velocity

        contact_force = elastic_force + damping_force

        rod_one.external_forces[..., index_one] += contact_force
        rod_two.external_forces[..., index_two] -= contact_force

        return

    def apply_torques(self, rod_one, index_one, rod_two, index_two):
        """
        Apply restoring joint torques to the connected rod objects.

        In FreeJoint class, this routine simply passes.

        Parameters
        ----------
        rod_one : object
            Rod-like object
        index_one : int
            Index of first rod for joint.
        rod_two : object
            Rod-like object
        index_two : int
            Index of second rod for joint.

        Returns
        -------

        """
        pass


class HingeJoint(FreeJoint):
    """
    This hinge joint class constrains the relative movement and rotation
    (only one axis defined by the user) between two nodes and elements
    (chosen by the user) by applying restoring forces and torques. For
    implementation details, refer to Zhang et. al. Nature
    Communications (2019).

        Attributes
        ----------
        k: float
            Stiffness coefficient of the joint.
        nu: float
            Damping coefficient of the joint.
        kt: float
            Rotational stiffness coefficient of the joint.
        normal_direction: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type. Constraint rotation direction.
    """

    # TODO: IN WRAPPER COMPUTE THE NORMAL DIRECTION OR ASK USER TO GIVE INPUT, IF NOT THROW ERROR
    def __init__(self, k, nu, kt, normal_direction):
        """

        Parameters
        ----------
        k: float
            Stiffness coefficient of the joint.
        nu: float
            Damping coefficient of the joint.
        kt: float
            Rotational stiffness coefficient of the joint.
        normal_direction: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type. Constraint rotation direction.
        """
        super().__init__(k, nu)
        # normal direction of the constrain plane
        # for example for yz plane (1,0,0)
        # unitize the normal vector
        self.normal_direction = normal_direction / np.linalg.norm(normal_direction)
        # additional in-plane constraint through restoring torque
        # stiffness of the restoring constraint -- tuned empirically
        self.kt = kt

    # Apply force is same as free joint
    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        return super().apply_forces(rod_one, index_one, rod_two, index_two)

    def apply_torques(self, rod_one, index_one, rod_two, index_two):
        # current direction of the first element of link two
        # also NOTE: - rod two is hinged at first element
        link_direction = (
            rod_two.position_collection[..., index_two + 1]
            - rod_two.position_collection[..., index_two]
        )

        # projection of the link direction onto the plane normal
        force_direction = (
            -np.dot(link_direction, self.normal_direction) * self.normal_direction
        )

        # compute the restoring torque
        torque = self.kt * np.cross(link_direction, force_direction)

        # The opposite torque will be applied on link one
        rod_one.external_torques[..., index_one] -= (
            rod_one.director_collection[..., index_one] @ torque
        )
        rod_two.external_torques[..., index_two] += (
            rod_two.director_collection[..., index_two] @ torque
        )


class FixedJoint(FreeJoint):
    """
    The fixed joint class restricts the relative movement and rotation
    between two nodes and elements by applying restoring forces and torques.
    For implementation details, refer to Zhang et al. Nature
    Communications (2019).

        Attributes
        ----------
        k: float
            Stiffness coefficient of the joint.
        nu: float
            Damping coefficient of the joint.
        kt: float
            Rotational stiffness coefficient of the joint.
    """

    def __init__(self, k, nu, kt):
        """

        Parameters
        ----------
        k: float
            Stiffness coefficient of the joint.
        nu: float
            Damping coefficient of the joint.
        kt: float
            Rotational stiffness coefficient of the joint.
        """
        super().__init__(k, nu)
        # additional in-plane constraint through restoring torque
        # stiffness of the restoring constraint -- tuned empirically
        self.kt = kt

    # Apply force is same as free joint
    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        return super().apply_forces(rod_one, index_one, rod_two, index_two)

    def apply_torques(self, rod_one, index_one, rod_two, index_two):
        # current direction of the first element of link two
        # also NOTE: - rod two is fixed at first element
        link_direction = (
            rod_two.position_collection[..., index_two + 1]
            - rod_two.position_collection[..., index_two]
        )

        # To constrain the orientation of link two, the second node of link two should align with
        # the direction of link one. Thus, we compute the desired position of the second node of link two
        # as check1, and the current position of the second node of link two as check2. Check1 and check2
        # should overlap.

        tgt_destination = (
            rod_one.position_collection[..., index_one]
            + rod_two.rest_lengths[index_two] * rod_one.tangents[..., index_one]
        )  # dl of rod 2 can be different than rod 1 so use rest length of rod 2

        curr_destination = rod_two.position_collection[
            ..., index_two + 1
        ]  # second element of rod2

        # Compute the restoring torque
        forcedirection = -self.kt * (
            curr_destination - tgt_destination
        )  # force direction is between rod2 2nd element and rod1
        torque = np.cross(link_direction, forcedirection)

        # The opposite torque will be applied on link one
        rod_one.external_torques[..., index_one] -= (
            rod_one.director_collection[..., index_one] @ torque
        )
        rod_two.external_torques[..., index_two] += (
            rod_two.director_collection[..., index_two] @ torque
        )


@numba.njit(cache=True)
def _dot_product(a, b):
    sum = 0.0
    for i in range(3):
        sum += a[i] * b[i]
    return sum


@numba.njit(cache=True)
def _norm(a):
    return sqrt(_dot_product(a, a))


@numba.njit(cache=True)
def _clip(x, low, high):
    return max(low, min(x, high))


# Can this be made more efficient than 2 comp, 1 or?
@numba.njit(cache=True)
def _out_of_bounds(x, low, high):
    return (x < low) or (x > high)


@numba.njit(cache=True)
def _find_min_dist(x1, e1, x2, e2):
    e1e1 = _dot_product(e1, e1)
    e1e2 = _dot_product(e1, e2)
    e2e2 = _dot_product(e2, e2)

    x1e1 = _dot_product(x1, e1)
    x1e2 = _dot_product(x1, e2)
    x2e1 = _dot_product(e1, x2)
    x2e2 = _dot_product(x2, e2)

    s = 0.0
    t = 0.0

    parallel = abs(1.0 - e1e2 ** 2 / (e1e1 * e2e2)) < 1e-6
    if parallel:
        # Some are parallel, so do processing
        t = (x2e1 - x1e1) / e1e1  # Comes from taking dot of e1 with a normal
        t = _clip(t, 0.0, 1.0)
        s = (x1e2 + t * e1e2 - x2e2) / e2e2  # Same as before
        s = _clip(s, 0.0, 1.0)
    else:
        # Using the Cauchy-Binet formula on eq(7) in docstring referenc
        s = (e1e1 * (x1e2 - x2e2) + e1e2 * (x2e1 - x1e1)) / (e1e1 * e2e2 - (e1e2) ** 2)
        t = (e1e2 * s + x2e1 - x1e1) / e1e1

        if _out_of_bounds(s, 0.0, 1.0) or _out_of_bounds(t, 0.0, 1.0):
            # potential_s = -100.0
            # potential_t = -100.0
            # potential_d = -100.0
            # overall_minimum_distance = 1e20

            # Fill in the possibilities
            potential_t = (x2e1 - x1e1) / e1e1
            s = 0.0
            t = _clip(potential_t, 0.0, 1.0)
            potential_d = _norm(x1 + e1 * t - x2)
            overall_minimum_distance = potential_d

            potential_t = (x2e1 + e1e2 - x1e1) / e1e1
            potential_t = _clip(potential_t, 0.0, 1.0)
            potential_d = _norm(x1 + e1 * potential_t - x2 - e2)
            if potential_d < overall_minimum_distance:
                s = 1.0
                t = potential_t
                overall_minimum_distance = potential_d

            potential_s = (x1e2 - x2e2) / e2e2
            potential_s = _clip(potential_s, 0.0, 1.0)
            potential_d = _norm(x2 + potential_s * e2 - x1)
            if potential_d < overall_minimum_distance:
                s = potential_s
                t = 0.0
                overall_minimum_distance = potential_d

            potential_s = (x1e2 + e1e2 - x2e2) / e2e2
            potential_s = _clip(potential_s, 0.0, 1.0)
            potential_d = _norm(x2 + potential_s * e2 - x1 - e1)
            if potential_d < overall_minimum_distance:
                s = potential_s
                t = 1.0

    return x2 + s * e2 - x1 - t * e1


@numba.njit(cache=True)
def _calculate_contact_forces(
    x_collection_rod,
    edge_collection_rod,
    x_cylinder,
    edge_cylinder,
    radii_sum,
    length_sum,
    internal_forces_rod,
    external_forces_rod,
    external_forces_cylinder,
    velocity_rod,
    velocity_cylinder,
    contact_k,
    contact_nu,
):
    # We already pass in only the first n_elem x
    n_points = x_collection_rod.shape[1]
    for i in range(n_points):
        # Element-wise bounding box
        x_selected = x_collection_rod[..., i]
        # x_cylinder is already a (,) array from outised
        del_x = x_selected - x_cylinder
        norm_del_x = _norm(del_x)

        # If outside then don't process
        if norm_del_x >= (radii_sum[i] + length_sum[i]):
            continue

        # find the shortest line segment between the two centerline
        # segments : differs from normal cylinder-cylinder intersection
        distance_vector = _find_min_dist(
            x_selected, edge_collection_rod[..., i], x_cylinder, edge_cylinder
        )
        distance_vector_length = _norm(distance_vector)
        distance_vector /= distance_vector_length

        gamma = radii_sum[i] - distance_vector_length

        # If distance is large, don't worry about it
        if gamma < -1e-5:
            continue

        rod_elemental_forces = 0.5 * (
            external_forces_rod[..., i]
            + external_forces_rod[..., i + 1]
            + internal_forces_rod[..., i]
            + internal_forces_rod[..., i + 1]
        )
        equilibrium_forces = -rod_elemental_forces + external_forces_cylinder[..., 0]

        normal_force = _dot_product(equilibrium_forces, distance_vector)
        # Following line same as np.where(normal_force < 0.0, -normal_force, 0.0)
        normal_force = abs(min(normal_force, 0.0))

        # CHECK FOR GAMMA > 0.0, heaviside but we need to overload it in numba
        # As a quick fix, use this instead
        mask = (gamma > 0.0) * 1.0

        contact_force = contact_k * gamma
        interpenetration_velocity = (
            0.5 * (velocity_rod[..., i] + velocity_rod[..., i + 1])
            - velocity_cylinder[..., 0]
        )
        contact_damping_force = contact_nu * _dot_product(
            interpenetration_velocity, distance_vector
        )

        # magnitude* direction
        net_contact_force = (
            normal_force + 0.5 * mask * (contact_damping_force + contact_force)
        ) * distance_vector

        # Add it to the rods at the end of the day
        if i == 0:
            external_forces_rod[..., i] -= 0.5 * net_contact_force
            external_forces_rod[..., i + 1] -= net_contact_force
            external_forces_cylinder[..., 0] += 1.5 * net_contact_force
        elif i == n_points:
            external_forces_rod[..., i] -= net_contact_force
            external_forces_rod[..., i + 1] -= 0.5 * net_contact_force
            external_forces_cylinder[..., 0] += 1.5 * net_contact_force
        else:
            external_forces_rod[..., i] -= net_contact_force
            external_forces_rod[..., i + 1] -= net_contact_force
            external_forces_cylinder[..., 0] += 2.0 * net_contact_force


@numba.njit(cache=True)
def _aabbs_not_intersecting(aabb_one, aabb_two):
    """Returns true if not intersecting else false"""
    if (aabb_one[0, 1] < aabb_two[0, 0]) | (aabb_one[0, 0] > aabb_two[0, 1]):
        return 1
    if (aabb_one[1, 1] < aabb_two[1, 0]) | (aabb_one[1, 0] > aabb_two[1, 1]):
        return 1
    if (aabb_one[2, 1] < aabb_two[2, 0]) | (aabb_one[2, 0] > aabb_two[2, 1]):
        return 1

    return 0


@numba.njit(cache=True)
def _prune_using_aabbs(
    rod_one_position,
    rod_one_radius_collection,
    rod_one_length_collection,
    cylinder_position,
    cylinder_director,
    cylinder_radius,
    cylinder_length,
):
    max_possible_dimension = np.zeros((3,))
    aabb_rod = np.empty((3, 2))
    aabb_cylinder = np.empty((3, 2))
    max_possible_dimension[...] = np.max(rod_one_radius_collection) + np.max(
        rod_one_length_collection
    )
    for i in range(3):
        aabb_rod[i, 0] = np.min(rod_one_position[i]) - max_possible_dimension[i]
        aabb_rod[i, 1] = np.max(rod_one_position[i]) + max_possible_dimension[i]

    # Is actually Q^T * d but numba complains about performance so we do
    # d^T @ Q
    cylinder_dimensions_in_local_FOR = np.array(
        [cylinder_radius, cylinder_radius, 0.5 * cylinder_length]
    )
    cylinder_dimensions_in_world_FOR = np.zeros_like(cylinder_dimensions_in_local_FOR)
    for i in range(3):
        for j in range(3):
            cylinder_dimensions_in_world_FOR[i] += (
                cylinder_director[j, i, 0] * cylinder_dimensions_in_local_FOR[j]
            )

    max_possible_dimension = np.abs(cylinder_dimensions_in_world_FOR)
    aabb_cylinder[..., 0] = cylinder_position[..., 0] - max_possible_dimension
    aabb_cylinder[..., 1] = cylinder_position[..., 0] + max_possible_dimension
    return _aabbs_not_intersecting(aabb_cylinder, aabb_rod)


class ExternalContact(FreeJoint):
    """
    Assumes that the second entity is a rigid body for now, can be
    changed at a later time

    Most of the cylinder-cylinder contact SHOULD be implemented
    as given in this paper:
    http://larochelle.sdsmt.edu/publications/2005-2009/Collision%20Detection%20of%20Cylindrical%20Rigid%20Bodies%20Using%20Line%20Geometry.pdf

    but, it isn't (the elastica-cpp kernels are implented)!
    This is maybe to speed-up the kernel, but it's
    potentially dangerous as it does not deal with "end" conditions
    correctly.
    """

    def __init__(self, k, nu):
        super().__init__(k, nu)

    def apply_forces(self, rod_one, index_one, cylinder_two, index_two):
        # del index_one, index_two

        # First, check for a global AABB bounding box, and see whether that
        # intersects
        if _prune_using_aabbs(
            rod_one.position_collection,
            rod_one.radius,
            rod_one.lengths,
            cylinder_two.position_collection,
            cylinder_two.director_collection,
            cylinder_two.radius[0],
            cylinder_two.length[0],
        ):
            return

        x_cyl = (
            cylinder_two.position_collection[..., 0]
            - 0.5 * cylinder_two.length * cylinder_two.director_collection[2, :, 0]
        )

        _calculate_contact_forces(
            rod_one.position_collection[..., :-1],
            rod_one.lengths * rod_one.tangents,
            x_cyl,
            cylinder_two.length * cylinder_two.director_collection[2, :, 0],
            rod_one.radius + cylinder_two.radius,
            rod_one.lengths + cylinder_two.length,
            rod_one.internal_forces,
            rod_one.external_forces,
            cylinder_two.external_forces,
            rod_one.velocity_collection,
            cylinder_two.velocity_collection,
            self.k,
            self.nu,
        )
