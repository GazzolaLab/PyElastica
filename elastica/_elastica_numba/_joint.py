__doc__ = """ Joint between rods module of Elastica Numba implementation """

import numpy as np
import numba

from elastica.joint import FreeJoint
from math import sqrt


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
    """ Returns true if not intersecting else false"""
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
            cylinder_two.radius,
            cylinder_two.length,
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
