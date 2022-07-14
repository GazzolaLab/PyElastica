__doc__ = """ Module containing joint classes to connect multiple rods together. """
__all__ = ["FreeJoint", "HingeJoint", "FixedJoint", "ExternalContact", "SelfContact"]
import numpy as np
import numba
from elastica.utils import Tolerance, MaxDimension
from elastica._linalg import _batch_product_k_ik_to_ik
from math import sqrt
from scipy.spatial.transform import Rotation


class FreeJoint:
    """
    This free joint class is the base class for all joints. Free or spherical
    joints constrains the relative movement between two nodes (chosen by the user)
    by applying restoring forces. For implementation details, refer to Zhang et al. Nature Communications (2019).

    Notes
    -----
    Every new joint class must be derived from the FreeJoint class.

        Attributes
        ----------
        k: float
            Stiffness coefficient of the joint.
        nu: float
            Damping coefficient of the joint.

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
        nut: float
            Rotational damping coefficient of the joint.
    """

    def __init__(self, k, nu, kt, nut=0.0):
        """

        Parameters
        ----------
        k: float
            Stiffness coefficient of the joint.
        nu: float
            Damping coefficient of the joint.
        kt: float
            Rotational stiffness coefficient of the joint.
        nut: float = 0.
            Rotational damping coefficient of the joint.
        """
        super().__init__(k, nu)
        # additional in-plane constraint through restoring torque
        # stiffness of the restoring constraint -- tuned empirically
        self.kt = kt
        self.nut = nut

    # Apply force is same as free joint
    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        return super().apply_forces(rod_one, index_one, rod_two, index_two)

    def apply_torques(self, system_one, index_one, system_two, index_two):
        # collect directors of systems one and two
        # note that systems can be either rods or rigid bodies
        system_one_director = system_one.director_collection[..., index_one]
        system_two_director = system_two.director_collection[..., index_two]

        # relative rotation matrix from system 1 to system 2: C_12 = C_1W @ C_W2 where W denotes the inertial frame
        error_rot = system_one_director @ system_two_director.T

        # compute rotation vector from system one to system two based on relative rotation matrix
        rot_vec = Rotation.from_matrix(error_rot).as_rotvec(degrees=False)

        # rotate rotation vector into inertial frame
        rot_vec = system_one_director.T @ rot_vec

        # we compute the constraining torque using a rotational spring - damper system in the inertial frame
        torque = self.kt * rot_vec

        # add damping torque
        if self.nut > 0.0:
            # error in rotation velocity between system 1 and system 2
            error_omega = (
                system_two.omega_collection[..., index_two]
                - system_one.omega_collection[..., index_one]
            )
            torque += self.nut * error_omega

        # The opposite torques will be applied to system one and two after rotating the torques into the local frame
        system_one.external_torques[..., index_one] += system_one_director @ torque
        system_two.external_torques[..., index_two] -= system_two_director @ torque


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
def _calculate_contact_forces_rod_rigid_body(
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
def _calculate_contact_forces_rod_rod(
    x_collection_rod_one,
    radius_rod_one,
    length_rod_one,
    tangent_rod_one,
    velocity_rod_one,
    internal_forces_rod_one,
    external_forces_rod_one,
    x_collection_rod_two,
    radius_rod_two,
    length_rod_two,
    tangent_rod_two,
    velocity_rod_two,
    internal_forces_rod_two,
    external_forces_rod_two,
    contact_k,
    contact_nu,
):
    # We already pass in only the first n_elem x
    n_points_rod_one = x_collection_rod_one.shape[1]
    n_points_rod_two = x_collection_rod_two.shape[1]
    edge_collection_rod_one = _batch_product_k_ik_to_ik(length_rod_one, tangent_rod_one)
    edge_collection_rod_two = _batch_product_k_ik_to_ik(length_rod_two, tangent_rod_two)

    for i in range(n_points_rod_one):
        for j in range(n_points_rod_two):
            radii_sum = radius_rod_one[i] + radius_rod_two[j]
            length_sum = length_rod_one[i] + length_rod_two[j]
            # Element-wise bounding box
            x_selected_rod_one = x_collection_rod_one[..., i]
            x_selected_rod_two = x_collection_rod_two[..., j]

            del_x = x_selected_rod_one - x_selected_rod_two
            norm_del_x = _norm(del_x)

            # If outside then don't process
            if norm_del_x >= (radii_sum + length_sum):
                continue

            # find the shortest line segment between the two centerline
            # segments : differs from normal cylinder-cylinder intersection
            distance_vector = _find_min_dist(
                x_selected_rod_one,
                edge_collection_rod_one[..., i],
                x_selected_rod_two,
                edge_collection_rod_two[..., j],
            )
            distance_vector_length = _norm(distance_vector)
            distance_vector /= distance_vector_length

            gamma = radii_sum - distance_vector_length

            # If distance is large, don't worry about it
            if gamma < -1e-5:
                continue

            rod_one_elemental_forces = 0.5 * (
                external_forces_rod_one[..., i]
                + external_forces_rod_one[..., i + 1]
                + internal_forces_rod_one[..., i]
                + internal_forces_rod_one[..., i + 1]
            )

            rod_two_elemental_forces = 0.5 * (
                external_forces_rod_two[..., j]
                + external_forces_rod_two[..., j + 1]
                + internal_forces_rod_two[..., j]
                + internal_forces_rod_two[..., j + 1]
            )

            equilibrium_forces = -rod_one_elemental_forces + rod_two_elemental_forces

            normal_force = _dot_product(equilibrium_forces, distance_vector)
            # Following line same as np.where(normal_force < 0.0, -normal_force, 0.0)
            normal_force = abs(min(normal_force, 0.0))

            # CHECK FOR GAMMA > 0.0, heaviside but we need to overload it in numba
            # As a quick fix, use this instead
            mask = (gamma > 0.0) * 1.0

            contact_force = contact_k * gamma
            interpenetration_velocity = 0.5 * (
                (velocity_rod_one[..., i] + velocity_rod_one[..., i + 1])
                - (velocity_rod_two[..., j] + velocity_rod_two[..., j + 1])
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
                external_forces_rod_one[..., i] -= net_contact_force * 2 / 3
                external_forces_rod_one[..., i + 1] -= net_contact_force * 4 / 3
            elif i == n_points_rod_one:
                external_forces_rod_one[..., i] -= net_contact_force * 4 / 3
                external_forces_rod_one[..., i + 1] -= net_contact_force * 2 / 3
            else:
                external_forces_rod_one[..., i] -= net_contact_force
                external_forces_rod_one[..., i + 1] -= net_contact_force

            if j == 0:
                external_forces_rod_two[..., j] += net_contact_force * 2 / 3
                external_forces_rod_two[..., j + 1] += net_contact_force * 4 / 3
            elif j == n_points_rod_two:
                external_forces_rod_two[..., j] += net_contact_force * 4 / 3
                external_forces_rod_two[..., j + 1] += net_contact_force * 2 / 3
            else:
                external_forces_rod_two[..., j] += net_contact_force
                external_forces_rod_two[..., j + 1] += net_contact_force


@numba.njit(cache=True)
def _calculate_contact_forces_self_rod(
    x_collection_rod,
    radius_rod,
    length_rod,
    tangent_rod,
    velocity_rod,
    external_forces_rod,
    contact_k,
    contact_nu,
):
    # We already pass in only the first n_elem x
    n_points_rod = x_collection_rod.shape[1]
    edge_collection_rod_one = _batch_product_k_ik_to_ik(length_rod, tangent_rod)

    for i in range(n_points_rod):
        skip = 1 + np.ceil(0.8 * np.pi * radius_rod[i] / length_rod[i])
        for j in range(i - skip, -1, -1):
            radii_sum = radius_rod[i] + radius_rod[j]
            length_sum = length_rod[i] + length_rod[j]
            # Element-wise bounding box
            x_selected_rod_index_i = x_collection_rod[..., i]
            x_selected_rod_index_j = x_collection_rod[..., j]

            del_x = x_selected_rod_index_i - x_selected_rod_index_j
            norm_del_x = _norm(del_x)

            # If outside then don't process
            if norm_del_x >= (radii_sum + length_sum):
                continue

            # find the shortest line segment between the two centerline
            # segments : differs from normal cylinder-cylinder intersection
            distance_vector = _find_min_dist(
                x_selected_rod_index_i,
                edge_collection_rod_one[..., i],
                x_selected_rod_index_j,
                edge_collection_rod_one[..., j],
            )
            distance_vector_length = _norm(distance_vector)
            distance_vector /= distance_vector_length

            gamma = radii_sum - distance_vector_length

            # If distance is large, don't worry about it
            if gamma < -1e-5:
                continue

            # CHECK FOR GAMMA > 0.0, heaviside but we need to overload it in numba
            # As a quick fix, use this instead
            mask = (gamma > 0.0) * 1.0

            contact_force = contact_k * gamma
            interpenetration_velocity = 0.5 * (
                (velocity_rod[..., i] + velocity_rod[..., i + 1])
                - (velocity_rod[..., j] + velocity_rod[..., j + 1])
            )
            contact_damping_force = contact_nu * _dot_product(
                interpenetration_velocity, distance_vector
            )

            # magnitude* direction
            net_contact_force = (
                0.5 * mask * (contact_damping_force + contact_force)
            ) * distance_vector

            # Add it to the rods at the end of the day
            # if i == 0:
            #     external_forces_rod[...,i] -= net_contact_force *2/3
            #     external_forces_rod[...,i+1] -= net_contact_force * 4/3
            if i == n_points_rod:
                external_forces_rod[..., i] -= net_contact_force * 4 / 3
                external_forces_rod[..., i + 1] -= net_contact_force * 2 / 3
            else:
                external_forces_rod[..., i] -= net_contact_force
                external_forces_rod[..., i + 1] -= net_contact_force

            if j == 0:
                external_forces_rod[..., j] += net_contact_force * 2 / 3
                external_forces_rod[..., j + 1] += net_contact_force * 4 / 3
            # elif j == n_points_rod:
            #     external_forces_rod[..., j] += net_contact_force * 4/3
            #     external_forces_rod[..., j+1] += net_contact_force * 2/3
            else:
                external_forces_rod[..., j] += net_contact_force
                external_forces_rod[..., j + 1] += net_contact_force


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
def _prune_using_aabbs_rod_rigid_body(
    rod_one_position_collection,
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
        aabb_rod[i, 0] = (
            np.min(rod_one_position_collection[i]) - max_possible_dimension[i]
        )
        aabb_rod[i, 1] = (
            np.max(rod_one_position_collection[i]) + max_possible_dimension[i]
        )

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


@numba.njit(cache=True)
def _prune_using_aabbs_rod_rod(
    rod_one_position_collection,
    rod_one_radius_collection,
    rod_one_length_collection,
    rod_two_position_collection,
    rod_two_radius_collection,
    rod_two_length_collection,
):
    max_possible_dimension = np.zeros((3,))
    aabb_rod_one = np.empty((3, 2))
    aabb_rod_two = np.empty((3, 2))
    max_possible_dimension[...] = np.max(rod_one_radius_collection) + np.max(
        rod_one_length_collection
    )
    for i in range(3):
        aabb_rod_one[i, 0] = (
            np.min(rod_one_position_collection[i]) - max_possible_dimension[i]
        )
        aabb_rod_one[i, 1] = (
            np.max(rod_one_position_collection[i]) + max_possible_dimension[i]
        )

    max_possible_dimension[...] = np.max(rod_two_radius_collection) + np.max(
        rod_two_length_collection
    )

    for i in range(3):
        aabb_rod_two[i, 0] = (
            np.min(rod_two_position_collection[i]) - max_possible_dimension[i]
        )
        aabb_rod_two[i, 1] = (
            np.max(rod_two_position_collection[i]) + max_possible_dimension[i]
        )

    return _aabbs_not_intersecting(aabb_rod_two, aabb_rod_one)


class ExternalContact(FreeJoint):
    """
    Assumes that the second entity is a rigid body for now, can be
    changed at a later time

    Most of the cylinder-cylinder contact SHOULD be implemented
    as given in this `paper <http://larochelle.sdsmt.edu/publications/2005-2009/Collision%20Detection%20of%20Cylindrical%20Rigid%20Bodies%20Using%20Line%20Geometry.pdf>`_.

    but, it isn't (the elastica-cpp kernels are implented)!
    This is maybe to speed-up the kernel, but it's
    potentially dangerous as it does not deal with "end" conditions
    correctly.
    """

    def __init__(self, k, nu):
        super().__init__(k, nu)

    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        # del index_one, index_two

        # TODO: raise error during the initialization if rod one is rigid body.

        # If rod two has one element, then it is rigid body.
        if rod_two.n_elems == 1:
            cylinder_two = rod_two
            # First, check for a global AABB bounding box, and see whether that
            # intersects
            if _prune_using_aabbs_rod_rigid_body(
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

            _calculate_contact_forces_rod_rigid_body(
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

        else:
            # First, check for a global AABB bounding box, and see whether that
            # intersects

            if _prune_using_aabbs_rod_rod(
                rod_one.position_collection,
                rod_one.radius,
                rod_one.lengths,
                rod_two.position_collection,
                rod_two.radius,
                rod_two.lengths,
            ):
                return

            _calculate_contact_forces_rod_rod(
                rod_one.position_collection[
                    ..., :-1
                ],  # Discount last node, we want element start position
                rod_one.radius,
                rod_one.lengths,
                rod_one.tangents,
                rod_one.velocity_collection,
                rod_one.internal_forces,
                rod_one.external_forces,
                rod_two.position_collection[
                    ..., :-1
                ],  # Discount last node, we want element start position
                rod_two.radius,
                rod_two.lengths,
                rod_two.tangents,
                rod_two.velocity_collection,
                rod_two.internal_forces,
                rod_two.external_forces,
                self.k,
                self.nu,
            )


class SelfContact(FreeJoint):
    """
    This class is modeling self contact of rod.

    """

    def __init__(self, k, nu):
        super().__init__(k, nu)

    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        # del index_one, index_two

        _calculate_contact_forces_self_rod(
            rod_one.position_collection[
                ..., :-1
            ],  # Discount last node, we want element start position
            rod_one.radius,
            rod_one.lengths,
            rod_one.tangents,
            rod_one.velocity_collection,
            rod_one.external_forces,
            self.k,
            self.nu,
        )
