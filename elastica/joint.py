__doc__ = """ Module containing joint classes to connect multiple rods together. """
__all__ = ["FreeJoint", "HingeJoint", "FixedJoint", "ExternalContact", "SelfContact"]
from elastica._linalg import _batch_product_k_ik_to_ik
from elastica._rotations import _inv_rotate
from math import sqrt
import numba
import numpy as np


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
        elastic_force = self.k * end_distance_vector

        relative_velocity = (
            rod_two.velocity_collection[..., index_two]
            - rod_one.velocity_collection[..., index_one]
        )
        damping_force = self.nu * relative_velocity

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
    def apply_forces(self, system_one, index_one, system_two, index_two):
        return super().apply_forces(system_one, index_one, system_two, index_two)

    def apply_torques(self, system_one, index_one, system_two, index_two):
        # current tangent direction of the `index_two` element of system two
        system_two_tangent = system_two.director_collection[2, :, index_two]

        # projection of the tangent of system two onto the plane normal
        force_direction = (
            -np.dot(system_two_tangent, self.normal_direction) * self.normal_direction
        )

        # compute the restoring torque
        torque = self.kt * np.cross(system_two_tangent, force_direction)

        # The opposite torque will be applied on link one
        system_one.external_torques[..., index_one] -= (
            system_one.director_collection[..., index_one] @ torque
        )
        system_two.external_torques[..., index_two] += (
            system_two.director_collection[..., index_two] @ torque
        )


class FixedJoint(FreeJoint):
    """
    The fixed joint class restricts the relative movement and rotation
    between two nodes and elements by applying restoring forces and torques.
    For implementation details, refer to Zhang et al. Nature
    Communications (2019).

        Notes
        -----
        Issue #131 : Add constraint in twisting, add rest_rotation_matrix (v0.3.0)

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
        rest_rotation_matrix: np.array
            2D (3,3) array containing data with 'float' type.
            Rest 3x3 rotation matrix from system one to system two at the connected elements.
            Instead of aligning the directors of both systems directly, a desired rest rotational matrix labeled C_12*
            is enforced.
    """

    def __init__(self, k, nu, kt, nut=0.0, rest_rotation_matrix=None):
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
        rest_rotation_matrix: np.array
            2D (3,3) array containing data with 'float' type.
            Rest 3x3 rotation matrix from system one to system two at the connected elements.
            If provided, the rest rotation matrix is enforced between the two systems throughout the simulation.
            If not provided, `rest_rotation_matrix` is initialized to the identity matrix,
            which means that a restoring torque will be applied to align the directors of both systems directly.
            (default=None)
        """
        super().__init__(k, nu)
        # additional in-plane constraint through restoring torque
        # stiffness of the restoring constraint -- tuned empirically
        self.kt = kt
        self.nut = nut

        # TODO: compute the rest rotation matrix directly during initialization
        #  as soon as systems (e.g. `rod_one` and `rod_two`) and indices (e.g. `index_one` and `index_two`)
        #  are available in the __init__
        if rest_rotation_matrix is None:
            rest_rotation_matrix = np.eye(3)
        assert rest_rotation_matrix.shape == (3, 3), "Rest rotation matrix must be 3x3"
        self.rest_rotation_matrix = rest_rotation_matrix

    # Apply force is same as free joint
    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        return super().apply_forces(rod_one, index_one, rod_two, index_two)

    def apply_torques(self, system_one, index_one, system_two, index_two):
        # collect directors of systems one and two
        # note that systems can be either rods or rigid bodies
        system_one_director = system_one.director_collection[..., index_one]
        system_two_director = system_two.director_collection[..., index_two]

        # rel_rot: C_12 = C_1I @ C_I2
        # C_12 is relative rotation matrix from system 1 to system 2
        # C_1I is the rotation from system 1 to the inertial frame (i.e. the world frame)
        # C_I2 is the rotation from the inertial frame to system 2 frame (inverse of system_two_director)
        rel_rot = system_one_director @ system_two_director.T
        # error_rot: C_22* = C_21 @ C_12*
        # C_22* is rotation matrix from current orientation of system 2 to desired orientation of system 2
        # C_21 is the inverse of C_12, which describes the relative (current) rotation from system 1 to system 2
        # C_12* is the desired rotation between systems one and two, which is saved in the static_rotation attribute
        dev_rot = rel_rot.T @ self.rest_rotation_matrix

        # compute rotation vectors based on C_22*
        # scipy implementation
        # rot_vec = Rotation.from_matrix(dev_rot).as_rotvec()
        #
        # implementation using custom _inv_rotate compiled with numba
        # rotation vector between identity matrix and C_22*
        rot_vec = _inv_rotate(np.dstack([np.eye(3), dev_rot.T])).squeeze()

        # rotate rotation vector into inertial frame
        rot_vec_inertial_frame = system_two_director.T @ rot_vec

        # deviation in rotation velocity between system 1 and system 2
        # first convert to inertial frame, then take differences
        dev_omega = (
            system_two_director.T @ system_two.omega_collection[..., index_two]
            - system_one_director.T @ system_one.omega_collection[..., index_one]
        )

        # we compute the constraining torque using a rotational spring - damper system in the inertial frame
        torque = self.kt * rot_vec_inertial_frame - self.nut * dev_omega

        # The opposite torques will be applied to system one and two after rotating the torques into the local frame
        system_one.external_torques[..., index_one] -= system_one_director @ torque
        system_two.external_torques[..., index_two] += system_two_director @ torque


def get_relative_rotation_two_systems(system_one, index_one, system_two, index_two):
    """
    Compute the relative rotation matrix C_12 between system one and system two at the specified elements.

    Examples
    ----------
    How to get the relative rotation between two systems (e.g. the rotation from end of rod one to base of rod two):

        >>> rel_rot_mat = get_relative_rotation_two_systems(rod1, -1, rod2, 0)

    How to initialize a FixedJoint with a rest rotation between the two systems,
    which is enforced throughout the simulation:

        >>> simulator.connect(
        ...    first_rod=rod1, second_rod=rod2, first_connect_idx=-1, second_connect_idx=0
        ... ).using(
        ...    FixedJoint,
        ...    ku=1e6, nu=0.0, kt=1e3, nut=0.0,
        ...    rest_rotation_matrix=get_relative_rotation_two_systems(rod1, -1, rod2, 0)
        ... )

    See Also
    ---------
    FixedJoint

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
    relative_rotation_matrix : np.array
        Relative rotation matrix C_12 between the two systems for their current state.
    """
    return (
        system_one.director_collection[..., index_one]
        @ system_two.director_collection[..., index_two].T
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

    # Return distance, contact point of system 2, contact point of system 1
    return x2 + s * e2 - x1 - t * e1, x2 + s * e2, x1 - t * e1


@numba.njit(cache=True)
def _calculate_contact_forces_rod_rigid_body(
    x_collection_rod,
    edge_collection_rod,
    x_cylinder_center,
    x_cylinder_tip,
    edge_cylinder,
    radii_sum,
    length_sum,
    internal_forces_rod,
    external_forces_rod,
    external_forces_cylinder,
    external_torques_cylinder,
    cylinder_director_collection,
    velocity_rod,
    velocity_cylinder,
    contact_k,
    contact_nu,
    velocity_damping_coefficient,
    friction_coefficient,
):
    # We already pass in only the first n_elem x
    n_points = x_collection_rod.shape[1]
    cylinder_total_contact_forces = np.zeros((3))
    cylinder_total_contact_torques = np.zeros((3))
    for i in range(n_points):
        # Element-wise bounding box
        x_selected = x_collection_rod[..., i]
        # x_cylinder is already a (,) array from outised
        del_x = x_selected - x_cylinder_tip
        norm_del_x = _norm(del_x)

        # If outside then don't process
        if norm_del_x >= (radii_sum[i] + length_sum[i]):
            continue

        # find the shortest line segment between the two centerline
        # segments : differs from normal cylinder-cylinder intersection
        distance_vector, x_cylinder_contact_point, _ = _find_min_dist(
            x_selected, edge_collection_rod[..., i], x_cylinder_tip, edge_cylinder
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

        # Compute contact spring force
        contact_force = contact_k * gamma * distance_vector
        interpenetration_velocity = velocity_cylinder[..., 0] - 0.5 * (
            velocity_rod[..., i] + velocity_rod[..., i + 1]
        )
        # Compute contact damping
        normal_interpenetration_velocity = (
            _dot_product(interpenetration_velocity, distance_vector) * distance_vector
        )
        contact_damping_force = -contact_nu * normal_interpenetration_velocity

        # magnitude* direction
        net_contact_force = 0.5 * mask * (contact_damping_force + contact_force)

        # Compute friction
        slip_interpenetration_velocity = (
            interpenetration_velocity - normal_interpenetration_velocity
        )
        slip_interpenetration_velocity_mag = np.linalg.norm(
            slip_interpenetration_velocity
        )
        slip_interpenetration_velocity_unitized = slip_interpenetration_velocity / (
            slip_interpenetration_velocity_mag + 1e-14
        )
        # Compute friction force in the slip direction.
        damping_force_in_slip_direction = (
            velocity_damping_coefficient * slip_interpenetration_velocity_mag
        )
        # Compute Coulombic friction
        coulombic_friction_force = friction_coefficient * np.linalg.norm(
            net_contact_force
        )
        # Compare damping force in slip direction and kinetic friction and minimum is the friction force.
        friction_force = (
            -min(damping_force_in_slip_direction, coulombic_friction_force)
            * slip_interpenetration_velocity_unitized
        )
        # Update contact force
        net_contact_force += friction_force

        # Torques acting on the cylinder
        moment_arm = x_cylinder_contact_point - x_cylinder_center

        # Add it to the rods at the end of the day
        if i == 0:
            external_forces_rod[..., i] -= 2 / 3 * net_contact_force
            external_forces_rod[..., i + 1] -= 4 / 3 * net_contact_force
            cylinder_total_contact_forces += 2.0 * net_contact_force
            cylinder_total_contact_torques += np.cross(
                moment_arm, 2.0 * net_contact_force
            )
        elif i == n_points:
            external_forces_rod[..., i] -= 4 / 3 * net_contact_force
            external_forces_rod[..., i + 1] -= 2 / 3 * net_contact_force
            cylinder_total_contact_forces += 2.0 * net_contact_force
            cylinder_total_contact_torques += np.cross(
                moment_arm, 2.0 * net_contact_force
            )
        else:
            external_forces_rod[..., i] -= net_contact_force
            external_forces_rod[..., i + 1] -= net_contact_force
            cylinder_total_contact_forces += 2.0 * net_contact_force
            cylinder_total_contact_torques += np.cross(
                moment_arm, 2.0 * net_contact_force
            )

    # Update the cylinder external forces and torques
    external_forces_cylinder[..., 0] += cylinder_total_contact_forces
    external_torques_cylinder[..., 0] += (
        cylinder_director_collection @ cylinder_total_contact_torques
    )


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
            distance_vector, _, _ = _find_min_dist(
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
            distance_vector, _, _ = _find_min_dist(
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
    This class is for applying contact forces between rod-cylinder and rod-rod.
    If you are want to apply contact forces between rod and cylinder, first system is always rod and second system
    is always cylinder.
    In addition to the contact forces, user can define apply friction forces between rod and cylinder that
    are in contact. For details on friction model refer to this [1]_.
    TODO: Currently friction force is between rod-cylinder, in future implement friction forces between rod-rod.

    Notes
    -----
    The `velocity_damping_coefficient` is set to a high value (e.g. 1e4) to minimize slip and simulate stiction
    (static friction), while friction_coefficient corresponds to the Coulombic friction coefficient.

    Examples
    --------
    How to define contact between rod and cylinder.

    >>> simulator.connect(rod, cylinder).using(
    ...    ExternalContact,
    ...    k=1e4,
    ...    nu=10,
    ...    velocity_damping_coefficient=10,
    ...    kinetic_friction_coefficient=10,
    ... )

    How to define contact between rod and rod.

    >>> simulator.connect(rod, rod).using(
    ...    ExternalContact,
    ...    k=1e4,
    ...    nu=10,
    ... )

    .. [1] Preclik T., Popa Constantin., Rude U., Regularizing a Time-Stepping Method for Rigid Multibody Dynamics, Multibody Dynamics 2011, ECCOMAS. URL: https://www10.cs.fau.de/publications/papers/2011/Preclik_Multibody_Ext_Abstr.pdf
    """

    # Dev note:
    # Most of the cylinder-cylinder contact SHOULD be implemented
    # as given in this `paper <http://larochelle.sdsmt.edu/publications/2005-2009/Collision%20Detection%20of%20Cylindrical%20Rigid%20Bodies%20Using%20Line%20Geometry.pdf>`,
    # but the elastica-cpp kernels are implemented.
    # This is maybe to speed-up the kernel, but it's
    # potentially dangerous as it does not deal with "end" conditions
    # correctly.

    def __init__(self, k, nu, velocity_damping_coefficient=0, friction_coefficient=0):
        """

        Parameters
        ----------
        k : float
            Contact spring constant.
        nu : float
            Contact damping constant.
        velocity_damping_coefficient : float
            Velocity damping coefficient between rigid-body and rod contact is used to apply friction force in the
            slip direction.
        friction_coefficient : float
            For Coulombic friction coefficient for rigid-body and rod contact.
        """
        super().__init__(k, nu)
        self.velocity_damping_coefficient = velocity_damping_coefficient
        self.friction_coefficient = friction_coefficient

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

            rod_element_position = 0.5 * (
                rod_one.position_collection[..., 1:]
                + rod_one.position_collection[..., :-1]
            )
            _calculate_contact_forces_rod_rigid_body(
                rod_element_position,
                rod_one.lengths * rod_one.tangents,
                cylinder_two.position_collection[..., 0],
                x_cyl,
                cylinder_two.length * cylinder_two.director_collection[2, :, 0],
                rod_one.radius + cylinder_two.radius,
                rod_one.lengths + cylinder_two.length,
                rod_one.internal_forces,
                rod_one.external_forces,
                cylinder_two.external_forces,
                cylinder_two.external_torques,
                cylinder_two.director_collection[:, :, 0],
                rod_one.velocity_collection,
                cylinder_two.velocity_collection,
                self.k,
                self.nu,
                self.velocity_damping_coefficient,
                self.friction_coefficient,
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
