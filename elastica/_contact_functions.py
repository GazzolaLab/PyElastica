__doc__ = """ Numba implementation module containing contact force calculation functions between rods and rigid bodies and other rods, rigid bodies or surfaces."""

from elastica.contact_utils import (
    _dot_product,
    _norm,
    _find_min_dist,
    _find_slipping_elements,
    _node_to_element_mass_or_force,
    _elements_to_nodes_inplace,
    _node_to_element_position,
    _node_to_element_velocity,
)
from elastica._linalg import (
    _batch_matvec,
    _batch_cross,
    _batch_norm,
    _batch_dot,
    _batch_product_i_k_to_ik,
    _batch_product_i_ik_to_k,
    _batch_product_k_ik_to_ik,
    _batch_vector_sum,
    _batch_matrix_transpose,
    _batch_vec_oneD_vec_cross,
)
import numpy as np
from numpy.typing import NDArray

from numba import njit


@njit(cache=True)  # type: ignore
def _calculate_contact_forces_rod_cylinder(
    x_collection_rod: NDArray[np.float64],
    edge_collection_rod: NDArray[np.float64],
    x_cylinder_center: NDArray[np.float64],
    x_cylinder_tip: NDArray[np.float64],
    edge_cylinder: NDArray[np.float64],
    radii_sum: NDArray[np.float64],
    length_sum: NDArray[np.float64],
    internal_forces_rod: NDArray[np.float64],
    external_forces_rod: NDArray[np.float64],
    external_forces_cylinder: NDArray[np.float64],
    external_torques_cylinder: NDArray[np.float64],
    cylinder_director_collection: NDArray[np.float64],
    velocity_rod: NDArray[np.float64],
    velocity_cylinder: NDArray[np.float64],
    contact_k: np.float64,
    contact_nu: np.float64,
    velocity_damping_coefficient: np.float64,
    friction_coefficient: np.float64,
) -> None:
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
            -np.minimum(damping_force_in_slip_direction, coulombic_friction_force)
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
        elif i == n_points - 1:
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


@njit(cache=True)  # type: ignore
def _calculate_contact_forces_rod_rod(
    x_collection_rod_one: NDArray[np.float64],
    radius_rod_one: NDArray[np.float64],
    length_rod_one: NDArray[np.float64],
    tangent_rod_one: NDArray[np.float64],
    velocity_rod_one: NDArray[np.float64],
    internal_forces_rod_one: NDArray[np.float64],
    external_forces_rod_one: NDArray[np.float64],
    x_collection_rod_two: NDArray[np.float64],
    radius_rod_two: NDArray[np.float64],
    length_rod_two: NDArray[np.float64],
    tangent_rod_two: NDArray[np.float64],
    velocity_rod_two: NDArray[np.float64],
    internal_forces_rod_two: NDArray[np.float64],
    external_forces_rod_two: NDArray[np.float64],
    contact_k: np.float64,
    contact_nu: np.float64,
) -> None:
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

            """FIX ME: Remove normal force and tune rod-rod contact example"""
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
            elif i == n_points_rod_one - 1:
                external_forces_rod_one[..., i] -= net_contact_force * 4 / 3
                external_forces_rod_one[..., i + 1] -= net_contact_force * 2 / 3
            else:
                external_forces_rod_one[..., i] -= net_contact_force
                external_forces_rod_one[..., i + 1] -= net_contact_force

            if j == 0:
                external_forces_rod_two[..., j] += net_contact_force * 2 / 3
                external_forces_rod_two[..., j + 1] += net_contact_force * 4 / 3
            elif j == n_points_rod_two - 1:
                external_forces_rod_two[..., j] += net_contact_force * 4 / 3
                external_forces_rod_two[..., j + 1] += net_contact_force * 2 / 3
            else:
                external_forces_rod_two[..., j] += net_contact_force
                external_forces_rod_two[..., j + 1] += net_contact_force


@njit(cache=True)  # type: ignore
def _calculate_contact_forces_self_rod(
    x_collection_rod: NDArray[np.float64],
    radius_rod: NDArray[np.float64],
    length_rod: NDArray[np.float64],
    tangent_rod: NDArray[np.float64],
    velocity_rod: NDArray[np.float64],
    external_forces_rod: NDArray[np.float64],
    contact_k: np.float64,
    contact_nu: np.float64,
) -> None:
    # We already pass in only the first n_elem x
    n_points_rod = x_collection_rod.shape[1]
    edge_collection_rod_one = _batch_product_k_ik_to_ik(length_rod, tangent_rod)

    for i in range(n_points_rod):
        skip = int(1 + np.ceil(0.8 * np.pi * radius_rod[i] / length_rod[i]))
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
            if i == n_points_rod - 1:
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


@njit(cache=True)  # type: ignore
def _calculate_contact_forces_rod_sphere(
    x_collection_rod: NDArray[np.float64],
    edge_collection_rod: NDArray[np.float64],
    x_sphere_center: NDArray[np.float64],
    x_sphere_tip: NDArray[np.float64],
    edge_sphere: NDArray[np.float64],
    radii_sum: NDArray[np.float64],
    length_sum: NDArray[np.float64],
    internal_forces_rod: NDArray[np.float64],
    external_forces_rod: NDArray[np.float64],
    external_forces_sphere: NDArray[np.float64],
    external_torques_sphere: NDArray[np.float64],
    sphere_director_collection: NDArray[np.float64],
    velocity_rod: NDArray[np.float64],
    velocity_sphere: NDArray[np.float64],
    contact_k: np.float64,
    contact_nu: np.float64,
    velocity_damping_coefficient: np.float64,
    friction_coefficient: np.float64,
) -> None:
    # We already pass in only the first n_elem x
    n_points = x_collection_rod.shape[1]
    sphere_total_contact_forces = np.zeros((3))
    sphere_total_contact_torques = np.zeros((3))
    for i in range(n_points):
        # Element-wise bounding box
        x_selected = x_collection_rod[..., i]
        # x_sphere is already a (,) array from outside
        del_x = x_selected - x_sphere_tip
        norm_del_x = _norm(del_x)

        # If outside then don't process
        if norm_del_x >= (radii_sum[i] + length_sum[i]):
            continue

        # find the shortest line segment between the two centerline
        distance_vector, x_sphere_contact_point, _ = _find_min_dist(
            x_selected, edge_collection_rod[..., i], x_sphere_tip, edge_sphere
        )
        distance_vector_length = _norm(distance_vector)
        distance_vector /= distance_vector_length

        gamma = radii_sum[i] - distance_vector_length

        # If distance is large, don't worry about it
        if gamma < -1e-5:
            continue

        # CHECK FOR GAMMA > 0.0, heaviside but we need to overload it in numba
        # As a quick fix, use this instead
        mask = (gamma > 0.0) * 1.0

        # Compute contact spring force
        contact_force = contact_k * gamma * distance_vector
        interpenetration_velocity = velocity_sphere[..., 0] - 0.5 * (
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
            -np.minimum(damping_force_in_slip_direction, coulombic_friction_force)
            * slip_interpenetration_velocity_unitized
        )
        # Update contact force
        net_contact_force += friction_force

        # Torques acting on the cylinder
        moment_arm = x_sphere_contact_point - x_sphere_center

        # Add it to the rods at the end of the day
        if i == 0:
            external_forces_rod[..., i] -= 2 / 3 * net_contact_force
            external_forces_rod[..., i + 1] -= 4 / 3 * net_contact_force
            sphere_total_contact_forces += 2.0 * net_contact_force
            sphere_total_contact_torques += np.cross(
                moment_arm, 2.0 * net_contact_force
            )
        elif i == n_points - 1:
            external_forces_rod[..., i] -= 4 / 3 * net_contact_force
            external_forces_rod[..., i + 1] -= 2 / 3 * net_contact_force
            sphere_total_contact_forces += 2.0 * net_contact_force
            sphere_total_contact_torques += np.cross(
                moment_arm, 2.0 * net_contact_force
            )
        else:
            external_forces_rod[..., i] -= net_contact_force
            external_forces_rod[..., i + 1] -= net_contact_force
            sphere_total_contact_forces += 2.0 * net_contact_force
            sphere_total_contact_torques += np.cross(
                moment_arm, 2.0 * net_contact_force
            )

    # Update the cylinder external forces and torques
    external_forces_sphere[..., 0] += sphere_total_contact_forces
    external_torques_sphere[..., 0] += (
        sphere_director_collection @ sphere_total_contact_torques
    )


@njit(cache=True)  # type: ignore
def _calculate_contact_forces_rod_plane(
    plane_origin: NDArray[np.float64],
    plane_normal: NDArray[np.float64],
    surface_tol: np.float64,
    k: np.float64,
    nu: np.float64,
    radius: NDArray[np.float64],
    mass: NDArray[np.float64],
    position_collection: NDArray[np.float64],
    velocity_collection: NDArray[np.float64],
    internal_forces: NDArray[np.float64],
    external_forces: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """
    This function computes the plane force response on the element, in the
    case of contact. Contact model given in Eqn 4.8 Gazzola et. al. RSoS 2018 paper
    is used.

    Parameters
    ----------
    system

    Returns
    -------
    magnitude of the plane response
    """

    # Compute plane response force
    nodal_total_forces = _batch_vector_sum(internal_forces, external_forces)
    element_total_forces = _node_to_element_mass_or_force(nodal_total_forces)

    force_component_along_normal_direction = _batch_product_i_ik_to_k(
        plane_normal, element_total_forces
    )
    forces_along_normal_direction = _batch_product_i_k_to_ik(
        plane_normal, force_component_along_normal_direction
    )

    # If the total force component along the plane normal direction is greater than zero that means,
    # total force is pushing rod away from the plane not towards the plane. Thus, response force
    # applied by the surface has to be zero.
    forces_along_normal_direction[
        ..., np.where(force_component_along_normal_direction > 0)[0]
    ] = 0.0
    # Compute response force on the element. Plane response force
    # has to be away from the surface and towards the element. Thus
    # multiply forces along normal direction with negative sign.
    plane_response_force = -forces_along_normal_direction

    # Elastic force response due to penetration
    element_position = _node_to_element_position(position_collection)
    distance_from_plane = _batch_product_i_ik_to_k(
        plane_normal, (element_position - plane_origin)
    )
    plane_penetration = np.minimum(distance_from_plane - radius, 0.0)
    elastic_force = -k * _batch_product_i_k_to_ik(plane_normal, plane_penetration)

    # Damping force response due to velocity towards the plane
    element_velocity = _node_to_element_velocity(
        mass=mass, node_velocity_collection=velocity_collection
    )
    normal_component_of_element_velocity = _batch_product_i_ik_to_k(
        plane_normal, element_velocity
    )
    damping_force = -nu * _batch_product_i_k_to_ik(
        plane_normal, normal_component_of_element_velocity
    )

    # Compute total plane response force
    plane_response_force_total = plane_response_force + elastic_force + damping_force

    # Check if the rod elements are in contact with plane.
    no_contact_point_idx = np.where((distance_from_plane - radius) > surface_tol)[0]
    # If rod element does not have any contact with plane, plane cannot apply response
    # force on the element. Thus lets set plane response force to 0.0 for the no contact points.
    plane_response_force[..., no_contact_point_idx] = 0.0
    plane_response_force_total[..., no_contact_point_idx] = 0.0

    # Update the external forces
    _elements_to_nodes_inplace(plane_response_force_total, external_forces)

    return (_batch_norm(plane_response_force), no_contact_point_idx)


@njit(cache=True)  # type: ignore
def _calculate_contact_forces_rod_plane_with_anisotropic_friction(
    plane_origin: NDArray[np.float64],
    plane_normal: NDArray[np.float64],
    surface_tol: np.float64,
    slip_velocity_tol: np.float64,
    k: np.float64,
    nu: np.float64,
    kinetic_mu_forward: np.float64,
    kinetic_mu_backward: np.float64,
    kinetic_mu_sideways: np.float64,
    static_mu_forward: np.float64,
    static_mu_backward: np.float64,
    static_mu_sideways: np.float64,
    radius: NDArray[np.float64],
    mass: NDArray[np.float64],
    tangents: NDArray[np.float64],
    position_collection: NDArray[np.float64],
    director_collection: NDArray[np.float64],
    velocity_collection: NDArray[np.float64],
    omega_collection: NDArray[np.float64],
    internal_forces: NDArray[np.float64],
    external_forces: NDArray[np.float64],
    internal_torques: NDArray[np.float64],
    external_torques: NDArray[np.float64],
) -> None:
    (
        plane_response_force_mag,
        no_contact_point_idx,
    ) = _calculate_contact_forces_rod_plane(
        plane_origin,
        plane_normal,
        surface_tol,
        k,
        nu,
        radius,
        mass,
        position_collection,
        velocity_collection,
        internal_forces,
        external_forces,
    )

    # First compute component of rod tangent in plane. Because friction forces acts in plane not out of plane. Thus
    # axial direction has to be in plane, it cannot be out of plane. We are projecting rod element tangent vector in
    # to the plane. So friction forces can only be in plane forces and not out of plane.
    tangent_along_normal_direction = _batch_product_i_ik_to_k(plane_normal, tangents)
    tangent_perpendicular_to_normal_direction = tangents - _batch_product_i_k_to_ik(
        plane_normal, tangent_along_normal_direction
    )
    tangent_perpendicular_to_normal_direction_mag = _batch_norm(
        tangent_perpendicular_to_normal_direction
    )
    # Normalize tangent_perpendicular_to_normal_direction. This is axial direction for plane. Here we are adding
    # small tolerance (1e-10) for normalization, in order to prevent division by 0.
    axial_direction = _batch_product_k_ik_to_ik(
        1 / (tangent_perpendicular_to_normal_direction_mag + 1e-14),
        tangent_perpendicular_to_normal_direction,
    )
    element_velocity = _node_to_element_velocity(
        mass=mass, node_velocity_collection=velocity_collection
    )
    # first apply axial kinetic friction
    velocity_mag_along_axial_direction = _batch_dot(element_velocity, axial_direction)
    velocity_along_axial_direction = _batch_product_k_ik_to_ik(
        velocity_mag_along_axial_direction, axial_direction
    )

    # Friction forces depends on the direction of velocity, in other words sign
    # of the velocity vector.
    velocity_sign_along_axial_direction = np.sign(velocity_mag_along_axial_direction)
    # Check top for sign convention
    kinetic_mu = 0.5 * (
        kinetic_mu_forward * (1 + velocity_sign_along_axial_direction)
        + kinetic_mu_backward * (1 - velocity_sign_along_axial_direction)
    )
    # Call slip function to check if elements slipping or not
    slip_function_along_axial_direction = _find_slipping_elements(
        velocity_along_axial_direction, slip_velocity_tol
    )
    # Now rolling kinetic friction
    rolling_direction = _batch_vec_oneD_vec_cross(axial_direction, plane_normal)
    torque_arm = _batch_product_i_k_to_ik(-plane_normal, radius)
    velocity_along_rolling_direction = _batch_dot(element_velocity, rolling_direction)
    directors_transpose = _batch_matrix_transpose(director_collection)
    # w_rot = Q.T @ omega @ Q @ r
    rotation_velocity = _batch_matvec(
        directors_transpose,
        _batch_cross(omega_collection, _batch_matvec(director_collection, torque_arm)),
    )
    rotation_velocity_along_rolling_direction = _batch_dot(
        rotation_velocity, rolling_direction
    )
    slip_velocity_mag_along_rolling_direction = (
        velocity_along_rolling_direction + rotation_velocity_along_rolling_direction
    )
    slip_velocity_along_rolling_direction = _batch_product_k_ik_to_ik(
        slip_velocity_mag_along_rolling_direction, rolling_direction
    )
    slip_function_along_rolling_direction = _find_slipping_elements(
        slip_velocity_along_rolling_direction, slip_velocity_tol
    )
    # Compute unitized total slip velocity vector. We will use this to distribute the weight of the rod in axial
    # and rolling directions.
    unitized_total_velocity = (
        slip_velocity_along_rolling_direction + velocity_along_axial_direction
    )
    unitized_total_velocity /= _batch_norm(unitized_total_velocity + 1e-14)
    # Apply kinetic friction in axial direction.
    kinetic_friction_force_along_axial_direction = -(
        (1.0 - slip_function_along_axial_direction)
        * kinetic_mu
        * plane_response_force_mag
        * _batch_dot(unitized_total_velocity, axial_direction)
        * axial_direction
    )
    # If rod element does not have any contact with plane, plane cannot apply friction
    # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
    kinetic_friction_force_along_axial_direction[..., no_contact_point_idx] = 0.0
    _elements_to_nodes_inplace(
        kinetic_friction_force_along_axial_direction, external_forces
    )
    # Apply kinetic friction in rolling direction.
    kinetic_friction_force_along_rolling_direction = -(
        (1.0 - slip_function_along_rolling_direction)
        * kinetic_mu_sideways
        * plane_response_force_mag
        * _batch_dot(unitized_total_velocity, rolling_direction)
        * rolling_direction
    )
    # If rod element does not have any contact with plane, plane cannot apply friction
    # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
    kinetic_friction_force_along_rolling_direction[..., no_contact_point_idx] = 0.0
    _elements_to_nodes_inplace(
        kinetic_friction_force_along_rolling_direction, external_forces
    )
    # torque = Q @ r @ Fr
    external_torques += _batch_matvec(
        director_collection,
        _batch_cross(torque_arm, kinetic_friction_force_along_rolling_direction),
    )

    # now axial static friction
    nodal_total_forces = _batch_vector_sum(internal_forces, external_forces)
    element_total_forces = _node_to_element_mass_or_force(nodal_total_forces)
    force_component_along_axial_direction = _batch_dot(
        element_total_forces, axial_direction
    )
    force_component_sign_along_axial_direction = np.sign(
        force_component_along_axial_direction
    )
    # check top for sign convention
    static_mu = 0.5 * (
        static_mu_forward * (1 + force_component_sign_along_axial_direction)
        + static_mu_backward * (1 - force_component_sign_along_axial_direction)
    )
    max_friction_force = (
        slip_function_along_axial_direction * static_mu * plane_response_force_mag
    )
    # friction = min(mu N, pushing force)
    static_friction_force_along_axial_direction = -(
        np.minimum(np.fabs(force_component_along_axial_direction), max_friction_force)
        * force_component_sign_along_axial_direction
        * axial_direction
    )
    # If rod element does not have any contact with plane, plane cannot apply friction
    # force on the element. Thus lets set static friction force to 0.0 for the no contact points.
    static_friction_force_along_axial_direction[..., no_contact_point_idx] = 0.0
    _elements_to_nodes_inplace(
        static_friction_force_along_axial_direction, external_forces
    )

    # now rolling static friction
    # there is some normal, tangent and rolling directions inconsitency from Elastica
    total_torques = _batch_matvec(
        directors_transpose, (internal_torques + external_torques)
    )
    # Elastica has opposite defs of tangents in interaction.h and rod.cpp
    total_torques_along_axial_direction = _batch_dot(total_torques, axial_direction)
    force_component_along_rolling_direction = _batch_dot(
        element_total_forces, rolling_direction
    )
    noslip_force = -(
        (
            radius * force_component_along_rolling_direction
            - 2.0 * total_torques_along_axial_direction
        )
        / 3.0
        / radius
    )
    max_friction_force = (
        slip_function_along_rolling_direction
        * static_mu_sideways
        * plane_response_force_mag
    )
    noslip_force_sign = np.sign(noslip_force)
    static_friction_force_along_rolling_direction = (
        np.minimum(np.fabs(noslip_force), max_friction_force)
        * noslip_force_sign
        * rolling_direction
    )
    # If rod element does not have any contact with plane, plane cannot apply friction
    # force on the element. Thus lets set plane static friction force to 0.0 for the no contact points.
    static_friction_force_along_rolling_direction[..., no_contact_point_idx] = 0.0
    _elements_to_nodes_inplace(
        static_friction_force_along_rolling_direction, external_forces
    )
    external_torques += _batch_matvec(
        director_collection,
        _batch_cross(torque_arm, static_friction_force_along_rolling_direction),
    )


@njit(cache=True)  # type: ignore
def _calculate_contact_forces_cylinder_plane(
    plane_origin: NDArray[np.float64],
    plane_normal: NDArray[np.float64],
    surface_tol: np.float64,
    k: np.float64,
    nu: np.float64,
    length: NDArray[np.float64],
    position_collection: NDArray[np.float64],
    velocity_collection: NDArray[np.float64],
    external_forces: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:

    # Compute plane response force
    # total_forces = system.internal_forces + system.external_forces
    total_forces = external_forces
    force_component_along_normal_direction = _batch_product_i_ik_to_k(
        plane_normal, total_forces
    )
    forces_along_normal_direction = _batch_product_i_k_to_ik(
        plane_normal, force_component_along_normal_direction
    )
    # If the total force component along the plane normal direction is greater than zero that means,
    # total force is pushing rod away from the plane not towards the plane. Thus, response force
    # applied by the surface has to be zero.
    forces_along_normal_direction[
        ..., np.where(force_component_along_normal_direction > 0)[0]
    ] = 0.0
    # Compute response force on the element. Plane response force
    # has to be away from the surface and towards the element. Thus
    # multiply forces along normal direction with negative sign.
    plane_response_force = -forces_along_normal_direction

    # Elastic force response due to penetration
    element_position = position_collection
    distance_from_plane = _batch_product_i_ik_to_k(
        plane_normal, (element_position - plane_origin)
    )
    plane_penetration = np.minimum(distance_from_plane - length / 2, 0.0)
    elastic_force = -k * _batch_product_i_k_to_ik(plane_normal, plane_penetration)

    # Damping force response due to velocity towards the plane
    element_velocity = velocity_collection
    normal_component_of_element_velocity = _batch_product_i_ik_to_k(
        plane_normal, element_velocity
    )
    damping_force = -nu * _batch_product_i_k_to_ik(
        plane_normal, normal_component_of_element_velocity
    )

    # Compute total plane response force
    plane_response_force_total = plane_response_force + elastic_force + damping_force

    # Check if the rigid body is in contact with plane.
    no_contact_point_idx = np.where((distance_from_plane - length / 2) > surface_tol)[0]
    # If rod element does not have any contact with plane, plane cannot apply response
    # force on the element. Thus lets set plane response force to 0.0 for the no contact points.
    plane_response_force[..., no_contact_point_idx] = 0.0
    plane_response_force_total[..., no_contact_point_idx] = 0.0

    # Update the external forces
    external_forces += plane_response_force_total

    return (_batch_norm(plane_response_force), no_contact_point_idx)
