__doc__ = """Rod plane contact with anistropic friction (no static friction)"""
from typing import Type

import numpy as np
from elastica._linalg import (
    _batch_norm,
    _batch_product_i_k_to_ik,
    _batch_product_k_ik_to_ik,
    _batch_vec_oneD_vec_cross,
    _batch_matvec,
    _batch_product_i_ik_to_k,
    _batch_dot,
    _batch_matrix_transpose,
    _batch_cross,
    _batch_vector_sum,
)

from elastica.contact_utils import (
    _node_to_element_position,
    _node_to_element_velocity,
    _find_slipping_elements,
    _elements_to_nodes_inplace,
    _node_to_element_mass_or_force,
)
from numba import njit
from elastica.rod.rod_base import RodBase
from elastica.surface import Plane
from elastica.surface.surface_base import SurfaceBase
from elastica.contact_forces import NoContact
from elastica.typing import RodType, SystemType


@njit(cache=True)
def apply_normal_force_numba(
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
):
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
    plane_response_force_total = elastic_force + damping_force

    # Check if the rod elements are in contact with plane.
    no_contact_point_idx = np.where((distance_from_plane - radius) > surface_tol)[0]
    # If rod element does not have any contact with plane, plane cannot apply response
    # force on the element. Thus lets set plane response force to 0.0 for the no contact points.
    plane_response_force[..., no_contact_point_idx] = 0.0
    plane_response_force_total[..., no_contact_point_idx] = 0.0

    # Update the external forces
    _elements_to_nodes_inplace(plane_response_force_total, external_forces)

    return (_batch_norm(plane_response_force), no_contact_point_idx)


@njit(cache=True)
def anisotropic_friction(
    plane_origin,
    plane_normal,
    surface_tol,
    slip_velocity_tol,
    k,
    nu,
    kinetic_mu_forward,
    kinetic_mu_backward,
    kinetic_mu_sideways,
    radius,
    mass,
    tangents,
    position_collection,
    director_collection,
    velocity_collection,
    omega_collection,
    internal_forces,
    external_forces,
):
    plane_response_force_mag, no_contact_point_idx = apply_normal_force_numba(
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

    unitized_total_velocity = element_velocity / _batch_norm(element_velocity + 1e-14)
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


class SnakeRodPlaneContact(NoContact):
    """
    This class is for applying contact forces between a snake rod and a plane with friction.
    First system is always rod and second system is always plane.

    How to define contact between rod and plane.
    >>> simulator.detect_contact_between(rod, plane).using(
    ...    SnakeRodPlaneContact,
    ...    k=1e4,
    ...    nu=10,
    ...    slip_velocity_tol = 1e-4,
    ...    kinetic_mu_array = np.array([1.0,2.0,3.0]),
    ... )
    """

    def __init__(
        self,
        k: float,
        nu: float,
        slip_velocity_tol: float,
        kinetic_mu_array: np.ndarray,
    ):
        """
        Parameters
        ----------
        k : float
                Contact spring constant.
        nu : float
                Contact damping constant.
        slip_velocity_tol: float
                Velocity tolerance to determine if the element is slipping or not.
        kinetic_mu_array: numpy.ndarray
                1D (3,) array containing data with 'float' type.
                [forward, backward, sideways] kinetic friction coefficients.
        """
        super().__init__()
        self.k = k
        self.nu = nu
        self.surface_tol = 1e-4
        self.slip_velocity_tol = slip_velocity_tol
        (
            self.kinetic_mu_forward,
            self.kinetic_mu_backward,
            self.kinetic_mu_sideways,
        ) = kinetic_mu_array

    @property
    def _allowed_system_two(self) -> list[Type]:
        # Modify this list to include the allowed system types for contact
        return [SurfaceBase]

    def apply_contact(self, system_one: RodType, system_two: SystemType) -> None:
        """
        In the case of contact with the plane, this function computes the plane reaction force on the element.

        Parameters
        ----------
        system_one: object
                Rod object.
        system_two: object
                Plane object.

        """
        anisotropic_friction(
            system_two.origin,
            system_two.normal,
            self.surface_tol,
            self.slip_velocity_tol,
            self.k,
            self.nu,
            self.kinetic_mu_forward,
            self.kinetic_mu_backward,
            self.kinetic_mu_sideways,
            system_one.radius,
            system_one.mass,
            system_one.tangents,
            system_one.position_collection,
            system_one.director_collection,
            system_one.velocity_collection,
            system_one.omega_collection,
            system_one.internal_forces,
            system_one.external_forces,
        )
