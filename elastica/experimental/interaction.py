__doc__ = """ Experimental interaction implementation."""
__all__ = [
    "AnisotropicFrictionalPlaneRigidBody",
]


import numpy as np
from elastica.external_forces import NoForces
from elastica.interaction import *
from elastica.interaction import (
    find_slipping_elements,
    apply_normal_force_numba_rigid_body,
    InteractionPlaneRigidBody,
)

import numba
from numba import njit
from elastica._linalg import (
    _batch_matmul,
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


class AnisotropicFrictionalPlaneRigidBody(NoForces, InteractionPlaneRigidBody):
    def __init__(
        self,
        k,
        nu,
        plane_origin,
        plane_normal,
        slip_velocity_tol,
        static_mu_array,
        kinetic_mu_array,
    ):
        InteractionPlaneRigidBody.__init__(self, k, nu, plane_origin, plane_normal)
        self.slip_velocity_tol = slip_velocity_tol
        (
            self.static_mu_forward,
            self.static_mu_backward,
            self.static_mu_sideways,
        ) = static_mu_array
        (
            self.kinetic_mu_forward,
            self.kinetic_mu_backward,
            self.kinetic_mu_sideways,
        ) = kinetic_mu_array

    # kinetic and static friction should separate functions
    # for now putting them together to figure out common variables
    def apply_forces(self, system, time=0.0):
        anisotropic_friction_numba_rigid_body(
            self.plane_origin,
            self.plane_normal,
            self.surface_tol,
            self.slip_velocity_tol,
            self.k,
            self.nu,
            self.kinetic_mu_forward,
            self.kinetic_mu_backward,
            self.kinetic_mu_sideways,
            self.static_mu_forward,
            self.static_mu_backward,
            self.static_mu_sideways,
            system.length,
            system.position_collection,
            system.director_collection,
            system.velocity_collection,
            system.omega_collection,
            system.external_forces,
            system.external_torques,
        )


@njit(cache=True)
def anisotropic_friction_numba_rigid_body(
    plane_origin,
    plane_normal,
    surface_tol,
    slip_velocity_tol,
    k,
    nu,
    kinetic_mu_forward,
    kinetic_mu_backward,
    kinetic_mu_sideways,
    static_mu_forward,
    static_mu_backward,
    static_mu_sideways,
    length,
    position_collection,
    director_collection,
    velocity_collection,
    omega_collection,
    external_forces,
    external_torques,
):
    # calculate axial and rolling directions
    # plane_response_force_mag, no_contact_point_idx = self.apply_normal_force(system)
    (
        plane_response_force_mag,
        no_contact_point_idx,
    ) = apply_normal_force_numba_rigid_body(
        plane_origin,
        plane_normal,
        surface_tol,
        k,
        nu,
        length,
        position_collection,
        velocity_collection,
        external_forces,
    )
    # FIXME: In future change the below part we should be able to compute the normal
    axial_direction = director_collection[0]  # rigid_body_normal  # system.tangents
    element_velocity = velocity_collection

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
    slip_function_along_axial_direction = find_slipping_elements(
        velocity_along_axial_direction, slip_velocity_tol
    )
    kinetic_friction_force_along_axial_direction = -(
        (1.0 - slip_function_along_axial_direction)
        * kinetic_mu
        * plane_response_force_mag
        * velocity_sign_along_axial_direction
        * axial_direction
    )

    binormal_direction = director_collection[1]  # rigid_body_binormal
    velocity_mag_along_binormal_direction = _batch_dot(
        element_velocity, binormal_direction
    )
    velocity_along_binormal_direction = _batch_product_k_ik_to_ik(
        velocity_mag_along_binormal_direction, binormal_direction
    )
    # Friction forces depends on the direction of velocity, in other words sign
    # of the velocity vector.
    velocity_sign_along_binormal_direction = np.sign(
        velocity_mag_along_binormal_direction
    )
    # Check top for sign convention
    kinetic_mu = 0.5 * (
        kinetic_mu_forward * (1 + velocity_sign_along_binormal_direction)
        + kinetic_mu_backward * (1 - velocity_sign_along_binormal_direction)
    )
    # Call slip function to check if elements slipping or not
    slip_function_along_binormal_direction = find_slipping_elements(
        velocity_along_binormal_direction, slip_velocity_tol
    )
    kinetic_friction_force_along_binormal_direction = -(
        (1.0 - slip_function_along_binormal_direction)
        * kinetic_mu
        * plane_response_force_mag
        * velocity_mag_along_binormal_direction
        * binormal_direction
    )

    # If rod element does not have any contact with plane, plane cannot apply friction
    # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
    kinetic_friction_force_along_axial_direction[..., no_contact_point_idx] = 0.0
    kinetic_friction_force_along_binormal_direction[..., no_contact_point_idx] = 0.0
    external_forces += (
        kinetic_friction_force_along_axial_direction
        + kinetic_friction_force_along_binormal_direction
    )
