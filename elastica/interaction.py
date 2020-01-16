__doc__ = """ Interaction module """

import numpy as np

from ._linalg import _batch_matmul, _batch_matvec, _batch_cross
from elastica.utils import MaxDimension
from elastica.external_forces import NoForces


def find_slipping_elements(velocity_slip, velocity_threshold):
    """
    This function takes the velocity of elements and checks if they are larger
    than the threshold velocity. If velocity of elements are larger than
    threshold velocity, that means those elements are slipping, in other words
    kinetic friction will be acting on those elements not static friction. This
    function output an array called slip function, this array has a size of number
    of elements. If velocity of element is smaller than the threshold velocity slip
    function value for that element is 1, which means static friction is acting on
    that element. If velocity of element is larger than the threshold velocity slip
    function value for that element is between 0 and 1, which means kinetic friction
    is acting on that element.

    Parameters
    ----------
    velocity_slip
    velocity_threshold

    Returns
    -------
    slip function

    """
    abs_velocity_slip = np.sqrt(np.einsum("ij, ij->j", velocity_slip, velocity_slip))
    slip_points = np.where(np.fabs(abs_velocity_slip) > velocity_threshold)
    slip_function = np.ones((velocity_slip.shape[1]))
    slip_function[slip_points] = np.fabs(
        1.0 - np.minimum(1.0, abs_velocity_slip[slip_points] / velocity_threshold - 1.0)
    )
    return slip_function


# TODO: node_to_elements only used in friction, so that it is located here, we can change it.
# Converting forces on nodes to elements
def nodes_to_elements(input):
    # TODO: find a way with out initialzing output vector
    output = np.zeros((input.shape[0], input.shape[1] - 1))
    output[..., :-1] += 0.5 * input[..., 1:-1]
    output[..., 1:] += 0.5 * input[..., 1:-1]
    output[..., 0] += input[..., 0]
    output[..., -1] += input[..., -1]
    return output


# base class for interaction
# only applies normal force no friction
class InteractionPlane:
    def __init__(self, k, nu, plane_origin, plane_normal):
        self.k = k
        self.nu = nu
        self.plane_origin = plane_origin.reshape(3, 1)
        self.plane_normal = plane_normal.reshape(3)
        self.surface_tol = 1e-4

    def apply_normal_force(self, system):
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
        # TODO: How should we compute internal forces here? Call _compute_internal_forces?
        # nodal_total_forces = system.internal_forces + system.external_forces
        nodal_total_forces = system._compute_internal_forces() + system.external_forces
        element_total_forces = nodes_to_elements(nodal_total_forces)
        force_component_along_normal_direction = np.einsum(
            "i, ij->j", self.plane_normal, element_total_forces
        )
        forces_along_normal_direction = np.einsum(
            "i, j->ij", self.plane_normal, force_component_along_normal_direction
        )
        # If the total force component along the plane normal direction is greater than zero that means,
        # total force is pushing rod away from the plane not towards the plane. Thus, response force
        # applied by the surface has to be zero.
        forces_along_normal_direction[
            ..., np.where(force_component_along_normal_direction > 0)
        ] = 0.0
        # Compute response force on the element. Plane response force
        # has to be away from the surface and towards the element. Thus
        # multiply forces along normal direction with negative sign.
        plane_response_force = -forces_along_normal_direction

        # Elastic force response due to penetration
        element_position = 0.5 * (
            system.position_collection[..., :-1] + system.position_collection[..., 1:]
        )
        distance_from_plane = np.einsum(
            "i, ij->j", self.plane_normal, (element_position - self.plane_origin)
        )
        plane_penetration = np.minimum(distance_from_plane - system.radius, 0.0)
        elastic_force = -self.k * np.einsum(
            "i, j->ij", self.plane_normal, plane_penetration
        )

        # Damping force response due to velocity towards the plane
        element_velocity = 0.5 * (
            system.velocity_collection[..., :-1] + system.velocity_collection[..., 1:]
        )
        normal_component_of_element_velocity = np.einsum(
            "i, ij->j", self.plane_normal, element_velocity
        )
        damping_force = -self.nu * np.einsum(
            "i, j->ij", self.plane_normal, normal_component_of_element_velocity
        )

        # Compute total plane response force
        plane_response_force_total = (
            plane_response_force + elastic_force + damping_force
        )

        # Check if the rod elements are in contact with plane.
        no_contact_point_idx = np.where(
            (distance_from_plane - system.radius) > self.surface_tol
        )
        # If rod element does not have any contact with plane, plane cannot apply response
        # force on the element. Thus lets set plane response force to 0.0 for the no contact points.
        plane_response_force[..., no_contact_point_idx] = 0.0
        plane_response_force_total[..., no_contact_point_idx] = 0.0

        system.external_forces[..., :-1] += 0.5 * plane_response_force_total
        system.external_forces[..., 1:] += 0.5 * plane_response_force_total

        return np.sqrt(
            np.einsum("ij, ij->j", plane_response_force, plane_response_force)
        )


# class for anisotropic frictional plane
# NOTE: friction coefficients are passed as arrays in the order
# mu_forward : mu_backward : mu_sideways
# head is at x[0] and forward means head to tail
# same convention for kinetic and static
# mu named as to which direction it opposes
class AnistropicFrictionalPlane(NoForces, InteractionPlane):
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
        InteractionPlane.__init__(self, k, nu, plane_origin, plane_normal)
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
        # calculate axial and rolling directions
        plane_response_force_mag = self.apply_normal_force(system)
        normal_plane_collection = np.repeat(
            self.plane_normal.reshape(3, 1), plane_response_force_mag.shape[0], axis=1
        )
        axial_direction = system.tangents
        element_velocity = 0.5 * (
            system.velocity_collection[..., :-1] + system.velocity_collection[..., 1:]
        )

        # first apply axial kinetic friction
        velocity_mag_along_axial_direction = np.einsum(
            "ij,ij->j", element_velocity, axial_direction
        )
        velocity_along_axial_direction = np.einsum(
            "j, ij->ij", velocity_mag_along_axial_direction, axial_direction
        )
        # Friction forces depends on the direction of velocity, in other words sign
        # of the velocity vector.
        velocity_sign_along_axial_direction = np.sign(
            velocity_mag_along_axial_direction
        )
        # Check top for sign convention
        kinetic_mu = 0.5 * (
            self.kinetic_mu_forward * (1 + velocity_sign_along_axial_direction)
            + self.kinetic_mu_backward * (1 - velocity_sign_along_axial_direction)
        )
        # Call slip function to check if elements slipping or not
        slip_function_along_axial_direction = find_slipping_elements(
            velocity_along_axial_direction, self.slip_velocity_tol
        )
        kinetic_friction_force_along_axial_direction = -(
            (1.0 - slip_function_along_axial_direction)
            * kinetic_mu
            * plane_response_force_mag
            * velocity_sign_along_axial_direction
            * axial_direction
        )
        system.external_forces[..., :-1] += (
            0.5 * kinetic_friction_force_along_axial_direction
        )
        system.external_forces[..., 1:] += (
            0.5 * kinetic_friction_force_along_axial_direction
        )

        # Now rolling kinetic friction
        rolling_direction = _batch_cross(normal_plane_collection, axial_direction)
        torque_arm = -system.radius * normal_plane_collection
        velocity_along_rolling_direction = np.einsum(
            "ij ,ij ->j ", element_velocity, rolling_direction
        )
        directors_transpose = np.einsum("ijk -> jik", system.director_collection)
        # w_rot = Q.T @ omega @ Q @ r
        rotation_velocity = _batch_matvec(
            directors_transpose,
            _batch_cross(
                system.omega_collection,
                _batch_matvec(system.director_collection, torque_arm),
            ),
        )
        rotation_velocity_along_rolling_direction = np.einsum(
            "ij,ij->j", rotation_velocity, rolling_direction
        )
        slip_velocity_mag_along_rolling_direction = (
            velocity_along_rolling_direction + rotation_velocity_along_rolling_direction
        )
        slip_velocity_along_rolling_direction = np.einsum(
            "j, ij->ij", slip_velocity_mag_along_rolling_direction, rolling_direction
        )
        slip_velocity_sign_along_rolling_direction = np.sign(
            slip_velocity_mag_along_rolling_direction
        )
        slip_function_along_rolling_direction = find_slipping_elements(
            slip_velocity_along_rolling_direction, self.slip_velocity_tol
        )
        kinetic_friction_force_along_rolling_direction = -(
            (1.0 - slip_function_along_rolling_direction)
            * self.kinetic_mu_sideways
            * plane_response_force_mag
            * slip_velocity_sign_along_rolling_direction
            * rolling_direction
        )
        system.external_forces[..., :-1] += (
            0.5 * kinetic_friction_force_along_rolling_direction
        )
        system.external_forces[..., 1:] += (
            0.5 * kinetic_friction_force_along_rolling_direction
        )
        # torque = Q @ r @ Fr
        system.external_torques += _batch_matvec(
            system.director_collection,
            _batch_cross(torque_arm, kinetic_friction_force_along_rolling_direction),
        )

        # now axial static friction
        # TODO: How should we compute internal forces here? Call _compute_internal_forces? But they are already computed in update acceleration?
        # nodal_total_forces = system.internal_forces + system.external_forces
        nodal_total_forces = system._compute_internal_forces() + system.external_forces
        element_total_forces = nodes_to_elements(nodal_total_forces)
        force_component_along_axial_direction = np.einsum(
            "ij,ij->j", element_total_forces, axial_direction
        )
        force_component_sign_along_axial_direction = np.sign(
            force_component_along_axial_direction
        )
        # check top for sign convention
        static_mu = 0.5 * (
            self.static_mu_forward * (1 + force_component_sign_along_axial_direction)
            + self.static_mu_backward * (1 - force_component_sign_along_axial_direction)
        )
        max_friction_force = (
            slip_function_along_axial_direction * static_mu * plane_response_force_mag
        )
        # friction = min(mu N, pushing force)
        static_friction_force_along_axial_direction = -(
            np.minimum(
                np.fabs(force_component_along_axial_direction), max_friction_force
            )
            * force_component_sign_along_axial_direction
            * axial_direction
        )
        system.external_forces[..., :-1] += (
            0.5 * static_friction_force_along_axial_direction
        )
        system.external_forces[..., 1:] += (
            0.5 * static_friction_force_along_axial_direction
        )

        # now rolling static friction
        # there is some normal, tangent and rolling directions inconsitency from Elastica
        # TODO: Make sure when we are here internal torques already computed
        # total_torques = _batch_matvec(
        #     directors_transpose, (system.internal_torques + system.external_torques)
        # )
        total_torques = _batch_matvec(
            directors_transpose,
            (system._compute_internal_torques() + system.external_torques),
        )
        # Elastica has opposite defs of tangents in interaction.h and rod.cpp
        total_torques_along_axial_direction = np.einsum(
            "ij,ij->j", total_torques, system.tangents
        )
        force_component_along_rolling_direction = np.einsum(
            "ij,ij->j", element_total_forces, rolling_direction
        )
        noslip_force = -(
            (
                system.radius * force_component_along_rolling_direction
                - 2.0 * total_torques_along_axial_direction
            )
            / 3.0
            / system.radius
        )
        max_friction_force = (
            slip_function_along_rolling_direction
            * self.static_mu_sideways
            * plane_response_force_mag
        )
        noslip_force_sign = np.sign(noslip_force)
        static_friction_force_along_rolling_direction = (
            np.minimum(np.fabs(noslip_force), max_friction_force)
            * noslip_force_sign
            * rolling_direction
        )
        system.external_forces[..., :-1] += (
            0.5 * static_friction_force_along_rolling_direction
        )
        system.external_forces[..., 1:] += (
            0.5 * static_friction_force_along_rolling_direction
        )
        system.external_torques += _batch_matvec(
            system.director_collection,
            _batch_cross(torque_arm, static_friction_force_along_rolling_direction),
        )
