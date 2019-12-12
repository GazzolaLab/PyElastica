__doc__ = """ Interaction module """

import numpy as np

from ._linalg import _batch_matmul, _batch_matvec, _batch_cross
from elastica.utils import MaxDimension


def linear_interpolation_slip(velocity_slip, velocity_threshold):
    """
    This function takes the velocity of elements and checks if they are
    larger than the threshold velocity. If velocity of elements larger than
    threshold velocity then slip occurs.
    :param velocity_slip:
    :param velocity_threshold:
    :return:
    """
    abs_velocity_slip = np.sqrt(np.einsum("ij, ij->j", velocity_slip, velocity_slip))
    slip_points = np.where(np.fabs(abs_velocity_slip) > velocity_threshold)
    slip_function = np.ones((velocity_slip.shape[1]))
    slip_function[slip_points[:]] = np.fabs(
        1.0
        - np.minimum(1.0, abs_velocity_slip[slip_points[:]] / velocity_threshold - 1.0)
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
    def __init__(self, k, nu, origin_plane, normal_plane):
        self.k = k
        self.nu = nu
        self.origin_plane = origin_plane.reshape(3, 1)
        self.normal_plane = normal_plane.reshape(3)
        self.surface_tol = 1e-4

    def apply_normal_force(self, system):
        """
        This function computes the plane force response on the element, in the
        case of contact. Contact model given in Eqn 4.8 Gazzola et. al. RSoS 2018 paper
        is used.
        :param system:
        :return: magnitude of the plane response
        """
        element_position = 0.5 * (
            system.position_collection[..., :-1] + system.position_collection[..., 1:]
        )
        distance_from_plane = np.einsum(
            "i, ij->j", self.normal_plane, (element_position - self.origin_plane)
        )
        no_contact_point_idx = np.where(
            (distance_from_plane - system.radius) > self.surface_tol
        )
        # TODO: How should we compute internal forces here? Call _compute_internal_forces?
        # nodal_total_forces = system.internal_forces + system.external_forces
        nodal_total_forces = system._compute_internal_forces() + system.external_forces
        total_forces = nodes_to_elements(nodal_total_forces)

        force_component_along_normal_direction = np.einsum(
            "i, ij->j", self.normal_plane, total_forces
        )
        forces_along_normal_direction = np.einsum(
            "i, j->ij", self.normal_plane, force_component_along_normal_direction
        )
        # If the total force component along the plane normal direction is greater than zero that means,
        # total force is pushing rod away from the plane not towards the surface. Thus, response force
        # applied by the surface has to be zero.
        forces_along_normal_direction[
            ..., np.where(force_component_along_normal_direction > 0)
        ] = 0.0
        plane_penetration = np.minimum(distance_from_plane - system.radius, 0.0)
        elastic_force = -self.k * np.einsum(
            "i, j->ij", self.normal_plane, plane_penetration
        )
        element_velocity = 0.5 * (
            system.velocity_collection[..., :-1] + system.velocity_collection[..., 1:]
        )
        normal_component_of_element_velocity = np.einsum(
            "i, ij->j", self.normal_plane, element_velocity
        )
        damping_force = -self.nu * np.einsum(
            "i, j->ij", self.normal_plane, normal_component_of_element_velocity
        )
        plane_response_force = -forces_along_normal_direction
        # If rod element does not have any contact with plane, plane cannot apply response force on the
        # element. Thus lets set plane response force to 0.0 for the no contact points.
        plane_response_force[..., no_contact_point_idx] = 0.0
        plane_response_force_total = (
            plane_response_force + elastic_force + damping_force
        )
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
class AnistropicFrictionalPlane(InteractionPlane):
    def __init__(
        self,
        k,
        nu,
        origin_plane,
        normal_plane,
        slip_velocity_tol,
        static_mu_array,
        kinetic_mu_array,
    ):
        InteractionPlane.__init__(self, k, nu, origin_plane, normal_plane)
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
    def apply_force(self, system):
        # calculate axial and rolling directions
        plane_response_force_mag = self.apply_normal_force(system)
        normal_plane_array = np.repeat(
            self.normal_plane.reshape(3, 1), plane_response_force_mag.shape[0], axis=1
        )
        axial_direction = system.tangents
        element_v = 0.5 * (
            system.velocity_collection[..., :-1] + system.velocity_collection[..., 1:]
        )

        # first apply axial kinetic friction
        # dot product
        axial_velocity_mag = np.einsum("ij,ij->j", element_v, axial_direction)
        axial_velocity = np.einsum("j, ij->ij", axial_velocity_mag, axial_direction)
        axial_velocity_sign = np.sign(axial_velocity_mag)
        # check top for sign convention
        kinetic_mu = 0.5 * (
            self.kinetic_mu_forward * (1 + axial_velocity_sign)
            + self.kinetic_mu_backward * (1 - axial_velocity_sign)
        )
        axial_slip_function = linear_interpolation_slip(
            axial_velocity, self.slip_velocity_tol
        )
        axial_kinetic_friction_force = -(
            (1.0 - axial_slip_function)
            * kinetic_mu
            * plane_response_force_mag
            * axial_velocity_sign
            * axial_direction
        )
        system.external_forces[..., :-1] += 0.5 * axial_kinetic_friction_force
        system.external_forces[..., 1:] += 0.5 * axial_kinetic_friction_force

        # now rolling kinetic friction
        rolling_direction = _batch_cross(normal_plane_array, axial_direction)
        torque_arm = -system.radius * normal_plane_array
        rolling_velocity = np.einsum("ij ,ij ->j ", element_v, rolling_direction)
        directors_transpose = np.einsum("ijk -> jik", system.director_collection)
        # v_rot = Q.T @ omega @ Q @ r
        rotation_velocity = _batch_matvec(
            directors_transpose,
            _batch_cross(
                system.omega_collection,
                _batch_matvec(system.director_collection, torque_arm),
            ),
        )
        rolling_rotation_velocity = np.einsum(
            "ij,ij->j", rotation_velocity, rolling_direction
        )
        rolling_slip_velocity_mag = rolling_velocity + rolling_rotation_velocity
        rolling_slip_velocity = np.einsum(
            "j, ij->ij", rolling_slip_velocity_mag, rolling_direction
        )
        rolling_slip_velocity_sign = np.sign(rolling_slip_velocity_mag)
        rolling_slip_function = linear_interpolation_slip(
            rolling_slip_velocity, self.slip_velocity_tol
        )
        rolling_kinetic_friction_force = -(
            (1.0 - rolling_slip_function)
            * self.kinetic_mu_sideways
            * plane_response_force_mag
            * rolling_slip_velocity_sign
            * rolling_direction
        )
        system.external_forces[..., :-1] += 0.5 * rolling_kinetic_friction_force
        system.external_forces[..., 1:] += 0.5 * rolling_kinetic_friction_force
        # torque = Q @ r @ Fr
        system.external_torques += _batch_matvec(
            system.director_collection,
            _batch_cross(torque_arm, rolling_kinetic_friction_force),
        )

        # now axial static friction
        # TODO: How should we compute internal forces here? Call _compute_internal_forces? But they are already computed in update acceleration?
        # nodal_total_forces = system.internal_forces + system.external_forces
        nodal_total_forces = system._compute_internal_forces() + system.external_forces
        total_forces = nodes_to_elements(nodal_total_forces)
        projection = np.einsum("ij,ij->j", total_forces, axial_direction)
        projection_sign = np.sign(projection)
        # check top for sign convention
        static_mu = 0.5 * (
            self.static_mu_forward * (1 + projection_sign)
            + self.static_mu_backward * (1 - projection_sign)
        )
        max_friction_force = axial_slip_function * static_mu * plane_response_force_mag
        # friction = min(mu N, pushing force)
        axial_static_friction_force = -(
            np.minimum(np.fabs(projection), max_friction_force)
            * projection_sign
            * axial_direction
        )
        system.external_forces[..., :-1] += 0.5 * axial_static_friction_force
        system.external_forces[..., 1:] += 0.5 * axial_static_friction_force

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
        tangential_torques = np.einsum("ij,ij->j", total_torques, system.tangents)
        projection = np.einsum("ij,ij->j", total_forces, rolling_direction)
        noslip_force = -(
            (system.radius * projection - 2.0 * tangential_torques)
            / 3.0
            / system.radius
        )
        max_friction_force = (
            rolling_slip_function * self.static_mu_sideways * plane_response_force_mag
        )
        noslip_force_sign = np.sign(noslip_force)
        rolling_static_friction_force = (
            np.minimum(np.fabs(noslip_force), max_friction_force)
            * noslip_force_sign
            * rolling_direction
        )
        system.external_forces[..., :-1] += 0.5 * rolling_static_friction_force
        system.external_forces[..., 1:] += 0.5 * rolling_static_friction_force
        system.external_torques += _batch_matvec(
            system.director_collection,
            _batch_cross(torque_arm, rolling_static_friction_force),
        )
