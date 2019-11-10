__doc__ = """ Interaction module """

import numpy as np

from ._linalg import _batch_matmul, _batch_matvec, _batch_cross


# interpolator for slip velocity for kinetic friction
def linear_interpolation_slip(velocity_slip, velocity_threshold):
    abs_velocity_slip = np.fabs(velocity_slip)
    # velocity_threshold_array = velocity_threshold * np.array((3, velocity_slip.shape[1]))
    slip_function = np.ones((1, velocity_slip.shape[1]))
    slip_points = np.where(np.fabs(abs_velocity_slip) > velocity_threshold)
    slip_function[0, slip_points] = np.fabs(
        1.0 - np.minimum(1.0, abs_velocity_slip / velocity_threshold - 1.0)
    )
    return slip_function


# base class for interaction
# only applies normal force no firctoon
class InteractionPlane:
    def __init__(self, k, nu, origin_plane, normal_plane):
        self.k = k
        self.nu = nu
        self.origin_plane = origin_plane
        self.normal_plane = normal_plane
        self.surface_tol = 1e-4

    def apply_normal_force(self, rod):
        element_x = 0.5 * (rod.position[..., :-1] + rod.position[..., 1:])
        distance_from_plane = self.normal_plane @ (element_x - self.origin_plane)
        no_contact_pts = np.where(distance_from_plane > self.surface_tol)
        nodal_total_forces = rod.internal_forces + rod.external_forces
        total_forces = 0.5 * (
            nodal_total_forces[..., :-1] + nodal_total_forces[..., 1:]
        )
        forces_normal_direction = self.normal_plane @ total_forces
        forces_normal = np.outer(self.normal_plane, forces_normal_direction)
        forces_normal[..., np.where(forces_normal_direction > 0)] = 0
        plane_penetration = np.minimum(distance_from_plane - rod.r, 0.0)
        elastic_force = -self.k * np.outer(self.normal_plane, plane_penetration)
        element_v = 0.5 * (rod.velocity[..., :-1] + rod.velocity[..., 1:])
        normal_v = self.normal_plane @ element_v
        damping_force = -self.nu * np.outer(self.normal_plane, normal_v)
        normal_force_plane = -forces_normal
        normal_force_plane[..., no_contact_pts[1]] = 0
        total_force_plane = normal_force_plane + elastic_force + damping_force
        rod.external_forces[..., :-1] += 0.5 * total_force_plane
        rod.external_forces[..., 1:] += 0.5 * total_force_plane
        return np.fabs(normal_force_plane)


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
    def apply_friction(self, rod):
        # calculate axial and rolling directions
        normal_force_plane = self.apply_normal_force(self, rod)
        normal_plane_array = np.outer(
            self.normal_plane, np.ones((1, normal_force_plane.shape[1]))
        )
        axial_direction = rod.tangents
        element_v = 0.5 * (rod.velocity[..., :-1] + rod.velocity[..., 1:])
        # first apply axial kinetic friction
        # dot product
        axial_velocity = np.einsum("ijk,ijk->jk", element_v, axial_direction)
        axial_velocity_sign = np.sign(axial_velocity)
        # check top for sign convention
        kinetic_mu = 0.5 * (
            self.kinetic_mu_forward * (1 + axial_velocity_sign)
            + self.kinetic_mu_backward * (1 - axial_velocity_sign)
        )
        axial_slip_function = (axial_velocity, self.slip_velocity_tol)
        axial_kinetic_friction_force = -(
            (1.0 - axial_slip_function)
            * kinetic_mu
            * normal_force_plane
            * axial_velocity_sign
            * axial_direction
        )
        rod.external_forces[..., :-1] += 0.5 * axial_kinetic_friction_force
        rod.external_forces[..., 1:] += 0.5 * axial_kinetic_friction_force

        # now rolling kinetic friction
        rolling_direction = _batch_cross(normal_plane_array, axial_direction)
        torque_arm = -rod.radius * normal_plane_array
        rolling_velocity = np.einsum("ijk,ijk->jk", element_v, rolling_direction)
        directors_transpose = np.einsum("jik", rod.directors)
        # v_rot = Q.T @ omega @ Q @ r
        rotation_velocity = _batch_matvec(
            directors_transpose,
            _batch_cross(rod.omega, _batch_matvec(rod.directors, torque_arm)),
        )
        rolling_rotation_velocity = np.einsum(
            "ijk,ijk->jk", rotation_velocity, rolling_direction
        )
        rolling_slip_velocity = rolling_velocity + rolling_rotation_velocity
        rolling_slip_velocity_sign = np.sign(rolling_slip_velocity)
        rolling_slip_function = (rolling_slip_velocity, self.slip_velocity_tol)
        rolling_kinetic_friction_force = -(
            (1.0 - rolling_slip_function)
            * self.kinetic_mu_sideways
            * normal_force_plane
            * rolling_slip_velocity_sign
            * rolling_direction
        )
        rod.external_forces[..., :-1] += 0.5 * rolling_kinetic_friction_force
        rod.external_forces[..., 1:] += 0.5 * rolling_kinetic_friction_force
        # torque = Q @ r @ Fr
        rod.external_torques += _batch_matvec(
            rod.directors, _batch_cross(torque_arm, rolling_kinetic_friction_force)
        )

        # now axial static friction
        nodal_total_forces = rod.internal_forces + rod.external_forces
        total_forces = 0.5 * (
            nodal_total_forces[..., :-1] + nodal_total_forces[..., 1:]
        )
        projection = np.einsum("ijk,ijk->jk", total_forces, axial_direction)
        projection_sign = np.sign(projection)
        # check top for sign convention
        static_mu = 0.5 * (
            self.static_mu_forward * (1 + projection_sign)
            + self.static_mu_backward * (1 - projection_sign)
        )
        max_friction_force = axial_slip_function * static_mu * normal_force_plane
        # friction = min(mu N, pushing force)
        axial_static_friction_force = -(
            np.minimum(np.fabs(projection), max_friction_force)
            * projection_sign
            * axial_direction
        )
        rod.external_forces[..., :-1] += 0.5 * axial_static_friction_force
        rod.external_forces[..., 1:] += 0.5 * axial_static_friction_force

        # now rolling static friction
        # there is some normal, tangent and rolling directions inconsitency from Elastica
        total_torques = _batch_matvec(
            directors_transpose, (rod.internal_torques + rod.external_torques)
        )
        # Elastica has opposite defs of tangents in interaction.h and rod.cpp
        tangential_torques = np.einsum("ijk,ijk->jk", total_torques, rod.tangents)
        projection = np.einsum("ijk,ijk->jk", total_forces, rolling_direction)
        noslip_force = -(
            (rod.radius * projection - 2.0 * tangential_torques) / 3.0 / rod.radius
        )
        max_friction_force = (
            rolling_slip_function * self.static_mu_sideways * normal_force_plane
        )
        noslip_force_sign = np.sign(noslip_force)
        rolling_static_friction_force = (
            np.minimum(np.fabs(noslip_force), max_friction_force)
            * noslip_force_sign
            * rolling_direction
        )
        rod.external_forces[..., :-1] += 0.5 * rolling_static_friction_force
        rod.external_forces[..., 1:] += 0.5 * rolling_static_friction_force
        rod.external_torques += _batch_matvec(
            rod.directors, _batch_cross(torque_arm, rolling_static_friction_force)
        )
