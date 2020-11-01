__doc__ = """ Numpy implementation module containing interactions between a rod and its environment. """

import numpy as np
from elastica.utils import MaxDimension
from elastica.external_forces import NoForces
from elastica._elastica_numpy._linalg import _batch_matvec, _batch_matmul, _batch_cross


def find_slipping_elements(velocity_slip, velocity_threshold):
    """
    This function takes the velocity of elements and checks if they are larger than the threshold velocity.
    If the velocity of elements is larger than threshold velocity, that means those elements are slipping.
    In other words, kinetic friction will be acting on those elements, not static friction.
    This function outputs an array called slip function, this array has a size of the number of elements.
    If the velocity of the element is smaller than the threshold velocity slip function value for that element is 1,
    which means static friction is acting on that element. If the velocity of the element is larger than
    the threshold velocity slip function value for that element is between 0 and 1, which means kinetic friction is acting
    on that element.

    Parameters
    ----------
    velocity_slip : numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
        Rod-like object element velocity.
    velocity_threshold : float
        Threshold velocity to determine slip.

    Returns
    -------
    slip_function : numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
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
    """
    This function converts the rod-like object dofs on nodes to
    dofs on elements. For example, node velocity is converted to
    element velocity.

    Parameters
    ----------
    input: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.

    Returns
    -------
    output: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
    """
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
    """
    The interaction plane class computes the plane reaction
    force on a rod-like object.  For more details regarding the contact module refer to
    Eqn 4.8 of Gazzola et al. RSoS (2018).

        Attributes
        ----------
        k: float
            Stiffness coefficient between the plane and the rod-like object.
        nu: float
            Dissipation coefficient between the plane and the rod-like object.
        plane_origin: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Origin of the plane.
        plane_normal: numpy.ndarray
           2D (dim, 1) array containing data with 'float' type.
           The normal vector of the plane.
        surface_tol: float
            Penetration tolerance between the plane and the rod-like object.

    """

    def __init__(self, k, nu, plane_origin, plane_normal):
        """

        Parameters
        ----------
        k: float
            Stiffness coefficient between the plane and the rod-like object.
        nu: float
            Dissipation coefficient between the plane and the rod-like object.
        plane_origin: numpy.ndarray
           2D (dim, 1) array containing data with 'float' type.
           Origin of the plane.
        plane_normal: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            The normal vector of the plane.
        """
        self.k = k
        self.nu = nu
        self.plane_origin = plane_origin.reshape(3, 1)
        self.plane_normal = plane_normal.reshape(3)
        self.surface_tol = 1e-4

    def apply_normal_force(self, system):
        """
        In the case of contact with the plane, this function computes the plane reaction force on the element.

        Parameters
        ----------
        system: object
            Rod-like object.

        Returns
        -------
        plane_response_force_mag : numpy.ndarray
            1D (blocksize) array containing data with 'float' type.
            Magnitude of plane response force acting on rod-like object.
        no_contact_point_idx : numpy.ndarray
            1D (blocksize) array containing data with 'int' type.
            Index of rod-like object elements that are not in contact with the plane.
        """

        # Compute plane response force
        nodal_total_forces = system.internal_forces + system.external_forces
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

        return (
            np.sqrt(np.einsum("ij, ij->j", plane_response_force, plane_response_force)),
            no_contact_point_idx,
        )


# class for anisotropic frictional plane
# NOTE: friction coefficients are passed as arrays in the order
# mu_forward : mu_backward : mu_sideways
# head is at x[0] and forward means head to tail
# same convention for kinetic and static
# mu named as to which direction it opposes
class AnisotropicFrictionalPlane(NoForces, InteractionPlane):
    """
    This anisotropic friction plane class is for computing
    anisotropic friction forces on rods.
    A detailed explanation of the implemented equations
    can be found in Gazzola et al. RSoS. (2018).

        Attributes
        ----------
        k: float
            Stiffness coefficient between the plane and the rod-like object.
        nu: float
            Dissipation coefficient between the plane and the rod-like object.
        plane_origin: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Origin of the plane.
        plane_normal: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            The normal vector of the plane.
        slip_velocity_tol: float
            Velocity tolerance to determine if the element is slipping or not.
        static_mu_array: numpy.ndarray
            1D (3,) array containing data with 'float' type.
            [forward, backward, sideways] static friction coefficients.
        kinetic_mu_array: numpy.ndarray
            1D (3,) array containing data with 'float' type.
            [forward, backward, sideways] kinetic friction coefficients.
    """

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
        """

        Parameters
        ----------
        k: float
            Stiffness coefficient between the plane and the rod-like object.
        nu: float
            Dissipation coefficient between the plane and the rod-like object.
        plane_origin: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Origin of the plane.
        plane_normal: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            The normal vector of the plane.
        slip_velocity_tol: float
            Velocity tolerance to determine if the element is slipping or not.
        static_mu_array: numpy.ndarray
            1D (3,) array containing data with 'float' type.
            [forward, backward, sideways] static friction coefficients.
        kinetic_mu_array: numpy.ndarray
            1D (3,) array containing data with 'float' type.
            [forward, backward, sideways] kinetic friction coefficients.
        """
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
        plane_response_force_mag, no_contact_point_idx = self.apply_normal_force(system)
        normal_plane_collection = np.repeat(
            self.plane_normal.reshape(3, 1), plane_response_force_mag.shape[0], axis=1
        )
        # First compute component of rod tangent in plane. Because friction forces acts in plane not out of plane. Thus
        # axial direction has to be in plane, it cannot be out of plane. We are projecting rod element tangent vector in
        # to the plane. So friction forces can only be in plane forces and not out of plane.
        tangent_along_normal_direction = np.einsum(
            "ij, ij->j", system.tangents, normal_plane_collection
        )
        tangent_perpendicular_to_normal_direction = system.tangents - np.einsum(
            "j, ij->ij", tangent_along_normal_direction, normal_plane_collection
        )
        tangent_perpendicular_to_normal_direction_mag = np.einsum(
            "ij, ij->j",
            tangent_perpendicular_to_normal_direction,
            tangent_perpendicular_to_normal_direction,
        )
        # Normalize tangent_perpendicular_to_normal_direction. This is axial direction for plane. Here we are adding
        # small tolerance (1e-10) for normalization, in order to prevent division by 0.
        axial_direction = np.einsum(
            "ij, j-> ij",
            tangent_perpendicular_to_normal_direction,
            1 / (tangent_perpendicular_to_normal_direction_mag + 1e-14),
        )
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
        # If rod element does not have any contact with plane, plane cannot apply friction
        # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
        kinetic_friction_force_along_axial_direction[..., no_contact_point_idx] = 0.0
        system.external_forces[..., :-1] += (
            0.5 * kinetic_friction_force_along_axial_direction
        )
        system.external_forces[..., 1:] += (
            0.5 * kinetic_friction_force_along_axial_direction
        )

        # Now rolling kinetic friction
        rolling_direction = _batch_cross(axial_direction, normal_plane_collection)
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
        # If rod element does not have any contact with plane, plane cannot apply friction
        # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
        kinetic_friction_force_along_rolling_direction[..., no_contact_point_idx] = 0.0
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
        nodal_total_forces = system.internal_forces + system.external_forces
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
        # If rod element does not have any contact with plane, plane cannot apply friction
        # force on the element. Thus lets set static friction force to 0.0 for the no contact points.
        static_friction_force_along_axial_direction[..., no_contact_point_idx] = 0.0
        system.external_forces[..., :-1] += (
            0.5 * static_friction_force_along_axial_direction
        )
        system.external_forces[..., 1:] += (
            0.5 * static_friction_force_along_axial_direction
        )

        # now rolling static friction
        # there is some normal, tangent and rolling directions inconsitency from Elastica
        total_torques = _batch_matvec(
            directors_transpose, (system.internal_torques + system.external_torques)
        )
        # Elastica has opposite defs of tangents in interaction.h and rod.cpp
        total_torques_along_axial_direction = np.einsum(
            "ij,ij->j", total_torques, axial_direction
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
        # If rod element does not have any contact with plane, plane cannot apply friction
        # force on the element. Thus lets set plane static friction force to 0.0 for the no contact points.
        static_friction_force_along_rolling_direction[..., no_contact_point_idx] = 0.0
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


# slender body theory
class SlenderBodyTheory(NoForces):
    """
    This slender body theory class is for flow-structure
    interaction problems. This class applies hydrodynamic
    forces on the body using the slender body theory given in
    Eq. 4.13 of Gazzola et al. RSoS (2018).

        Attributes
        ----------
        dynamic_viscosity: float
            Dynamic viscosity of the fluid.

    """

    def __init__(self, dynamic_viscosity):
        """

        Parameters
        ----------
        dynamic_viscosity : float
            Dynamic viscosity of the fluid.
        """
        super(SlenderBodyTheory, self).__init__()
        self.dynamic_viscosity = dynamic_viscosity

    def apply_forces(self, system, time=0.0):
        total_length = system.lengths.sum()

        element_velocity = 0.5 * (
            system.velocity_collection[..., :-1] + system.velocity_collection[..., 1:]
        )
        tangent_tangent_transpose = -0.5 * np.einsum(
            "ik, jk -> ijk", system.tangents, system.tangents
        )
        # Do I-0.5*t*t'
        np.einsum("iij->ij", tangent_tangent_transpose, optimize=False)[...] += 1.0
        factor = (
            -4.0
            * np.pi
            * self.dynamic_viscosity
            / np.log(total_length / system.radius)
            * system.lengths
        )
        stokes_force = factor * np.einsum(
            "ijk, jk -> ik", tangent_tangent_transpose, element_velocity
        )

        system.external_forces[..., :-1] += 0.5 * stokes_force
        system.external_forces[..., 1:] += 0.5 * stokes_force


# TODO: Test cases needed for the rigid body interaction
#
# # base class for interaction
# # only applies normal force no friction
# class InteractionPlaneRigidBody:
#     def __init__(self, k, nu, plane_origin, plane_normal):
#         self.k = k
#         self.nu = nu
#         self.plane_origin = plane_origin.reshape(3, 1)
#         self.plane_normal = plane_normal.reshape(3)
#         self.surface_tol = 1e-4
#
#     def apply_normal_force(self, system):
#         """
#         This function computes the plane force response on the rigid body, in the
#         case of contact. Contact model given in Eqn 4.8 Gazzola et. al. RSoS 2018 paper
#         is used.
#         Parameters
#         ----------
#         system
#
#         Returns
#         -------
#         magnitude of the plane response
#         """
#
#         # Compute plane response force
#         total_forces = system.internal_forces + system.external_forces
#         force_component_along_normal_direction = np.einsum(
#             "i, ij->j", self.plane_normal, total_forces
#         )
#         forces_along_normal_direction = np.einsum(
#             "i, j->ij", self.plane_normal, force_component_along_normal_direction
#         )
#         # If the total force component along the plane normal direction is greater than zero that means,
#         # total force is pushing rod away from the plane not towards the plane. Thus, response force
#         # applied by the surface has to be zero.
#         forces_along_normal_direction[
#             ..., np.where(force_component_along_normal_direction > 0)
#         ] = 0.0
#         # Compute response force on the element. Plane response force
#         # has to be away from the surface and towards the element. Thus
#         # multiply forces along normal direction with negative sign.
#         plane_response_force = -forces_along_normal_direction
#
#         # Elastic force response due to penetration
#         element_position = system.position_collection
#
#         distance_from_plane = np.einsum(
#             "i, ij->j", self.plane_normal, (element_position - self.plane_origin)
#         )
#         plane_penetration = np.minimum(distance_from_plane - system.length, 0.0)
#         elastic_force = -self.k * np.einsum(
#             "i, j->ij", self.plane_normal, plane_penetration
#         )
#
#         # Damping force response due to velocity towards the plane
#         element_velocity = system.velocity_collection
#
#         normal_component_of_element_velocity = np.einsum(
#             "i, ij->j", self.plane_normal, element_velocity
#         )
#         damping_force = -self.nu * np.einsum(
#             "i, j->ij", self.plane_normal, normal_component_of_element_velocity
#         )
#
#         # Compute total plane response force
#         plane_response_force_total = (
#             plane_response_force + elastic_force + damping_force
#         )
#
#         # Check if the rod elements are in contact with plane.
#         base_length = np.linalg.norm(system.length) / 2
#         no_contact_point_idx = np.where(
#             (distance_from_plane - base_length) > self.surface_tol
#         )
#         # If rod element does not have any contact with plane, plane cannot apply response
#         # force on the element. Thus lets set plane response force to 0.0 for the no contact points.
#         plane_response_force[..., no_contact_point_idx] = 0.0
#         plane_response_force_total[..., no_contact_point_idx] = 0.0
#
#         system.external_forces += plane_response_force_total
#
#         return (
#             np.sqrt(np.einsum("ij, ij->j", plane_response_force, plane_response_force)),
#             no_contact_point_idx,
#         )
#
#
#
# class AnisotropicFrictionalPlaneRigidBody(NoForces, InteractionPlaneRigidBody):
#     def __init__(
#         self,
#         k,
#         nu,
#         plane_origin,
#         plane_normal,
#         slip_velocity_tol,
#         static_mu_array,
#         kinetic_mu_array,
#     ):
#         InteractionPlaneRigidBody.__init__(self, k, nu, plane_origin, plane_normal)
#         self.slip_velocity_tol = slip_velocity_tol
#         (
#             self.static_mu_forward,
#             self.static_mu_backward,
#             self.static_mu_sideways,
#         ) = static_mu_array
#         (
#             self.kinetic_mu_forward,
#             self.kinetic_mu_backward,
#             self.kinetic_mu_sideways,
#         ) = kinetic_mu_array
#
#     # kinetic and static friction should separate functions
#     # for now putting them together to figure out common variables
#     def apply_forces(self, system, time=0.0):
#         # calculate axial and rolling directions
#         plane_response_force_mag, no_contact_point_idx = self.apply_normal_force(system)
#         # normal_plane_collection = np.repeat(
#         #     self.plane_normal.reshape(3, 1), plane_response_force_mag.shape[0], axis=1
#         # )
#         # First compute component of rod tangent in plane. Because friction forces acts in plane not out of plane. Thus
#         # axial direction has to be in plane, it cannot be out of plane. We are projecting rod element tangent vector in
#         # to the plane. So friction forces can only be in plane forces and not out of plane.
#         # tangent_along_normal_direction = np.einsum(
#         #     "ij, ij->j", system.tangents, normal_plane_collection
#         # )
#         # tangent_perpendicular_to_normal_direction = system.tangents - np.einsum(
#         #     "j, ij->ij", tangent_along_normal_direction, normal_plane_collection
#         # )
#         # tangent_perpendicular_to_normal_direction_mag = np.einsum(
#         #     "ij, ij->j",
#         #     tangent_perpendicular_to_normal_direction,
#         #     tangent_perpendicular_to_normal_direction,
#         # )
#         # # Normalize tangent_perpendicular_to_normal_direction. This is axial direction for plane. Here we are adding
#         # # small tolerance (1e-10) for normalization, in order to prevent division by 0.
#         # axial_direction = np.einsum(
#         #     "ij, j-> ij",
#         #     tangent_perpendicular_to_normal_direction,
#         #     1 / (tangent_perpendicular_to_normal_direction_mag + 1e-14),
#         # )
#         # FIXME: In future change the below part we should be able to compute the normal
#         axial_direction = system.normal  # system.tangents
#         element_velocity = system.velocity_collection
#
#         # first apply axial kinetic friction
#         velocity_mag_along_axial_direction = np.einsum(
#             "ij,ij->j", element_velocity, axial_direction
#         )
#         velocity_along_axial_direction = np.einsum(
#             "j, ij->ij", velocity_mag_along_axial_direction, axial_direction
#         )
#         # Friction forces depends on the direction of velocity, in other words sign
#         # of the velocity vector.
#         velocity_sign_along_axial_direction = np.sign(
#             velocity_mag_along_axial_direction
#         )
#         # Check top for sign convention
#         kinetic_mu = 0.5 * (
#             self.kinetic_mu_forward * (1 + velocity_sign_along_axial_direction)
#             + self.kinetic_mu_backward * (1 - velocity_sign_along_axial_direction)
#         )
#         # Call slip function to check if elements slipping or not
#         slip_function_along_axial_direction = find_slipping_elements(
#             velocity_along_axial_direction, self.slip_velocity_tol
#         )
#         kinetic_friction_force_along_axial_direction = -(
#             (1.0 - slip_function_along_axial_direction)
#             * kinetic_mu
#             * plane_response_force_mag
#             * velocity_sign_along_axial_direction
#             * axial_direction
#         )
#         # If rod element does not have any contact with plane, plane cannot apply friction
#         # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
#         kinetic_friction_force_along_axial_direction[..., no_contact_point_idx] = 0.0
#         system.external_forces += kinetic_friction_force_along_axial_direction
#
#         # # Now rolling kinetic friction
#         # rolling_direction = _batch_cross(axial_direction, normal_plane_collection)
#         # torque_arm = -system.radius * normal_plane_collection
#         # velocity_along_rolling_direction = np.einsum(
#         #     "ij ,ij ->j ", element_velocity, rolling_direction
#         # )
#         # directors_transpose = np.einsum("ijk -> jik", system.director_collection)
#         # # w_rot = Q.T @ omega @ Q @ r
#         # rotation_velocity = _batch_matvec(
#         #     directors_transpose,
#         #     _batch_cross(
#         #         system.omega_collection,
#         #         _batch_matvec(system.director_collection, torque_arm),
#         #     ),
#         # )
#         # rotation_velocity_along_rolling_direction = np.einsum(
#         #     "ij,ij->j", rotation_velocity, rolling_direction
#         # )
#         # slip_velocity_mag_along_rolling_direction = (
#         #     velocity_along_rolling_direction + rotation_velocity_along_rolling_direction
#         # )
#         # slip_velocity_along_rolling_direction = np.einsum(
#         #     "j, ij->ij", slip_velocity_mag_along_rolling_direction, rolling_direction
#         # )
#         # slip_velocity_sign_along_rolling_direction = np.sign(
#         #     slip_velocity_mag_along_rolling_direction
#         # )
#         # slip_function_along_rolling_direction = find_slipping_elements(
#         #     slip_velocity_along_rolling_direction, self.slip_velocity_tol
#         # )
#         # kinetic_friction_force_along_rolling_direction = -(
#         #     (1.0 - slip_function_along_rolling_direction)
#         #     * self.kinetic_mu_sideways
#         #     * plane_response_force_mag
#         #     * slip_velocity_sign_along_rolling_direction
#         #     * rolling_direction
#         # )
#         # # If rod element does not have any contact with plane, plane cannot apply friction
#         # # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
#         # kinetic_friction_force_along_rolling_direction[..., no_contact_point_idx] = 0.0
#         # system.external_forces += kinetic_friction_force_along_rolling_direction
#         #
#         # # torque = Q @ r @ Fr
#         # system.external_torques += _batch_matvec(
#         #     system.director_collection,
#         #     _batch_cross(torque_arm, kinetic_friction_force_along_rolling_direction),
#         # )
#
#         # now axial static friction
#         element_total_forces = system.internal_forces + system.external_forces
#         force_component_along_axial_direction = np.einsum(
#             "ij,ij->j", element_total_forces, axial_direction
#         )
#         force_component_sign_along_axial_direction = np.sign(
#             force_component_along_axial_direction
#         )
#         # check top for sign convention
#         static_mu = 0.5 * (
#             self.static_mu_forward * (1 + force_component_sign_along_axial_direction)
#             + self.static_mu_backward * (1 - force_component_sign_along_axial_direction)
#         )
#         max_friction_force = (
#             slip_function_along_axial_direction * static_mu * plane_response_force_mag
#         )
#         # friction = min(mu N, pushing force)
#         static_friction_force_along_axial_direction = -(
#             np.minimum(
#                 np.fabs(force_component_along_axial_direction), max_friction_force
#             )
#             * force_component_sign_along_axial_direction
#             * axial_direction
#         )
#         # If rod element does not have any contact with plane, plane cannot apply friction
#         # force on the element. Thus lets set static friction force to 0.0 for the no contact points.
#         static_friction_force_along_axial_direction[..., no_contact_point_idx] = 0.0
#         system.external_forces += static_friction_force_along_axial_direction
#
#         # # now rolling static friction
#         # # there is some normal, tangent and rolling directions inconsitency from Elastica
#         # total_torques = _batch_matvec(
#         #     directors_transpose, (system.internal_torques + system.external_torques)
#         # )
#         # # Elastica has opposite defs of tangents in interaction.h and rod.cpp
#         # total_torques_along_axial_direction = np.einsum(
#         #     "ij,ij->j", total_torques, axial_direction
#         # )
#         # force_component_along_rolling_direction = np.einsum(
#         #     "ij,ij->j", element_total_forces, rolling_direction
#         # )
#         # noslip_force = -(
#         #     (
#         #         system.radius * force_component_along_rolling_direction
#         #         - 2.0 * total_torques_along_axial_direction
#         #     )
#         #     / 3.0
#         #     / system.radius
#         # )
#         # max_friction_force = (
#         #     slip_function_along_rolling_direction
#         #     * self.static_mu_sideways
#         #     * plane_response_force_mag
#         # )
#         # noslip_force_sign = np.sign(noslip_force)
#         # static_friction_force_along_rolling_direction = (
#         #     np.minimum(np.fabs(noslip_force), max_friction_force)
#         #     * noslip_force_sign
#         #     * rolling_direction
#         # )
#         # # If rod element does not have any contact with plane, plane cannot apply friction
#         # # force on the element. Thus lets set plane static friction force to 0.0 for the no contact points.
#         # static_friction_force_along_rolling_direction[..., no_contact_point_idx] = 0.0
#         # system.external_forces += static_friction_force_along_rolling_direction
#
#         # system.external_torques += _batch_matvec(
#         #     system.director_collection,
#         #     _batch_cross(torque_arm, static_friction_force_along_rolling_direction),
#         # )
