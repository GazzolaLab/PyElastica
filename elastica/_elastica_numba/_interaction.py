__doc__ = """ Numba implementation module containing interactions between a rod and its environment."""

import numpy as np
from elastica.utils import MaxDimension
from elastica.external_forces import NoForces


import numba
from numba import njit
from elastica._elastica_numba._linalg import (
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


@njit(cache=True)
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
    """
    Developer Notes
    -----
    Benchmark results, for a blocksize of 100 using timeit
    python version: 18.9 µs ± 2.98 µs per loop
    this version: 1.96 µs ± 58.3 ns per loop
    """
    abs_velocity_slip = _batch_norm(velocity_slip)
    slip_points = np.where(np.fabs(abs_velocity_slip) > velocity_threshold)
    slip_function = np.ones((velocity_slip.shape[1]))
    slip_function[slip_points] = np.fabs(
        1.0 - np.minimum(1.0, abs_velocity_slip[slip_points] / velocity_threshold - 1.0)
    )
    return slip_function


@njit(cache=True)
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
    """
    Developer Notes
    -----
    Benchmark results, for a blocksize of 100 using timeit
    Python version: 18.1 µs ± 1.03 µs per loop
    This version: 1.55 µs ± 13.4 ns per loop
    """
    blocksize = input.shape[1] - 1  # nelem
    output = np.zeros((3, blocksize))
    for i in range(3):
        for k in range(0, blocksize):
            output[i, k] += 0.5 * (input[i, k] + input[i, k + 1])

            # Put extra care for the first and last element
    output[..., 0] += 0.5 * input[..., 0]
    output[..., -1] += 0.5 * input[..., -1]

    return output


@njit(cache=True)
def elements_to_nodes_inplace(vector_in_element_frame, vector_in_node_frame):
    """
    Updating nodal forces using the forces computed on elements
    Parameters
    ----------
    vector_in_element_frame
    vector_in_node_frame

    Returns
    -------
    Notes
    -----
    Benchmark results, for a blocksize of 100 using timeit
    Python version: 23.1 µs ± 7.57 µs per loop
    This version: 696 ns ± 10.2 ns per loop
    """
    for i in range(3):
        for k in range(vector_in_element_frame.shape[1]):
            vector_in_node_frame[i, k] += 0.5 * vector_in_element_frame[i, k]
            vector_in_node_frame[i, k + 1] += 0.5 * vector_in_element_frame[i, k]


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
        return apply_normal_force_numba(
            self.plane_origin,
            self.plane_normal,
            self.surface_tol,
            self.k,
            self.nu,
            system.radius,
            system.position_collection,
            system.velocity_collection,
            system.internal_forces,
            system.external_forces,
        )


@njit(cache=True)
def apply_normal_force_numba(
    plane_origin,
    plane_normal,
    surface_tol,
    k,
    nu,
    radius,
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
    element_total_forces = nodes_to_elements(nodal_total_forces)

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
    element_position = node_to_element_pos_or_vel(position_collection)
    distance_from_plane = _batch_product_i_ik_to_k(
        plane_normal, (element_position - plane_origin)
    )
    plane_penetration = np.minimum(distance_from_plane - radius, 0.0)
    elastic_force = -k * _batch_product_i_k_to_ik(plane_normal, plane_penetration)

    # Damping force response due to velocity towards the plane
    element_velocity = node_to_element_pos_or_vel(velocity_collection)
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
    elements_to_nodes_inplace(plane_response_force_total, external_forces)

    return (_batch_norm(plane_response_force), no_contact_point_idx)


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
        """
        Call numba implementation to apply friction forces
        Parameters
        ----------
        system
        time

        Returns
        -------

        """
        anisotropic_friction(
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
            system.radius,
            system.tangents,
            system.position_collection,
            system.director_collection,
            system.velocity_collection,
            system.omega_collection,
            system.internal_forces,
            system.external_forces,
            system.internal_torques,
            system.external_torques,
        )


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
    static_mu_forward,
    static_mu_backward,
    static_mu_sideways,
    radius,
    tangents,
    position_collection,
    director_collection,
    velocity_collection,
    omega_collection,
    internal_forces,
    external_forces,
    internal_torques,
    external_torques,
):
    plane_response_force_mag, no_contact_point_idx = apply_normal_force_numba(
        plane_origin,
        plane_normal,
        surface_tol,
        k,
        nu,
        radius,
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
    element_velocity = node_to_element_pos_or_vel(velocity_collection)
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
    # If rod element does not have any contact with plane, plane cannot apply friction
    # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
    kinetic_friction_force_along_axial_direction[..., no_contact_point_idx] = 0.0
    elements_to_nodes_inplace(
        kinetic_friction_force_along_axial_direction, external_forces
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
    slip_velocity_sign_along_rolling_direction = np.sign(
        slip_velocity_mag_along_rolling_direction
    )
    slip_function_along_rolling_direction = find_slipping_elements(
        slip_velocity_along_rolling_direction, slip_velocity_tol
    )
    kinetic_friction_force_along_rolling_direction = -(
        (1.0 - slip_function_along_rolling_direction)
        * kinetic_mu_sideways
        * plane_response_force_mag
        * slip_velocity_sign_along_rolling_direction
        * rolling_direction
    )
    # If rod element does not have any contact with plane, plane cannot apply friction
    # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
    kinetic_friction_force_along_rolling_direction[..., no_contact_point_idx] = 0.0
    elements_to_nodes_inplace(
        kinetic_friction_force_along_rolling_direction, external_forces
    )
    # torque = Q @ r @ Fr
    external_torques += _batch_matvec(
        director_collection,
        _batch_cross(torque_arm, kinetic_friction_force_along_rolling_direction),
    )

    # now axial static friction
    nodal_total_forces = _batch_vector_sum(internal_forces, external_forces)
    element_total_forces = nodes_to_elements(nodal_total_forces)
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
    elements_to_nodes_inplace(
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
    elements_to_nodes_inplace(
        static_friction_force_along_rolling_direction, external_forces
    )
    external_torques += _batch_matvec(
        director_collection,
        _batch_cross(torque_arm, static_friction_force_along_rolling_direction),
    )


# Slender body module
@njit(cache=True)
def sum_over_elements(input):
    """
    This function sums all elements of the input array.
    Using a Numba njit decorator shows better performance
    compared to python sum(), .sum() and np.sum()

    Parameters
    ----------
    input: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.

    Returns
    -------
    float

    """
    """
    Developer Note
    -----
    Faster than sum(), .sum() and np.sum()

    For blocksize = 200

    sum(): 36.9 µs ± 3.99 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    .sum(): 3.17 µs ± 90.1 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

    np.sum(): 5.17 µs ± 364 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

    This version: 513 ns ± 24.6 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    """

    output = 0.0
    for i in range(input.shape[0]):
        output += input[i]

    return output


@njit(cache=True)
def node_to_element_pos_or_vel(vector_in_node_frame):
    """
    This function computes the velocity of the elements.
    Here we define a separate function because benchmark results
    showed that using Numba, we get more than 3 times faster calculation.

    Parameters
    ----------
    vector_in_node_frame: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.

    Returns
    -------
    vector_in_element_frame: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
    """
    """
    Developer Notes
    -----
    Benchmark results, for a blocksize of 100,

    Python version: 3.5 µs ± 149 ns per loop

    This version: 729 ns ± 14.3 ns per loop

    """
    n_elem = vector_in_node_frame.shape[1] - 1
    vector_in_element_frame = np.empty((3, n_elem))
    for k in range(n_elem):
        vector_in_element_frame[0, k] = 0.5 * (
            vector_in_node_frame[0, k + 1] + vector_in_node_frame[0, k]
        )
        vector_in_element_frame[1, k] = 0.5 * (
            vector_in_node_frame[1, k + 1] + vector_in_node_frame[1, k]
        )
        vector_in_element_frame[2, k] = 0.5 * (
            vector_in_node_frame[2, k + 1] + vector_in_node_frame[2, k]
        )

    return vector_in_element_frame


@njit(cache=True)
def slender_body_forces(
    tangents, velocity_collection, dynamic_viscosity, lengths, radius
):
    r"""
    This function computes hydrodynamic forces on a body using slender body theory.
    The below implementation is from Eq. 4.13 in Gazzola et al. RSoS. (2018).

    .. math::
        F_{h}=\frac{-4\pi\mu}{\ln{(L/r)}}\left(\mathbf{I}-\frac{1}{2}\mathbf{t}^{\textrm{T}}\mathbf{t}\right)\mathbf{v}



    Parameters
    ----------
    tangents: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
        Rod-like element tangent directions.
    velocity_collection: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
        Rod-like object velocity collection.
    dynamic_viscosity: float
        Dynamic viscosity of the fluid.
    length: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod-like object element lengths.
    radius: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod-like object element radius.

    Returns
    -------
    stokes_force: numpy.ndarray
       2D (dim, blocksize) array containing data with 'float' type.
    """

    """
    Developer Note
    ----
    Faster than numpy einsum implementation for blocksize 100

    numpy: 39.5 µs ± 6.78 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    this version: 3.91 µs ± 310 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    """

    f = np.empty((tangents.shape[0], tangents.shape[1]))
    total_length = sum_over_elements(lengths)
    element_velocity = node_to_element_pos_or_vel(velocity_collection)

    for k in range(tangents.shape[1]):
        # compute the entries of t`t. a[#][#] are the the
        # entries of t`t matrix
        a11 = tangents[0, k] * tangents[0, k]
        a12 = tangents[0, k] * tangents[1, k]
        a13 = tangents[0, k] * tangents[2, k]

        a21 = tangents[1, k] * tangents[0, k]
        a22 = tangents[1, k] * tangents[1, k]
        a23 = tangents[1, k] * tangents[2, k]

        a31 = tangents[2, k] * tangents[0, k]
        a32 = tangents[2, k] * tangents[1, k]
        a33 = tangents[2, k] * tangents[2, k]

        # factor = - 4*pi*mu/ln(L/r)
        factor = (
            -4.0
            * np.pi
            * dynamic_viscosity
            / np.log(total_length / radius[k])
            * lengths[k]
        )

        # Fh = factor * ((I - 0.5 * a) * v)
        f[0, k] = factor * (
            (1.0 - 0.5 * a11) * element_velocity[0, k]
            + (0.0 - 0.5 * a12) * element_velocity[1, k]
            + (0.0 - 0.5 * a13) * element_velocity[2, k]
        )
        f[1, k] = factor * (
            (0.0 - 0.5 * a21) * element_velocity[0, k]
            + (1.0 - 0.5 * a22) * element_velocity[1, k]
            + (0.0 - 0.5 * a23) * element_velocity[2, k]
        )
        f[2, k] = factor * (
            (0.0 - 0.5 * a31) * element_velocity[0, k]
            + (0.0 - 0.5 * a32) * element_velocity[1, k]
            + (1.0 - 0.5 * a33) * element_velocity[2, k]
        )

    return f


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
        """
        This function applies hydrodynamic forces on body
        using the slender body theory given in
        Eq. 4.13 Gazzola et. al. RSoS 2018 paper

        Parameters
        ----------
        system

        Returns
        -------

        """

        stokes_force = slender_body_forces(
            system.tangents,
            system.velocity_collection,
            self.dynamic_viscosity,
            system.lengths,
            system.radius,
        )
        elements_to_nodes_inplace(stokes_force, system.external_forces)


# TODO: Test cases needed for the rigid body interaction

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
#         return apply_normal_force_numba_rigid_body(
#             self.plane_origin,
#             self.plane_normal,
#             self.surface_tol,
#             self.k,
#             self.nu,
#             system.length,
#             system.position_collection,
#             system.velocity_collection,
#             system.internal_forces,
#             system.external_forces,
#         )
#
#
# @njit(cache=True)
# def apply_normal_force_numba_rigid_body(
#     plane_origin,
#     plane_normal,
#     surface_tol,
#     k,
#     nu,
#     length,
#     position_collection,
#     velocity_collection,
#     internal_forces,
#     external_forces,
# ):
#
#     # Compute plane response force
#     # total_forces = system.internal_forces + system.external_forces
#     total_forces = _batch_vector_sum(internal_forces, external_forces)
#     force_component_along_normal_direction = _batch_product_i_ik_to_k(
#         plane_normal, total_forces
#     )
#     forces_along_normal_direction = _batch_product_i_k_to_ik(
#         plane_normal, force_component_along_normal_direction
#     )
#     # If the total force component along the plane normal direction is greater than zero that means,
#     # total force is pushing rod away from the plane not towards the plane. Thus, response force
#     # applied by the surface has to be zero.
#     forces_along_normal_direction[
#         ..., np.where(force_component_along_normal_direction > 0)[0]
#     ] = 0.0
#     # Compute response force on the element. Plane response force
#     # has to be away from the surface and towards the element. Thus
#     # multiply forces along normal direction with negative sign.
#     plane_response_force = -forces_along_normal_direction
#
#     # Elastic force response due to penetration
#     element_position = position_collection
#     distance_from_plane = _batch_product_i_ik_to_k(
#         plane_normal, (element_position - plane_origin)
#     )
#     plane_penetration = np.minimum(distance_from_plane - length / 2, 0.0)
#     elastic_force = -k * _batch_product_i_k_to_ik(plane_normal, plane_penetration)
#
#     # Damping force response due to velocity towards the plane
#     element_velocity = velocity_collection
#     normal_component_of_element_velocity = _batch_product_i_ik_to_k(
#         plane_normal, element_velocity
#     )
#     damping_force = -nu * _batch_product_i_k_to_ik(
#         plane_normal, normal_component_of_element_velocity
#     )
#
#     # Compute total plane response force
#     plane_response_force_total = plane_response_force + elastic_force + damping_force
#
#     # Check if the rigid body is in contact with plane.
#     no_contact_point_idx = np.where((distance_from_plane - length / 2) > surface_tol)[0]
#     # If rod element does not have any contact with plane, plane cannot apply response
#     # force on the element. Thus lets set plane response force to 0.0 for the no contact points.
#     plane_response_force[..., no_contact_point_idx] = 0.0
#     plane_response_force_total[..., no_contact_point_idx] = 0.0
#
#     # Update the external forces
#     external_forces += plane_response_force_total
#
#     return (_batch_norm(plane_response_force), no_contact_point_idx)
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
#         anisotropic_firction_numba_rigid_body(
#             self.plane_origin,
#             self.plane_normal,
#             self.surface_tol,
#             self.slip_velocity_tol,
#             self.k,
#             self.nu,
#             self.kinetic_mu_forward,
#             self.kinetic_mu_backward,
#             self.kinetic_mu_sideways,
#             self.static_mu_forward,
#             self.static_mu_backward,
#             self.static_mu_sideways,
#             system.length,
#             system.normal,
#             system.binormal,
#             system.position_collection,
#             system.director_collection,
#             system.velocity_collection,
#             system.omega_collection,
#             system.internal_forces,
#             system.external_forces,
#             system.internal_torques,
#             system.external_torques,
#         )
#
#
# @njit(cache=True)
# def anisotropic_firction_numba_rigid_body(
#     plane_origin,
#     plane_normal,
#     surface_tol,
#     slip_velocity_tol,
#     k,
#     nu,
#     kinetic_mu_forward,
#     kinetic_mu_backward,
#     kinetic_mu_sideways,
#     static_mu_forward,
#     static_mu_backward,
#     static_mu_sideways,
#     length,
#     rigid_body_normal,
#     rigid_body_binormal,
#     position_collection,
#     director_collection,
#     velocity_collection,
#     omega_collection,
#     internal_forces,
#     external_forces,
#     internal_torques,
#     external_torques,
# ):
#     # calculate axial and rolling directions
#     # plane_response_force_mag, no_contact_point_idx = self.apply_normal_force(system)
#     (
#         plane_response_force_mag,
#         no_contact_point_idx,
#     ) = apply_normal_force_numba_rigid_body(
#         plane_origin,
#         plane_normal,
#         surface_tol,
#         k,
#         nu,
#         length,
#         position_collection,
#         velocity_collection,
#         internal_forces,
#         external_forces,
#     )
#     # normal_plane_collection = np.repeat(
#     #     self.plane_normal.reshape(3, 1), plane_response_force_mag.shape[0], axis=1
#     # )
#     # First compute component of rod tangent in plane. Because friction forces acts in plane not out of plane. Thus
#     # axial direction has to be in plane, it cannot be out of plane. We are projecting rod element tangent vector in
#     # to the plane. So friction forces can only be in plane forces and not out of plane.
#     # tangent_along_normal_direction = np.einsum(
#     #     "ij, ij->j", system.tangents, normal_plane_collection
#     # )
#     # tangent_perpendicular_to_normal_direction = system.tangents - np.einsum(
#     #     "j, ij->ij", tangent_along_normal_direction, normal_plane_collection
#     # )
#     # tangent_perpendicular_to_normal_direction_mag = np.einsum(
#     #     "ij, ij->j",
#     #     tangent_perpendicular_to_normal_direction,
#     #     tangent_perpendicular_to_normal_direction,
#     # )
#     # # Normalize tangent_perpendicular_to_normal_direction. This is axial direction for plane. Here we are adding
#     # # small tolerance (1e-10) for normalization, in order to prevent division by 0.
#     # axial_direction = np.einsum(
#     #     "ij, j-> ij",
#     #     tangent_perpendicular_to_normal_direction,
#     #     1 / (tangent_perpendicular_to_normal_direction_mag + 1e-14),
#     # )
#     # FIXME: In future change the below part we should be able to compute the normal
#     axial_direction = rigid_body_normal  # system.tangents
#     element_velocity = velocity_collection
#
#     # first apply axial kinetic friction
#     # velocity_mag_along_axial_direction = np.einsum(
#     #     "ij,ij->j", element_velocity, axial_direction
#     # )
#     # velocity_along_axial_direction = np.einsum(
#     #     "j, ij->ij", velocity_mag_along_axial_direction, axial_direction
#     # )
#     velocity_mag_along_axial_direction = _batch_dot(element_velocity, axial_direction)
#     velocity_along_axial_direction = _batch_product_k_ik_to_ik(
#         velocity_mag_along_axial_direction, axial_direction
#     )
#     # Friction forces depends on the direction of velocity, in other words sign
#     # of the velocity vector.
#     velocity_sign_along_axial_direction = np.sign(velocity_mag_along_axial_direction)
#     # Check top for sign convention
#     kinetic_mu = 0.5 * (
#         kinetic_mu_forward * (1 + velocity_sign_along_axial_direction)
#         + kinetic_mu_backward * (1 - velocity_sign_along_axial_direction)
#     )
#     # Call slip function to check if elements slipping or not
#     slip_function_along_axial_direction = find_slipping_elements(
#         velocity_along_axial_direction, slip_velocity_tol
#     )
#     kinetic_friction_force_along_axial_direction = -(
#         (1.0 - slip_function_along_axial_direction)
#         * kinetic_mu
#         * plane_response_force_mag
#         * velocity_sign_along_axial_direction
#         * axial_direction
#     )
#
#     binormal_direction = rigid_body_binormal
#     velocity_mag_along_binormal_direction = _batch_dot(
#         element_velocity, binormal_direction
#     )
#     velocity_along_binormal_direction = _batch_product_k_ik_to_ik(
#         velocity_mag_along_binormal_direction, binormal_direction
#     )
#     # Friction forces depends on the direction of velocity, in other words sign
#     # of the velocity vector.
#     velocity_sign_along_binormal_direction = np.sign(
#         velocity_mag_along_binormal_direction
#     )
#     # Check top for sign convention
#     kinetic_mu = 0.5 * (
#         kinetic_mu_forward * (1 + velocity_sign_along_binormal_direction)
#         + kinetic_mu_backward * (1 - velocity_sign_along_binormal_direction)
#     )
#     # Call slip function to check if elements slipping or not
#     slip_function_along_binormal_direction = find_slipping_elements(
#         velocity_along_binormal_direction, slip_velocity_tol
#     )
#     kinetic_friction_force_along_binormal_direction = -(
#         (1.0 - slip_function_along_binormal_direction)
#         * kinetic_mu
#         * plane_response_force_mag
#         * velocity_mag_along_binormal_direction
#         * binormal_direction
#     )
#
#     # If rod element does not have any contact with plane, plane cannot apply friction
#     # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
#     kinetic_friction_force_along_axial_direction[..., no_contact_point_idx] = 0.0
#     kinetic_friction_force_along_binormal_direction[..., no_contact_point_idx] = 0.0
#     external_forces += (
#         kinetic_friction_force_along_axial_direction
#         + kinetic_friction_force_along_binormal_direction
#     )
#
#     # # Now rolling kinetic friction
#     # rolling_direction = _batch_cross(axial_direction, normal_plane_collection)
#     # torque_arm = -system.radius * normal_plane_collection
#     # velocity_along_rolling_direction = np.einsum(
#     #     "ij ,ij ->j ", element_velocity, rolling_direction
#     # )
#     # directors_transpose = np.einsum("ijk -> jik", system.director_collection)
#     # # w_rot = Q.T @ omega @ Q @ r
#     # rotation_velocity = _batch_matvec(
#     #     directors_transpose,
#     #     _batch_cross(
#     #         system.omega_collection,
#     #         _batch_matvec(system.director_collection, torque_arm),
#     #     ),
#     # )
#     # rotation_velocity_along_rolling_direction = np.einsum(
#     #     "ij,ij->j", rotation_velocity, rolling_direction
#     # )
#     # slip_velocity_mag_along_rolling_direction = (
#     #     velocity_along_rolling_direction + rotation_velocity_along_rolling_direction
#     # )
#     # slip_velocity_along_rolling_direction = np.einsum(
#     #     "j, ij->ij", slip_velocity_mag_along_rolling_direction, rolling_direction
#     # )
#     # slip_velocity_sign_along_rolling_direction = np.sign(
#     #     slip_velocity_mag_along_rolling_direction
#     # )
#     # slip_function_along_rolling_direction = find_slipping_elements(
#     #     slip_velocity_along_rolling_direction, self.slip_velocity_tol
#     # )
#     # kinetic_friction_force_along_rolling_direction = -(
#     #     (1.0 - slip_function_along_rolling_direction)
#     #     * self.kinetic_mu_sideways
#     #     * plane_response_force_mag
#     #     * slip_velocity_sign_along_rolling_direction
#     #     * rolling_direction
#     # )
#     # # If rod element does not have any contact with plane, plane cannot apply friction
#     # # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
#     # kinetic_friction_force_along_rolling_direction[..., no_contact_point_idx] = 0.0
#     # system.external_forces += kinetic_friction_force_along_rolling_direction
#     #
#     # # torque = Q @ r @ Fr
#     # system.external_torques += _batch_matvec(
#     #     system.director_collection,
#     #     _batch_cross(torque_arm, kinetic_friction_force_along_rolling_direction),
#     # )
#
#     # # now axial static friction
#     # element_total_forces = _batch_vector_sum(internal_forces, external_forces)
#     # # force_component_along_axial_direction = np.einsum(
#     # #     "ij,ij->j", element_total_forces, axial_direction
#     # # )
#     # force_component_along_axial_direction = _batch_dot(
#     #     element_total_forces, axial_direction
#     # )
#     # force_component_sign_along_axial_direction = np.sign(
#     #     force_component_along_axial_direction
#     # )
#     # # check top for sign convention
#     # static_mu = 0.5 * (
#     #     static_mu_forward * (1 + force_component_sign_along_axial_direction)
#     #     + static_mu_backward * (1 - force_component_sign_along_axial_direction)
#     # )
#     # max_friction_force = (
#     #     slip_function_along_axial_direction * static_mu * plane_response_force_mag
#     # )
#     # # friction = min(mu N, pushing force)
#     # static_friction_force_along_axial_direction = -(
#     #     np.minimum(
#     #         np.fabs(force_component_along_axial_direction), max_friction_force
#     #     )
#     #     * force_component_sign_along_axial_direction
#     #     * axial_direction
#     # )
#     # # If rod element does not have any contact with plane, plane cannot apply friction
#     # # force on the element. Thus lets set static friction force to 0.0 for the no contact points.
#     # static_friction_force_along_axial_direction[..., no_contact_point_idx] = 0.0
#     # external_forces += static_friction_force_along_axial_direction
#
#     # # now rolling static friction
#     # # there is some normal, tangent and rolling directions inconsitency from Elastica
#     # total_torques = _batch_matvec(
#     #     directors_transpose, (system.internal_torques + system.external_torques)
#     # )
#     # # Elastica has opposite defs of tangents in interaction.h and rod.cpp
#     # total_torques_along_axial_direction = np.einsum(
#     #     "ij,ij->j", total_torques, axial_direction
#     # )
#     # force_component_along_rolling_direction = np.einsum(
#     #     "ij,ij->j", element_total_forces, rolling_direction
#     # )
#     # noslip_force = -(
#     #     (
#     #         system.radius * force_component_along_rolling_direction
#     #         - 2.0 * total_torques_along_axial_direction
#     #     )
#     #     / 3.0
#     #     / system.radius
#     # )
#     # max_friction_force = (
#     #     slip_function_along_rolling_direction
#     #     * self.static_mu_sideways
#     #     * plane_response_force_mag
#     # )
#     # noslip_force_sign = np.sign(noslip_force)
#     # static_friction_force_along_rolling_direction = (
#     #     np.minimum(np.fabs(noslip_force), max_friction_force)
#     #     * noslip_force_sign
#     #     * rolling_direction
#     # )
#     # # If rod element does not have any contact with plane, plane cannot apply friction
#     # # force on the element. Thus lets set plane static friction force to 0.0 for the no contact points.
#     # static_friction_force_along_rolling_direction[..., no_contact_point_idx] = 0.0
#     # system.external_forces += static_friction_force_along_rolling_direction
#
#     # system.external_torques += _batch_matvec(
#     #     system.director_collection,
#     #     _batch_cross(torque_arm, static_friction_force_along_rolling_direction),
#     # )
