__doc__ = """ Numba implementation module containing interactions between a rod and its environment."""


import numpy as np
from elastica.external_forces import NoForces
from numba import njit
from elastica.contact_utils import (
    _elements_to_nodes_inplace,
    _node_to_element_velocity,
)
from elastica._contact_functions import (
    _calculate_contact_forces_rod_plane,
    _calculate_contact_forces_rod_plane_with_anisotropic_friction,
    _calculate_contact_forces_cylinder_plane,
)

from numpy.typing import NDArray

from elastica.typing import SystemType, RodType, RigidBodyType


# base class for interaction
# only applies normal force no friction
class InteractionPlane(NoForces):
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

    def __init__(
        self,
        k: float,
        nu: float,
        plane_origin: NDArray[np.float64],
        plane_normal: NDArray[np.float64],
    ) -> None:
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
        self.k = np.float64(k)
        self.nu = np.float64(nu)
        self.surface_tol = np.float64(1e-4)
        self.plane_origin = plane_origin.reshape(3, 1)
        self.plane_normal = plane_normal.reshape(3)

    def apply_forces(self, system: RodType, time: np.float64 = np.float64(0.0)) -> None:
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
        return _calculate_contact_forces_rod_plane(
            self.plane_origin,
            self.plane_normal,
            self.surface_tol,
            self.k,
            self.nu,
            system.radius,
            system.mass,
            system.position_collection,
            system.velocity_collection,
            system.internal_forces,
            system.external_forces,
        )


# class for anisotropic frictional plane
# NOTE: friction coefficients are passed as arrays in the order
# mu_forward : mu_backward : mu_sideways
# head is at x[0] and forward means head to tail
# same convention for kinetic and static
# mu named as to which direction it opposes
class AnisotropicFrictionalPlane(InteractionPlane):
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
        k: float,
        nu: float,
        plane_origin: NDArray[np.float64],
        plane_normal: NDArray[np.float64],
        slip_velocity_tol: float,
        static_mu_array: NDArray[np.float64],
        kinetic_mu_array: NDArray[np.float64],
    ) -> None:
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
        self.slip_velocity_tol = np.float64(slip_velocity_tol)
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
    def apply_forces(
        self, system: "RodType | RigidBodyType", time: np.float64 = np.float64(0.0)
    ) -> None:
        """
        Call numba implementation to apply friction forces
        Parameters
        ----------
        system : RodType | RigidBodyType
        time

        """
        _calculate_contact_forces_rod_plane_with_anisotropic_friction(
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
            system.mass,
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


# Slender body module
@njit(cache=True)  # type: ignore
def sum_over_elements(input: NDArray[np.float64]) -> np.float64:
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

    output: np.float64 = np.float64(0.0)
    for i in range(input.shape[0]):
        output += input[i]

    return output


@njit(cache=True)  # type: ignore
def slender_body_forces(
    tangents: NDArray[np.float64],
    velocity_collection: NDArray[np.float64],
    dynamic_viscosity: np.float64,
    lengths: NDArray[np.float64],
    radius: NDArray[np.float64],
    mass: NDArray[np.float64],
) -> NDArray[np.float64]:
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
    mass: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod-like object node mass.

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
    element_velocity = _node_to_element_velocity(
        mass=mass, node_velocity_collection=velocity_collection
    )

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

    def __init__(self, dynamic_viscosity: float) -> None:
        """

        Parameters
        ----------
        dynamic_viscosity : float
            Dynamic viscosity of the fluid.
        """
        super(SlenderBodyTheory, self).__init__()
        self.dynamic_viscosity = np.float64(dynamic_viscosity)

    def apply_forces(self, system: RodType, time: np.float64 = np.float64(0.0)) -> None:
        """
        This function applies hydrodynamic forces on body
        using the slender body theory given in
        Eq. 4.13 Gazzola et. al. RSoS 2018 paper

        Parameters
        ----------
        system

        """

        stokes_force = slender_body_forces(
            system.tangents,
            system.velocity_collection,
            self.dynamic_viscosity,
            system.lengths,
            system.radius,
            system.mass,
        )
        _elements_to_nodes_inplace(stokes_force, system.external_forces)


# base class for interaction
# only applies normal force no friction
class InteractionPlaneRigidBody(NoForces):
    def __init__(
        self,
        k: float,
        nu: float,
        plane_origin: NDArray[np.float64],
        plane_normal: NDArray[np.float64],
    ) -> None:
        self.k = np.float64(k)
        self.nu = np.float64(nu)
        self.surface_tol = np.float64(1e-4)
        self.plane_origin = plane_origin.reshape(3, 1)
        self.plane_normal = plane_normal.reshape(3)

    def apply_forces(
        self, system: RigidBodyType, time: np.float64 = np.float64(0.0)
    ) -> None:
        """
        This function computes the plane force response on the rigid body, in the
        case of contact. Contact model given in Eqn 4.8 Gazzola et. al. RSoS 2018 paper
        is used.
        Parameters
        ----------
        system

        Returns
        -------
        magnitude of the plane response
        """
        _calculate_contact_forces_cylinder_plane(
            self.plane_origin,
            self.plane_normal,
            self.surface_tol,
            self.k,
            self.nu,
            system.length,
            system.position_collection,
            system.velocity_collection,
            system.external_forces,
        )
