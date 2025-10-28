__doc__ = """ Numba implementation module containing interactions between a rod and its environment."""


import numpy as np
from elastica.external_forces import NoForces
from numba import njit
from elastica.contact_utils import (
    _elements_to_nodes_inplace,
    _node_to_element_velocity,
)
from elastica._contact_functions import (
    _calculate_contact_forces_cylinder_plane,
)

from numpy.typing import NDArray

from elastica.typing import SystemType, RodType, RigidBodyType


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
