_doc__ = """ Slender body module """

import numpy as np

from ._linalg import _batch_matmul, _batch_matvec, _batch_cross
from elastica.external_forces import NoForces
import numba
from numba import njit


@njit
def sum_over_elements(input):
    """
    This function sums all elements of input array,
    using a numba jit decorator shows better performance
    compared to python sum(), .sum() and np.sum()

    Parameters
    ----------
    input

    Returns
    -------

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


@njit
def node_to_element_velocity(node_velocity):
    """
    This function computes to velocity on the elements.
    Here we define a seperate function because benchmark results
    showed that using numba, we get almost 3 times faster calculation

    Parameters
    ----------
    node_velocity

    Returns
    -------
    element_velocity

    Note
    ___
    Faster than pure python for blocksize 100
    python: 3.81 µs ± 427 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    this version: 1.11 µs ± 19.3 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

    """
    element_velocity = 0.5 * (node_velocity[..., :-1] + node_velocity[..., 1:])
    return element_velocity


@njit
def slender_body_forces(
    tangents, velocity_collection, dynamic_viscosity, lengths, radius
):
    """
    This function computes hydrodynamic forces on body using slender body theory.

    Parameters
    ----------
    tangents
    velocity_collection
    dynamic_viscosity
    length
    radius

    Returns
    -------
    Faster than numpy einsum implementation for blocksize 100
    numpy: 39.5 µs ± 6.78 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    this version: 3.91 µs ± 310 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

    """

    l = tangents.shape[0]
    m = tangents.shape[1]
    f = np.empty((l, m))
    total_length = sum_over_elements(lengths)
    element_velocity = node_to_element_velocity(velocity_collection)

    for k in range(m):
        a11 = tangents[0, k] * tangents[0, k]
        a12 = tangents[0, k] * tangents[1, k]
        a13 = tangents[0, k] * tangents[2, k]

        a21 = tangents[1, k] * tangents[0, k]
        a22 = tangents[1, k] * tangents[1, k]
        a23 = tangents[1, k] * tangents[2, k]

        a31 = tangents[2, k] * tangents[0, k]
        a32 = tangents[2, k] * tangents[1, k]
        a33 = tangents[2, k] * tangents[2, k]

        factor = (
            -4.0
            * np.pi
            * dynamic_viscosity
            / np.log(total_length / radius[k])
            * lengths[k]
        )

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
    def __init__(self, dynamic_viscosity):
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

        system.external_forces[..., :-1] += 0.5 * stokes_force
        system.external_forces[..., 1:] += 0.5 * stokes_force
