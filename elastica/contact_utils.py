__doc__ = """ Helper functions for contact force calculation """

from math import sqrt
import numba
import numpy as np
from numpy.typing import NDArray
from elastica._linalg import (
    _batch_norm,
)
from typing import Literal, Sequence


@numba.njit(cache=True)  # type: ignore
def _dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    total: float = 0.0
    for i in range(3):
        total += a[i] * b[i]
    return total


@numba.njit(cache=True)  # type: ignore
def _norm(a: Sequence[float]) -> float:
    return sqrt(_dot_product(a, a))


@numba.njit(cache=True)  # type: ignore
def _clip(x: float, low: float, high: float) -> float:
    return max(low, min(x, high))


# Can this be made more efficient than 2 comp, 1 or?
@numba.njit(cache=True)  # type: ignore
def _out_of_bounds(x: float, low: float, high: float) -> bool:
    return bool((x < low) or (x > high))


@numba.njit(cache=True)  # type: ignore
def _find_min_dist(
    x1: NDArray[np.float64],
    e1: NDArray[np.float64],
    x2: NDArray[np.float64],
    e2: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    e1e1 = _dot_product(e1, e1)  # type: ignore
    e1e2 = _dot_product(e1, e2)  # type: ignore
    e2e2 = _dot_product(e2, e2)  # type: ignore

    x1e1 = _dot_product(x1, e1)  # type: ignore
    x1e2 = _dot_product(x1, e2)  # type: ignore
    x2e1 = _dot_product(e1, x2)  # type: ignore
    x2e2 = _dot_product(x2, e2)  # type: ignore

    s = 0.0
    t = 0.0

    parallel = abs(1.0 - e1e2**2 / (e1e1 * e2e2)) < 1e-6
    if parallel:
        # Some are parallel, so do processing
        t = (x2e1 - x1e1) / e1e1  # Comes from taking dot of e1 with a normal
        t = _clip(t, 0.0, 1.0)
        s = (x1e2 + t * e1e2 - x2e2) / e2e2  # Same as before
        s = _clip(s, 0.0, 1.0)
    else:
        # Using the Cauchy-Binet formula on eq(7) in docstring referenc
        s = (e1e1 * (x1e2 - x2e2) + e1e2 * (x2e1 - x1e1)) / (e1e1 * e2e2 - (e1e2) ** 2)
        t = (e1e2 * s + x2e1 - x1e1) / e1e1

        if _out_of_bounds(s, 0.0, 1.0) or _out_of_bounds(t, 0.0, 1.0):
            # potential_s = -100.0
            # potential_t = -100.0
            # potential_d = -100.0
            # overall_minimum_distance = 1e20

            # Fill in the possibilities
            potential_t = (x2e1 - x1e1) / e1e1
            s = 0.0
            t = _clip(potential_t, 0.0, 1.0)
            potential_d = _norm(x1 + e1 * t - x2)
            overall_minimum_distance = potential_d

            potential_t = (x2e1 + e1e2 - x1e1) / e1e1
            potential_t = _clip(potential_t, 0.0, 1.0)
            potential_d = _norm(x1 + e1 * potential_t - x2 - e2)
            if potential_d < overall_minimum_distance:
                s = 1.0
                t = potential_t
                overall_minimum_distance = potential_d

            potential_s = (x1e2 - x2e2) / e2e2
            potential_s = _clip(potential_s, 0.0, 1.0)
            potential_d = _norm(x2 + potential_s * e2 - x1)
            if potential_d < overall_minimum_distance:
                s = potential_s
                t = 0.0
                overall_minimum_distance = potential_d

            potential_s = (x1e2 + e1e2 - x2e2) / e2e2
            potential_s = _clip(potential_s, 0.0, 1.0)
            potential_d = _norm(x2 + potential_s * e2 - x1 - e1)
            if potential_d < overall_minimum_distance:
                s = potential_s
                t = 1.0

    # Return distance, contact point of system 2, contact point of system 1
    return x2 + s * e2 - x1 - t * e1, x2 + s * e2, x1 - t * e1


@numba.njit(cache=True)  # type: ignore
def _aabbs_not_intersecting(
    aabb_one: NDArray[np.float64], aabb_two: NDArray[np.float64]
) -> Literal[1, 0]:
    """Returns true if not intersecting else false"""
    if (aabb_one[0, 1] < aabb_two[0, 0]) | (aabb_one[0, 0] > aabb_two[0, 1]):
        return 1
    if (aabb_one[1, 1] < aabb_two[1, 0]) | (aabb_one[1, 0] > aabb_two[1, 1]):
        return 1
    if (aabb_one[2, 1] < aabb_two[2, 0]) | (aabb_one[2, 0] > aabb_two[2, 1]):
        return 1

    return 0


@numba.njit(cache=True)  # type: ignore
def _prune_using_aabbs_rod_cylinder(
    rod_one_position_collection: NDArray[np.float64],
    rod_one_radius_collection: NDArray[np.float64],
    rod_one_length_collection: NDArray[np.float64],
    cylinder_position: NDArray[np.float64],
    cylinder_director: NDArray[np.float64],
    cylinder_radius: NDArray[np.float64],
    cylinder_length: NDArray[np.float64],
) -> Literal[1, 0]:
    max_possible_dimension = np.zeros((3,))
    aabb_rod = np.empty((3, 2))
    aabb_cylinder = np.empty((3, 2))
    max_possible_dimension[...] = np.max(rod_one_radius_collection) + np.max(
        rod_one_length_collection
    )
    for i in range(3):
        aabb_rod[i, 0] = (
            np.min(rod_one_position_collection[i]) - max_possible_dimension[i]
        )
        aabb_rod[i, 1] = (
            np.max(rod_one_position_collection[i]) + max_possible_dimension[i]
        )

    # Is actually Q^T * d but numba complains about performance so we do
    # d^T @ Q
    cylinder_dimensions_in_local_FOR = np.array(
        [cylinder_radius, cylinder_radius, 0.5 * cylinder_length]
    )
    cylinder_dimensions_in_world_FOR = np.zeros_like(cylinder_dimensions_in_local_FOR)
    for i in range(3):
        for j in range(3):
            cylinder_dimensions_in_world_FOR[i] += (
                cylinder_director[j, i, 0] * cylinder_dimensions_in_local_FOR[j]
            )

    max_possible_dimension = np.abs(cylinder_dimensions_in_world_FOR)
    aabb_cylinder[..., 0] = cylinder_position[..., 0] - max_possible_dimension
    aabb_cylinder[..., 1] = cylinder_position[..., 0] + max_possible_dimension
    return _aabbs_not_intersecting(aabb_cylinder, aabb_rod)


@numba.njit(cache=True)  # type: ignore
def _prune_using_aabbs_rod_rod(
    rod_one_position_collection: NDArray[np.float64],
    rod_one_radius_collection: NDArray[np.float64],
    rod_one_length_collection: NDArray[np.float64],
    rod_two_position_collection: NDArray[np.float64],
    rod_two_radius_collection: NDArray[np.float64],
    rod_two_length_collection: NDArray[np.float64],
) -> Literal[1, 0]:
    max_possible_dimension = np.zeros((3,))
    aabb_rod_one = np.empty((3, 2))
    aabb_rod_two = np.empty((3, 2))
    max_possible_dimension[...] = np.max(rod_one_radius_collection) + np.max(
        rod_one_length_collection
    )
    for i in range(3):
        aabb_rod_one[i, 0] = (
            np.min(rod_one_position_collection[i]) - max_possible_dimension[i]
        )
        aabb_rod_one[i, 1] = (
            np.max(rod_one_position_collection[i]) + max_possible_dimension[i]
        )

    max_possible_dimension[...] = np.max(rod_two_radius_collection) + np.max(
        rod_two_length_collection
    )

    for i in range(3):
        aabb_rod_two[i, 0] = (
            np.min(rod_two_position_collection[i]) - max_possible_dimension[i]
        )
        aabb_rod_two[i, 1] = (
            np.max(rod_two_position_collection[i]) + max_possible_dimension[i]
        )

    return _aabbs_not_intersecting(aabb_rod_two, aabb_rod_one)


@numba.njit(cache=True)  # type: ignore
def _prune_using_aabbs_rod_sphere(
    rod_one_position_collection: NDArray[np.float64],
    rod_one_radius_collection: NDArray[np.float64],
    rod_one_length_collection: NDArray[np.float64],
    sphere_position: NDArray[np.float64],
    sphere_director: NDArray[np.float64],
    sphere_radius: NDArray[np.float64],
) -> Literal[1, 0]:
    max_possible_dimension = np.zeros((3,))
    aabb_rod = np.empty((3, 2))
    aabb_sphere = np.empty((3, 2))
    max_possible_dimension[...] = np.max(rod_one_radius_collection) + np.max(
        rod_one_length_collection
    )
    for i in range(3):
        aabb_rod[i, 0] = (
            np.min(rod_one_position_collection[i]) - max_possible_dimension[i]
        )
        aabb_rod[i, 1] = (
            np.max(rod_one_position_collection[i]) + max_possible_dimension[i]
        )

    sphere_dimensions_in_local_FOR = np.array(
        [sphere_radius, sphere_radius, sphere_radius]
    )
    sphere_dimensions_in_world_FOR = np.zeros_like(sphere_dimensions_in_local_FOR)
    for i in range(3):
        for j in range(3):
            sphere_dimensions_in_world_FOR[i] += (
                sphere_director[j, i, 0] * sphere_dimensions_in_local_FOR[j]
            )

    max_possible_dimension = np.abs(sphere_dimensions_in_world_FOR)
    aabb_sphere[..., 0] = sphere_position[..., 0] - max_possible_dimension
    aabb_sphere[..., 1] = sphere_position[..., 0] + max_possible_dimension
    return _aabbs_not_intersecting(aabb_sphere, aabb_rod)


@numba.njit(cache=True)  # type: ignore
def _find_slipping_elements(
    velocity_slip: NDArray[np.float64], velocity_threshold: np.float64
) -> NDArray[np.float64]:
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


@numba.njit(cache=True)  # type: ignore
def _node_to_element_mass_or_force(input: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    This function converts the mass/forces on rod nodes to
    elements, where special treatment is necessary at the ends.

    Parameters
    ----------
    input: numpy.ndarray
        2D (dim, blocksize) array containing nodal mass/forces
        with 'float' type.

    Returns
    -------
    output: numpy.ndarray
        2D (dim, blocksize) array containing elemental mass/forces
        with 'float' type.
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


@numba.njit(cache=True)  # type: ignore
def _elements_to_nodes_inplace(
    vector_in_element_frame: NDArray[np.float64],
    vector_in_node_frame: NDArray[np.float64],
) -> None:
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


@numba.njit(cache=True)  # type: ignore
def _node_to_element_position(
    node_position_collection: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    This function computes the position of the elements
    from the nodal values.
    Here we define a separate function because benchmark results
    showed that using Numba, we get more than 3 times faster calculation.

    Parameters
    ----------
    node_position_collection: numpy.ndarray
        2D (dim, blocksize) array containing nodal positions with
        'float' type.

    Returns
    -------
    element_position_collection: numpy.ndarray
        2D (dim, blocksize) array containing elemental positions with
        'float' type.
    """
    """
    Developer Notes
    -----
    Benchmark results, for a blocksize of 100,

    Python version: 3.5 µs ± 149 ns per loop

    This version: 729 ns ± 14.3 ns per loop

    """
    n_elem = node_position_collection.shape[1] - 1
    element_position_collection = np.empty((3, n_elem))
    for k in range(n_elem):
        element_position_collection[0, k] = 0.5 * (
            node_position_collection[0, k + 1] + node_position_collection[0, k]
        )
        element_position_collection[1, k] = 0.5 * (
            node_position_collection[1, k + 1] + node_position_collection[1, k]
        )
        element_position_collection[2, k] = 0.5 * (
            node_position_collection[2, k + 1] + node_position_collection[2, k]
        )

    return element_position_collection


@numba.njit(cache=True)  # type: ignore
def _node_to_element_velocity(
    mass: NDArray[np.float64], node_velocity_collection: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    This function computes the velocity of the elements
    from the nodal values. Uses the velocity of center of mass
    in order to conserve momentum during computation.

    Parameters
    ----------
    mass: numpy.ndarray
        2D (dim, blocksize) array containing nodal masses with
        'float' type.
    node_velocity_collection: numpy.ndarray
        2D (dim, blocksize) array containing nodal velocities with
        'float' type.

    Returns
    -------
    element_velocity_collection: numpy.ndarray
        2D (dim, blocksize) array containing elemental velocities with
        'float' type.
    """
    n_elem = node_velocity_collection.shape[1] - 1
    element_velocity_collection = np.empty((3, n_elem))
    for k in range(n_elem):
        element_velocity_collection[0, k] = (
            mass[k + 1] * node_velocity_collection[0, k + 1]
            + mass[k] * node_velocity_collection[0, k]
        )
        element_velocity_collection[1, k] = (
            mass[k + 1] * node_velocity_collection[1, k + 1]
            + mass[k] * node_velocity_collection[1, k]
        )
        element_velocity_collection[2, k] = (
            mass[k + 1] * node_velocity_collection[2, k + 1]
            + mass[k] * node_velocity_collection[2, k]
        )
        element_velocity_collection[:, k] /= mass[k + 1] + mass[k]

    return element_velocity_collection
