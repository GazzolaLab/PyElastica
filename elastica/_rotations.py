__doc__ = """ Rotation kernels """

import functools
from itertools import combinations

import numpy as np
from numpy import sin, cos, sqrt, arccos
from numpy.typing import NDArray

from numba import njit

from elastica.typing import RodType, RigidBodyType, ConnectionIndex
from elastica._linalg import _batch_matmul


@njit(cache=True)  # type: ignore
def _get_rotation_matrix(
    scale: np.float64, axis_collection: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute rotation matrices from axis-angle representation using Rodrigues' formula.

    Parameters
    ----------
    scale : float
        Scale factor applied to rotation angles. The actual rotation angle for each
        axis is scale * ||axis||.
    axis_collection : numpy.ndarray
        2D array of shape (dim, blocksize) containing rotation axes. Each column
        represents an axis of rotation.

    Returns
    -------
    rot_mat : numpy.ndarray
        3D array of shape (dim, dim, blocksize) containing rotation matrices computed
        using Rodrigues' rotation formula.

    Notes
    -----
    The axes are normalized before computing the rotation matrices. A small epsilon
    (1e-14) is added to prevent division by zero for zero-length axes.
    """
    blocksize = axis_collection.shape[1]
    rot_mat = np.empty((3, 3, blocksize))

    for k in range(blocksize):
        v0 = axis_collection[0, k]
        v1 = axis_collection[1, k]
        v2 = axis_collection[2, k]

        theta = sqrt(v0 * v0 + v1 * v1 + v2 * v2)

        v0 /= theta + 1e-14
        v1 /= theta + 1e-14
        v2 /= theta + 1e-14

        theta *= scale
        u_prefix = sin(theta)
        u_sq_prefix = 1.0 - cos(theta)

        rot_mat[0, 0, k] = 1.0 - u_sq_prefix * (v1 * v1 + v2 * v2)
        rot_mat[1, 1, k] = 1.0 - u_sq_prefix * (v0 * v0 + v2 * v2)
        rot_mat[2, 2, k] = 1.0 - u_sq_prefix * (v0 * v0 + v1 * v1)

        rot_mat[0, 1, k] = u_prefix * v2 + u_sq_prefix * v0 * v1
        rot_mat[1, 0, k] = -u_prefix * v2 + u_sq_prefix * v0 * v1
        rot_mat[0, 2, k] = -u_prefix * v1 + u_sq_prefix * v0 * v2
        rot_mat[2, 0, k] = u_prefix * v1 + u_sq_prefix * v0 * v2
        rot_mat[1, 2, k] = u_prefix * v0 + u_sq_prefix * v1 * v2
        rot_mat[2, 1, k] = -u_prefix * v0 + u_sq_prefix * v1 * v2

    return rot_mat


@njit(cache=True)  # type: ignore
def _rotate(
    director_collection: NDArray[np.float64],
    scale: np.float64,
    axis_collection: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Rotate director collection by specified axes and scale (alibi rotation).

    Performs alibi (active) rotations on a collection of director frames.
    Each director frame is rotated around its corresponding axis by an angle
    proportional to the scale factor. The rotation is applied using Rodrigues'
    rotation formula via `_get_rotation_matrix`.

    Parameters
    ----------
    director_collection : numpy.ndarray
        3D array of shape (dim, dim, blocksize) containing rotation matrices
        (director frames) to be rotated.
    scale : float
        Scale factor for rotation angles. The actual rotation angle for each
        frame is scale * ||axis||, where ||axis|| is the magnitude of the
        corresponding axis vector.
    axis_collection : numpy.ndarray
        2D array of shape (dim, blocksize) containing rotation axes for each
        director frame. Each column represents the axis of rotation for the
        corresponding director frame.

    Returns
    -------
    rotated_directors : numpy.ndarray
        3D array of shape (dim, dim, blocksize) containing the rotated director
        frames. Each frame is rotated around its corresponding axis by the
        scaled angle.

    Notes
    -----
    This function performs alibi (active) rotations, meaning the coordinate
    system is rotated. For more information on rotation matrix ambiguities, see:
    https://en.wikipedia.org/wiki/Rotation_matrix#Ambiguities

    The rotation is computed as: R(scale * axis) @ director, where R is the
    rotation matrix computed from the axis-angle representation.
    """
    # return _batch_matmul(
    #     director_collection, _get_rotation_matrix(scale, axis_collection)
    # )
    return _batch_matmul(
        _get_rotation_matrix(scale, axis_collection), director_collection
    )


@njit(cache=True)  # type: ignore
def _inv_rotate(director_collection: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute rotation axes between consecutive director frames using Rodrigues' formula.

    Calculates the rotation axis (in axis-angle representation) that transforms
    each director frame to the next one. This is the inverse operation of rotating
    directors and is used to extract the relative rotation between consecutive
    elements.

    Parameters
    ----------
    director_collection : numpy.ndarray
        The collection of frames/directors at every element, of shape (dim, dim, n)
        where n is the number of director frames.

    Returns
    -------
    vector_collection : numpy.ndarray
        The collection of rotation axes, of shape (dim, n-1). Each column represents
        the axis of rotation (scaled by angle) that transforms director[k] to director[k+1].

    Notes
    -----
    The output has n-1 elements because it computes the relative rotation between
    consecutive pairs of directors. The rotation axis is computed using the trace
    of the relative rotation matrix Q_{k+1} @ Q_k^T.
    """
    blocksize = director_collection.shape[2] - 1
    vector_collection = np.empty((3, blocksize))

    for k in range(blocksize):
        # Q_{i+i}Q^T_{i} collection
        vector_collection[0, k] = (
            director_collection[2, 0, k + 1] * director_collection[1, 0, k]
            + director_collection[2, 1, k + 1] * director_collection[1, 1, k]
            + director_collection[2, 2, k + 1] * director_collection[1, 2, k]
        ) - (
            director_collection[1, 0, k + 1] * director_collection[2, 0, k]
            + director_collection[1, 1, k + 1] * director_collection[2, 1, k]
            + director_collection[1, 2, k + 1] * director_collection[2, 2, k]
        )

        vector_collection[1, k] = (
            director_collection[0, 0, k + 1] * director_collection[2, 0, k]
            + director_collection[0, 1, k + 1] * director_collection[2, 1, k]
            + director_collection[0, 2, k + 1] * director_collection[2, 2, k]
        ) - (
            director_collection[2, 0, k + 1] * director_collection[0, 0, k]
            + director_collection[2, 1, k + 1] * director_collection[0, 1, k]
            + director_collection[2, 2, k + 1] * director_collection[0, 2, k]
        )

        vector_collection[2, k] = (
            director_collection[1, 0, k + 1] * director_collection[0, 0, k]
            + director_collection[1, 1, k + 1] * director_collection[0, 1, k]
            + director_collection[1, 2, k + 1] * director_collection[0, 2, k]
        ) - (
            director_collection[0, 0, k + 1] * director_collection[1, 0, k]
            + director_collection[0, 1, k + 1] * director_collection[1, 1, k]
            + director_collection[0, 2, k + 1] * director_collection[1, 2, k]
        )

        trace = (
            (
                director_collection[0, 0, k + 1] * director_collection[0, 0, k]
                + director_collection[0, 1, k + 1] * director_collection[0, 1, k]
                + director_collection[0, 2, k + 1] * director_collection[0, 2, k]
            )
            + (
                director_collection[1, 0, k + 1] * director_collection[1, 0, k]
                + director_collection[1, 1, k + 1] * director_collection[1, 1, k]
                + director_collection[1, 2, k + 1] * director_collection[1, 2, k]
            )
            + (
                director_collection[2, 0, k + 1] * director_collection[2, 0, k]
                + director_collection[2, 1, k + 1] * director_collection[2, 1, k]
                + director_collection[2, 2, k + 1] * director_collection[2, 2, k]
            )
        )

        # Clip the trace to between -1 and 3.
        # Any deviation beyond this is numerical error
        trace = min(trace, 3.0)
        trace = max(trace, -1.0)
        theta = arccos(0.5 * trace - 0.5) + 1e-14
        magnitude = -0.5 * theta / sin(theta)

        vector_collection[0, k] *= magnitude
        vector_collection[1, k] *= magnitude
        vector_collection[2, k] *= magnitude

    return vector_collection


_generate_skew_map_sentinel = (0, 0, 0)


# TODO: Below contains numpy-only implementations
@functools.lru_cache(maxsize=1)
def _generate_skew_map(dim: int) -> list[tuple[int, int, int]]:
    """
    Generate mapping indices for converting vectors to skew-symmetric matrices.

    Creates a mapping that defines how vector elements are arranged in a
    flattened skew-symmetric matrix representation. This is used for efficient
    conversion between vector and matrix forms in dimension-agnostic operations.

    Notes
    -----
    The mapping handles the conversion from a vector v = [x, y, z] to a
    skew-symmetric matrix M where M[i,j] = -M[j,i] and the off-diagonal
    elements correspond to vector components.

    The formula used (dim - (i + j)) works correctly for dimensions 2 and 3,
    but may need verification for higher dimensions.
    """
    # Preallocate
    mapping_list = [_generate_skew_map_sentinel] * ((dim**2 - dim) // 2)
    # Indexing (i,j), j is the fastest changing
    # r = 2, r here is rank, we deal with only matrices
    for index, (i, j) in enumerate(combinations(range(dim), r=2)):
        # matrix indices
        tgt_idx = dim * i + j
        # Sign-bit to check order of entries
        sign = (-1) ** tgt_idx
        # idx in v
        # TODO Wrong formulae, but works for two and three dimensions
        src_idx = dim - (i + j)

        # Check order to fill in the list
        if sign < 0:
            entry_t = (src_idx, j, i)
        else:
            entry_t = (src_idx, i, j)

        mapping_list[index] = entry_t

    return mapping_list


@functools.lru_cache(maxsize=1)
def _get_skew_map(dim: int) -> tuple[tuple[int, int, int], ...]:
    """Generates mapping from src to target skew-symmetric operator

    For input vector V and output Matrix M (represented in lexicographical index),
    we calculate mapping from

        |x|        |0 -z y|
    V = |y| to M = |z 0 -x|
        |z|        |-y x 0|

    in a dimension agnostic way.

    """
    mapping_list = _generate_skew_map(dim)

    # sort for non-strided access in source dimension, potentially faster copies
    mapping_list.sort(key=lambda tup: tup[0])

    # return iterator
    return tuple(mapping_list)


@functools.lru_cache(maxsize=1)
def _get_inv_skew_map(dim: int) -> tuple[tuple[int, int, int], ...]:
    """
    Generate inverse mapping for extracting vectors from skew-symmetric matrices.

    Creates a mapping that defines how to extract vector elements from a
    flattened skew-symmetric matrix representation. This is the inverse
    operation of `_generate_skew_map`.

    Notes
    -----
    This mapping is used to extract vector components from skew-symmetric
    matrices. The mapping is generated by inverting the tuple element order
    from `_generate_skew_map`.
    """
    # (vec_src, mat_i, mat_j, sign)
    mapping_list = _generate_skew_map(dim)

    # invert tuple elements order to have
    #             (mat_i, mat_j, vec_tgt, sign)
    return tuple((t[1], t[2], t[0]) for t in mapping_list)


@functools.lru_cache(maxsize=1)
def _get_diag_map(dim: int) -> tuple[int, ...]:
    """Generates lexicographic mapping to diagonal in a serialized matrix-type

    For input dimension dim  we calculate mapping to * in Matrix M below

        |* 0 0|
    M = |0 * 0|
        |0 0 *|

    in a dimension agnostic way.

    """
    return tuple([dim_iter * (dim + 1) for dim_iter in range(dim)])


def _skew_symmetrize(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert vector collection to skew-symmetric matrix collection.

    Notes
    -----
    Gets close to the hard-coded implementation in time but with slightly
    high memory requirement for iteration.

    For blocksize=128,
    hardcoded : 5.9 µs ± 186 ns per loop
    this : 6.19 µs ± 79.2 ns per loop

    """
    dim, blocksize = vector.shape
    skewed = np.zeros((dim, dim, blocksize))

    # Iterate over generated indices and put stuff from v to m
    for src_index, tgt_i, tgt_j in _get_skew_map(dim):
        skewed[tgt_i, tgt_j] = vector[src_index]
        skewed[tgt_j, tgt_i] = -skewed[tgt_i, tgt_j]

    return skewed


# This is purely for testing and optimization sake
# While calculating u^2, use u with einsum instead, as it is tad bit faster
def _skew_symmetrize_sq(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Generate the square of a skew-symmetric matrix from vector elements.

    Computes u^2 where u is the skew-symmetric matrix corresponding to the input
    vector. This is used in Rodrigues' rotation formula.

    Parameters
    ----------
    vector : numpy.ndarray
        Input vector collection of shape (dim, blocksize).

    Returns
    -------
    output : numpy.ndarray
        Square of skew-symmetric matrices of shape (dim, dim, blocksize).
        For a 3D vector [x, y, z], the corresponding matrix u^2 is:
        [[-(y^2+z^2), xy, xz],
         [yx, -(x^2+z^2), yz],
         [zx, zy, -(x^2+y^2)]]

    Note
    ----
    Faster than hard-coded implementation in time with slightly high
    memory requirement for einsum calculation.

    For blocksize=128,
    hardcoded : 23.1 µs ± 481 ns per loop
    this version: 14.1 µs ± 96.9 ns per loop
    """

    # First generate array of [x^2, xy, xz, yx, y^2, yz, zx, zy, z^2]
    # across blocksize
    # This is slightly faster than doing v[np.newaxis,:,:] * v[:,np.newaxis,:]
    products_xy: NDArray[np.float64] = np.einsum("ik,jk->ijk", vector, vector)

    # No copy made here, as we do not change memory layout
    # products_xy = products_xy.reshape((dim * dim, -1))

    # Now calculate (x^2 + y^2 + z^2) across blocksize
    # Interpret this as a contraction ji,ij->j with v.T, v
    mag = np.einsum("ij,ij->j", vector, vector)

    # Iterate over only the diagonal and subtract mag
    # Somewhat faster (5us for 128 blocksize) but more memory efficient than doing :
    # > eye_arr = np.ravel(np.eye(dim, dim))
    # > eye_arr = eye_arr[:, np.newaxis] * mag[np.newaxis, :]
    # > products_xy - mag

    # This version is faster for smaller blocksizes <= 128
    # Efficiently extracts only diagonal elements
    # reshape returns a view in this case
    np.einsum("iij->ij", products_xy)[...] -= mag

    # # This version is faster for larger blocksizes > 256
    # for diag_idx in _get_diag_map(dim):
    #     products_xy[diag_idx, :] -= mag

    # We expect this version to be superior, but due to numpy's advanced
    # indexing always returning a copy, rather than a view, it turns out
    # to be more expensive.
    #     products_xy[_get_diag_map(dim, :] -= mag

    return products_xy


def _get_skew_symmetric_pair(
    vector_collection: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute both the skew-symmetric matrix and its square from vector collection.

    This is a convenience function that computes both u and u^2 where u is the
    skew-symmetric matrix corresponding to the input vectors. These are commonly
    used together in Rodrigues' rotation formula.

    """
    u = _skew_symmetrize(vector_collection)
    u_sq = np.einsum("ijk,jlk->ilk", u, u)
    return u, u_sq


def _inv_skew_symmetrize(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Return the vector elements from a skew-symmetric matrix M.

    Parameters
    ----------
    matrix : numpy.ndarray
        3D (dim, dim, blocksize) array containing skew-symmetric matrices.

    Returns
    -------
    vector : numpy.ndarray
        2D (dim, blocksize) array containing the extracted vector elements.

    Notes
    -----
    Hardcoded: 2.28 µs ± 63.3 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    This: 2.91 µs ± 58.3 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    """
    dim, dim, blocksize = matrix.shape

    vector = np.zeros((dim, blocksize))

    # Iterate over generated indices and put stuff from v to m
    # The original skew_mapping function takes consecutive
    # indices in v and puts them in the matrix, so we skip
    # indices here
    for src_i, src_j, tgt_index in _get_inv_skew_map(dim):
        vector[tgt_index] = matrix[src_i, src_j]

    return vector


def get_relative_rotation_two_systems(
    system_one: "RodType | RigidBodyType",
    index_one: "ConnectionIndex",
    system_two: "RodType | RigidBodyType",
    index_two: "ConnectionIndex",
) -> NDArray[np.float64]:
    """
    Compute the relative rotation matrix C_12 between system one and system two at the specified elements.

    Examples
    --------
    How to get the relative rotation between two systems (e.g. the rotation from end of rod one to base of rod two):

        >>> rel_rot_mat = get_relative_rotation_two_systems(system1, -1, system2, 0)

    How to initialize a FixedJoint with a rest rotation between the two systems,
    which is enforced throughout the simulation:

        >>> simulator.connect(
        ...    first_rod=system1, second_rod=system2, first_connect_idx=-1, second_connect_idx=0
        ... ).using(
        ...    FixedJoint,
        ...    ku=1e6, nu=0.0, kt=1e3, nut=0.0,
        ...    rest_rotation_matrix=get_relative_rotation_two_systems(system1, -1, system2, 0)
        ... )

    See Also
    --------
    FixedJoint

    Parameters
    ----------
    system_one : RodType | RigidBodyType
        Rod or rigid-body object
    index_one : ConnectionIndex
        Index of first system for connection.
    system_two : RodType | RigidBodyType
        Rod or rigid-body object
    index_two : ConnectionIndex
        Index of second system for connection.

    Returns
    -------
    relative_rotation_matrix : numpy.ndarray
        2D (3, 3) array containing the relative rotation matrix C_12 between the two systems
        for their current state.
    """
    director_one = system_one.director_collection[..., index_one]
    director_two = system_two.director_collection[..., index_two]
    return director_one @ director_two.T
