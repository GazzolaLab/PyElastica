__doc__ = """ Rotation kernels """
import functools
from itertools import combinations

import numpy as np

from .utils import isqrt


# TODO Check feasiblity of Quaternions here


@functools.lru_cache(maxsize=1)
def _generate_skew_map(dim: int):
    # TODO Documentation
    # Preallocate
    mapping_list = [None] * ((dim ** 2 - dim) // 2)
    # Indexing (i,j), j is the fastest changing
    # r = 2, r here is rank, we deal with only matrices
    for index, (i, j) in enumerate(combinations(range(dim), r=2)):
        # matrix indices
        tgt_idx = dim * i + j
        # Sign-bit to check order of entries
        sign = (-1) ** (tgt_idx)
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
def _get_skew_map(dim):
    """ Generates mapping from src to target skew-symmetric operator

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
def _get_inv_skew_map(dim):
    # TODO Documentation
    # (vec_src, mat_i, mat_j, sign)
    mapping_list = _generate_skew_map(dim)

    # invert tuple elements order to have
    #             (mat_i, mat_j, vec_tgt, sign)
    return tuple((t[1], t[2], t[0]) for t in mapping_list)


@functools.lru_cache(maxsize=1)
def _get_diag_map(dim):
    """ Generates lexicographic mapping to diagonal in a serialized matrix-type

    For input dimension dim  we calculate mapping to * in Matrix M below

        |* 0 0|
    M = |0 * 0|
        |0 0 *|

    in a dimension agnostic way.

    """
    # Preallocate
    mapping_list = [None] * dim

    # Store linear indices
    for dim_iter in range(dim):
        mapping_list[dim_iter] = dim_iter * (dim + 1)

    return tuple(mapping_list)


def _skew_symmetrize(vector):
    """

    Parameters
    ----------
    vector : numpy.ndarray of shape (dim, blocksize)

    Returns
    -------
    output : numpy.ndarray of shape (dim*dim, blocksize) corresponding to
             [0, -z, y, z, 0, -x, -y , x, 0]

    Note
    ----
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
def _skew_symmetrize_sq(vector):
    """
    Generate the square of an orthogonal matrix from vector elements

    Parameters
    ----------
    vector : numpy.ndarray of shape (dim, blocksize)

    Returns
    -------
    output : numpy.ndarray of shape (dim*dim, blocksize) corresponding to
             [-(y^2+z^2), xy, xz, yx, -(x^2+z^2), yz, zx, zy, -(x^2+y^2)]

    Note
    ----
    Faster than hard-coded implementation in time with slightly high
    memory requirement for einsum calculation.

    For blocksize=128,
    hardcoded : 23.1 µs ± 481 ns per loop
    this version: 14.1 µs ± 96.9 ns per loop
    """
    dim, _ = vector.shape

    # First generate array of [x^2, xy, xz, yx, y^2, yz, zx, zy, z^2]
    # across blocksize
    # This is slightly faster than doing v[np.newaxis,:,:] * v[:,np.newaxis,:]
    products_xy = np.einsum("ik,jk->ijk", vector, vector)

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


def _get_skew_symmetric_pair(vector_collection):
    """

    Parameters
    ----------
    vector_collection

    Returns
    -------

    """
    u = _skew_symmetrize(vector_collection)
    u_sq = np.einsum("ijk,jlk->ilk", u, u)
    return u, u_sq


def _inv_skew_symmetrize(matrix):
    """
    Return the vector elements from a skew-symmetric matrix M

    Parameters
    ----------
    matrix : np.ndarray of dimension (dim, dim, blocksize)

    Returns
    -------
    vector : np.ndarray of dimension (dim, blocksize)

    Note
    ----
    Harcoded : 2.28 µs ± 63.3 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    This : 2.91 µs ± 58.3 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
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


def _construct_rotation_matrix(dt: float, omega_collection):
    """

    Parameters
    ----------
    dt
    omega_collection

    Returns
    -------

    """
    # First normalize omega
    theta = np.sqrt(np.einsum("ij,ij->j", omega_collection, omega_collection))
    omega_collection /= theta

    # Multiply theta by dt (efficient, as 1D) and get prefixes
    theta *= dt
    u_prefix = np.sin(theta)
    u_sq_prefix = 1.0 - np.cos(theta)

    # Get skew symmetric U and U * U
    u, u_sq = _get_skew_symmetric_pair(omega_collection)

    # Start rotate_mat minus the \delta_ij
    rot_mat = u_prefix * u + u_sq_prefix * u_sq

    """Both these versions are almost equivalent, both in time and memory
    keeping second for ease of us"""

    # dim, _ = omega_collection.shape
    # for idx in iters(dim):
    #     rot_mat[idx, :] += 1.0

    np.einsum("iij->ij", rot_mat)[...] += 1.0

    return rot_mat


def _rotate(director_collection, dt: float, omega_collection):
    """

    Parameters
    ----------
    director_collection
    dt
    omega_collection

    Returns
    -------

    """
    # TODO Finish documentation
    return np.einsum(
        "ijk,jlk->ilk",
        _construct_rotation_matrix(dt, omega_collection),
        director_collection,
    )
