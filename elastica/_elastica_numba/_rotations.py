__doc__ = """ Rotation kernels Numba implementation """

import numpy as np
from numpy import sin
from numpy import cos
from numpy import sqrt
from numpy import arccos

import numba
from numba import njit

from elastica._linalg import _batch_matmul


@njit(cache=True)
def _get_rotation_matrix(scale: float, axis_collection):
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


def _rotate(director_collection, scale: float, axis_collection):
    """
    Does alibi rotations
    https://en.wikipedia.org/wiki/Rotation_matrix#Ambiguities

    Parameters
    ----------
    director_collection
    scale
    axis_collection

    Returns
    -------

    # TODO Finish documentation
    """
    # return _batch_matmul(
    #     director_collection, _get_rotation_matrix(scale, axis_collection)
    # )
    return _batch_matmul(
        _get_rotation_matrix(scale, axis_collection), director_collection
    )


@njit(cache=True)
def _inv_rotate(director_collection):
    blocksize = director_collection.shape[2] - 1
    vector_collection = np.empty((3, blocksize))

    for k in range(blocksize):
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

        # TODO HARDCODED bugfix has to be changed. Remove 1e-14 tolerance
        theta = arccos(0.5 * trace - 0.5 - 1e-10)

        vector_collection[0, k] *= -0.5 * theta / sin(theta + 1e-14)
        vector_collection[1, k] *= -0.5 * theta / sin(theta + 1e-14)
        vector_collection[2, k] *= -0.5 * theta / sin(theta + 1e-14)

    return vector_collection
