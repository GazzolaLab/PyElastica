__doc__ = """ Rotation interface functions"""
__all__ = ["skew_symmetrize", "inv_skew_symmetrize"]

import numpy as np

from elastica._rotations import _inv_skew_symmetrize, _skew_symmetrize

from .utils import MaxDimension, isqrt


# TODO Check feasiblity of Quaternions here


def skew_symmetrize(vector):
    n_dim = vector.ndim

    if n_dim == 1:
        # Shape is (dim,)
        vector = np.expand_dims(vector, axis=1)
    elif n_dim == 2:
        # First possibilty, shape is (blocksize, dim), with dim
        # Soft fix : resize always so that first dimension is least
        if vector.shape[0] > max(MaxDimension.value(), vector.shape[1]):
            vector = vector.T
    elif n_dim > 2:
        raise RuntimeError("Vector dimensions >2 are not supported")

    # Check for pure 3D cases for now
    assert vector.shape[0] == MaxDimension.value(), "Need first dimension = 3"

    return _skew_symmetrize(vector)


def inv_skew_symmetrize(matrix_collection):
    """ Safe wrapper around inv_skew_symmetrize that does checking
    on type of matrix_collection (is it skew-symmetric and so on?)
    """
    n_dim = matrix_collection.ndim

    def assert_proper_square(num):
        sqrt_num = isqrt(num)
        assert sqrt_num ** 2 == num, "Matrix dimension passed is not a perfect square"
        return sqrt_num

    if n_dim == 1:
        # Shape is (dim**2, )
        # Check if dim**2 is not a perfect square
        dim = assert_proper_square(matrix_collection.shape[0])

        # Now reshape matrix accordingly to fit (dim, dim, 1)
        matrix_collection = np.atleast_3d(matrix_collection).reshape(dim, dim, 1)

    if n_dim == 2:
        # First possibilty, shape is (blocksize, dim**2)
        # Soft fix : resize always so that first dimension is least
        if matrix_collection.shape[0] > max(
            MaxDimension.value() ** 2, matrix_collection.shape[1]
        ):
            matrix_collection = matrix_collection.T

        # Check if dim**2 is not a perfect square
        dim = assert_proper_square(matrix_collection.shape[0])

        # Expand to three dimensions
        # inp : (dim**2, bs)
        # op : (dim, dim, bs)
        matrix_collection = matrix_collection.reshape(dim, dim, -1)

    if n_dim == 3:
        # First possibilty, shape is (blocksize, dim, dim)
        if matrix_collection.shape[0] > max(
            MaxDimension.value() ** 2, matrix_collection.shape[1]
        ) and matrix_collection.shape[0] > max(
            MaxDimension.value() ** 2, matrix_collection.shape[2]
        ):
            matrix_collection = matrix_collection.T

        # Given (dim, dim, bs) array, check if dimensions are equal
        assert (
            matrix_collection.shape[0] == matrix_collection.shape[1]
        ), "Matrix shapes along 1 and 2 are not equal"

        # Obtain dimensions for checking
        dim = matrix_collection.shape[0]

    elif n_dim > 3:
        raise RuntimeError("Matrix dimensions >3 are not supported")

    assert dim == MaxDimension.value(), "Need dimension = 3"

    # No memory allocated
    matrix_collection_t = np.einsum("ijk->jik", matrix_collection)

    # Checks, but 'b' argument allocates memory
    if np.allclose(matrix_collection, -matrix_collection_t):
        return _inv_skew_symmetrize(matrix_collection)
    else:
        raise ValueError("matrix_collection passed is not skew-symmetric")


# def multiply():
#     # (dim, dim, n) and (dim, dim, n)


# def rotate_kernel(input, theta, axis):
#     """
#     n leading dimension because these are all independent computations
#     and so we benefit least from vectorisation if we have n at the end,
#     if we are using pure numpy functions

#     v_input = (n, dim, dim) array
#     v_theta = (n, ) array, magnitude
#     v_axis =  (n, dim) array, v_axis[:, i] should have unit norm
#     """
#     '''
#     def normalize(v):
#         """ Normalize a vector/ matrix """
#         norm = np.linalg.norm(v)
#         if np.isclose(norm, 0.0, atol = Tolerance.tol()):
#             return np.zeros(3)
#         return v / norm
#     '''

#     dimensions = axis.shape(1)
#     blocksize = theta.shape(0)

#     u_square_prefix = 1.0 - np.cos(theta)
#     u_prefix = np.sin(theta)

#     # Build rotation matrix from scratch
#     eye = np.eye(dimensions)[np.newaxis, :]
#     rot_matrix = np.repeat(eye, blocksize, axis=0)

#     mag = np.linalg.norm(u, axis=1)
#     theta *= mag

#     rot_idx = np.invert(np.isclose(mag, 0.0, atol=Tolerance.tol()))

#     unorm = np.zeros_like(u)
#     if np.any(rot_idx):
#         unorm[rot_idx] = u[rot_idx] / mag[rot_idx, None]


#         U_mat = np.array([skew_symmetrize(uu) for uu in unorm])
#         nrot_idx = np.sum(rot_idx)
#         tmp_eye = np.array([np.eye(3) for i in range(nrot_idx)])
#         rot_matrix[rot_idx] = tmp_eye + U_mat[rot_idx] @ (
#             s_angle[rot_idx, None, None] * tmp_eye
#             + (1.0 - c_angle[rot_idx, None, None]) * U_mat[rot_idx]
#         )

#     return rot_matrix
