__doc__ = """ Rotation interface functions"""
__all__ = ["skew_symmetrize", "inv_skew_symmetrize", "rotate"]

import numpy as np

from elastica._rotations import (
    _inv_skew_symmetrize,
    _skew_symmetrize,
    _rotate,
)

from .utils import MaxDimension, isqrt


# TODO Complete, but nicer interface, evolve it eventually


def format_vector_shape(vector_collection):
    """
    Function for formatting vector shapes into correct format
    Parameters
    ----------
    vector_collection: numpy.ndarray
        Can be 1D or 2D.

    Returns
    -------
    output: numpy.ndarray
        Can be 1D or 2D.
    """
    n_dim = vector_collection.ndim

    if n_dim == 1:
        # Shape is (dim,)
        vector_collection = np.expand_dims(vector_collection, axis=1)
    elif n_dim == 2:
        # First possibilty, shape is (blocksize, dim), with dim
        # Soft fix : resize always so that first dimension is least
        if vector_collection.shape[0] > max(
            MaxDimension.value(), vector_collection.shape[1]
        ):
            vector_collection = vector_collection.T
        # Second possibility, shape is (blocksize,dim), with blocksize<dim
        # Example row vector (1,3),(2,3)
        if (
            vector_collection.shape[0] < MaxDimension.value()
            and vector_collection.shape[1] == MaxDimension.value()
        ):
            vector_collection = vector_collection.T

    elif n_dim > 2:
        raise RuntimeError("Vector collection dimensions >2 are not supported")

        # Check for pure 3D cases for now
    assert (
        vector_collection.shape[0] == MaxDimension.value()
    ), "Need first dimension = 3"

    return vector_collection


def format_matrix_shape(matrix_collection):
    """
    Formats input matrix into correct format
    Parameters
    ----------
    matrix_collection: numpy.ndarray
        Can be 1D, 2D, 3D.

    Returns
    -------

    """
    n_dim = matrix_collection.ndim

    # check first two dimensions are same and matrix is square
    # other possibility is one dimension is dim**2 and other is blocksize,
    # we need to convert the matrix in that case.
    def assert_proper_square(num1):
        sqrt_num = isqrt(num1)
        assert sqrt_num ** 2 == num1, "Matrix dimension passed is not a perfect square"
        return sqrt_num

    if n_dim == 1:
        # Shape is (dim**2, )
        # Check if dim**2 is a perfect square
        dim = assert_proper_square(matrix_collection.shape[0])

        # Now reshape matrix accordingly to fit (dim, dim, 1)
        matrix_collection = np.atleast_3d(matrix_collection).reshape(dim, dim, 1)

    if n_dim == 2:
        # Check if we already have a square matrix or not, i.e. (3,3)
        if matrix_collection.shape[0] == matrix_collection.shape[1]:
            dim = matrix_collection.shape[0]
        else:
            # First possibilty, shape is (blocksize, dim**2)
            # Soft fix : resize always so that first dimension is least
            if matrix_collection.shape[0] > max(
                MaxDimension.value() ** 2, matrix_collection.shape[1]
            ):
                matrix_collection = matrix_collection.T

            # Check if dim**2 is not a perfect square
            dim = assert_proper_square(matrix_collection.shape[0])

        # Expand to three dimensions
        # inp : (dim,dim) or (dim**2, bs)
        # op : (dim, dim, bs)
        matrix_collection = matrix_collection.reshape(dim, dim, -1)
    if n_dim == 3:
        # First possibilty, shape is (blocksize, dim, dim)
        if matrix_collection.shape[0] > max(
            MaxDimension.value(), matrix_collection.shape[1]
        ) and matrix_collection.shape[0] > max(
            MaxDimension.value(), matrix_collection.shape[2]
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

    assert (
        dim == MaxDimension.value()
    ), "Need matrix dimension = 3 for example (9,), (3,3), (3,3,1), (9,n), (3,3,n)"

    return matrix_collection


def skew_symmetrize(vector):
    vector = format_vector_shape(vector)
    return _skew_symmetrize(vector)


def inv_skew_symmetrize(matrix_collection):
    """
    Safe wrapper around inv_skew_symmetrize that does checking
    and formatting on type of matrix_collection using format_matrix_shape
    function.

    Parameters
    ----------
    matrix_collection: numpy.ndarray

    Returns
    -------

    """
    # format matrix collection into correct shape
    matrix_collection = format_matrix_shape(matrix_collection)
    # No memory allocated
    matrix_collection_t = np.einsum("ijk->jik", matrix_collection)

    # Checks, but 'b' argument allocates memory
    if np.allclose(matrix_collection, -matrix_collection_t):
        return _inv_skew_symmetrize(matrix_collection)
    else:
        raise ValueError("matrix_collection passed is not skew-symmetric")


def rotate(matrix, scale, axis):
    """
    This function takes single or multiple frames as matrix. Then rotates these frames
    around a single axis for all frames, or can rotate each frame around its own
    rotation axis as defined by user. Scale determines how much frames rotates
    around this axis.

    matrix: minimum shape = dim**2x1, supports shape = 3x3xn
    axis: minimum dim = 3x1, 1x3, supports dim = 3xn, nx3
    scale: minimum float, supports 1D vectors also dim = n

    """

    matrix = format_matrix_shape(matrix)
    axis = format_vector_shape(axis)

    return _rotate(matrix, scale, axis)
