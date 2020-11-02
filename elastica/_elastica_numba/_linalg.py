__doc__ = """ Convenient linear algebra kernels Numba implementation"""
import numpy as np
import numba
from numba import njit
from numpy import sqrt


@njit(cache=True)
def _batch_matvec(matrix_collection, vector_collection):
    """
    This function does batch matrix and batch vector product
    Parameters
    ----------
    matrix_collection
    vector_collection

    Returns
    -------
    Notes
    -----
    Benchmark results, for a blocksize of 100, using timeit
    Python einsum: 9.74 µs ± 4.72 µs per loop
    This version: 1.73 µs ± 452 ns per loop
    """
    blocksize = vector_collection.shape[1]
    output_vector = np.zeros((3, blocksize))

    for i in range(3):
        for j in range(3):
            for k in range(blocksize):
                output_vector[i, k] += (
                    matrix_collection[i, j, k] * vector_collection[j, k]
                )

    return output_vector


@njit(cache=True)
def _batch_matmul(first_matrix_collection, second_matrix_collection):
    """
    This is batch matrix matrix multiplication function. Only batch
    of 3x3 matrices can be multiplied.
    Parameters
    ----------
    first_matrix_collection
    second_matrix_collection

    Returns
    -------
    Notes
    Microbenchmark results showed that for a block size of 200,
    %timeit np.einsum("ijk,jlk->ilk", first_matrix_collection, second_matrix_collection)
    12.8 µs ± 136 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    This version is
    4.41 µs ± 395 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    """
    blocksize = first_matrix_collection.shape[2]
    output_matrix = np.zeros((3, 3, blocksize))

    for k in range(blocksize):
        output_matrix[0, 0, k] = (
            first_matrix_collection[0, 0, k] * second_matrix_collection[0, 0, k]
            + first_matrix_collection[0, 1, k] * second_matrix_collection[1, 0, k]
            + first_matrix_collection[0, 2, k] * second_matrix_collection[2, 0, k]
        )
        output_matrix[0, 1, k] = (
            first_matrix_collection[0, 0, k] * second_matrix_collection[0, 1, k]
            + first_matrix_collection[0, 1, k] * second_matrix_collection[1, 1, k]
            + first_matrix_collection[0, 2, k] * second_matrix_collection[2, 1, k]
        )
        output_matrix[0, 2, k] = (
            first_matrix_collection[0, 0, k] * second_matrix_collection[0, 2, k]
            + first_matrix_collection[0, 1, k] * second_matrix_collection[1, 2, k]
            + first_matrix_collection[0, 2, k] * second_matrix_collection[2, 2, k]
        )
        output_matrix[1, 0, k] = (
            first_matrix_collection[1, 0, k] * second_matrix_collection[0, 0, k]
            + first_matrix_collection[1, 1, k] * second_matrix_collection[1, 0, k]
            + first_matrix_collection[1, 2, k] * second_matrix_collection[2, 0, k]
        )
        output_matrix[1, 1, k] = (
            first_matrix_collection[1, 0, k] * second_matrix_collection[0, 1, k]
            + first_matrix_collection[1, 1, k] * second_matrix_collection[1, 1, k]
            + first_matrix_collection[1, 2, k] * second_matrix_collection[2, 1, k]
        )
        output_matrix[1, 2, k] = (
            first_matrix_collection[1, 0, k] * second_matrix_collection[0, 2, k]
            + first_matrix_collection[1, 1, k] * second_matrix_collection[1, 2, k]
            + first_matrix_collection[1, 2, k] * second_matrix_collection[2, 2, k]
        )
        output_matrix[2, 0, k] = (
            first_matrix_collection[2, 0, k] * second_matrix_collection[0, 0, k]
            + first_matrix_collection[2, 1, k] * second_matrix_collection[1, 0, k]
            + first_matrix_collection[2, 2, k] * second_matrix_collection[2, 0, k]
        )
        output_matrix[2, 1, k] = (
            first_matrix_collection[2, 0, k] * second_matrix_collection[0, 1, k]
            + first_matrix_collection[2, 1, k] * second_matrix_collection[1, 1, k]
            + first_matrix_collection[2, 2, k] * second_matrix_collection[2, 1, k]
        )
        output_matrix[2, 2, k] = (
            first_matrix_collection[2, 0, k] * second_matrix_collection[0, 2, k]
            + first_matrix_collection[2, 1, k] * second_matrix_collection[1, 2, k]
            + first_matrix_collection[2, 2, k] * second_matrix_collection[2, 2, k]
        )

    return output_matrix


@njit(cache=True)
def _batch_cross(first_vector_collection, second_vector_collection):
    """
    This function does cross product between two batch vectors.
    Parameters
    ----------
    first_vector_collection
    second_vector_collection

    Returns
    -------
    Notes
    ----
    Benchmark results, for a blocksize of 100 using timeit
    Python einsum: 14 µs ± 8.96 µs per loop
    This version: 1.18 µs ± 141 ns per loop
    """
    blocksize = first_vector_collection.shape[1]
    output_vector = np.empty((3, blocksize))

    for k in range(blocksize):
        output_vector[0, k] = (
            first_vector_collection[1, k] * second_vector_collection[2, k]
            - first_vector_collection[2, k] * second_vector_collection[1, k]
        )

        output_vector[1, k] = (
            first_vector_collection[2, k] * second_vector_collection[0, k]
            - first_vector_collection[0, k] * second_vector_collection[2, k]
        )

        output_vector[2, k] = (
            first_vector_collection[0, k] * second_vector_collection[1, k]
            - first_vector_collection[1, k] * second_vector_collection[0, k]
        )

    return output_vector


@njit(cache=True)
def _batch_vec_oneD_vec_cross(first_vector_collection, second_vector):
    """
    This function does cross product between batch vector and a 1D vector.
    Idea of having this function is that, for friction calculations, we dont
    want to repate and expand 1D plane normal vector. Thus instead we are writting
    a new cross product operation, so we are not allocating unnecesary memory.

    Parameters
    ----------
    first_vector_collection
    second_vector

    Returns
    -------
    Notes
    -----
    Benchmark results, for a blocksize of 100 using timeit
    Python einsum: 16.6 µs ± 2.69 µs per loop
    This version: 1.1 µs ± 132 ns per loop

    """
    blocksize = first_vector_collection.shape[1]
    output_vector = np.empty((3, blocksize))

    for k in range(blocksize):
        output_vector[0, k] = (
            first_vector_collection[1, k] * second_vector[2]
            - first_vector_collection[2, k] * second_vector[1]
        )

        output_vector[1, k] = (
            first_vector_collection[2, k] * second_vector[0]
            - first_vector_collection[0, k] * second_vector[2]
        )

        output_vector[2, k] = (
            first_vector_collection[0, k] * second_vector[1]
            - first_vector_collection[1, k] * second_vector[0]
        )

    return output_vector


@njit(cache=True)
def _batch_dot(first_vector, second_vector):
    """
    This function does batch vec and batch vec dot product.
    Parameters
    ----------
    first_vector
    second_vector

    Returns
    -------
    Notes
    -----
    Benchmark results, for a blocksize of 100 using timeit
    Python einsum: 3.89 µs ± 815 ns per loop
    This version: 938 ns ± 14.1 ns per loop
    """
    blocksize = first_vector.shape[1]
    output_vector = np.empty((blocksize))

    for k in range(blocksize):
        output_vector[k] = (
            first_vector[0, k] * second_vector[0, k]
            + first_vector[1, k] * second_vector[1, k]
            + first_vector[2, k] * second_vector[2, k]
        )

    return output_vector


@njit(cache=True)
def _batch_norm(vector):
    """
    This function computes norm of a batch vector
    Parameters
    ----------
    vector

    Returns
    -------
    Notes
    -----
    Benchmark results, for a blocksize of 100 using timeit
    Python einsum: 5.03 µs ± 864 ns per loop
    This version: 1.02 µs ± 104 ns per loop
    """
    blocksize = vector.shape[1]
    output_vector = np.empty((blocksize))

    for k in range(blocksize):
        output_vector[k] = sqrt(
            vector[0, k] * vector[0, k]
            + vector[1, k] * vector[1, k]
            + vector[2, k] * vector[2, k]
        )

    return output_vector


@njit(cache=True)
def _batch_product_i_k_to_ik(vector1, vector2):
    """
    This function does outer product following 'i,k->ik'.
    vector1 has shape of 3 and vector 2 has shape of blocksize
    Parameters
    ----------
    vector1
    vector2

    Returns
    -------
    Notes
    -----
    Benchmark results, for a blocksize of 100 using timeit
    Python einsum: 3.61 µs ± 55.3 ns per loop
    This version: 961 ns ± 53.7 ns per loop
    """
    blocksize = vector2.shape[0]
    # Assert check to see if given input vector dimensions are correct
    assert vector1.shape[0] == 3
    output_vector = np.empty((3, blocksize))
    for i in range(3):
        for k in range(blocksize):
            output_vector[i, k] = vector1[i] * vector2[k]

    return output_vector


@njit(cache=True)
def _batch_product_i_ik_to_k(vector1, vector2):
    """
    This function does the following product 'i,ik->k'
    This function do dot product between a vector of 3 elements
    with a batch vector.
    Parameters
    ----------
    vector1
    vector2

    Returns
    -------
    Notes
    -----
    Benchmark results, for a blocksize of 100 using timeit
    Python einsum: 3.31 µs ± 104 ns per loop
    This version: 958 ns ± 60.3 ns per loop
    """
    blocksize = vector2.shape[1]
    assert vector1.shape[0] == 3
    assert vector2.shape[0] == 3
    output_vector = np.zeros((blocksize))

    for i in range(3):
        for k in range(blocksize):
            output_vector[k] += vector1[i] * vector2[i, k]

    return output_vector


@njit(cache=True)
def _batch_product_k_ik_to_ik(vector1, vector2):
    """
    This function does the following product 'k, ik->ik'
    Parameters
    ----------
    vector1
    vector2

    Returns
    -------
    Notes
    ----
    Benchmark results, for a blocksize of 100 using timeit
    Python einsum: 3.69 µs ± 67.9 ns per loop
    This version: 876 ns ± 11.1 ns per loop
    """
    blocksize = vector2.shape[1]
    assert vector2.shape[0] == 3
    assert vector1.shape[0] == blocksize
    output_vector = np.empty((3, blocksize))

    for i in range(3):
        for k in range(blocksize):
            output_vector[i, k] = vector1[k] * vector2[i, k]

    return output_vector


@njit(cache=True)
def _batch_vector_sum(vector1, vector2):
    """
    This function is for summing up two vectors. Although
    this function is not faster than pure python implementation
    when we add a numba njit decorator around the pure python
    implementation we code slows down. Thus, for high level
    numba njit function, using this function is beneficial.
    Parameters
    ----------
    vector1
    vector2

    Returns
    -------
    Benchmark results, for a blocksize of 100 using timeit
    Pure python: 802 ns ± 63.6 ns per loop
    Numba njit decorator on pure python: 1.14 µs ± 41.3 ns per loop
    This version: 884 ns ± 11.9 ns per loop
    """
    blocksize = vector1.shape[1]
    output_vector = np.empty((3, blocksize))

    for i in range(3):
        for k in range(blocksize):
            output_vector[i, k] = vector1[i, k] + vector2[i, k]

    return output_vector


@njit(cache=True)
def _batch_matrix_transpose(input_matrix):
    """
    This function takes an batch input matrix and transpose it.
    Parameters
    ----------
    input_matrix

    Returns
    -------
    Notes
    ----
    Benchmark results,
    Einsum: 2.08 µs ± 553 ns per loop
    This version: 817 ns ± 15.2 ns per loop
    """
    output_matrix = np.empty(input_matrix.shape)
    for i in range(input_matrix.shape[0]):
        for j in range(input_matrix.shape[1]):
            for k in range(input_matrix.shape[2]):
                output_matrix[j, i, k] = input_matrix[i, j, k]
    return output_matrix
