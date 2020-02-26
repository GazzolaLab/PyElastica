__doc__ = """ Convenient linear algebra kernels """
import numpy as np
from numpy import sqrt

# import numba
# from numba import njit
# import functools
# from itertools import permutations

# from .utils import perm_parity


# @functools.lru_cache(maxsize=1)
# def levi_civita_tensor(dim):
#     """
#
#     Parameters
#     ----------
#     dim
#
#     Returns
#     -------
#
#     """
#     epsilon = np.zeros((dim,) * dim)
#
#     for index_tup in permutations(range(dim), dim):
#         epsilon[index_tup] = perm_parity(list(index_tup))
#
#     return epsilon

try:
    import numba
    from numba import njit

    @njit()
    def _batch_matvec(matrix_collection, vector_collection):
        blocksize = vector_collection.shape[1]
        output_vector = np.zeros((3, blocksize))

        for i in range(3):
            for j in range(3):
                for k in range(blocksize):
                    output_vector[i, k] += (
                        matrix_collection[i, j, k] * vector_collection[j, k]
                    )

        return output_vector

    @njit()
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

    @njit()
    def _batch_cross(first_vector_collection, second_vector_collection):
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

    @njit()
    def _batch_dot(first_vector, second_vector):
        blocksize = first_vector.shape[1]
        output_vector = np.empty((blocksize))

        for k in range(blocksize):
            output_vector[k] = (
                first_vector[0, k] * second_vector[0, k]
                + first_vector[1, k] * second_vector[1, k]
                + first_vector[2, k] * second_vector[2, k]
            )

        return output_vector

    @njit()
    def _batch_norm(vector):
        blocksize = vector.shape[1]
        output_vector = np.empty((blocksize))

        for k in range(blocksize):
            output_vector[k] = sqrt(
                vector[0, k] * vector[0, k]
                + vector[1, k] * vector[1, k]
                + vector[2, k] * vector[2, k]
            )

        return output_vector


except ImportError:
    import functools
    from itertools import permutations
    from .utils import perm_parity

    @functools.lru_cache(maxsize=1)
    def levi_civita_tensor(dim):
        """

        Parameters
        ----------
        dim

        Returns
        -------

        """
        epsilon = np.zeros((dim,) * dim)

        for index_tup in permutations(range(dim), dim):
            epsilon[index_tup] = perm_parity(list(index_tup))

        return epsilon

    def _batch_matvec(matrix_collection, vector_collection):
        return np.einsum("ijk,jk->ik", matrix_collection, vector_collection)

    def _batch_matmul(first_matrix_collection, second_matrix_collection):
        return np.einsum(
            "ijk,jlk->ilk", first_matrix_collection, second_matrix_collection
        )

    def _batch_cross(first_vector_collection, second_vector_collection):
        """

        Parameters
        ----------
        first_vector_collection
        second_vector_collection

        Returns
        -------

        Note
        ----
        If we hardcode np.einsum as follows, the timing data is
        %timeit np.einsum('ijk,jl,kl->il',levi_civita_tensor(3), first_vector_collection, second_vector_collection)
        9.45 µs ± 55.9 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        Using batch_cross, the timing data is
        9.98 µs ± 233 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        For reference, using np.cross as follows:
        %timeit np.cross(first_vector_collection, second_vector_collection, axisa=0, axisb=0).T
        where the transpose is needed because cross switches axes, the microbenchmark is
        42.2 µs ± 3.27 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
        """
        return np.einsum(
            "ijk,jl,kl->il",
            levi_civita_tensor(first_vector_collection.shape[0]),
            first_vector_collection,
            second_vector_collection,
        )


# def _batch_matvec(matrix_collection, vector_collection):
#     return np.einsum("ijk,jk->ik", matrix_collection, vector_collection)


# @njit()
# def _batch_matvec(matrix_collection, vector_collection):
#     """
#     This is batch matrix vector multiplication function. Only
#     batch of 3x3 matrix and 3x1 vector can be multiplied.
#     Parameters
#     ----------
#     matrix_collection
#     vector_collection
#
#     Returns
#     -------
#     Notes
#     Microbenchmark results showed that for a block size of 200,
#     %timeit np.einsum("ijk,jk->ik", matrix_collection, vector_collection)
#     5.72 µs ± 707 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
#     This version is
#     1.76 µs ± 17.5 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
#     """
#     blocksize = vector_collection.shape[1]
#     output_vector = np.empty((3, blocksize))
#
#     for k in range(blocksize):
#         output_vector[0, k] = matrix_collection[0, 0, k] * vector_collection[0, k] + matrix_collection[0, 1, k] * \
#                               vector_collection[1, k] + matrix_collection[0, 2, k] * vector_collection[2, k]
#         output_vector[1, k] = matrix_collection[1, 0, k] * vector_collection[0, k] + matrix_collection[1, 1, k] * \
#                               vector_collection[1, k] + matrix_collection[1, 2, k] * vector_collection[2, k]
#         output_vector[2, k] = matrix_collection[2, 0, k] * vector_collection[0, k] + matrix_collection[2, 1, k] * \
#                               vector_collection[1, k] + matrix_collection[2, 2, k] * vector_collection[2, k]
#
#     return output_vector


# @njit()
# def _batch_matvec(matrix_collection, vector_collection):
#     blocksize = vector_collection.shape[1]
#     output_vector = np.zeros((3, blocksize))
#
#     for i in range(3):
#         for j in range(3):
#             for k in range(blocksize):
#                 output_vector[i, k] += (
#                     matrix_collection[i, j, k] * vector_collection[j, k]
#                 )
#
#     return output_vector


# def _batch_matmul(first_matrix_collection, second_matrix_collection):
#     return np.einsum("ijk,jlk->ilk", first_matrix_collection, second_matrix_collection)


# @njit()
# def _batch_matmul(first_matrix_collection, second_matrix_collection):
#     """
#     This is batch matrix matrix multiplication function. Only batch
#     of 3x3 matrices can be multiplied.
#     Parameters
#     ----------
#     first_matrix_collection
#     second_matrix_collection
#
#     Returns
#     -------
#     Notes
#     Microbenchmark results showed that for a block size of 200,
#     %timeit np.einsum("ijk,jlk->ilk", first_matrix_collection, second_matrix_collection)
#     12.8 µs ± 136 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
#     This version is
#     4.41 µs ± 395 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
#     """
#     blocksize = first_matrix_collection.shape[2]
#     output_matrix = np.zeros((3, 3, blocksize))
#
#     for k in range(blocksize):
#         output_matrix[0, 0, k] = (
#             first_matrix_collection[0, 0, k] * second_matrix_collection[0, 0, k]
#             + first_matrix_collection[0, 1, k] * second_matrix_collection[1, 0, k]
#             + first_matrix_collection[0, 2, k] * second_matrix_collection[2, 0, k]
#         )
#         output_matrix[0, 1, k] = (
#             first_matrix_collection[0, 0, k] * second_matrix_collection[0, 1, k]
#             + first_matrix_collection[0, 1, k] * second_matrix_collection[1, 1, k]
#             + first_matrix_collection[0, 2, k] * second_matrix_collection[2, 1, k]
#         )
#         output_matrix[0, 2, k] = (
#             first_matrix_collection[0, 0, k] * second_matrix_collection[0, 2, k]
#             + first_matrix_collection[0, 1, k] * second_matrix_collection[1, 2, k]
#             + first_matrix_collection[0, 2, k] * second_matrix_collection[2, 2, k]
#         )
#         output_matrix[1, 0, k] = (
#             first_matrix_collection[1, 0, k] * second_matrix_collection[0, 0, k]
#             + first_matrix_collection[1, 1, k] * second_matrix_collection[1, 0, k]
#             + first_matrix_collection[1, 2, k] * second_matrix_collection[2, 0, k]
#         )
#         output_matrix[1, 1, k] = (
#             first_matrix_collection[1, 0, k] * second_matrix_collection[0, 1, k]
#             + first_matrix_collection[1, 1, k] * second_matrix_collection[1, 1, k]
#             + first_matrix_collection[1, 2, k] * second_matrix_collection[2, 1, k]
#         )
#         output_matrix[1, 2, k] = (
#             first_matrix_collection[1, 0, k] * second_matrix_collection[0, 2, k]
#             + first_matrix_collection[1, 1, k] * second_matrix_collection[1, 2, k]
#             + first_matrix_collection[1, 2, k] * second_matrix_collection[2, 2, k]
#         )
#         output_matrix[2, 0, k] = (
#             first_matrix_collection[2, 0, k] * second_matrix_collection[0, 0, k]
#             + first_matrix_collection[2, 1, k] * second_matrix_collection[1, 0, k]
#             + first_matrix_collection[2, 2, k] * second_matrix_collection[2, 0, k]
#         )
#         output_matrix[2, 1, k] = (
#             first_matrix_collection[2, 0, k] * second_matrix_collection[0, 1, k]
#             + first_matrix_collection[2, 1, k] * second_matrix_collection[1, 1, k]
#             + first_matrix_collection[2, 2, k] * second_matrix_collection[2, 1, k]
#         )
#         output_matrix[2, 2, k] = (
#             first_matrix_collection[2, 0, k] * second_matrix_collection[0, 2, k]
#             + first_matrix_collection[2, 1, k] * second_matrix_collection[1, 2, k]
#             + first_matrix_collection[2, 2, k] * second_matrix_collection[2, 2, k]
#         )
#
#     return output_matrix


# def _batch_cross(first_vector_collection, second_vector_collection):
#     """
#
#     Parameters
#     ----------
#     first_vector_collection
#     second_vector_collection
#
#     Returns
#     -------
#
#     Note
#     ----
#     If we hardcode np.einsum as follows, the timing data is
#     %timeit np.einsum('ijk,jl,kl->il',levi_civita_tensor(3), first_vector_collection, second_vector_collection)
#     9.45 µs ± 55.9 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
#     Using batch_cross, the timing data is
#     9.98 µs ± 233 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
#     For reference, using np.cross as follows:
#     %timeit np.cross(first_vector_collection, second_vector_collection, axisa=0, axisb=0).T
#     where the transpose is needed because cross switches axes, the microbenchmark is
#     42.2 µs ± 3.27 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
#     """
#     return np.einsum(
#         "ijk,jl,kl->il",
#         levi_civita_tensor(first_vector_collection.shape[0]),
#         first_vector_collection,
#         second_vector_collection,
#     )


# @njit()
# def _batch_cross(first_vector_collection, second_vector_collection):
#     blocksize = first_vector_collection.shape[1]
#     output_vector = np.empty((3, blocksize))
#
#     for k in range(blocksize):
#         output_vector[0, k] = (
#             first_vector_collection[1, k] * second_vector_collection[2, k]
#             - first_vector_collection[2, k] * second_vector_collection[1, k]
#         )
#
#         output_vector[1, k] = (
#             first_vector_collection[2, k] * second_vector_collection[0, k]
#             - first_vector_collection[0, k] * second_vector_collection[2, k]
#         )
#
#         output_vector[2, k] = (
#             first_vector_collection[0, k] * second_vector_collection[1, k]
#             - first_vector_collection[1, k] * second_vector_collection[0, k]
#         )
#
#     return output_vector
#
#
# @njit()
# def _batch_dot(first_vector, second_vector):
#     blocksize = first_vector.shape[1]
#     output_vector = np.empty((blocksize))
#
#     for k in range(blocksize):
#         output_vector[k] = (
#             first_vector[0, k] * second_vector[0, k]
#             + first_vector[1, k] * second_vector[1, k]
#             + first_vector[2, k] * second_vector[2, k]
#         )
#
#     return output_vector
#
#
# @njit()
# def _batch_norm(vector):
#     blocksize = vector.shape[1]
#     output_vector = np.empty((blocksize))
#
#     for k in range(blocksize):
#         output_vector[k] = sqrt(
#             vector[0, k] * vector[0, k]
#             + vector[1, k] * vector[1, k]
#             + vector[2, k] * vector[2, k]
#         )
#
#     return output_vector
