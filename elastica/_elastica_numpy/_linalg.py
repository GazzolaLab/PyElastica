__doc__ = """ Convenient linear algebra kernels Numba implementation"""
import numpy as np
import functools
from itertools import permutations
from elastica.utils import perm_parity


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
    return np.einsum("ijk,jlk->ilk", first_matrix_collection, second_matrix_collection)


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


def _batch_dot(first_vector_collection, second_vector_collection):
    """
    This function does batch vector batch vector dot product.
    Parameters
    ----------
    first_vector_collection
    second_vector_collection

    Returns
    -------
    Note
    ----
    Benchmark results, for a blocksize of 100 using timeit
    This version: 3.89 µs ± 815 ns per loop
    Numba version: 938 ns ± 14.1 ns per loop
    """
    return np.einsum("ij,ij->j", first_vector_collection, second_vector_collection)


def _batch_norm(vector_collection):
    return np.sqrt(np.einsum("ij,ij->j", vector_collection, vector_collection))
