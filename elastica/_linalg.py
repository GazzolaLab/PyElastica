__doc__ = """ Convenient linear algebra kernels """

from elastica import IMPORT_NUMBA

if IMPORT_NUMBA:
    from elastica._elastica_numba._linalg import (
        _batch_matvec,
        _batch_matmul,
        _batch_cross,
        _batch_vec_oneD_vec_cross,
        _batch_dot,
        _batch_norm,
        _batch_product_i_k_to_ik,
        _batch_product_i_ik_to_k,
        _batch_product_k_ik_to_ik,
        _batch_vector_sum,
        _batch_matrix_transpose,
    )
else:
    from elastica._elastica_numpy._linalg import (
        levi_civita_tensor,
        _batch_matvec,
        _batch_matmul,
        _batch_cross,
        _batch_dot,
        _batch_norm,
    )
