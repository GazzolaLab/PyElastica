import warnings
from elastica._linalg import (
    levi_civita_tensor,
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

warnings.warn(
    "The numba-implementation is included in the default elastica module. Please import without _elastica_numba.",
    DeprecationWarning,
)
