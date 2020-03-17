__doc__ = """ Rotation kernels """


from elastica import IMPORT_NUMBA

if IMPORT_NUMBA:
    from elastica._elastica_numba._rotations import (
        _get_rotation_matrix,
        _rotate,
        _inv_rotate,
    )
else:
    from elastica._elastica_numpy._rotations import (
        _generate_skew_map,
        _get_skew_map,
        _get_inv_skew_map,
        _get_diag_map,
        _skew_symmetrize,
        _skew_symmetrize_sq,
        _get_skew_symmetric_pair,
        _inv_skew_symmetrize,
        _get_rotation_matrix,
        _rotate,
        _inv_rotate,
    )
