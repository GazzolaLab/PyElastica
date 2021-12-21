__doc__ = """ Quadrature and difference kernels """


from elastica import IMPORT_NUMBA

if IMPORT_NUMBA:
    from elastica._elastica_numba._calculus import (
        _trapezoidal,
        _two_point_difference,
        _difference,
        _average,
        _clip_array,
        _isnan_check,
        _trapezoidal_for_block_structure,
        _two_point_difference_for_block_structure,
    )

    position_difference_kernel = _difference
    position_average = _average

else:
    from elastica._elastica_numpy._calculus import (
        _trapezoidal,
        _two_point_difference,
        _clip_array,
        _isnan_check,
        _get_zero_array,
        _trapezoidal_for_block_structure,
        _two_point_difference_for_block_structure,
    )

quadrature_kernel = _trapezoidal
difference_kernel = _two_point_difference
quadrature_kernel_for_block_structure = _trapezoidal_for_block_structure
difference_kernel_for_block_structure = _two_point_difference_for_block_structure
