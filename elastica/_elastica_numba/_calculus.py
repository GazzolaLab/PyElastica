import warnings
from elastica._calculus import (
    _trapezoidal,
    _two_point_difference,
    _difference,
    _average,
    _clip_array,
    _isnan_check,
    _get_zero_array,
    _trapezoidal_for_block_structure,
    _two_point_difference_for_block_structure,
)

warnings.warn(
    "The numba-implementation is included in the default elastica module. Please import without _elastica_numba.",
    DeprecationWarning,
)

position_difference_kernel = _difference
position_average = _average
quadrature_kernel = _trapezoidal
difference_kernel = _two_point_difference
quadrature_kernel_for_block_structure = _trapezoidal_for_block_structure
difference_kernel_for_block_structure = _two_point_difference_for_block_structure
