import warnings
from elastica._rotations import _get_rotation_matrix, _inv_rotate, _rotate

warnings.warn(
    "The numba-implementation is included in the default elastica module. Please import without _elastica_numba.",
    DeprecationWarning,
)
