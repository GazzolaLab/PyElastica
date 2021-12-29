import warnings
import numpy as np
from elastica._rotations import _get_rotation_matrix, _rotate
from elastica.rod.data_structures import (
    _RodSymplecticStepperMixin,
    _bootstrap_from_data,
    _State,
    _DerivativeState,
    _KinematicState,
    _DynamicState,
)

warnings.warn(
    "The numba-implementation is included in the default elastica module. Please import without _elastica_numba.",
    DeprecationWarning,
)
