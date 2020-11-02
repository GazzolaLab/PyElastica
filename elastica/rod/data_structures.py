__doc__ = "Data structure wrapper for rod components"
__all__ = [
    "_RodSymplecticStepperMixin",
    "_bootstrap_from_data",
    "_State",
    "_DerivativeState",
    "_KinematicState",
    "_DynamicState",
]
import numpy as np

from elastica._rotations import _get_rotation_matrix, _rotate
from elastica import IMPORT_NUMBA

if IMPORT_NUMBA:
    from elastica._elastica_numba._rod._data_structures import (
        _RodSymplecticStepperMixin,
        _bootstrap_from_data,
        _State,
        _DerivativeState,
        _KinematicState,
        _DynamicState,
    )
else:
    from elastica._elastica_numpy._rod._data_structures import (
        _RodSymplecticStepperMixin,
        _bootstrap_from_data,
        _State,
        _DerivativeState,
        _KinematicState,
        _DynamicState,
    )
