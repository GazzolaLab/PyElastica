__doc__ = """Symplectic timesteppers and concepts of Elastica Numba implementation"""

import warnings
from elastica.timestepper.symplectic_steppers import (
    SymplecticStepperTag,
    PositionVerlet,
    PEFRL,
)


warnings.warn(
    "The numba-implementation is included in the default elastica module. Please import without _elastica_numba.",
    DeprecationWarning,
)
