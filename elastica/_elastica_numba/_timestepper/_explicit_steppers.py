__doc__ = """Explicit timesteppers  and concepts of Elastica Numba implementation"""

import warnings
from elastica.timestepper.explicit_steppers import (
    ExplicitStepperTag,
    RungeKutta4,
    EulerForward,
)


warnings.warn(
    "The numba-implementation is included in the default elastica module. Please import without _elastica_numba.",
    DeprecationWarning,
)
