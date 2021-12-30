__doc__ = """Timestepping utilities to be used with Rod and RigidBody classes of Elastica Numba implementation"""

import warnings
from elastica.timestepper import extend_stepper_interface, integrate


warnings.warn(
    "The numba-implementation is included in the default elastica module. Please import without _elastica_numba.",
    DeprecationWarning,
)
