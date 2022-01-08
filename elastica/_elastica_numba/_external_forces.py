import warnings
from elastica.external_forces import (
    NoForces,
    GravityForces,
    EndpointForces,
    UniformTorques,
    UniformForces,
    MuscleTorques,
)

warnings.warn(
    "The numba-implementation is included in the default elastica module. Please import without _elastica_numba.",
    DeprecationWarning,
)
