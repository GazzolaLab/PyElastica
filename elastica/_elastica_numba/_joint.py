import warnings

from elastica.joint import (
    FreeJoint,
    HingeJoint,
    FixedJoint,
    ExternalContact,
    SelfContact,
)

warnings.warn(
    "The numba-implementation is included in the default elastica module. Please import without _elastica_numba.",
    DeprecationWarning,
)
