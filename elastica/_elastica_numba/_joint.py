import warnings
from elastica.joint import ExternalContact

warnings.warn(
    "The numba-implementation is included in the default elastica module. Please import without _elastica_numba.",
    DeprecationWarning,
)
