import warnings
from elastica import FreeRod, OneEndFixedRod, HelicalBucklingBC

warnings.warn(
    "The numba-implementation is included in the default elastica module. Please import without _elastica_numba.",
    DeprecationWarning,
)
