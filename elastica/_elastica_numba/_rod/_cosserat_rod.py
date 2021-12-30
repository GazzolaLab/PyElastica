import warnings
from elastica.rod.cosserat_rod import CosseratRod

warnings.warn(
    "The numba-implementation is included in the default elastica module. Please import without _elastica_numba.",
    DeprecationWarning,
)
