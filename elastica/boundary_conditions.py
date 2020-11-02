__doc__ = """ Boundary conditions for rod """
__all__ = ["FreeRod", "OneEndFixedRod", "HelicalBucklingBC"]

from elastica import IMPORT_NUMBA

if IMPORT_NUMBA:
    from elastica._elastica_numba._boundary_conditions import (
        FreeRod,
        OneEndFixedRod,
        HelicalBucklingBC,
    )
else:
    from elastica._elastica_numpy._boundary_conditions import (
        FreeRod,
        OneEndFixedRod,
        HelicalBucklingBC,
    )
