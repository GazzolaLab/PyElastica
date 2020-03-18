__doc__ = """ External forcing for rod """
__all__ = [
    "NoForces",
    "GravityForces",
    "EndpointForces",
    "UniformTorques",
    "UniformForces",
    "MuscleTorques",
]

from elastica import IMPORT_NUMBA

if IMPORT_NUMBA:
    from elastica._elastica_numba._external_forces import (
        NoForces,
        GravityForces,
        EndpointForces,
        UniformTorques,
        UniformForces,
        MuscleTorques,
    )

else:
    from elastica._elastica_numpy._external_forces import (
        NoForces,
        GravityForces,
        EndpointForces,
        UniformTorques,
        UniformForces,
        MuscleTorques,
    )
