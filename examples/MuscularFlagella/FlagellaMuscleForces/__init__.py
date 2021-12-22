__doc__ = """Import flagella muscle forces class depending on presence of Numba in the environment. """
__all__ = ["MuscleForces"]

from elastica import IMPORT_NUMBA

if IMPORT_NUMBA:
    from examples.MuscularFlagella.FlagellaMuscleForces.muscle_forces_flagella_numba import (
        MuscleForces,
    )
else:
    from examples.MuscularFlagella.FlagellaMuscleForces.muscle_forces_flagella_numpy import (
        MuscleForces,
    )
