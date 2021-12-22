__doc__ = """Import flagella connection class depending on presence of Numba in the environment. """
__all__ = ["MuscularFlagellConnection"]

from elastica import IMPORT_NUMBA

if IMPORT_NUMBA:
    from examples.MuscularFlagella.MuscularFlagellaConnection.connection_flagella_numba import (
        MuscularFlagellConnection,
    )
else:
    from examples.MuscularFlagella.MuscularFlagellaConnection.connection_flagella_numpy import (
        MuscularFlagellConnection,
    )
