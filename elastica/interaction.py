__doc__ = """ Interaction module """
__all__ = [
    "AnisotropicFrictionalPlane",
    # "AnisotropicFrictionalPlaneRigidBody",
    "SlenderBodyTheory",
]
from elastica import IMPORT_NUMBA

if IMPORT_NUMBA:
    from elastica._elastica_numba._interaction import (
        AnisotropicFrictionalPlane,
        InteractionPlane,
        # AnisotropicFrictionalPlaneRigidBody,
        # InteractionPlaneRigidBody,
        SlenderBodyTheory,
    )
else:
    from elastica._elastica_numpy._interaction import (
        AnisotropicFrictionalPlane,
        InteractionPlane,
        # AnisotropicFrictionalPlaneRigidBody,
        # InteractionPlaneRigidBody,
        SlenderBodyTheory,
    )
