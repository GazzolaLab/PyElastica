import warnings
from elastica.interaction import (
    AnisotropicFrictionalPlane,
    SlenderBodyTheory,
    # AnisotropicFrictionalPlaneRigidBody,
    # InteractionPlane,
    # InteractionPlaneRigidBody,
    elements_to_nodes_inplace,
    node_to_element_pos_or_vel,
    nodes_to_elements,
    sum_over_elements,
)

warnings.warn(
    "The numba-implementation is included in the default elastica module. Please import without _elastica_numba.",
    DeprecationWarning,
)
