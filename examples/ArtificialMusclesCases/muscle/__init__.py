from examples.ArtificialMusclesCases.muscle.muscle_base import CoiledMuscle
from elastica.experimental.connection_contact_joint.parallel_connection import (
    get_connection_vector_straight_straight_rod,
)
from examples.ArtificialMusclesCases.muscle.connect_straight_rods import (
    ContactSurfaceJoint,
    SurfaceJointSideBySide,
    ParallelJointInterior,
)
from examples.ArtificialMusclesCases.muscle.muscle_forcing import (
    PointSpring,
    MeshRigidBodyPointSpring,
)
from examples.ArtificialMusclesCases.muscle.muscle_library import *
from examples.ArtificialMusclesCases.muscle.artificial_muscle_actuation import (
    ArtficialMuscleActuation,
    ManualArtficialMuscleActuation,
    ArtficialMuscleActuationDecoupled,
)
from examples.ArtificialMusclesCases.muscle.muscle_boundary_conditions import (
    IsometricBC,
    IsometricStrainBC,
    CoilTwistBC,
)
from examples.ArtificialMusclesCases.muscle.memory_block_connections import (
    MemoryBlockConnections,
)
