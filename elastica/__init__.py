from collections import defaultdict
from elastica.rod.knot_theory import (
    KnotTheory,
    KnotTheoryCompatibleProtocol,
    compute_link,
    compute_twist,
    compute_writhe,
)
from elastica.rod.rod_base import RodBase
from elastica.rod.cosserat_rod import CosseratRod
from elastica.rigidbody.rigid_body import RigidBodyBase
from elastica.rigidbody.cylinder import Cylinder
from elastica.rigidbody.sphere import Sphere
from elastica.surface.plane import Plane
from elastica.boundary_conditions import (
    ConstraintBase,
    FreeBC,
    OneEndFixedBC,
    GeneralConstraint,
    FixedConstraint,
    HelicalBucklingBC,
    FreeRod,
    OneEndFixedRod,
)
from elastica.external_forces import (
    NoForces,
    EndpointForces,
    GravityForces,
    UniformForces,
    UniformTorques,
    MuscleTorques,
    EndpointForcesSinusoidal,
)
from elastica.interaction import (
    AnisotropicFrictionalPlane,
    InteractionPlane,
    SlenderBodyTheory,
)
from elastica.joint import (
    FreeJoint,
    ExternalContact,
    FixedJoint,
    HingeJoint,
    SelfContact,
)
from elastica.contact_forces import (
    NoContact,
    RodRodContact,
    RodCylinderContact,
    RodSelfContact,
    RodSphereContact,
    RodPlaneContact,
    RodPlaneContactWithAnisotropicFriction,
    CylinderPlaneContact,
)
from elastica.callback_functions import CallBackBaseClass, ExportCallBack, MyCallBack
from elastica.dissipation import (
    DamperBase,
    AnalyticalLinearDamper,
    LaplaceDissipationFilter,
)
from elastica.modules.base_system import BaseSystemCollection
from elastica.modules.callbacks import CallBacks
from elastica.modules.connections import Connections
from elastica.modules.constraints import Constraints
from elastica.modules.forcing import Forcing
from elastica.modules.damping import Damping
from elastica.modules.contact import Contact

from elastica.transformations import inv_skew_symmetrize
from elastica.transformations import rotate
from elastica._calculus import (
    position_difference_kernel,
    position_average,
    quadrature_kernel,
    difference_kernel,
    quadrature_kernel_for_block_structure,
    difference_kernel_for_block_structure,
)
from elastica._linalg import levi_civita_tensor
from elastica.utils import isqrt
from elastica.typing import RodType, SystemType, AllowedContactType
from elastica.timestepper import (
    integrate,
    PositionVerlet,
    PEFRL,
    RungeKutta4,
    EulerForward,
    extend_stepper_interface,
)
from elastica.memory_block.memory_block_rigid_body import MemoryBlockRigidBody
from elastica.memory_block.memory_block_rod import MemoryBlockCosseratRod
from elastica.restart import save_state, load_state
from elastica.mesh.mesh_initializer import Mesh
