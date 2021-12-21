__doc__ = """
This function is a wrapper to construct memory blocks for different types of systems, such as
Cosserat Rods, Rigid Body etc.
"""

from elastica.rod import RodBase
from elastica.rigidbody import RigidBodyBase
from elastica.memory_block import MemoryBlockCosseratRod, MemoryBlockRigidBody


def construct_memory_block_structures(systems):
    """
    This function takes the systems (rod or rigid body) appended to the simulator class and
    separates them into lists depending on if system is Cosserat rod or rigid body. Then using
    these separated out systems it creates the memory blocks for Cosserat rods and rigid bodies.

    Returns
    -------

    """
    _memory_blocks = []
    temp_list_for_cosserat_rod_systems = []
    temp_list_for_rigid_body_systems = []

    for system_idx, sys_to_be_added in enumerate(systems):

        if issubclass(sys_to_be_added.__class__, RodBase):
            temp_list_for_cosserat_rod_systems.append(sys_to_be_added)

        elif issubclass(sys_to_be_added.__class__, RigidBodyBase):
            temp_list_for_rigid_body_systems.append(sys_to_be_added)

        else:
            raise TypeError(
                "{0}\n"
                "is not a system passing validity\n"
                "checks for constructing block structure. If you are sure that\n"
                "{0}\n"
                "satisfies all criteria for being a system, please add\n"
                "it here with correct memory block implementation.\n"
                "The allowed types are\n"
                "{1} {2}".format(sys_to_be_added.__class__, RodBase, RigidBodyBase)
            )

    if temp_list_for_cosserat_rod_systems:
        _memory_blocks.append(
            MemoryBlockCosseratRod(temp_list_for_cosserat_rod_systems)
        )

    if temp_list_for_rigid_body_systems:
        _memory_blocks.append(MemoryBlockRigidBody(temp_list_for_rigid_body_systems))

    return _memory_blocks
