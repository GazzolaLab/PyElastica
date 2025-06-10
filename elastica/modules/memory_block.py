__doc__ = """
This function is a module to construct memory blocks for different types of systems, such as
Cosserat Rods, Rigid Body etc.
"""
from typing import cast
from elastica.typing import (
    RodType,
    RigidBodyType,
    SurfaceType,
    StaticSystemType,
    SystemIdxType,
    BlockSystemType,
)

from elastica.rod.rod_base import RodBase
from elastica.rigidbody.rigid_body import RigidBodyBase
from elastica.surface.surface_base import SurfaceBase
from elastica.memory_block.memory_block_rod import MemoryBlockCosseratRod
from elastica.memory_block.memory_block_rigid_body import MemoryBlockRigidBody


def construct_memory_block_structures(
    systems: list[StaticSystemType],
) -> list[BlockSystemType]:
    """
    This function takes the systems (rod or rigid body) appended to the simulator class and
    separates them into lists depending on if system is Cosserat rod or rigid body. Then using
    these separated out systems it creates the memory blocks for Cosserat rods and rigid bodies.

    Returns
    -------

    """
    _memory_blocks: list[BlockSystemType] = []
    temp_list_for_cosserat_rod_systems: list[RodType] = []
    temp_list_for_rigid_body_systems: list[RigidBodyType] = []
    temp_list_for_cosserat_rod_systems_idx: list[SystemIdxType] = []
    temp_list_for_rigid_body_systems_idx: list[SystemIdxType] = []

    for system_idx, sys_to_be_added in enumerate(systems):

        if isinstance(sys_to_be_added, RodBase):
            rod_system = cast(RodType, sys_to_be_added)
            temp_list_for_cosserat_rod_systems.append(rod_system)
            temp_list_for_cosserat_rod_systems_idx.append(system_idx)

        elif isinstance(sys_to_be_added, RigidBodyBase):
            rigid_body_system = cast(RigidBodyType, sys_to_be_added)
            temp_list_for_rigid_body_systems.append(rigid_body_system)
            temp_list_for_rigid_body_systems_idx.append(system_idx)

        elif isinstance(sys_to_be_added, SurfaceBase):
            pass
            # surface_system = cast(SurfaceType, sys_to_be_added)
            # raise NotImplementedError(
            #     "Surfaces are not yet implemented in memory block construction."
            # )

        else:
            raise TypeError(
                "{0}\n"
                "is not a system passing validity\n"
                "checks for constructing block structure. If you are sure that\n"
                "{0}\n"
                "satisfies all criteria for being a system, please add\n"
                "it here with correct memory block implementation.\n"
                "The allowed types are\n"
                "{1} {2} {3}".format(
                    sys_to_be_added.__class__, RodBase, RigidBodyBase, SurfaceBase
                )
            )

    if temp_list_for_cosserat_rod_systems:
        _memory_blocks.append(
            MemoryBlockCosseratRod(
                temp_list_for_cosserat_rod_systems,
                temp_list_for_cosserat_rod_systems_idx,
            )
        )

    if temp_list_for_rigid_body_systems:
        _memory_blocks.append(
            MemoryBlockRigidBody(
                temp_list_for_rigid_body_systems, temp_list_for_rigid_body_systems_idx
            )
        )

    return list(_memory_blocks)
