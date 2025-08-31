__doc__ = """
This function is a module to construct memory blocks for different types of systems, such as
Cosserat Rods, Rigid Body etc.
"""
from typing import cast, Type
from elastica.typing import (
    SystemType,
    SystemIdxType,
    BlockSystemType,
)

from collections import defaultdict


def construct_memory_block_structures(
    systems: list[SystemType],
    associated_block_types: dict[Type[SystemType], Type[BlockSystemType] | bool],
) -> list[BlockSystemType]:
    """
    This function takes the systems (rod or rigid body) appended to the simulator class and
    separates them into lists depending on if system is Cosserat rod or rigid body. Then using
    these separated out systems it creates the memory blocks for Cosserat rods and rigid bodies.

    Returns
    -------

    """
    _memory_blocks: list[BlockSystemType] = []

    types = []
    system_list = defaultdict(list)
    index_list = defaultdict(list)
    for system_idx, system in enumerate(systems):
        block_type = associated_block_types[type(system)]
        if block_type is False:
            # System is not part of time stepping integration.
            continue
        elif block_type is True:
            # System is part of time stepping integration.
            # No need for block support.
            _memory_blocks.append(cast(BlockSystemType, system))
        else:
            # System is part of time stepping integration.
            # Need for block support.
            types.append(block_type)
            system_list[block_type].append(system)
            index_list[block_type].append(system_idx)

    for block_type in types:
        block = block_type(system_list[block_type], index_list[block_type])
        _memory_blocks.append(block)

    return _memory_blocks
