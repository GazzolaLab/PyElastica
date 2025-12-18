__doc__ = """
This function is a module to construct memory blocks for different types of systems, such as
Cosserat Rods, Rigid Body etc.
"""

from typing import Type, TYPE_CHECKING
from collections import defaultdict
from elastica.systems.protocol import SystemProtocol

if TYPE_CHECKING:
    from elastica.typing import (
        SystemType,
        StaticSystemType,
        SystemIdxType,
        BlockSystemType,
    )


def construct_memory_block_structures(
    systems: list["StaticSystemType"],
    block_supports: dict[Type["BlockSystemType"], list[Type["SystemType"]]],
) -> tuple[list["BlockSystemType"], list["SystemType"]]:
    """
    This function takes the systems appended to the simulator class and
    separates them into groups based on their block support. Then using
    these grouped systems it creates the memory blocks.

    Parameters
    ----------
    systems : list[StaticSystemType]
        List of systems to be grouped into memory blocks.
    block_supports : dict[Type[BlockSystemType], list[Type[SystemType]]]
        Dictionary mapping block types to the list of system types that support it.

    Returns
    -------
    list[BlockSystemType]
        List of memory block structures created from the systems.

    Notes
    -----
    Systems that don't have an associated block type in the dictionary will be
    skipped (no block constructed), but they are still allowed to be appended
    to the system collection.
    """
    _memory_blocks: list["BlockSystemType"] = []
    _non_blocked_systems: list[SystemProtocol] = []

    # Group systems by their block type
    system_list: dict[Type["BlockSystemType"], list["StaticSystemType"]] = defaultdict(
        list
    )
    index_list: dict[Type["BlockSystemType"], list["SystemIdxType"]] = defaultdict(list)

    for system_idx, system in enumerate(systems):
        # Find the matching system type in block_supports
        block_type = None
        for bt, system_types in block_supports.items():
            if (
                type(system) in system_types
            ):  # Explicit check for *exact* system type, not subclasses.
                block_type = bt
                break

        if block_type is not None:
            # If block type found, group the system
            system_list[block_type].append(system)
            index_list[block_type].append(system_idx)
        elif isinstance(system, SystemProtocol):
            _non_blocked_systems.append(system)

    # Create blocks for each block type
    for block_type, systems_for_block in system_list.items():
        # block_type is a concrete class with constructor (systems, system_idx_list)
        block: BlockSystemType = block_type(systems_for_block, index_list[block_type])
        _memory_blocks.append(block)

    return _memory_blocks, _non_blocked_systems
