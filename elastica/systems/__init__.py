from typing import Type

from elastica.typing import SystemType, SystemCollectionType


def is_system_a_collection(system: "SystemType | SystemCollectionType") -> bool:
    # Check if system is a "collection" of smaller systems
    # by checking for the [] method
    """

    Parameters
    ----------
    system: object

    Returns
    -------

    """
    """
    Developer Note
    --------------
    Motivation :
    System collection is useful you have a collection of rods (small
    systems) that can be independently integrated for the
    most part, you can lump the into one larger system (`system_collection`).
    Then the collection is only responsible for coordination
    (how to communicate between these systems/rods for,
    say, utilizing a joint).

    Checking :
    Why do we check `__getitiem__` as attribute?

    That's one of the requirements for a system collection.
    There can be others (see the latest PR #24), which will
    evolve as the interface evolves. Then we can add those
    requirements on the interface here.
    """
    from elastica.modules import BaseSystemCollection

    __sys_get_item = getattr(system, "__getitem__", None)
    return isinstance(system, BaseSystemCollection) or callable(__sys_get_item)
