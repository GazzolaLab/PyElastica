from typing import Iterator, TypeVar, Generic, Type
from elastica.timestepper.protocol import ExplicitStepperProtocol
from elastica.typing import SystemCollectionType

from copy import copy


def is_system_a_collection(system: object) -> bool:
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
    return issubclass(system.__class__, BaseSystemCollection) or callable(
        __sys_get_item
    )


# FIXME: Move memory related functions to separate module or as part of the timestepper


# TODO: Use MemoryProtocol
def make_memory_for_explicit_stepper(
    stepper: ExplicitStepperProtocol, system: SystemCollectionType
) -> "MemoryCollection":
    # TODO Automated logic (class creation, memory management logic) agnostic of stepper details (RK, AB etc.)

    from elastica.timestepper.explicit_steppers import (
        RungeKutta4,
        EulerForward,
    )

    # is_this_system_a_collection = is_system_a_collection(system)

    memory_cls: Type
    if RungeKutta4 in stepper.__class__.mro():
        # Bad way of doing it, introduces tight coupling
        # this should rather be taken from the class itself
        class MemoryRungeKutta4:
            def __init__(self) -> None:
                self.initial_state = None
                self.k_1 = None
                self.k_2 = None
                self.k_3 = None
                self.k_4 = None

        memory_cls = MemoryRungeKutta4
    elif EulerForward in stepper.__class__.mro():

        class MemoryEulerForward:
            def __init__(self) -> None:
                self.initial_state = None
                self.k = None

        memory_cls = MemoryEulerForward
    else:
        raise NotImplementedError("Making memory for other types not supported")

    return MemoryCollection(memory_cls(), len(system))


M = TypeVar("M", bound="MemoryCollection")


class MemoryCollection(Generic[M]):
    """Slots of memories for timestepper in a cohesive unit.

    A `MemoryCollection` object is meant to be used in conjunction
    with a `SystemCollection`, where each independent `System` to
    be integrated has its own `Memory`.

    Example
    -------

    A RK4 integrator needs to store k_1, k_2, k_3, k_4 (intermediate
    results from four stages) for each `System`. The restriction for
    having a memory slot arises because the `Systems` are usually
    not independent of one another and may need communication after
    every stage.
    """

    def __init__(self, memory: M, n_memory_slots: int):
        super(MemoryCollection, self).__init__()

        self.__memories: list[M] = []
        for _ in range(n_memory_slots - 1):
            self.__memories.append(copy(memory))
        self.__memories.append(memory)

    def __getitem__(self, idx: int) -> M:
        return self.__memories[idx]

    def __len__(self) -> int:
        return len(self.__memories)

    def __iter__(self) -> Iterator[M]:
        return self.__memories.__iter__()
