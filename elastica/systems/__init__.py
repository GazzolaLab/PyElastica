def is_system_a_collection(system):
    # Check if system is a "collection" of smaller systems
    # by checking for the [] method
    __sys_get_item = getattr(system, "__getitem__", None)
    return callable(__sys_get_item)


def make_memory_for_explicit_stepper(stepper, system):
    # TODO Automated logic (class creation, memory management logic) agnostic of stepper details (RK, AB etc.)

    from ..timestepper.explicit_steppers import RungeKutta4

    is_this_system_a_collection = is_system_a_collection(system)

    if RungeKutta4 in stepper.__class__.mro():
        # Bad way of doing it, introduces tight coupling
        # this should rather be taken from the class itself
        class MemoryRungeKutta4:
            def __init__(self):
                super(MemoryRungeKutta4, self).__init__()
                self.initial_state = None
                self.k_1 = None
                self.k_2 = None
                self.k_3 = None
                self.k_4 = None

        memory_cls = MemoryRungeKutta4
    else:
        raise NotImplementedError("Making memory for other types not supported")

    return (
        MemoryCollection(memory_cls(), len(system))
        if is_this_system_a_collection
        else memory_cls()
    )


class MemoryCollection:
    def __init__(self, memory, n_memory_slots):
        super(MemoryCollection, self).__init__()

        self.__memories = [None] * n_memory_slots

        from copy import copy

        for i_slot in range(n_memory_slots - 1):
            self.__memories[i_slot] = copy(memory)

        # Save final copy
        self.__memories[-1] = memory

    def __getitem__(self, idx):
        return self.__memories[idx]

    def __len__(self):
        return len(self.__memories)

    def __iter__(self):
        return self.__memories.__iter__()
