"""
base_system
-----------

basic coordinating multiple, smaller systems that have an independently integrable
interface (ie. works with symplectic or explicit routines `timestepper.py`.)
"""
from collections.abc import MutableSequence

from elastica._rod import RodBase


class BaseSystemCollection(MutableSequence):
    """
    Base System

    Technical note : We can directly subclass a list for the
    most part, but this is a bad idea, as List is non abstract
    https://stackoverflow.com/q/3945940
    """

    def __init__(self):
        # We need to initialize our mixin classes
        super(BaseSystemCollection, self).__init__()
        # List of system types/bases that are allowed
        self.allowed_sys_types = (RodBase,)
        # List of systems to be integrated
        self._systems = []

    def _check_type(self, sys_to_be_added):
        assert issubclass(sys_to_be_added.__class__, self.allowed_sys_types), (
            "{0} is not a valid system that can be added into "
            "BaseSystem. If you are sure that {0} satisfies all"
            "promises of a system, add it using "
            "BaseSystem.extend_allowed_types".format(sys_to_be_added.__class__)
        )

    def __len__(self):
        return len(self._systems)

    def __getitem__(self, idx):
        return self._systems[idx]

    def __delitem__(self, idx):
        del self._systems[idx]

    def __setitem__(self, idx, system):
        self._check_type(system)
        self._systems[idx] = system

    def insert(self, idx, system):
        self._check_type(system)
        self._systems.insert(idx, system)

    def __str__(self):
        return str(self._systems)

    def extend_allowed_types(self, additional_types):
        self.allowed_sys_types += additional_types

    def override_allowed_types(self, allowed_types):
        self.allowed_sys_types = allowed_types

    def synchronize(self, time):
        pass
