__doc__ = """
Base System
-----------

Basic coordinating for multiple, smaller systems that have an independently integrable
interface (i.e. works with symplectic or explicit routines `timestepper.py`.)
"""
from collections.abc import MutableSequence
from itertools import chain

from elastica.rod import RodBase
from elastica.rigidbody import RigidBodyBase
from elastica.wrappers.memory_block import construct_memory_block_structures


class BaseSystemCollection(MutableSequence):
    """
    Base System for simulator classes. Every simulation class written by the user
    must be derived from the BaseSystemCollection class; otherwise the simulation will
    proceed.

        Attributes
        ----------
        allowed_sys_types: tuple
            Tuple of allowed type rod-like objects. Here use a base class for objects, i.e. RodBase.
        _systems: list
            List of rod-like objects.
        _features: list
            List of classes acting on the rod-like object, such as external forces classes.

    """

    """
    Developer Note
    -----
    Note
    ----
    We can directly subclass a list for the
    most part, but this is a bad idea, as List is non abstract
    https://stackoverflow.com/q/3945940
    """

    def __init__(self):
        # We need to initialize our mixin classes
        super(BaseSystemCollection, self).__init__()
        # List of system types/bases that are allowed
        self.allowed_sys_types = (RodBase, RigidBodyBase)
        # List of systems to be integrated
        self._systems = []
        # List of feature calls, such as those coming
        # from Controllers, Environments which are
        # tacked on to the SystemCollection in a sim.
        self._features = NotImplemented

    def _check_type(self, sys_to_be_added):
        if not issubclass(sys_to_be_added.__class__, self.allowed_sys_types):
            raise TypeError(
                "{0}\n"
                "is not a system passing validity\n"
                "checks, that can be added into BaseSystem. If you are sure that\n"
                "{0}\n"
                "satisfies all criteria for being a system, please add\n"
                "it using BaseSystem.extend_allowed_types.\n"
                "The allowed types are\n"
                "{1}".format(sys_to_be_added.__class__, self.allowed_sys_types)
            )
        return True

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

    def _get_sys_idx_if_valid(self, sys_to_be_added):
        from numpy import int_ as npint

        n_systems = len(self._systems)  # Total number of systems from mixed-in class

        if isinstance(sys_to_be_added, (int, npint)):
            # 1. If they are indices themselves, check range
            assert (
                -n_systems <= sys_to_be_added < n_systems
            ), "Rod index {} exceeds number of registered rodtems".format(
                sys_to_be_added
            )
            sys_idx = sys_to_be_added
        elif self._check_type(sys_to_be_added):
            # 2. If they are rod objects (most likely), lookup indices
            # index might have some problems : https://stackoverflow.com/a/176921
            try:
                sys_idx = self._systems.index(sys_to_be_added)
            except ValueError:
                raise ValueError(
                    "Rod {} was not found, did you append it to the system?".format(
                        sys_to_be_added
                    )
                )

        return sys_idx

    def finalize(self):
        """
        This method finalizes the simulator class. When it is called, it is assumed that the user has appended
        all rod-like objects to the simulator as well as all boundary conditions, callbacks, etc.,
        acting on these rod-like objects. After the finalize method called,
        the user cannot add new features to the simulator class.

        Returns
        -------

        """

        def get_methods_from_feature_classes(method_name: str):
            methods = [
                [v for (k, v) in cls.__dict__.items() if k.endswith(method_name)]
                for cls in self.__class__.__bases__
            ]
            return list(chain.from_iterable(methods))

        self._features = get_methods_from_feature_classes("__call__")

        self._features_that_constrain_values = get_methods_from_feature_classes(
            "_constrain_values"
        )
        self._features_that_constrain_rates = get_methods_from_feature_classes(
            "_constrain_rates"
        )
        self._callback_features = get_methods_from_feature_classes("_callBack")
        finalize_methods = get_methods_from_feature_classes("_finalize")

        # construct memory block
        self._memory_blocks = construct_memory_block_structures(self._systems)

        for finalize in finalize_methods:
            finalize(self)

    def synchronize(self, time):
        # Calls all , connections, controls etc.
        for feature in self._features:
            feature(self, time)

    def constrain_values(self, time):
        # Calls all constraints, connections, controls etc.
        for feature in self._features_that_constrain_values:
            feature(self, time)

    def constrain_rates(self, time):
        # Calls all constraints, connections, controls etc.
        for feature in self._features_that_constrain_rates:
            feature(self, time)

    def apply_callbacks(self, time, current_step: int):
        # Calls call back functions at the end of time-step
        for feature in self._callback_features:
            feature(self, time, current_step)
