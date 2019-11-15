"""
constraints
-----------

Provides the Constraints interface to enforce boundary conditions (see `boundary_conditions.py`).
"""

from elastica.boundary_conditions import FreeRod


class Constraints:
    def __init__(self):
        self._constraints = []

    def constrain(self, system):
        sys_idx = self._get_sys_idx_if_valid(system)

        # Create _Constraint object, cache it and return to user
        _constraint = _Constraint(sys_idx)
        self._constraints.append(_constraint)

        return _constraint

    def _finalize(self):
        # From stored _Constraint objects, instantiate the boundary conditions
        # inplace : https://stackoverflow.com/a/1208792

        # dev : the first index stores the rod index to apply the boundary condition
        # to. Technically we can use another array but it its one more book-keeping
        # step. Being lazy, I put them both in the same array
        self._constraints[:] = [
            (constraint.id(), constraint(self._systems[constraint.id()]))
            for constraint in self._constraints
        ]

        # Sort from lowest id to highest id for potentially better memory access
        self._constraints.sort()

    def __call__(self, time, *args, **kwargs):
        for sys_id, constraint in self._constraints:
            constraint.constrain_values(self._systems[sys_id], time, *args, **kwargs)
            constraint.constrain_rates(self._systems[sys_id], time, *args, **kwargs)


class _Constraint:
    def __init__(self, sys_idx: int):
        self._sys_idx = sys_idx
        self._bc_cls = None
        self._args = ()
        self._kwargs = {}

    def using(self, bc_cls, *args, **kwargs):
        assert issubclass(
            bc_cls, FreeRod
        ), "{} is not a valid boundary condition. Did you forget to derive from FreeRod?".format(
            bc_cls
        )
        self._bc_cls = bc_cls
        self._args = args
        self._kwargs = kwargs
        return self

    def id(self):
        return self._sys_idx

    def __call__(self, rod, *args, **kwargs):
        """ Constructs a constraint after checks

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        if not self._bc_cls:
            raise RuntimeError(
                "No boundary condition provided to constrain rod"
                "id {0} at {1}, but a BC was intended. Did you"
                "forget to call the `using` method?".format(self.id(), rod)
            )

        # If there is position, director in kwargs, deal with it first
        # Returns None if not found
        pos_indices = self._kwargs.pop(
            "positions", None
        )  # calculate position indices as a tuple
        director_indices = self._kwargs.pop(
            "directors", None
        )  # calculate director indices as a tuple

        # If pos_indices is not None, construct list else empty list
        # IMPORTANT : do copy for memory-safe operations
        positions = (
            [rod.position[..., idx].copy() for idx in pos_indices]
            if pos_indices
            else []
        )
        directors = (
            [rod.directors[..., idx].copy() for idx in director_indices]
            if director_indices
            else []
        )
        try:
            return self._bc_cls(*positions, *directors, *self._args, **self._kwargs)
        except TypeError:
            raise TypeError(
                r"Unable to construct boundary condition class. Note that:\n"
                r"1. Any rod properties needed should be placed first\n"
                r"in the boundary_condition __init__ like so (pos_one, pos_two, <other_args>)\n"
                r"2. Number of requested position and directors such as (1, 2) should match\n"
                r"the __init__ method. eg MyBC.__init__(pos_one, director_one, director_two)\n"
                r"should have the `using` call as .using(MyBC, position=(1,), director=(1,-1))\n"
            )
