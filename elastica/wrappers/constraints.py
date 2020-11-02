__doc__ = """
Constraints
-----------

Provides the constraints interface to enforce displacement boundary conditions (see `boundary_conditions.py`).
"""

from elastica.boundary_conditions import FreeRod


class Constraints:
    """
    The Constraints class is a wrapper for enforcing displacement boundary conditions.
    To enforce boundary conditions on rod-like objects, the simulator class
    must be derived from Constraints class.

        Attributes
        ----------
        _constraints: list
            List of boundary condition classes defined for rod-like objects.
    """

    def __init__(self):
        self._constraints = []
        super(Constraints, self).__init__()

    def constrain(self, system):
        """
        This method enforces a displacement boundary conditions to the relevant user-defined
        system or rod-like object. You must input the system or rod-like
        object that you want to enforce boundary condition on.

        Parameters
        ----------
        system: object
            System is a rod-like object.

        Returns
        -------

        """
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
        # _constraints contains list of tuples. First element of tuple is rod number and
        # following elements are the type of boundary condition such as
        # [(0, FreeRod, OneEndFixedRod), (1, HelicalBucklingBC), ... ]
        # Thus using lambda we iterate over the list of tuples and use rod number (x[0])
        # to sort constraints.
        self._constraints.sort(key=lambda x: x[0])

        # At t=0.0, constrain all the boundary conditions (for compatability with
        # initial conditions)
        # TODO: you may need to change naming of _callBC
        self._constrain_values(time=0.0)
        self._constrain_rates(time=0.0)
        # self._callBC(time=0.0)

    # # TODO: same as above naming of _callBC function
    # def _callBC(self, time, *args, **kwargs):
    #     for sys_id, constraint in self._constraints:
    #         constraint.constrain_values(self._systems[sys_id], time, *args, **kwargs)
    #         constraint.constrain_rates(self._systems[sys_id], time, *args, **kwargs)

    def _constrain_values(self, time, *args, **kwargs):
        for sys_id, constraint in self._constraints:
            constraint.constrain_values(self._systems[sys_id], time, *args, **kwargs)

    def _constrain_rates(self, time, *args, **kwargs):
        for sys_id, constraint in self._constraints:
            constraint.constrain_rates(self._systems[sys_id], time, *args, **kwargs)


class _Constraint:
    """
    Constraint wrapper private class

    Attributes
    ----------
    _sys_idx: int
    _bc_cls: list
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.
    """

    def __init__(self, sys_idx: int):
        """

        Parameters
        ----------
        sys_idx: int

        """
        self._sys_idx = sys_idx
        self._bc_cls = None
        self._args = ()
        self._kwargs = {}

    def using(self, bc_cls, *args, **kwargs):
        """
        This method is a wrapper to set which boundary condition class is used to
        enforce boundary condition from user defined rod-like objects.

        Parameters
        ----------
        bc_cls : object
            User defined boundary condition class.
        *args
            Variable length argument list
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------

        """
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
        """Constructs a constraint after checks

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
            "constrained_position_idx", None
        )  # calculate position indices as a tuple
        director_indices = self._kwargs.pop(
            "constrained_director_idx", None
        )  # calculate director indices as a tuple

        # If pos_indices is not None, construct list else empty list
        # IMPORTANT : do copy for memory-safe operations
        positions = (
            [rod.position_collection[..., idx].copy() for idx in pos_indices]
            if pos_indices
            else []
        )
        directors = (
            [rod.director_collection[..., idx].copy() for idx in director_indices]
            if director_indices
            else []
        )
        try:
            return self._bc_cls(*positions, *directors, *self._args, **self._kwargs)
        except (TypeError, IndexError):
            raise TypeError(
                "Unable to construct boundary condition class. Note that:\n"
                "1. Any rod properties needed should be placed first\n"
                "in the boundary_condition __init__ like so (pos_one, pos_two, <other_args>)\n"
                "2. Number of requested position and directors such as (1, 2) should match\n"
                "the __init__ method. eg MyBC.__init__(pos_one, director_one, director_two)\n"
                "should have the `using` call as .using(MyBC, positions=(1,), directors=(1,-1))\n"
            )
