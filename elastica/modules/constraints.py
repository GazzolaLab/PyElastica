__doc__ = """
Constraints
-----------

Provides the constraints interface to enforce displacement boundary conditions (see `boundary_conditions.py`).
"""
from typing import Any, Type, cast
from typing_extensions import Self

import functools

import numpy as np

from elastica.boundary_conditions import ConstraintBase

from elastica.typing import (
    SystemIdxType,
    ConstrainingIndex,
    RigidBodyType,
    RodType,
)
from elastica.memory_block.protocol import BlockRodProtocol
from .protocol import ConstrainedSystemCollectionProtocol, ModuleProtocol


class Constraints:
    """
    The Constraints class is a module for enforcing displacement boundary conditions.
    To enforce boundary conditions on rod-like objects, the simulator class
    must be derived from Constraints class.

        Attributes
        ----------
        _constraints: list
            List of boundary condition classes defined for rod-like objects.
    """

    def __init__(self: ConstrainedSystemCollectionProtocol) -> None:
        self._constraints_list: list[ModuleProtocol] = []
        super(Constraints, self).__init__()
        self._feature_group_finalize.append(self._finalize_constraints)

    def constrain(
        self: ConstrainedSystemCollectionProtocol, system: "RodType | RigidBodyType"
    ) -> ModuleProtocol:
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
        sys_idx = self.get_system_index(system)

        # Create _Constraint object, cache it and return to user
        _constraint: ModuleProtocol = _Constraint(sys_idx)
        self._constraints_list.append(_constraint)
        self._feature_group_constrain_values.append_id(_constraint)
        self._feature_group_constrain_rates.append_id(_constraint)

        return _constraint

    def _finalize_constraints(self: ConstrainedSystemCollectionProtocol) -> None:
        """
        In case memory block have ring rod, then periodic boundaries have to be synched. In order to synchronize
        periodic boundaries, a new constrain for memory block rod added called as _ConstrainPeriodicBoundaries. This
        constrain will synchronize the only periodic boundaries of position, director, velocity and omega variables.
        """

        for block in self.block_systems():
            # append the memory block to the simulation as a system. Memory block is the final system in the simulation.
            if hasattr(block, "ring_rod_flag"):
                from elastica._synchronize_periodic_boundary import (
                    _ConstrainPeriodicBoundaries,
                )

                # Apply the constrain to synchronize the periodic boundaries of the memory rod. Find the memory block
                # sys idx among other systems added and then apply boundary conditions.
                memory_block_idx = self.get_system_index(block)
                block_system = cast(BlockRodProtocol, self[memory_block_idx])
                self.constrain(block_system).using(
                    _ConstrainPeriodicBoundaries,
                )

        # From stored _Constraint objects, instantiate the boundary conditions
        # inplace : https://stackoverflow.com/a/1208792

        # dev : the first index stores the rod index to apply the boundary condition
        # to.
        # Sort from lowest id to highest id for potentially better memory access
        # _constraints contains list of tuples. First element of tuple is rod number and
        # following elements are the type of boundary condition such as
        # [(0, ConstraintBase, OneEndFixedBC), (1, HelicalBucklingBC), ... ]
        # Thus using lambda we iterate over the list of tuples and use rod number (x[0])
        # to sort constraints.
        self._constraints_list.sort(key=lambda x: x.id())
        for constraint in self._constraints_list:
            sys_id = constraint.id()
            constraint_instance = constraint.instantiate(self[sys_id])

            constrain_values = functools.partial(
                constraint_instance.constrain_values, system=self[sys_id]
            )
            constrain_rates = functools.partial(
                constraint_instance.constrain_rates, system=self[sys_id]
            )

            self._feature_group_constrain_values.add_operators(
                constraint, [constrain_values]
            )
            self._feature_group_constrain_rates.add_operators(
                constraint, [constrain_rates]
            )

        # At t=0.0, constrain all the boundary conditions (for compatability with
        # initial conditions)
        self.constrain_values(time=np.float64(0.0))
        self.constrain_rates(time=np.float64(0.0))

        self._constraints_list = []
        del self._constraints_list


class _Constraint:
    """
    Constraint module private class

    Attributes
    ----------
    _sys_idx: int
    _bc_cls: Type[ConstraintBase]
    constrained_position_idx: ConstrainingIndex
    constrained_director_idx: ConstrainingIndex
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.
    """

    def __init__(self, sys_idx: SystemIdxType) -> None:
        """

        Parameters
        ----------
        sys_idx: int

        """
        self._sys_idx = sys_idx
        self._bc_cls: Type[ConstraintBase]
        self._args: Any
        self._kwargs: Any
        self.constrained_position_idx: ConstrainingIndex
        self.constrained_director_idx: ConstrainingIndex

    def using(
        self,
        cls: Type[ConstraintBase],
        *args: Any,
        constrained_position_idx: ConstrainingIndex = (),
        constrained_director_idx: ConstrainingIndex = (),
        **kwargs: Any,
    ) -> Self:
        """
        This method is a module to set which boundary condition class is used to
        enforce boundary condition from user defined rod-like objects.

        Parameters
        ----------
        cls : Type[ConstraintBase]
            User defined boundary condition class.
        *args
            Variable length argument list
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------

        """
        assert issubclass(
            cls, ConstraintBase
        ), "{} is not a valid constraint. Constraint must be driven from ConstraintBase.".format(
            cls
        )
        self._bc_cls = cls
        self.constrained_position_idx = constrained_position_idx
        self.constrained_director_idx = constrained_director_idx
        self._args = args
        self._kwargs = kwargs
        return self

    def id(self) -> SystemIdxType:
        return self._sys_idx

    def instantiate(self, system: "RodType | RigidBodyType") -> ConstraintBase:
        """Constructs a constraint after checks"""
        if not hasattr(self, "_bc_cls"):
            raise RuntimeError(
                "No boundary condition provided to constrain rod"
                "id {0} at {1}, but a BC was intended. Did you"
                "forget to call the `using` method?".format(self.id(), system)
            )

        # IMPORTANT : do copy for memory-safe operations
        positions = (
            [
                system.position_collection[..., idx].copy()
                for idx in self.constrained_position_idx
            ]
            if self.constrained_position_idx
            else []
        )
        directors = (
            [
                system.director_collection[..., idx].copy()
                for idx in self.constrained_director_idx
            ]
            if self.constrained_director_idx
            else []
        )
        try:
            bc = self._bc_cls(
                *positions,
                *directors,
                *self._args,
                _system=system,
                constrained_position_idx=self.constrained_position_idx,
                constrained_director_idx=self.constrained_director_idx,
                **self._kwargs,
            )
            return bc
        except (TypeError, IndexError):
            raise TypeError(
                "Unable to construct boundary condition class. Note that:\n"
                "1. Any rod properties needed should be placed first\n"
                "in the boundary_condition __init__ like so (pos_one, pos_two, <other_args>)\n"
                "2. Number of requested position and directors such as (1, 2) should match\n"
                "the __init__ method. eg MyBC.__init__(pos_one, director_one, director_two)\n"
                "should have the `using` call as .using(MyBC, positions=(1,), directors=(1,-1))\n"
            )
