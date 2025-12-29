from __future__ import annotations

__doc__ = """
CollisionEnvironment
--------------------

Provides the collision environment interface to configure collision detection and
detection resolution for Cosserat rods in the simulation.
"""
__all__ = ["CollisionEnvironment"]
from typing import Any, Type, Literal
import functools

from elastica.external_forces import NoForces
from elastica.typing import SystemType, SystemIdxType
from elastica.systems.protocol import SystemProtocol
from elastica.systems.protocol import SystemCollectionProtocol, ModuleProtocol

from .memory_block_rod import MemoryBlockCosseratRod
from .collision_physics import CollisionPhysics

CoarseDetectionType = Literal["hash_grid"]
FineDetectionType = Literal["sphere_sphere"]
BatchingType = Literal["union_find"]  # , "single_batch", "hybrid_batch"]


class CollisionEnvironment(SystemCollectionProtocol):
    """
    The CollisionEnvironment class enables collision detection and resolution for Cosserat rods in the simulation.
    By including CollisionEnvironment as a mixin, the simulator gains support for discrete element method (DEM)-based contact handling.
    Use this environment together with C++ backend support to activate and configure collision pipeline stages.

    Notes
    -----
    Currently, the CollisionEnvironment is only compatible when used with the
    C++ backend.
    Currently, the feature does not support variations in the collision detection
    methods, as it requires re-compilation of the C++ backend. This must be done
    by the user.

    Examples
    --------
    User can include the CollisionEnvironment in the simulator class to enable
    collision detection and reaction-forces for all Cosserat rods.

    >>> import elastica as ea
    >>> import elasticapp as epp
    ...
    >>> # Include the CollisionEnvironment in the simulator class
    >>> class Simulator(ea.BaseSystemCollection, epp.CollisionEnvironment, ...):
    ...     pass
    ...
    >>> simulator = Simulator()
    >>> # Enable the C++ block supports for the Cosserat rods
    >>> simulator.enable_block_supports(ea.CosseratRod, epp.MemoryBlockCosseratRod)

    To change the configuration of the collision physics, use `.configure_collision`.

    >>> simulator \
        .configure_collision_detection() \
        .using(
            epp.LinearSpringDashpot,
            k_normal=1.0,
            eta_normal=0.1,
            friction=0.5
        )

    Attributes
    ----------
    _ext_forces_torques: list
        List of forcing class defined for rod-like objects.
    """

    def __init__(self) -> None:
        super().__init__()
        self._feature_group_finalize.append(self._finalize_block_collision)

        # Initial configuration
        self._collision_coarse_detection: CoarseDetectionType
        self._collision_fine_detection: FineDetectionType
        self._collision_batching: BatchingType
        self._collision_controller: "_CollisionController"
        self.configure_collision_detection()

    def configure_collision_detection(
        self,
        coarse_detection: CoarseDetectionType = "hash_grid",
        fine_detection: FineDetectionType = "sphere_sphere",
        batching: BatchingType = "union_find",
    ) -> ModuleProtocol:
        """
        This method applies external forces and torques on the relevant
        user-defined system or rod-like object. You must input the system
        or rod-like object that you want to apply external forces and torques on.

        Parameters
        ----------
        coarse_detection: CoarseDetectionType
            The coarse detection algorithm to use.
        fine_detection: FineDetectionType
            The fine detection algorithm to use.
        batching: BatchingType
            The batching algorithm to use.
        """
        self._collision_coarse_detection = coarse_detection
        self._collision_fine_detection = fine_detection
        self._collision_batching = batching

        # Create _Constraint object, cache it and return to user
        self._collision_controller = _CollisionController()
        self._feature_group_synchronize.append_id(self._collision_controller)

        return self._collision_controller

    def _finalize_block_collision(self) -> None:
        controller = self._collision_controller.instantiate()
        del self._collision_controller
        # Find block rods in the system
        for sys in self.final_systems():
            if isinstance(sys, MemoryBlockCosseratRod):
                block = sys._block
                break
        else:
            raise ValueError(
                "No block rods found in the system. Collision requies at least one rod in the simulator."
            )

        compute_collision = functools.partial(
            controller.resolve_collision, system=block
        )

        self._feature_group_synchronize.add_operators(controller, [compute_collision])

        return controller


class _CollisionController:
    """
    Collision controller module private class
    """

    _forcing_cls: Type[NoForces]
    _args: Any
    _kwargs: Any

    def using(self, cls: Type[NoForces], *args: Any, **kwargs: Any) -> None:
        """
        This method sets which forcing class is used to apply forcing
        to user defined rod-like objects.

        Parameters
        ----------
        cls: Type[Any]
            User defined forcing class.
        *args: Any
            Variable length argument list.
        **kwargs: Any
            Arbitrary keyword arguments.

        Returns
        -------

        """
        assert issubclass(
            cls, CollisionPhysics
        ), "{} is not a valid collision physics. Did you forget to derive from CollisionPhysics?".format(
            cls
        )
        self._cls = cls
        self._args = args
        self._kwargs = kwargs

    def id(self) -> SystemIdxType:
        return None  # type: ignore

    def instantiate(self) -> "CollisionPhysics":
        """Constructs a constraint after checks"""
        if not hasattr(self, "_cls"):
            raise RuntimeError(
                "No collision physics provided to act on block rod"
                "but a collision physics was registered. Did you forget to call"
                "the `using` method"
            )

        try:
            return self._cls(*self._args, **self._kwargs)
        except (TypeError, IndexError):
            raise TypeError(
                r"Unable to construct collision physics class.\n"
                r"Did you provide all necessary collision physics parameters?"
            )
