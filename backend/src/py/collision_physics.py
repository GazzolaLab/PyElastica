from __future__ import annotations
from abc import ABC, abstractmethod
from elasticapp._memory_block import BlockRodSystem
from elasticapp._collision import CollisionSystem

from .module_collision import CoarseDetectionType, FineDetectionType, BatchingType


class CollisionPhysics(ABC):
    """
    Abstract base class for block-wise collision physics.
    """

    def __init__(
        self,
        coarse_detection: CoarseDetectionType,
        fine_detection: FineDetectionType,
        batching: BatchingType,
    ) -> None:
        super().__init__()
        self.coarse_detection = coarse_detection
        self.fine_detection = fine_detection
        self.batching = batching

    @abstractmethod
    def resolve_collision(self, system: "BlockRodSystem") -> None:
        pass


class LinearSpringDashpot(CollisionPhysics):
    """
    Linear spring-dashpot collision physics.
    """

    def __init__(
        self,
        *args,
        k_normal: float = 1.0,
        eta_normal: float = 0.1,
        k_tangential: float = 0.0,
        eta_tangential: float | None = None,
        friction: float = 0.5,
        detect_every: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        from elasticapp._collision import LinearSpringDashpot as Model

        if eta_tangential is None and k_tangential == 0.0:
            # Use 3-parameter constructor (eta_tangential = eta_normal, k_tangential = 0.0)
            model = Model(k_normal=k_normal, eta_normal=eta_normal, friction=friction)
        elif k_tangential == 0.0:
            # Use 4-parameter constructor (explicit eta_tangential, k_tangential = 0.0)
            model = Model(
                k_normal=k_normal,
                eta_normal=eta_normal,
                eta_tangential=eta_tangential,
                friction=friction,
            )
        else:
            # Use 5-parameter constructor (explicit k_tangential and eta_tangential)
            if eta_tangential is None:
                eta_tangential = eta_normal  # Default to eta_normal if not provided
            model = Model(
                k_normal=k_normal,
                eta_normal=eta_normal,
                k_tangential=k_tangential,
                eta_tangential=eta_tangential,
                friction=friction,
            )
        self.collision_system = CollisionSystem(model, detect_every=detect_every)

    def resolve_collision(self, system: "BlockRodSystem") -> None:
        self.collision_system.resolve(system)


class NoInteraction(CollisionPhysics):
    """
    NoInteraction collision physics model for testing purposes.

    This model returns zero force for all contacts, allowing collision detection
    to be tested without applying any forces. Useful for:
    - Testing collision detection algorithms
    - Validating contact geometry
    - Debugging collision pipeline without force effects
    """

    def __init__(self, *args, detect_every: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        from elasticapp._collision import NoInteraction as Model

        model = Model()
        self.collision_system = CollisionSystem(model, detect_every=detect_every)

    def resolve_collision(self, system: "BlockRodSystem") -> None:
        self.collision_system.resolve(system)
