__doc__ = """Base class for elastica system"""

from typing import Protocol, Type, runtime_checkable

from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class StaticSystemProtocol(Protocol):
    """
    Protocol for all static elastica system. Minimal requirement interface
    to be included in the simulator.
    """

    REQUISITE_MODULES: list[Type]


@runtime_checkable
class SystemProtocol(StaticSystemProtocol, Protocol):
    """
    Protocol for all dynamic elastica system.
    """

    @abstractmethod
    def compute_internal_forces_and_torques(self, time: np.float64) -> None: ...

    @abstractmethod
    def update_accelerations(self, time: np.float64) -> None: ...

    @abstractmethod
    def zeroed_out_external_forces_and_torques(self, time: np.float64) -> None: ...


class SymplecticSystemProtocol(SystemProtocol, Protocol):
    """
    Protocol defining the required interface for symplectic time integration.
    Typically, implementation of these properties are provided in data_structures.py
    for the specific system, and use to build the block structure.

    Any class used with the symplectic timesteppers in :mod:`elastica.timestepper`
    (e.g., :class:`PositionVerlet`, :class:`PEFRL`) must satisfy this protocol.

    The symplectic stepper accesses:
        - ``n_nodes``
        - ``update_kinematics`` and ``update_dynamics``: called by the timestepper

    See Also
    --------
    elastica.timestepper.symplectic_steppers : Symplectic stepper implementations
    elastica.rod.CosseratRod : A concrete implementation satisfying this protocol

    """

    n_nodes: int

    def update_kinematics(self, time: np.float64, prefac: np.float64) -> None:
        """Update kinematic state. Typically called after compute_internal_forces_and_torques."""
        ...

    def update_dynamics(self, time: np.float64, prefac: np.float64) -> None:
        """Update dynamic state. Typically called after ``update_accelerations``."""
        ...
