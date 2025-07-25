__doc__ = """Base class for elastica system"""

from typing import Protocol, Type
from elastica.typing import StateType, SystemType

from elastica.rod.data_structures import _KinematicState, _DynamicState

import numpy as np
from numpy.typing import NDArray


class StaticSystemProtocol(Protocol):
    REQUISITE_MODULES: list[Type]


class SystemProtocol(StaticSystemProtocol, Protocol):
    """
    Protocol for all dynamic elastica system
    """

    def compute_internal_forces_and_torques(self, time: np.float64) -> None: ...

    def update_accelerations(self, time: np.float64) -> None: ...

    def zeroed_out_external_forces_and_torques(self, time: np.float64) -> None: ...


class SlenderBodyGeometryProtocol(Protocol):
    @property
    def n_nodes(self) -> int: ...

    @property
    def n_elems(self) -> int: ...

    position_collection: NDArray[np.float64]
    velocity_collection: NDArray[np.float64]
    acceleration_collection: NDArray[np.float64]

    omega_collection: NDArray[np.float64]
    alpha_collection: NDArray[np.float64]
    director_collection: NDArray[np.float64]

    external_forces: NDArray[np.float64]
    external_torques: NDArray[np.float64]

    internal_forces: NDArray[np.float64]
    internal_torques: NDArray[np.float64]


class SymplecticSystemProtocol(SystemProtocol, SlenderBodyGeometryProtocol, Protocol):
    """
    Protocol for system with symplectic state variables
    """

    v_w_collection: NDArray[np.float64]
    dvdt_dwdt_collection: NDArray[np.float64]

    @property
    def kinematic_states(self) -> _KinematicState: ...

    @property
    def dynamic_states(self) -> _DynamicState: ...

    def kinematic_rates(
        self, time: np.float64, prefac: np.float64
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    def dynamic_rates(
        self, time: np.float64, prefac: np.float64
    ) -> NDArray[np.float64]: ...
