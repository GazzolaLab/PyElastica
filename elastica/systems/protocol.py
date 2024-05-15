__doc__ = """Base class for elastica system"""

from typing import Protocol, Type
from elastica.typing import StateType, SystemType
from elastica.modules.protocol import ModuleProtocol

from elastica.rod.data_structures import _KinematicState, _DynamicState

import numpy as np
from numpy.typing import NDArray


class SystemProtocol(Protocol):
    """
    Protocol for all elastica system
    """

    REQUISITE_MODULES: list[Type[ModuleProtocol]]

    @property
    def n_nodes(self) -> int: ...

    @property
    def position_collection(self) -> NDArray: ...

    @property
    def velocity_collection(self) -> NDArray: ...

    @property
    def acceleration_collection(self) -> NDArray: ...

    @property
    def omega_collection(self) -> NDArray: ...

    @property
    def alpha_collection(self) -> NDArray: ...

    @property
    def external_forces(self) -> NDArray: ...

    @property
    def external_torques(self) -> NDArray: ...

    def update_internal_forces_and_torques(self, time: np.floating) -> None: ...


class SymplecticSystemProtocol(SystemProtocol, Protocol):
    """
    Protocol for system with symplectic state variables
    """

    @property
    def kinematic_states(self) -> _KinematicState: ...

    @property
    def dynamic_states(self) -> _DynamicState: ...

    @property
    def rate_collection(self) -> NDArray: ...

    @property
    def dvdt_dwdt_collection(self) -> NDArray: ...

    def kinematic_rates(
        self, time: np.floating, prefac: np.floating
    ) -> tuple[NDArray, NDArray]: ...

    def dynamic_rates(
        self, time: np.floating, prefac: np.floating
    ) -> tuple[NDArray]: ...


class ExplicitSystemProtocol(SystemProtocol, Protocol):
    # TODO: Temporarily made to handle explicit stepper.
    # Need to be refactored as the explicit stepper is further developed.
    def __call__(self, time: np.floating, dt: np.floating) -> np.floating: ...
    @property
    def state(self) -> StateType: ...
    @state.setter
    def state(self, state: StateType) -> None: ...
