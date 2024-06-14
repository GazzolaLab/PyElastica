__doc__ = """Base class for elastica system"""

from typing import Protocol, Type
from elastica.typing import StateType, SystemType

from elastica.rod.data_structures import _KinematicState, _DynamicState

import numpy as np
from numpy.typing import NDArray


class _SystemWithEnergy(Protocol):
    def compute_translational_energy(self) -> NDArray[np.float64]: ...

    def compute_rotational_energy(self) -> NDArray[np.float64]: ...


class _SystemWithCenterOfMass(Protocol):
    def compute_velocity_center_of_mass(self) -> NDArray[np.float64]: ...

    def compute_position_center_of_mass(self) -> NDArray[np.float64]: ...


class SystemProtocol(_SystemWithEnergy, _SystemWithCenterOfMass, Protocol):
    """
    Protocol for all elastica system
    """

    REQUISITE_MODULES: list[Type]

    @property
    def n_nodes(self) -> int: ...

    @property
    def n_elems(self) -> int: ...

    position_collection: NDArray[np.floating]

    velocity_collection: NDArray[np.floating]

    acceleration_collection: NDArray[np.floating]
    director_collection: NDArray[np.floating]

    omega_collection: NDArray[np.floating]
    alpha_collection: NDArray[np.floating]

    internal_forces: NDArray[np.floating]
    internal_torques: NDArray[np.floating]

    external_forces: NDArray[np.floating]
    external_torques: NDArray[np.floating]

    def compute_internal_forces_and_torques(self, time: np.floating) -> None: ...

    def update_accelerations(self, time: np.floating) -> None: ...

    def zeroed_out_external_forces_and_torques(self, time: np.floating) -> None: ...


class SymplecticSystemProtocol(SystemProtocol, Protocol):
    """
    Protocol for system with symplectic state variables
    """

    v_w_collection: NDArray[np.floating]
    dvdt_dwdt_collection: NDArray[np.floating]

    @property
    def kinematic_states(self) -> _KinematicState: ...

    @property
    def dynamic_states(self) -> _DynamicState: ...

    def kinematic_rates(
        self, time: np.floating, prefac: np.floating
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]: ...

    def dynamic_rates(
        self, time: np.floating, prefac: np.floating
    ) -> NDArray[np.floating]: ...


class ExplicitSystemProtocol(SystemProtocol, Protocol):
    # TODO: Temporarily made to handle explicit stepper.
    # Need to be refactored as the explicit stepper is further developed.
    def __call__(self, time: np.floating, dt: np.floating) -> np.floating: ...
    @property
    def state(self) -> StateType: ...
    @state.setter
    def state(self, state: StateType) -> None: ...
    @property
    def n_elems(self) -> int: ...
