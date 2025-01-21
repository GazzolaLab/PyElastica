from typing import Protocol

from elastica.typing import StepType, StateType
from elastica.systems.protocol import SystemProtocol
from elastica.timestepper.protocol import StepperProtocol

from elastica.rod.data_structures import _KinematicState, _DynamicState

import numpy as np
from numpy.typing import NDArray


class ExplicitSystemProtocol(SystemProtocol, Protocol):
    """
    Protocol for system with state variables
    TODO: Maybe keep it same as symplectic system?
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


class ExplicitStepperProtocol(StepperProtocol, Protocol):
    """Explicit stepper protocol."""

    def stage(
        self, System: ExplicitSystemProtocol, time: np.float64, dt: np.float64
    ) -> np.float64: ...
