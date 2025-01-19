__doc__ = "Time stepper interface"

from typing import Protocol

from elastica.typing import (
    SteppersOperatorsType,
    StepType,
    SystemCollectionType,
)
from elastica.systems.protocol import SymplecticSystemProtocol

import numpy as np


class StepperProtocol(Protocol):
    """Protocol for all time-steppers"""

    steps_and_prefactors: SteppersOperatorsType

    def __init__(self) -> None: ...

    @property
    def n_stages(self) -> int: ...

    def step_methods(self) -> SteppersOperatorsType: ...

    def step(
        self, SystemCollection: SystemCollectionType, time: np.float64, dt: np.float64
    ) -> np.float64: ...

    def step_single_instance(
        self, System: SymplecticSystemProtocol, time: np.float64, dt: np.float64
    ) -> np.float64: ...


class SymplecticStepperProtocol(StepperProtocol, Protocol):
    """symplectic stepper protocol."""

    def get_steps(self) -> list[StepType]: ...

    def get_prefactors(self) -> list[StepType]: ...
