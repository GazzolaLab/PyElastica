__doc__ = "Time stepper interface"

from typing import Protocol

from elastica.typing import (
    SystemType,
    SteppersOperatorsType,
    OperatorType,
    SystemCollectionType,
)

import numpy as np


class StepperProtocol(Protocol):
    """Protocol for all time-steppers"""

    steps_and_prefactors: SteppersOperatorsType

    def __init__(self) -> None: ...

    def build_step_methods(self) -> SteppersOperatorsType: ...

    def step(
        self, SystemCollection: SystemCollectionType, time: np.float64, dt: np.float64
    ) -> np.float64: ...

    def step_single_instance(
        self, SystemCollection: SystemType, time: np.float64, dt: np.float64
    ) -> np.float64: ...


class SymplecticStepperProtocol(StepperProtocol, Protocol):
    """Symplectic stepper protocol."""

    def get_steps(self) -> list[OperatorType]: ...

    def get_prefactors(self) -> list[OperatorType]: ...

    @staticmethod
    def do_step(
        TimeStepper: "StepperProtocol",
        steps_and_prefactors: SteppersOperatorsType,
        SystemCollection: SystemCollectionType,
        time: np.float64,
        dt: np.float64,
    ) -> np.float64: ...


class ExplicitStepperProtocol(StepperProtocol, Protocol):
    """Explicit stepper protocol."""

    def stage(
        self, System: SystemType, time: np.float64, dt: np.float64
    ) -> np.float64: ...
