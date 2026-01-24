__doc__ = "Time stepper interface"

from typing import Protocol

from elastica.typing import (
    SystemType,
    SystemCollectionType,
)
from elastica.systems.protocol import SymplecticSystemProtocol

import numpy as np


class StepperProtocol(Protocol):
    """Protocol for all time-steppers"""

    def step(
        self,
        SystemCollection: SystemCollectionType,
        time: np.float64 | float,
        dt: np.float64 | float,
    ) -> np.float64: ...

    def step_single_instance(
        self, System: SystemType, time: np.float64, dt: np.float64
    ) -> np.float64: ...
