__doc__ = """Timestepping utilities to be used with Rod and RigidBody classes"""

from typing import Tuple, List, Callable, Type, Any
from elastica.typing import SystemType, SystemCollectionType, SteppersOperatorsType

import numpy as np
from tqdm import tqdm

from elastica.systems import is_system_a_collection

from .symplectic_steppers import PositionVerlet, PEFRL
from .explicit_steppers import RungeKutta4, EulerForward

from .tag import SymplecticStepperTag, ExplicitStepperTag
from .protocol import StepperProtocol, SymplecticStepperProtocol


# Deprecated: Remove in the future version
# Many script still uses this method to control timestep. Keep it for backward compatibility
def extend_stepper_interface(
    Stepper: StepperProtocol, System: SystemType | SystemCollectionType
) -> Tuple[
    Callable[
        [StepperProtocol, SystemCollectionType, np.floating, np.floating], np.floating
    ],
    SteppersOperatorsType,
]:
    # Check if system is a "collection" of smaller systems
    assert is_system_a_collection(System)

    stepper_methods: SteppersOperatorsType = Stepper.step_methods()
    do_step_method: Callable = Stepper.do_step  # type: ignore[attr-defined]
    return do_step_method, stepper_methods


def integrate(
    Stepper: StepperProtocol,
    SystemCollection: SystemCollectionType,
    final_time: float,
    n_steps: int = 1000,
    restart_time: float = 0.0,
    progress_bar: bool = True,
) -> np.floating:
    """

    Parameters
    ----------
    Stepper : StepperProtocol
        Stepper algorithm to use.
    SystemCollection : SystemType
        The elastica-system to simulate.
    final_time : float
        Total simulation time. The timestep is determined by final_time / n_steps.
    n_steps : int
        Number of steps for the simulation. (default: 1000)
    restart_time : float
        The timestamp of the first integration step. (default: 0.0)
    progress_bar : bool
        Toggle the tqdm progress bar. (default: True)
    """
    assert final_time > 0.0, "Final time is negative!"
    assert n_steps > 0, "Number of integration steps is negative!"
    assert is_system_a_collection(SystemCollection)

    dt = np.float_(float(final_time) / n_steps)
    time = np.float_(restart_time)

    for i in tqdm(range(n_steps), disable=(not progress_bar)):
        time = Stepper.step(SystemCollection, time, dt)

    print("Final time of simulation is : ", time)
    return time
