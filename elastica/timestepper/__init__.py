__doc__ = """Timestepping utilities to be used with Rod and RigidBody classes"""

from typing import Tuple, List, Callable, Type

import numpy as np
from tqdm import tqdm

from elastica.typing import SystemType
from elastica.systems import is_system_a_collection

from .symplectic_steppers import PositionVerlet, PEFRL
from .explicit_steppers import RungeKutta4, EulerForward

from .tag import SymplecticStepperTag, ExplicitStepperTag
from .protocol import StepperProtocol, StatefulStepperProtocol
from .protocol import MethodCollectorProtocol


# TODO: Both extend_stepper_interface and integrate should be in separate file.
# __init__ is probably not an ideal place to have these scripts.
def extend_stepper_interface(
    Stepper: StepperProtocol, System: SystemType
) -> Tuple[Callable, Tuple[Callable]]:

    # StepperMethodCollector: Type[MethodCollectorProtocol]
    # SystemStepper: Type[StepperProtocol]
    if isinstance(Stepper.Tag, SymplecticStepperTag):
        from elastica.timestepper.symplectic_steppers import (
            _SystemInstanceStepper,
            _SystemCollectionStepper,
            SymplecticStepperMethods,
        )

        StepperMethodCollector = SymplecticStepperMethods
    elif isinstance(Stepper.Tag, ExplicitStepperTag):  # type: ignore[no-redef]
        from elastica.timestepper.explicit_steppers import (
            _SystemInstanceStepper,
            _SystemCollectionStepper,
            ExplicitStepperMethods,
        )

        StepperMethodCollector = ExplicitStepperMethods
    else:
        raise NotImplementedError(
            "Only explicit and symplectic steppers are supported, given stepper is {}".format(
                Stepper.__class__.__name__
            )
        )

    # Check if system is a "collection" of smaller systems
    if is_system_a_collection(System):
        SystemStepper = _SystemCollectionStepper
    else:
        SystemStepper = _SystemInstanceStepper

    stepper_methods: Tuple[Callable] = StepperMethodCollector(Stepper).step_methods()
    do_step_method: Callable = SystemStepper.do_step
    return do_step_method, stepper_methods


def integrate(
    StatefulStepper: StatefulStepperProtocol,
    System: SystemType,
    final_time: float,
    n_steps: int = 1000,
    restart_time: float = 0.0,
    progress_bar: bool = True,
) -> float:
    """

    Parameters
    ----------
    StatefulStepper : StatefulStepperProtocol
        Stepper algorithm to use.
    System : SystemType
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

    # Extend the stepper's interface after introspecting the properties
    # of the system. If system is a collection of small systems (whose
    # states cannot be aggregated), then stepper now loops over the system
    # state
    do_step, stages_and_updates = extend_stepper_interface(StatefulStepper, System)

    dt = np.float64(float(final_time) / n_steps)
    time = restart_time

    for i in tqdm(range(n_steps), disable=(not progress_bar)):
        time = do_step(StatefulStepper, stages_and_updates, System, time, dt)

    print("Final time of simulation is : ", time)
    return time
