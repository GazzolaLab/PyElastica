__doc__ = """Timestepping utilities to be used with Rod and RigidBody classes"""
__all__ = [
    "integrate",
    "PositionVerlet",
    "PEFRL",
    "RungeKutta4",
    "EulerForward",
    "extend_stepper_interface",
]

import numpy as np
from tqdm import tqdm
from elastica.timestepper.symplectic_steppers import (
    SymplecticStepperTag,
    PositionVerlet,
    PEFRL,
)
from elastica.timestepper.explicit_steppers import (
    ExplicitStepperTag,
    RungeKutta4,
    EulerForward,
)


# TODO: Both extend_stepper_interface and integrate should be in separate file.
# __init__ is probably not an ideal place to have these scripts.
def extend_stepper_interface(Stepper, System):
    from elastica.utils import extend_instance
    from elastica.systems import is_system_a_collection

    # Check if system is a "collection" of smaller systems
    # by checking for the [] method
    is_this_system_a_collection = is_system_a_collection(System)

    """
    # Stateful steppers are no more used so remove them
    ConcreteStepper = (
        Stepper.stepper if _StatefulStepper in Stepper.__class__.mro() else Stepper
    )
    """
    ConcreteStepper = Stepper

    if type(ConcreteStepper.Tag) == SymplecticStepperTag:
        from elastica.timestepper.symplectic_steppers import (
            _SystemInstanceStepper,
            _SystemCollectionStepper,
            SymplecticStepperMethods as StepperMethodCollector,
        )
    elif type(ConcreteStepper.Tag) == ExplicitStepperTag:
        from elastica.timestepper.explicit_steppers import (
            _SystemInstanceStepper,
            _SystemCollectionStepper,
            ExplicitStepperMethods as StepperMethodCollector,
        )
    # elif SymplecticCosseratRodStepper in ConcreteStepper.__class__.mro():
    #    return  # hacky fix for now. remove HybridSteppers in a future version.
    else:
        raise NotImplementedError(
            "Only explicit and symplectic steppers are supported, given stepper is {}".format(
                ConcreteStepper.__class__.__name__
            )
        )

    stepper_methods = StepperMethodCollector(ConcreteStepper)
    do_step_method = (
        _SystemCollectionStepper.do_step
        if is_this_system_a_collection
        else _SystemInstanceStepper.do_step
    )
    return do_step_method, stepper_methods.step_methods()


# TODO Improve interface of this function to take args and kwargs for ease of use
def integrate(
    StatefulStepper,
    System,
    final_time: float,
    n_steps: int = 1000,
    restart_time: float = 0.0,
    progress_bar: bool = True,
    **kwargs,
):
    """

    Parameters
    ----------
    StatefulStepper :
        Stepper algorithm to use.
    System :
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
