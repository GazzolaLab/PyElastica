__doc__ = """Timestepping utilities to be used with Rod and RigidBody classes of Elastica Numpy implementation"""

import numpy as np

# from ._explicit_steppers import ExplicitStepper
# from ._symplectic_steppers import SymplecticStepper
from ._explicit_steppers import ExplicitStepperTag
from ._symplectic_steppers import SymplecticStepperTag

# from elastica.timesteppers.hybrid_rod_steppers import SymplecticCosseratRodStepper
# from elastica.timestepper._stepper_interface import _StatefulStepper


def extend_stepper_interface(Stepper, System):
    from elastica.utils import extend_instance
    from elastica._elastica_numpy._systems import is_system_a_collection

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
def integrate(StatefulStepper, System, final_time: float, n_steps: int = 1000):
    assert final_time > 0.0, "Final time is negative!"
    assert n_steps > 0, "Number of integration steps is negative!"

    # Extend the stepper's interface after introspecting the properties
    # of the system. If system is a collection of small systems (whose
    # states cannot be aggregated), then stepper now loops over the system
    # state
    do_step, stages_and_updates = extend_stepper_interface(StatefulStepper, System)

    dt = np.float64(float(final_time) / n_steps)
    time = np.float64(0.0)

    from tqdm import tqdm

    for i in tqdm(range(n_steps)):
        time = do_step(StatefulStepper, stages_and_updates, System, time, dt)

    print("Final time of simulation is : ", time)
    return
