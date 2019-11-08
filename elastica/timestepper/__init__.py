__doc__ = """Timestepping utilities to be used with Rod and RigidBody classes"""
import numpy as np

from .explicit_steppers import ExplicitStepper
from .symplectic_steppers import SymplecticStepper
from .hybrid_rod_steppers import SymplecticCosseratRodStepper
from ._stepper_interface import _StatefulStepper


def extend_stepper_interface(Stepper, System):
    from ..utils import extend_instance

    # Check if system is a "collection" of smaller systems
    # by checking for the [] method
    __sys_get_item = getattr(System, "__getitem__", None)
    is_system_a_collection = callable(__sys_get_item)

    ConcreteStepper = (
        Stepper.stepper if _StatefulStepper in Stepper.__class__.mro() else Stepper
    )

    if SymplecticStepper in ConcreteStepper.__class__.mro():
        from .symplectic_steppers import (
            _SystemInstanceStepper,
            _SystemCollectionStepper,
        )
    elif ExplicitStepper in ConcreteStepper.__class__.mro():
        from .explicit_steppers import _SystemInstanceStepper, _SystemCollectionStepper
    elif SymplecticCosseratRodStepper in ConcreteStepper.__class__.mro():
        return  # hacky fix for now. remove HybridSteppers in a future version.
    else:
        raise NotImplementedError(
            "Only explicit and symplectic steppers are supported, given stepper is {}".format(
                ConcreteStepper.__class__.__name__
            )
        )

    ExtendClass = (
        _SystemCollectionStepper if is_system_a_collection else _SystemInstanceStepper
    )
    extend_instance(ConcreteStepper, ExtendClass)


# TODO Improve interface of this function to take args and kwargs for ease of use
def integrate(StatefulStepper, System, final_time: float, n_steps: int = 1000):
    assert final_time > 0.0, "Final time is negative!"
    assert n_steps > 0, "Number of integration steps is negative!"

    from ..utils import Tolerance

    # Extend the stepper's interface after introspecting the properties
    # of the system. If system is a collection of small systems (whose
    # states cannot be aggregated), then stepper now loops over the system
    # state
    extend_stepper_interface(StatefulStepper, System)

    dt = np.float64(float(final_time) / n_steps)
    time = np.float64(0.0)
    tol = Tolerance.atol()
    while np.abs(final_time - time) > 1e5 * tol:
        time = StatefulStepper.do_step(System, time, dt)
