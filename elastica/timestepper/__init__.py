__doc__ = """Timestepping utilities to be used with Rod and RigidBody classes"""
import numpy as np

from .explicit_steppers import ExplicitStepper
from .symplectic_steppers import SymplecticStepper
from .hybrid_rod_steppers import SymplecticCosseratRodStepper
from ._stepper_interface import _StatefulStepper


def extend_stepper_interface(Stepper, System):
    from ..utils import extend_instance
    from ..systems import is_system_a_collection

    # Check if system is a "collection" of smaller systems
    # by checking for the [] method
    is_this_system_a_collection = is_system_a_collection(System)

    ConcreteStepper = (
        Stepper.stepper if _StatefulStepper in Stepper.__class__.mro() else Stepper
    )

    if SymplecticStepper in ConcreteStepper.__class__.mro():
        from .symplectic_steppers import (
            _SystemInstanceStepperMixin,
            _SystemCollectionStepperMixin,
        )
    elif ExplicitStepper in ConcreteStepper.__class__.mro():
        from .explicit_steppers import (
            _SystemInstanceStepperMixin,
            _SystemCollectionStepperMixin,
        )
    elif SymplecticCosseratRodStepper in ConcreteStepper.__class__.mro():
        return  # hacky fix for now. remove HybridSteppers in a future version.
    else:
        raise NotImplementedError(
            "Only explicit and symplectic steppers are supported, given stepper is {}".format(
                ConcreteStepper.__class__.__name__
            )
        )

    ExtendClass = (
        _SystemCollectionStepperMixin
        if is_this_system_a_collection
        else _SystemInstanceStepperMixin
    )
    extend_instance(ConcreteStepper, ExtendClass)


# TODO Add features to this experimental callback class
class CallBack:
    def __init__(self, step_skip: int, sys_idx, state_key):
        self.every = step_skip
        self.system_idx = sys_idx
        self.state_key = state_key
        self.callable = None

    @classmethod
    def make_callback(cls, state_key):
        return cls(step_skip=1, sys_idx=0, state_key=state_key)

    def will_be_called_at_every(self, step_count: int):
        self.every = step_count

    def needs(self, state_key: str):
        self.state_key = state_key
        return self

    def of(self, sys_idx):
        self.system_idx = sys_idx

    def register(self, callable):
        self.callable = callable


# TODO Improve interface of this function to take args and kwargs for ease of use
def integrate(
    StatefulStepper,
    System,
    final_time: float,
    n_steps: int = 1000,
    callbacks: list = None,
):
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
    # tol = Tolerance.atol()
    # while np.abs(final_time - time) > 1e5 * tol:
    from tqdm import tqdm

    # TODO Remove in a future version, use callback instead
    pos_memory_list = []
    velocity_memory_list = []
    dir_memory_list = []

    for i in tqdm(range(n_steps)):
        time = StatefulStepper.do_step(System, time, dt)

        # TODO Remove in a future version, use callback instead
        # Uncomment if needed, causes tests to fail for
        # simple systems, as expected
        # if i % 1000 == 0:
        #     pos_memory_list.append(System[0].position_collection.copy())
        #     dir_memory_list.append(System[0].director_collection.copy())
        #     velocity_memory_list.append(System[0].velocity_collection.copy())

    print("Final time of simulation is : ", time)
    return pos_memory_list, dir_memory_list, velocity_memory_list
