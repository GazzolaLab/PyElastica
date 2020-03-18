__doc__ = """Timestepping utilities to be used with Rod and RigidBody classes"""
__all__ = ["integrate", "PositionVerlet", "PEFRL", "RungeKutta4", "EulerForward"]
from elastica.timestepper.symplectic_steppers import PositionVerlet, PEFRL
from elastica.timestepper.explicit_steppers import RungeKutta4, EulerForward

# import numpy as np

# from .explicit_steppers import ExplicitStepper
# from .symplectic_steppers import SymplecticStepper
# from .explicit_steppers import ExplicitStepperTag
# from .symplectic_steppers import SymplecticStepperTag

# from .hybrid_rod_steppers import SymplecticCosseratRodStepper
# from ._stepper_interface import _StatefulStepper

from elastica import IMPORT_NUMBA

if IMPORT_NUMBA:
    from elastica._elastica_numba._timestepper import (
        extend_stepper_interface,
        integrate,
    )
else:
    from elastica._elastica_numpy._timestepper import (
        extend_stepper_interface,
        integrate,
    )


#
# def extend_stepper_interface(Stepper, System):
#     from ..utils import extend_instance
#     from ..systems import is_system_a_collection
#
#     # Check if system is a "collection" of smaller systems
#     # by checking for the [] method
#     is_this_system_a_collection = is_system_a_collection(System)
#
#     ConcreteStepper = (
#         Stepper.stepper if _StatefulStepper in Stepper.__class__.mro() else Stepper
#     )
#     try:
#         # from numba import typeof
#         # In order to by pass jit classes try to import something
#         from numba import something, typeof
#
#         if typeof(ConcreteStepper.Tag) == SymplecticStepperTag.class_type.instance_type:
#             from .symplectic_steppers import (
#                 _SystemInstanceStepper,
#                 _SystemCollectionStepper,
#                 SymplecticStepperMethods as StepperMethodCollector,
#             )
#         elif typeof(ConcreteStepper.Tag) == ExplicitStepperTag.class_type.instance_type:
#             from .explicit_steppers import (
#                 _SystemInstanceStepper,
#                 _SystemCollectionStepper,
#                 ExplicitStepperMethods as StepperMethodCollector,
#             )
#         else:
#             raise NotImplementedError(
#                 "Only explicit and symplectic steppers are supported, given stepper is {}".format(
#                     ConcreteStepper.__class__.__name__
#                 )
#             )
#     except ImportError:
#         if type(ConcreteStepper.Tag) == SymplecticStepperTag:
#             from .symplectic_steppers import (
#                 _SystemInstanceStepper,
#                 _SystemCollectionStepper,
#                 SymplecticStepperMethods as StepperMethodCollector,
#             )
#         elif type(ConcreteStepper.Tag) == ExplicitStepperTag:
#             from .explicit_steppers import (
#                 _SystemInstanceStepper,
#                 _SystemCollectionStepper,
#                 ExplicitStepperMethods as StepperMethodCollector,
#             )
#         # elif SymplecticCosseratRodStepper in ConcreteStepper.__class__.mro():
#         #    return  # hacky fix for now. remove HybridSteppers in a future version.
#         else:
#             raise NotImplementedError(
#                 "Only explicit and symplectic steppers are supported, given stepper is {}".format(
#                     ConcreteStepper.__class__.__name__
#                 )
#             )
#
#     stepper_methods = StepperMethodCollector(ConcreteStepper)
#     do_step_method = (
#         _SystemCollectionStepper.do_step
#         if is_this_system_a_collection
#         else _SystemInstanceStepper.do_step
#     )
#     return do_step_method, stepper_methods.step_methods()


# # TODO Improve interface of this function to take args and kwargs for ease of use
# def integrate(
#     StatefulStepper, System, final_time: float, n_steps: int = 1000,
# ):
#     assert final_time > 0.0, "Final time is negative!"
#     assert n_steps > 0, "Number of integration steps is negative!"
#
#     # Extend the stepper's interface after introspecting the properties
#     # of the system. If system is a collection of small systems (whose
#     # states cannot be aggregated), then stepper now loops over the system
#     # state
#     do_step, stages_and_updates = extend_stepper_interface(StatefulStepper, System)
#     print(stages_and_updates)
#
#     dt = np.float64(float(final_time) / n_steps)
#     time = np.float64(0.0)
#
#     from tqdm import tqdm
#
#     for i in tqdm(range(n_steps)):
#         time = do_step(StatefulStepper, stages_and_updates, System, time, dt)
#
#     print("Final time of simulation is : ", time)
#     return
