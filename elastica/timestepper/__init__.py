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
