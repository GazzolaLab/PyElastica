__doc__ = "Hybrid rod steppers"

import numpy as np
from itertools import chain

from .explicit_steppers import ExplicitLinearExponentialIntegrator, ExplicitStepper
from .symplectic_steppers import (
    SymplecticLinearExponentialIntegrator,
    SymplecticStepper,
    PositionVerlet,
)
from ..utils import grouper


class CosseratRodStepper:
    def __init__(self, stepper):
        self.stepper = stepper
        if SymplecticStepper in stepper.__bases__:
            self.linear_stepper = SymplecticLinearExponentialIntegrator
        elif ExplicitStepper in stepper.__bases__:
            self.linear_stepper = ExplicitLinearExponentialIntegrator


class SymplecticCosseratRodStepper:
    """
    Follows the facade pattern
    Steps in a hybrid fashion for time-marching the dynamics of Cosserat Rods
    1. Does
    """

    def __init__(self, symplectic_stepper=PositionVerlet()):
        self.stepper = symplectic_stepper
        self.linear_stepper = SymplecticLinearExponentialIntegrator()

        """ We need to mix stepper and linear_stepper here,
        where linear_stepper "follows" the pattern set by stepper
        Pattern
        -------
        Stepper : A B C D C B A
        LinearStepper : X
        HybridStepper : X A B X C D X C B X A
        """
        # Adapted from https://stackoverflow.com/a/31040952
        # Interval to interleave LinearStepper into HybridStepper
        # Typically 2, for one kinematic + dynamic step
        interleave_interval = 2
        self._steps = self.linear_stepper._steps + list(
            chain(
                *[
                    self.stepper._steps[i : i + interleave_interval]
                    + self.linear_stepper._steps
                    if len(self.stepper._steps[i : i + interleave_interval])
                    == interleave_interval
                    else self.stepper._steps[i : i + interleave_interval]
                    for i in range(0, self.stepper.n_stages, interleave_interval)
                ]
            )
        )

        """ HybridStepper after reversal:  A X B C X D C X B A X
        which is the order we want as exponential always follows symplectic"""
        self._steps.reverse()

        self.__n_steps = len(self._steps)

        """ We now group it into (A,X,B), (C,X,D), .... , (A,X)
        to use it in the time loop. Note that the last component always contains
        the kinematic step and needs to be independently applied
        """
        step_group_size = 3  # One kinematic step, one exp. step, one dynamic step
        self._steps = list(grouper(self._steps, step_group_size))

        # Last step is (A, X) so we pop it
        self._last_kinematic_step, self._last_exponential_step = self._steps.pop()
        # self._steps = tuple(self._steps)

        self._prefactors = self.stepper._prefactors.copy()
        prefactor_group_size = (
            2  # One prefactor for kin. + exp. step, one for dyn. step
        )
        self._prefactors = list(grouper(self._prefactors, prefactor_group_size))

        # Last step is one prefactor (prefaca, ) for (A, X) so we pop it
        (self._last_kin_prefactor,) = self._prefactors.pop()
        # self._prefactors = tuple(self._prefactors)

        self._steps_and_prefactors = tuple(zip(self._prefactors, self._steps))

    @property
    def n_stages(self):
        return self.__n_steps

    def do_step(self, System, time: np.float64, dt: np.float64):
        """
        Parameters
        ----------
        System: rod object
        time: float
        dt: float

        Returns
        -------

        Caveats
        -------
        2x SLOWER than an equivalent symplectic stepper because of indirections.
        """
        # Peel over all steps and prefactors
        for (
            (kin_prefactor, dyn_prefactor),
            (kin_step, exp_step, dyn_step),
        ) in self._steps_and_prefactors:
            prefactor = kin_prefactor(self.stepper, dt)
            # exp_step call does not change the time, so putting it first
            time = exp_step(self.linear_stepper, System, time, prefactor)
            # kin_step call modifies time according to prefactor, so it comes next
            time = kin_step(self.stepper, System, time, prefactor)
            prefactor = dyn_prefactor(self.stepper, dt)
            time = dyn_step(self.stepper, System, time, prefactor)

        # Deal with the remnant steps and prefactors
        prefactor = self._last_kin_prefactor(self.stepper, dt)
        time = self._last_exponential_step(self.linear_stepper, System, time, prefactor)
        # kin_step call modifies time according to prefactor
        time = self._last_kinematic_step(self.stepper, System, time, prefactor)

        return time
