__doc__ = """Symplectic time steppers and concepts for integrating the kinematic and dynamic equations of rod-like objects.  """

import numpy as np

# from elastica.timestepper._stepper_interface import (
#     _TimeStepper,
#     _LinearExponentialIntegratorMixin,
# )
from elastica import IMPORT_NUMBA

"""
Developer Note
--------------

For the reasons why we define Mixin classes here, the developer
is referred to the same section on `explicit_steppers.py`.
"""


class _SystemInstanceStepper:
    @staticmethod
    def do_step(
        TimeStepper, _steps_and_prefactors, System, time: np.float64, dt: np.float64
    ):
        for (kin_prefactor, kin_step, dyn_step) in _steps_and_prefactors[:-1]:
            kin_step(TimeStepper, System, time, dt)
            time += kin_prefactor(TimeStepper, dt)
            System.update_internal_forces_and_torques(time)
            dyn_step(TimeStepper, System, time, dt)

        # Peel the last kinematic step and prefactor alone
        last_kin_prefactor = _steps_and_prefactors[-1][0]
        last_kin_step = _steps_and_prefactors[-1][1]

        last_kin_step(TimeStepper, System, time, dt)
        return time + last_kin_prefactor(TimeStepper, dt)


class _SystemCollectionStepper:
    """
    Symplectic stepper collection class
    """

    @staticmethod
    def do_step(
        TimeStepper,
        _steps_and_prefactors,
        SystemCollection,
        time: np.float64,
        dt: np.float64,
    ):
        """
        Function for doing symplectic stepper over the user defined rods (system).

        Parameters
        ----------
        SystemCollection: rod object
        time: float
        dt: float

        Returns
        -------

        """
        for (kin_prefactor, kin_step, dyn_step) in _steps_and_prefactors[:-1]:

            for system in SystemCollection:
                kin_step(TimeStepper, system, time, dt)

            time += kin_prefactor(TimeStepper, dt)

            # TODO: remove below line, it should be some other function synchronizeBC
            # SystemCollection.synchronizeBC(time)
            # Constrain only values
            SystemCollection.constrain_values(time)

            # We need internal forces and torques because they are used by interaction module.
            for system in SystemCollection:
                system.update_internal_forces_and_torques(time)
                # system.update_internal_forces_and_torques()

            # Add external forces, controls etc.
            SystemCollection.synchronize(time)

            for system in SystemCollection:
                dyn_step(TimeStepper, system, time, dt)

            # TODO: remove below line, it should be some other function synchronizeBC
            # Constrain only rates
            SystemCollection.constrain_rates(time)

        # Peel the last kinematic step and prefactor alone
        last_kin_prefactor = _steps_and_prefactors[-1][0]
        last_kin_step = _steps_and_prefactors[-1][1]

        for system in SystemCollection:
            last_kin_step(TimeStepper, system, time, dt)
        time += last_kin_prefactor(TimeStepper, dt)
        SystemCollection.constrain_values(time)

        # Call back function, will call the user defined call back functions and store data
        SystemCollection.apply_callbacks(time, int(time / dt))

        return time


class SymplecticStepperMethods:
    def __init__(self, timestepper_instance):
        take_methods_from = timestepper_instance
        # Let the total number of steps for the Symplectic method
        # be (2*n + 1) (for time-symmetry). What we do is collect
        # the first n + 1 entries down in _steps and _prefac below, and then
        # reverse and append it to itself.
        self._steps = [
            v
            for (k, v) in take_methods_from.__class__.__dict__.items()
            if k.endswith("step")
        ]
        # Prefac here is necessary because the linear-exponential integrator
        # needs only the prefactor and not the dt.
        self._prefactors = [
            v
            for (k, v) in take_methods_from.__class__.__dict__.items()
            if k.endswith("prefactor")
        ]

        # # We are getting function named as _update_internal_forces_torques from dictionary,
        # # it turns a list.
        # self._update_internal_forces_torques = [
        #     v
        #     for (k, v) in take_methods_from.__class__.__dict__.items()
        #     if k.endswith("forces_torques")
        # ]

        def mirror(in_list):
            """Mirrors an input list ignoring the last element
            If steps = [A, B, C]
            then this call makes it [A, B, C, B, A]

            Parameters
            ----------
            in_list : input list to be mirrored, modified in-place

            Returns
            -------

            """
            #  syntax is very ugly
            if len(in_list) > 1:
                in_list.extend(in_list[-2::-1])
            elif in_list:
                in_list.append(in_list[0])

        mirror(self._steps)
        mirror(self._prefactors)

        assert (
            len(self._steps) == 2 * len(self._prefactors) - 1
        ), "Size mismatch in the number of steps and prefactors provided for a Symplectic Stepper!"

        self._kinematic_steps = self._steps[::2]
        self._dynamic_steps = self._steps[1::2]

        # Avoid this check for MockClasses
        if len(self._kinematic_steps) > 0:
            assert (
                len(self._kinematic_steps) == len(self._dynamic_steps) + 1
            ), "Size mismatch in the number of kinematic and dynamic steps provided for a Symplectic Stepper!"

        from itertools import zip_longest

        def NoOp(*args):
            pass

        self._steps_and_prefactors = tuple(
            zip_longest(
                self._prefactors,
                self._kinematic_steps,
                self._dynamic_steps,
                fillvalue=NoOp,
            )
        )

    def step_methods(self):
        return self._steps_and_prefactors

    @property
    def n_stages(self):
        return len(self._steps_and_prefactors)


if IMPORT_NUMBA:
    from elastica._elastica_numba._timestepper._symplectic_steppers import (
        SymplecticStepperTag,
        PositionVerlet,
        PEFRL,
    )
else:
    from elastica._elastica_numpy._timestepper._symplectic_steppers import (
        SymplecticStepperTag,
        PositionVerlet,
        PEFRL,
    )
