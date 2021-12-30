__doc__ = """Symplectic time steppers and concepts for integrating the kinematic and dynamic equations of rod-like objects.  """

import numpy as np

# from elastica._elastica_numba._timestepper._symplectic_steppers import (
#     SymplecticStepperTag,
#     PositionVerlet,
#     PEFRL,
# )

# from elastica.timestepper._stepper_interface import (
#     _TimeStepper,
#     _LinearExponentialIntegratorMixin,
# )
from elastica.rod.data_structures import (
    overload_operator_kinematic_numba,
    overload_operator_dynamic_numba,
)

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

            for system in SystemCollection._memory_blocks:
                kin_step(TimeStepper, system, time, dt)

            time += kin_prefactor(TimeStepper, dt)

            # Constrain only values
            SystemCollection.constrain_values(time)

            # We need internal forces and torques because they are used by interaction module.
            for system in SystemCollection._memory_blocks:
                system.update_internal_forces_and_torques(time)
                # system.update_internal_forces_and_torques()

            # Add external forces, controls etc.
            SystemCollection.synchronize(time)

            for system in SystemCollection._memory_blocks:
                dyn_step(TimeStepper, system, time, dt)

            # Constrain only rates
            SystemCollection.constrain_rates(time)

        # Peel the last kinematic step and prefactor alone
        last_kin_prefactor = _steps_and_prefactors[-1][0]
        last_kin_step = _steps_and_prefactors[-1][1]

        for system in SystemCollection._memory_blocks:
            last_kin_step(TimeStepper, system, time, dt)
        time += last_kin_prefactor(TimeStepper, dt)
        SystemCollection.constrain_values(time)

        # Call back function, will call the user defined call back functions and store data
        SystemCollection.apply_callbacks(time, int(time / dt))

        # Zero out the external forces and torques
        for system in SystemCollection._memory_blocks:
            system.reset_external_forces_and_torques(time)

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


class SymplecticStepperTag:
    def __init__(self):
        pass


class PositionVerlet:
    """
    Position Verlet symplectic time stepper class, which
    includes methods for second-order position Verlet.
    """

    Tag = SymplecticStepperTag()

    def __init__(self):
        pass

    def _first_prefactor(self, dt):
        return 0.5 * dt

    def _first_kinematic_step(self, System, time: np.float64, dt: np.float64):
        prefac = self._first_prefactor(dt)

        overload_operator_kinematic_numba(
            System.n_nodes,
            prefac,
            System.kinematic_states.position_collection,
            System.kinematic_states.director_collection,
            System.velocity_collection,
            System.omega_collection,
        )

    def _first_dynamic_step(self, System, time: np.float64, dt: np.float64):

        overload_operator_dynamic_numba(
            System.dynamic_states.rate_collection,
            System.dynamic_rates(time, dt),
        )


class PEFRL:
    """
    Position Extended Forest-Ruth Like Algorithm of
    I.M. Omelyan, I.M. Mryglod and R. Folk, Computer Physics Communications 146, 188 (2002),
    http://arxiv.org/abs/cond-mat/0110585
    """

    # xi and chi are confusing, but be careful!
    ξ = np.float64(0.1786178958448091e0)  # ξ
    λ = -np.float64(0.2123418310626054e0)  # λ
    χ = -np.float64(0.6626458266981849e-1)  # χ

    # Pre-calculate other coefficients
    lambda_dash_coeff = 0.5 * (1.0 - 2.0 * λ)
    xi_chi_dash_coeff = 1.0 - 2.0 * (ξ + χ)

    Tag = SymplecticStepperTag()

    def __init__(self):
        pass

    def _first_kinematic_prefactor(self, dt):
        return self.ξ * dt

    def _first_kinematic_step(self, System, time: np.float64, dt: np.float64):
        prefac = self._first_kinematic_prefactor(dt)
        overload_operator_kinematic_numba(
            System.n_nodes,
            prefac,
            System.kinematic_states.position_collection,
            System.kinematic_states.director_collection,
            System.velocity_collection,
            System.omega_collection,
        )
        # System.kinematic_states += prefac * System.kinematic_rates(time, prefac)

    def _first_dynamic_step(self, System, time: np.float64, dt: np.float64):
        prefac = self.lambda_dash_coeff * dt
        overload_operator_dynamic_numba(
            System.dynamic_states.rate_collection,
            System.dynamic_rates(time, prefac),
        )
        # System.dynamic_states += prefac * System.dynamic_rates(time, prefac)

    def _second_kinematic_prefactor(self, dt):
        return self.χ * dt

    def _second_kinematic_step(self, System, time: np.float64, dt: np.float64):
        prefac = self._second_kinematic_prefactor(dt)
        overload_operator_kinematic_numba(
            System.n_nodes,
            prefac,
            System.kinematic_states.position_collection,
            System.kinematic_states.director_collection,
            System.velocity_collection,
            System.omega_collection,
        )
        # System.kinematic_states += prefac * System.kinematic_rates(time, prefac)

    def _second_dynamic_step(self, System, time: np.float64, dt: np.float64):
        prefac = self.λ * dt
        overload_operator_dynamic_numba(
            System.dynamic_states.rate_collection,
            System.dynamic_rates(time, prefac),
        )
        # System.dynamic_states += prefac * System.dynamic_rates(time, prefac)

    def _third_kinematic_prefactor(self, dt):
        return self.xi_chi_dash_coeff * dt

    def _third_kinematic_step(self, System, time: np.float64, dt: np.float64):
        prefac = self._third_kinematic_prefactor(dt)
        # Need to fill in
        overload_operator_kinematic_numba(
            System.n_nodes,
            prefac,
            System.kinematic_states.position_collection,
            System.kinematic_states.director_collection,
            System.velocity_collection,
            System.omega_collection,
        )
        # System.kinematic_states += prefac * System.kinematic_rates(time, prefac)
