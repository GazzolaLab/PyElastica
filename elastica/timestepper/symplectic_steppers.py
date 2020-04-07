__doc__ = """Symplectic time steppers and concepts for integrating the kinematic and dynamic equations of rod-like objects.  """
import numpy as np

from elastica.timestepper._stepper_interface import (
    _TimeStepper,
    _LinearExponentialIntegratorMixin,
)

"""
Developer Note
--------------

For the reasons why we define Mixin classes here, the developer
is referred to the same section on `explicit_steppers.py`.
"""


class _SystemInstanceStepperMixin:
    def do_step(self, System, time: np.float64, dt: np.float64):
        for (
            kin_prefactor,
            kin_step,
            dyn_prefactor,
            dyn_step,
        ) in self._steps_and_prefactors[:-1]:
            prefac = kin_prefactor(self, dt)
            time = kin_step(self, System, time, prefac)
            prefac = dyn_prefactor(self, dt)
            time = dyn_step(self, System, time, prefac)

        # Peel the last kinematic step and prefactor alone
        last_kin_prefactor = self._steps_and_prefactors[-1][0]
        last_kin_step = self._steps_and_prefactors[-1][1]

        prefac = last_kin_prefactor(self, dt)
        time = last_kin_step(self, System, time, prefac)
        return time


class _SystemCollectionStepperMixin:
    """
    Symplectic stepper mixin collection class
    """

    def do_step(self, SystemCollection, time: np.float64, dt: np.float64):
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
        for (
            kin_prefactor,
            kin_step,
            dyn_prefactor,
            dyn_step,
        ) in self._steps_and_prefactors[:-1]:
            prefac = kin_prefactor(self, dt)
            for system in SystemCollection[:-1]:
                _ = kin_step(self, system, time, prefac)
            time = kin_step(self, SystemCollection[-1], time, prefac)

            # TODO: remove below lines and try to find a better call option to compute internal forces and torques
            # We need internal forces and torques because they are used by interaction module.
            update_internal_forces_torques = self._update_internal_forces_torques[
                0
            ]  # here 0 because you have one item in list
            for system in SystemCollection[:-1]:
                _ = update_internal_forces_torques(self, system, time)
            time = update_internal_forces_torques(self, SystemCollection[-1], time)

            # BoCos, External forces, controls etc.
            SystemCollection.synchronize(time)
            # TODO: remove below line, it should be some other function synchronizeBC
            SystemCollection.synchronizeBC(time)
            prefac = dyn_prefactor(self, dt)
            for system in SystemCollection[:-1]:
                _ = dyn_step(self, system, time, prefac)
            time = dyn_step(self, SystemCollection[-1], time, prefac)

            # TODO: remove below line, it should be some other function synchronizeBC
            SystemCollection.synchronizeBC(time)

        # Peel the last kinematic step and prefactor alone
        last_kin_prefactor = self._steps_and_prefactors[-1][0]
        last_kin_step = self._steps_and_prefactors[-1][1]

        prefac = last_kin_prefactor(self, dt)
        for system in SystemCollection[:-1]:
            _ = last_kin_step(self, system, time, prefac)
        time = last_kin_step(self, SystemCollection[-1], time, prefac)

        # Call back function, will call the user defined call back functions and store data
        SystemCollection.callBack(time, int(time / dt))

        return time


class SymplecticStepper(_TimeStepper):
    """
    Symplectic stepper constructor.

    Attributes
    ----------
    _steps: list
        List containing methods of symplectic time stepper.
    _prefactors: list
        List containing prefactors of symplectic time stepper.
    _update_internal_forces_torques: list
        List containing methods for computing internal forces and torques.
    """

    def __init__(self, cls=None):
        super(SymplecticStepper, self).__init__()
        take_methods_from = self if cls is None else cls()
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

        # We are getting function named as _update_internal_forces_torques from dictionary,
        # it turns a list.
        self._update_internal_forces_torques = [
            v
            for (k, v) in take_methods_from.__class__.__dict__.items()
            if k.endswith("forces_torques")
        ]

        def mirror(in_list):
            """ Mirrors an input list ignoring the last element
            If steps = [A, B, C]
            then this call makes it [A, B, C, B, A]

            Parameters
            ----------
            in_list : input list to be mirrored, modified in-place

            Returns
            -------

            """
            #  syntax is very ugly
            in_list.extend(in_list[-2::-1])

        mirror(self._steps)
        mirror(self._prefactors)

        assert len(self._steps) == len(
            self._prefactors
        ), "Size mismatch in the number of steps and prefactors provided for a Symplectic Stepper!"

        self._kinematic_steps = self._steps[::2]
        self._dynamic_steps = self._steps[1::2]
        self._kinematic_prefactors = self._prefactors[::2]
        self._dynamic_prefactors = self._prefactors[1::2]

        # Avoid this check for MockClasses
        if len(self._kinematic_steps) > 0:
            assert (
                len(self._kinematic_steps) == len(self._dynamic_steps) + 1
            ), "Size mismatch in the number of kinematic and dynamic steps provided for a Symplectic Stepper!"
            assert (
                len(self._kinematic_prefactors) == len(self._dynamic_prefactors) + 1
            ), "Size mismatch in the number of kinematic and dynamic prefactors provided for a Symplectic Stepper!"

        from itertools import zip_longest

        self._steps_and_prefactors = tuple(
            zip_longest(
                self._kinematic_prefactors,
                self._kinematic_steps,
                self._dynamic_prefactors,
                self._dynamic_steps,
            )
        )

    @property
    def n_stages(self):
        return len(self._steps_and_prefactors)


class PositionVerlet(SymplecticStepper):
    """
    Position Verlet symplectic time stepper class, which
    includes methods for second-order position Verlet.
    """

    def __init__(self):
        super(PositionVerlet, self).__init__()

    def _first_kinematic_prefactor(self, dt):
        return 0.5 * dt

    def _first_kinematic_step(self, System, time: np.float64, prefac: np.float64):
        System.kinematic_states += prefac * System.kinematic_rates(time, prefac)
        return time + prefac

    def _first_dynamic_prefactor(self, dt):
        return dt

    def _first_dynamic_step(self, System, time: np.float64, prefac: np.float64):
        System.dynamic_states += prefac * System.dynamic_rates(
            time, prefac
        )  # TODO : Why should we pass dt into System again?
        return time

    # TODO: find a better place for this or a better call option. We need to compute internal forces and torques before external because interaction uses it!
    def _update_internal_forces_torques(self, System, time: np.float64):
        System.update_internal_forces_and_torques(time)
        return time

    # Note : we don't need the second half of the calls as it simply forwards
    # to its equivalent first half. This is taken care in the base class

    # def _second_kinematic_step(self, System, time: np.float64, dt: np.float64):
    #     return self._first_kinematic_step(System, time, dt)


class PEFRL(SymplecticStepper):
    """
    Position Extended Forest-Ruth Like Algorithm of
    I.M. Omelyan, I.M. Mryglod, and R. Folk, Computer Physics Communications 146, 188 (2002),
    http://arxiv.org/abs/cond-mat/0110585
    """

    # xi and chi are confusing, but be careful!
    ξ = np.float64(0.1786178958448091e0)  # ξ
    λ = -np.float64(0.2123418310626054e0)  # λ
    χ = -np.float64(0.6626458266981849e-1)  # χ

    # Pre-calculate other coefficients
    lambda_dash_coeff = 0.5 * (1.0 - 2.0 * λ)
    xi_chi_dash_coeff = 1.0 - 2.0 * (ξ + χ)

    def __init__(self):
        super(PEFRL, self).__init__()

    def _first_kinematic_prefactor(self, dt):
        return self.ξ * dt

    def _first_kinematic_step(self, System, time: np.float64, prefac: np.float64):
        System.kinematic_states += prefac * System.kinematic_rates(time, prefac)
        return time + prefac

    def _first_dynamic_prefactor(self, dt):
        return self.lambda_dash_coeff * dt

    def _first_dynamic_step(self, System, time: np.float64, prefac: np.float64):
        System.dynamic_states += prefac * System.dynamic_rates(time, prefac)
        return time

    def _second_kinematic_prefactor(self, dt):
        return self.χ * dt

    def _second_kinematic_step(self, System, time: np.float64, prefac: np.float64):
        System.kinematic_states += prefac * System.kinematic_rates(time, prefac)
        return time + prefac

    def _second_dynamic_prefactor(self, dt):
        return self.λ * dt

    def _second_dynamic_step(self, System, time: np.float64, prefac: np.float64):
        System.dynamic_states += prefac * System.dynamic_rates(time, prefac)
        return time

    def _third_kinematic_prefactor(self, dt):
        return self.xi_chi_dash_coeff * dt

    def _third_kinematic_step(self, System, time: np.float64, prefac: np.float64):
        # Need to fill in
        System.kinematic_states += prefac * System.kinematic_rates(time, prefac)
        return time + prefac

    # TODO: find a better place for this or a better call option. We need to compute internal forces and torques before external because interaction uses it!
    def _update_internal_forces_torques(self, System, time: np.float64):
        System.update_internal_forces_and_torques(time)
        return time

    # Note : we don't need the second half of the calls as it simply forwards
    # to its equivalent first half. This is taken care in the base class

    # def _third_dynamic_step(self, System, time: np.float64, dt: np.float64):
    #     return self._second_dynamic_step(System, time, dt)
    #
    # def _fourth_kinematic_step(self, System, time: np.float64, dt: np.float64):
    #     return self._second_kinematic_step(System, time, dt)
    #
    # def _fourth_dynamic_step(self, System, time: np.float64, dt: np.float64):
    #     return self._first_dynamic_step(System, time, dt)
    #
    # def _fifth_kinematic_step(self, System, time: np.float64, dt: np.float64):
    #     return self._first_kinematic_step(System, time, dt)
    #     # return time + dt # To avoid numerical precision errors


class SymplecticLinearExponentialIntegrator(
    _LinearExponentialIntegratorMixin, SymplecticStepper
):
    def __init__(self):
        _LinearExponentialIntegratorMixin.__init__(self)
        SymplecticStepper.__init__(self, _LinearExponentialIntegratorMixin)
