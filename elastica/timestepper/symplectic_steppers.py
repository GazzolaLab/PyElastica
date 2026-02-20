__doc__ = """Symplectic time steppers and concepts for integrating the kinematic and dynamic equations of rod-like objects.  """

from typing import TYPE_CHECKING, Any, Callable

from itertools import zip_longest

from elastica.typing import (
    SystemCollectionType,
    SystemType,
    StepType,
    SteppersOperatorsType,
)

import numpy as np

from elastica.systems.protocol import SymplecticSystemProtocol
from .protocol import StepperProtocol

"""
Developer Note
--------------

For the reasons why we define Mixin classes here, the developer
is referred to the same section on `explicit_steppers.py`.
"""


class SymplecticStepperMixin:
    get_steps: Callable[[], list[StepType]]
    get_prefactors: Callable[[], list[StepType]]

    def __init__(self) -> None:
        self.steps_and_prefactors: SteppersOperatorsType = self.step_methods()

    def step_methods(self) -> SteppersOperatorsType:
        # Let the total number of steps for the Symplectic method
        # be (2*n + 1) (for time-symmetry).
        _steps: list[StepType] = self.get_steps()
        # Prefac here is necessary because the linear-exponential integrator
        # needs only the prefactor and not the dt.
        _prefactors: list[StepType] = self.get_prefactors()
        assert int(np.ceil(len(_steps) / 2)) == len(
            _prefactors
        ), f"{len(_steps)=}, {len(_prefactors)=}"

        # Separate the kinematic and dynamic steps
        _kinematic_steps: list[StepType] = _steps[::2]
        _dynamic_steps: list[StepType] = _steps[1::2]

        def no_operation(*args: Any) -> None:
            pass

        return tuple(
            zip_longest(
                _prefactors,
                _kinematic_steps,
                _dynamic_steps,
                fillvalue=no_operation,
            )
        )

    @property
    def n_stages(self) -> int:
        return len(self.steps_and_prefactors)

    def step(
        self,
        SystemCollection: SystemCollectionType,
        time: np.float64 | float,
        dt: np.float64 | float,
    ) -> np.float64:
        """
        Function for doing symplectic stepper over the user defined rods (system).

        Returns
        -------
        time: float
            The time after the integration step.

        """
        simulation_time = np.float64(time)
        simulation_dt = np.float64(dt)

        for kin_prefactor, kin_step, dyn_step in self.steps_and_prefactors[:-1]:
            for system in SystemCollection.final_systems():
                kin_step(system, simulation_time, simulation_dt)

            simulation_time += kin_prefactor(simulation_dt)

            # Constrain only values
            SystemCollection.constrain_values(simulation_time)

            # We need internal forces and torques because they are used by interaction module.
            for system in SystemCollection.final_systems():
                system.compute_internal_forces_and_torques(simulation_time)

            # Add external forces, controls etc.
            SystemCollection.synchronize(simulation_time)

            for system in SystemCollection.final_systems():
                dyn_step(system, simulation_time, simulation_dt)

            # Constrain only rates
            SystemCollection.constrain_rates(simulation_time)

        # Peel the last kinematic step and prefactor alone
        last_kin_prefactor = self.steps_and_prefactors[-1][0]
        last_kin_step = self.steps_and_prefactors[-1][1]

        for system in SystemCollection.final_systems():
            last_kin_step(system, simulation_time, simulation_dt)
        simulation_time += last_kin_prefactor(simulation_dt)
        SystemCollection.constrain_values(simulation_time)

        # Call back function, will call the user defined call back functions and store data
        SystemCollection.apply_callbacks(
            simulation_time, round(simulation_time / simulation_dt)
        )

        # Zero out the external forces and torques
        for system in SystemCollection.final_systems():
            system.zeroed_out_external_forces_and_torques(simulation_time)

        return simulation_time

    @staticmethod
    def do_step(
        TimeStepper: StepperProtocol,
        steps_and_prefactors: SteppersOperatorsType,
        SystemCollection: SystemCollectionType,
        time: np.float64,
        dt: np.float64,
    ) -> np.float64:  # pragma: no cover
        from warning import warn

        warn("This method is deprecated. Use the instance method .step instead.")
        return TimeStepper.step(SystemCollection, time, dt)  # type: ignore

    def step_single_instance(
        self,
        System: SystemType,
        time: np.float64,
        dt: np.float64,
    ) -> np.float64:
        """
        (The function is used for single system instance, mainly for testing purposes.)
        """

        for kin_prefactor, kin_step, dyn_step in self.steps_and_prefactors[:-1]:
            kin_step(System, time, dt)
            time += kin_prefactor(dt)
            System.compute_internal_forces_and_torques(time)
            dyn_step(System, time, dt)

        # Peel the last kinematic step and prefactor alone
        last_kin_prefactor = self.steps_and_prefactors[-1][0]
        last_kin_step = self.steps_and_prefactors[-1][1]

        last_kin_step(System, time, dt)
        return time + last_kin_prefactor(dt)


class PositionVerlet(SymplecticStepperMixin):
    """
    Position Verlet symplectic time stepper class, which
    includes methods for second-order position Verlet.
    """

    def get_steps(self) -> list[StepType]:
        return [
            self._first_kinematic_step,
            self._first_dynamic_step,
            self._first_kinematic_step,
        ]

    def get_prefactors(self) -> list[StepType]:
        return [
            self._first_prefactor,
            self._first_prefactor,
        ]

    def _first_prefactor(self, dt: np.float64) -> np.float64:
        return 0.5 * dt

    def _first_kinematic_step(
        self, System: SymplecticSystemProtocol, time: np.float64, dt: np.float64
    ) -> None:
        prefac = self._first_prefactor(dt)
        System.update_kinematics(time, prefac)

    def _first_dynamic_step(
        self, System: SymplecticSystemProtocol, time: np.float64, dt: np.float64
    ) -> None:
        prefac = dt
        System.update_accelerations(time)
        System.update_dynamics(time, prefac)


class PEFRL(SymplecticStepperMixin):
    """
    Position Extended Forest-Ruth Like Algorithm of
    I.M. Omelyan, I.M. Mryglod and R. Folk, Computer Physics Communications 146, 188 (2002),
    http://arxiv.org/abs/cond-mat/0110585
    """

    # xi and chi are confusing, but be careful!
    ξ: np.float64 = np.float64(0.1786178958448091e0)  # ξ
    λ: np.float64 = -np.float64(0.2123418310626054e0)  # λ
    χ: np.float64 = -np.float64(0.6626458266981849e-1)  # χ

    # Pre-calculate other coefficients
    lambda_dash_coeff: np.float64 = 0.5 * (1.0 - 2.0 * λ)
    xi_chi_dash_coeff: np.float64 = 1.0 - 2.0 * (ξ + χ)

    def get_steps(self) -> list[StepType]:
        operators = [
            self._first_kinematic_step,
            self._first_dynamic_step,
            self._second_kinematic_step,
            self._second_dynamic_step,
            self._third_kinematic_step,
        ]
        return operators + operators[-2::-1]

    def get_prefactors(self) -> list[StepType]:
        return [
            self._first_kinematic_prefactor,
            self._second_kinematic_prefactor,
            self._third_kinematic_prefactor,
            self._second_kinematic_prefactor,
            self._first_kinematic_prefactor,
        ]

    def _first_kinematic_prefactor(self, dt: np.float64) -> np.float64:
        return self.ξ * dt

    def _first_kinematic_step(
        self, System: SymplecticSystemProtocol, time: np.float64, dt: np.float64
    ) -> None:
        prefac = self._first_kinematic_prefactor(dt)
        System.update_kinematics(time, prefac)
        # System.kinematic_states += prefac * System.kinematic_rates(time, prefac)

    def _first_dynamic_step(
        self, System: SymplecticSystemProtocol, time: np.float64, dt: np.float64
    ) -> None:
        # System.dynamic_states += prefac * System.dynamic_rates(time, prefac)
        prefac = self.lambda_dash_coeff * dt
        System.update_accelerations(time)
        System.update_dynamics(time, prefac)

    def _second_kinematic_prefactor(self, dt: np.float64) -> np.float64:
        return self.χ * dt

    def _second_kinematic_step(
        self, System: SymplecticSystemProtocol, time: np.float64, dt: np.float64
    ) -> None:
        prefac = self._second_kinematic_prefactor(dt)
        System.update_kinematics(time, prefac)
        # System.kinematic_states += prefac * System.kinematic_rates(time, prefac)

    def _second_dynamic_step(
        self, System: SymplecticSystemProtocol, time: np.float64, dt: np.float64
    ) -> None:
        # System.dynamic_states += prefac * System.dynamic_rates(time, prefac)
        prefac = self.λ * dt
        System.update_accelerations(time)
        System.update_dynamics(time, prefac)

    def _third_kinematic_prefactor(self, dt: np.float64) -> np.float64:
        return self.xi_chi_dash_coeff * dt

    def _third_kinematic_step(
        self, System: SymplecticSystemProtocol, time: np.float64, dt: np.float64
    ) -> None:
        prefac = self._third_kinematic_prefactor(dt)
        System.update_kinematics(time, prefac)
        # System.kinematic_states += prefac * System.kinematic_rates(time, prefac)


if TYPE_CHECKING:
    from .protocol import StepperProtocol

    _: StepperProtocol = PositionVerlet()
    _: StepperProtocol = PEFRL()  # type: ignore [no-redef]
