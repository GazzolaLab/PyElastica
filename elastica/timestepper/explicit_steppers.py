__doc__ = """Explicit timesteppers  and concepts"""

import numpy as np
from copy import deepcopy

from elastica.typing import (
    SystemType,
    SystemCollectionType,
    OperatorType,
    SteppersOperatorsType,
    StateType,
)
from elastica.systems.protocol import ExplicitSystemProtocol
from elastica.rod.data_structures import (
    overload_operator_kinematic_numba,
    overload_operator_dynamic_numba,
)
from .protocol import ExplicitStepperProtocol


class ExplicitStepperMixin:
    """Base class for all explicit steppers
    Can also be used as a mixin with optional cls argument below
    """

    def system_inplace_update(
        self: ExplicitStepperProtocol,
        system1: ExplicitSystemProtocol,
        system2: ExplicitSystemProtocol,
        time: np.float64,
        dt: np.float64,
    ):
        """
        y_n+1 = y_n + prefac * f(y_n, t_n)
        """
        overload_operator_kinematic_numba(
            system1.n_nodes,
            dt,
            system1.kinematic_states.position_collection,
            system1.kinematic_states.director_collection,
            system2.velocity_collection,
            system2.omega_collection,
        )

        overload_operator_dynamic_numba(
            system1.dynamic_states.rate_collection,
            system2.dynamic_rates(time, dt),
        )

    def system_rate_update(
        self, SystemCollection: SystemCollectionType, time: np.float64
    ):
        # Constrain
        SystemCollection.constrain_values(time)

        # We need internal forces and torques because they are used by interaction module.
        for system in SystemCollection.block_systems():
            system.compute_internal_forces_and_torques(time)

        # Add external forces, controls etc.
        SystemCollection.synchronize(time)

        # Constrain only rates
        SystemCollection.constrain_rates(time)

    def step(
        self: ExplicitStepperProtocol,
        SystemCollection: SystemCollectionType,
        time: np.float64,
        dt: np.float64,
    ) -> np.float64:
        self.stage(SystemCollection, time, dt)

        # Timestep update
        next_time = time + dt

        # Call back function, will call the user defined call back functions and store data
        SystemCollection.apply_callbacks(next_time, round(next_time / dt))

        # Zero out the external forces and torques
        for system in SystemCollection.block_systems():
            system.zeroed_out_external_forces_and_torques(next_time)

        return next_time


class EulerForward(ExplicitStepperMixin):
    """
    Classical Euler Forward stepper. Stateless, coordinates operations only.
    """

    def stage(
        self,
        SystemCollection: SystemCollectionType,
        time: np.float64,
        dt: np.float64,
    ) -> None:
        self.system_rate_update(SystemCollection, time)
        for system in SystemCollection.block_systems():
            self.system_inplace_update(system, system, time, dt)


class RungeKutta4(ExplicitStepperMixin):
    """
    Stateless runge-kutta4. coordinates operations only.
    """

    def stage(
        self,
        SystemCollection: SystemCollectionType,
        time: np.float64,
        dt: np.float64,
    ) -> None:
        """
        In-place update of system states with Runge-Kutta 4 integration
        """

        # TODO: Try to avoid deepcopying the whole system collection
        # Related to #33: save/restart module
        # Maybe search for low-storage RK scheme
        k1_system = SystemCollection  # Alias
        k2_system = deepcopy(SystemCollection)
        k3_system = deepcopy(SystemCollection)
        k4_system = deepcopy(SystemCollection)

        # First stage
        self.system_rate_update(k1_system, time)

        # Second stage
        for system1, system2 in zip(
            k2_system.block_systems(), k1_system.block_systems()
        ):
            self.system_inplace_update(system1, system2, time, dt / 2.0)
        self.system_rate_update(k2_system, time + dt / 2.0)

        # Third stage
        for system1, system2 in zip(
            k3_system.block_systems(), k2_system.block_systems()
        ):
            self.system_inplace_update(system1, system2, time, dt / 2.0)
        self.system_rate_update(k3_system, time + dt / 2.0)

        # Fourth stage
        for system1, system2 in zip(
            k4_system.block_systems(), k3_system.block_systems()
        ):
            self.system_inplace_update(system1, system2, time, dt)
        self.system_rate_update(k3_system, time + dt)

        # Combine stages
        for system1, k1, k2, k3, k4 in zip(
            SystemCollection.block_systems(),
            k1_system.block_systems(),
            k2_system.block_systems(),
            k3_system.block_systems(),
            k4_system.block_systems(),
        ):
            velocity_update = (
                k1.velocity_collection
                + 2 * k2.velocity_collection
                + 2 * k3.velocity_collection
                + k4.velocity_collection
            ) / 6
            omega_update = (
                k1.omega_collection
                + 2 * k2.omega_collection
                + 2 * k3.omega_collection
                + k4.omega_collection
            ) / 6
            dynamic_rates_update = (
                k1.dynamic_rates(time, dt / 6.0)
                + k2.dynamic_rates(time, dt / 3.0)
                + k3.dynamic_rates(time, dt / 3.0)
                + k4.dynamic_rates(time, dt / 6.0)
            )  # Time is dummy.

            overload_operator_kinematic_numba(
                system1.n_nodes,
                dt,
                system1.kinematic_states.position_collection,
                system1.kinematic_states.director_collection,
                velocity_update,
                omega_update,
            )

            overload_operator_dynamic_numba(
                system1.dynamic_states.rate_collection,
                dynamic_rates_update,
            )
