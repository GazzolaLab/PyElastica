__doc__ = """ Built in dissipation module implementations """
__all__ = [
    "DamperBase",
    "ExponentialDamper",
]

from typing import Type, Union
import numpy as np
from abc import ABC, abstractmethod
from elastica.rod import RodBase
from elastica.rigidbody import RigidBodyBase


class DamperBase(ABC):
    """Base class for damping module implementations.

    Notes
    -----
    All damper classes must inherit DamperBase class.


        Attributes
        ----------
        system : RodBase or RigidBodyBase

    """

    _system: Union[Type[RodBase], Type[RigidBodyBase]]

    def __init__(self, *args, **kwargs):
        """Initialize damping module"""
        try:
            self._system = kwargs["_system"]
        except KeyError:
            raise KeyError(
                "Please use simulator.dampen(...).using(...) syntax to establish "
                "damping."
            )

    @property
    def system(self) -> Union[Type[RodBase], Type[RigidBodyBase]]:
        """get system (rod or rigid body) reference"""
        return self._system

    @abstractmethod
    def constrain_rates(
        self, rod: Union[Type[RodBase], Type[RigidBodyBase]], time: float
    ) -> None:
        # TODO: In the future, we can remove rod and use self.system
        """
        Constrain rates (velocity and/or omega) of a rod object.

        Parameters
        ----------
        rod : Union[Type[RodBase], Type[RigidBodyBase]]
            Rod or rigid-body object.
        time : float
            The time of simulation.

        """
        pass


class ExponentialDamper(DamperBase):
    """
    Exponential damper.
    TODO: include the equations and math.

    Attributes
    ----------
    translational_exponential_damping_coefficient: numpy.ndarray
        1D array containing data with 'float' type.
        Damping coefficient acting on translational velocity.
    rotational_exponential_damping_coefficient : numpy.ndarray
        1D array containing data with 'float' type.
        Damping coefficient acting on rotational velocity.
    """

    def __init__(self, damping_constant, time_step, **kwargs):
        """
        Exponential damper initializer

        Parameters
        ----------
        damping_constant : float
            Damping constant for the exponential dampers.
        time_step : float
            Time-step of simulation
        """
        super().__init__(**kwargs)
        # Compute the damping coefficient for translational velocity
        nodal_mass = self._system.mass
        self.translational_exponential_damping_coefficient = np.exp(
            -damping_constant * time_step / nodal_mass
        )

        # Compute the damping coefficient for exponential velocity
        element_mass = 0.5 * (nodal_mass[1:] + nodal_mass[:-1])
        element_mass[0] += 0.5 * nodal_mass[0]
        element_mass[-1] += 0.5 * nodal_mass[-1]
        self.rotational_exponential_damping_coefficient = np.exp(
            -damping_constant * time_step / element_mass
        )

    def constrain_rates(
        self, rod: Union[Type[RodBase], Type[RigidBodyBase]], time: float
    ) -> None:
        rod.velocity_collection[:] = (
            rod.velocity_collection * self.translational_exponential_damping_coefficient
        )

        rod.omega_collection[:] = rod.omega_collection * np.power(
            self.rotational_exponential_damping_coefficient,
            rod.dilatation * np.diagonal(rod.inv_mass_second_moment_of_inertia).T,
        )
