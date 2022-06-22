__doc__ = """ Built in dissipation module implementations """
__all__ = [
    "DamperBase",
    "ExponentialDamper",
]
from abc import ABC, abstractmethod

from elastica.typing import SystemType

import numpy as np


class DamperBase(ABC):
    """Base class for damping module implementations.

    Notes
    -----
    All damper classes must inherit DamperBase class.


        Attributes
        ----------
        system : SystemType (RodBase or RigidBodyBase)

    """

    _system: SystemType

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
    def system(self) -> SystemType:
        """get system (rod or rigid body) reference"""
        return self._system

    @abstractmethod
    def dampen_rates(self, rod: SystemType, time: float) -> None:
        # TODO: In the future, we can remove rod and use self.system
        """
        Dampen rates (velocity and/or omega) of a rod object.

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
    Exponential damper class uses the following equations to damp the velocities.

    .. math::

        \\mathbf{v} = \\mathbf{v} \\exp \\left( -  \\nu dt  \\right)

        \\pmb{\\omega} = \\pmb{\\omega} \\exp \\left( - \\frac{{\\nu} m dt } { \\mathbf{J}} \\right)

    Examples
    --------
    How to set exponential damper for rod or rigid body:

    >>> simulator.dampin(rod).using(
    ...     ExponentialDamper,
    ...     damping_constant=0.1,
    ...     time_step = 1E-4,   # Simulation time-step
    ... )

    Notes
    -----
    Advantage of using Exponential Damper is you can set `damping_constant` as high as possible and simulation never
    blows up. You can start reducing `damping_constant` untill you feel dynamics are captured sufficiently. This gives
    you a direction to tune `damping_constant` while keeping simulation stable and at the same time capturing the
    dynamics.

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
            -damping_constant * time_step
        )

        # Compute the damping coefficient for exponential velocity
        element_mass = 0.5 * (nodal_mass[1:] + nodal_mass[:-1])
        element_mass[0] += 0.5 * nodal_mass[0]
        element_mass[-1] += 0.5 * nodal_mass[-1]
        self.rotational_exponential_damping_coefficient = np.exp(
            -damping_constant
            * time_step
            * element_mass
            * np.diagonal(self._system.inv_mass_second_moment_of_inertia).T
        )

    def dampen_rates(self, rod: SystemType, time: float) -> None:
        rod.velocity_collection[:] = (
            rod.velocity_collection * self.translational_exponential_damping_coefficient
        )

        rod.omega_collection[:] = rod.omega_collection * np.power(
            self.rotational_exponential_damping_coefficient, rod.dilatation
        )
