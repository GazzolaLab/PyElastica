__doc__ = """
(added in version 0.3.0)

Built in damper module implementations
"""
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
    def system(self): # -> SystemType: (Return type is not parsed with sphinx book.)
        """
        get system (rod or rigid body) reference
        
        Returns
        -------
        SystemType

        """
        return self._system

    @abstractmethod
    def dampen_rates(self, rod: SystemType, time: float):
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
    Exponential damper class. This class corresponds to the analytical version of
    a linear damper, and uses the following equations to damp translational and
    rotational velocities:

    .. math::

        \\mathbf{v}^{n+1} = \\mathbf{v}^n \\exp \\left( -  \\nu~dt  \\right)

        \\pmb{\\omega}^{n+1} = \\pmb{\\omega}^n \\exp \\left( - \\frac{{\\nu}~m~dt } { \\mathbf{J}} \\right)

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
    Since this class analytically treats the damping term, it is unconditionally stable
    from a timestep perspective, i.e. the presence of damping does not impose any additional
    restriction on the simulation timestep size. This implies that when using
    Exponential Damper, one can set `damping_constant` as high as possible, without worrying
    about the simulation becoming unstable. This now leads to a streamlined procedure
    for tuning the `damping_constant`:

    1. Set a high value for `damping_constant` to first acheive a stable simulation.
    2. If you feel the simulation is overdamped, reduce `damping_constant` until you
       feel the simulation is underdamped, and expected dynamics are recovered.

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

    def dampen_rates(self, rod: SystemType, time: float):
        rod.velocity_collection[:] = (
            rod.velocity_collection * self.translational_exponential_damping_coefficient
        )

        rod.omega_collection[:] = rod.omega_collection * np.power(
            self.rotational_exponential_damping_coefficient, rod.dilatation
        )
