import logging
from typing import Any, Callable, TypeAlias

import numpy as np

from elastica.typing import RodType
from elastica.dissipation.damper_base import DamperBase


DampenType: TypeAlias = Callable[[RodType], None]


class AnalyticalLinearDamper(DamperBase):
    """
    Analytical linear damper class. This class corresponds to the analytical version of
    a linear damper, and uses the following equations to damp translational and
    rotational velocities:

    .. math::

        \\mathbf{v}^{n+1} = \\mathbf{v}^n \\exp \\left( -  \\nu~dt  \\right)

        \\pmb{\\omega}^{n+1} = \\pmb{\\omega}^n \\exp \\left( - \\frac{{\\nu}~m~dt } { \\mathbf{J}} \\right)

    Examples
    --------
    How to set analytical linear damper for rod or rigid body:

    >>> simulator.dampen(rod).using(
    ...     AnalyticalLinearDamper,
    ...     damping_constant=0.1,
    ...     time_step = 1E-4,   # Simulation time-step
    ... )

    Notes
    -----
    Since this class analytically treats the damping term, it is unconditionally stable
    from a timestep perspective, i.e. the presence of damping does not impose any additional
    restriction on the simulation timestep size. This implies that when using
    AnalyticalLinearDamper, one can set `damping_constant` as high as possible, without worrying
    about the simulation becoming unstable. This now leads to a streamlined procedure
    for tuning the `damping_constant`:

    1. Set a high value for `damping_constant` to first acheive a stable simulation.
    2. If you feel the simulation is overdamped, reduce `damping_constant` until you
       feel the simulation is underdamped, and expected dynamics are recovered.

    Attributes
    ----------
    translational_damping_coefficient: numpy.ndarray
        1D array containing data with 'float' type.
        Damping coefficient acting on translational velocity.
    rotational_damping_coefficient : numpy.ndarray
        1D array containing data with 'float' type.
        Damping coefficient acting on rotational velocity.
    """

    def __init__(self, time_step: np.float64, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        damping_constant = kwargs.get("damping_constant", None)
        uniform_damping_constant = kwargs.get("uniform_damping_constant", None)
        translational_damping_constant = kwargs.get(
            "translational_damping_constant", None
        )
        rotational_damping_constant = kwargs.get("rotational_damping_constant", None)

        self._dampen_rates_protocol: DampenType

        if (
            (damping_constant is not None)
            and (uniform_damping_constant is None)
            and (translational_damping_constant is None)
            and (rotational_damping_constant is None)
        ):
            logging.warning(
                "Analytical linear damping using generic damping constant "
                "will be deprecated in 0.4.0"
            )
            self._dampen_rates_protocol = self._deprecated_damping_protocol(
                damping_constant=damping_constant, time_step=time_step
            )

        elif (
            (damping_constant is None)
            and (uniform_damping_constant is not None)
            and (translational_damping_constant is None)
            and (rotational_damping_constant is None)
        ):
            self._dampen_rates_protocol = self._uniform_damping_protocol(
                uniform_damping_constant=uniform_damping_constant, time_step=time_step
            )

        elif (
            (damping_constant is None)
            and (uniform_damping_constant is None)
            and (translational_damping_constant is not None)
            and (rotational_damping_constant is not None)
        ):
            self._dampen_rates_protocol = self._physical_damping_protocol(
                translational_damping_constant=translational_damping_constant,
                rotational_damping_constant=rotational_damping_constant,
                time_step=time_step,
            )

        else:
            message = (
                "AnalyticalLinearDamper usage:\n"
                "\tsimulator.dampen(rod).using(\n"
                "\t\tAnalyticalLinearDamper,\n"
                "\t\ttranslational_damping_constant=...,\n"
                "\t\trotational_damping_constant=...,\n"
                "\t\ttime_step=...,\n"
                "\t)\n"
                "\tor\n"
                "\tsimulator.dampen(rod).using(\n"
                "\t\tAnalyticalLinearDamper,\n"
                "\t\tuniform_damping_constant=...,\n"
                "\t\ttime_step=...,\n"
                "\t)\n"
                "\tor (deprecated in 0.4.0)\n"
                "\tsimulator.dampen(rod).using(\n"
                "\t\tAnalyticalLinearDamper,\n"
                "\t\tdamping_constant=...,\n"
                "\t\ttime_step=...,\n"
                "\t)\n"
            )
            raise ValueError(message)

    def _deprecated_damping_protocol(
        self, damping_constant: np.float64, time_step: np.float64
    ) -> DampenType:
        nodal_mass = self._system.mass
        self._translational_damping_coefficient = np.exp(-damping_constant * time_step)

        if self._system.ring_rod_flag:
            element_mass = nodal_mass
        else:
            element_mass = 0.5 * (nodal_mass[1:] + nodal_mass[:-1])
            element_mass[0] += 0.5 * nodal_mass[0]
            element_mass[-1] += 0.5 * nodal_mass[-1]
        self._rotational_damping_coefficient = np.exp(
            -damping_constant
            * time_step
            * element_mass
            * np.diagonal(self._system.inv_mass_second_moment_of_inertia).T
        )

        def dampen_rates_protocol(rod: RodType) -> None:
            rod.velocity_collection *= self._translational_damping_coefficient
            rod.omega_collection *= np.power(
                self._rotational_damping_coefficient, rod.dilatation
            )

        return dampen_rates_protocol

    def _uniform_damping_protocol(
        self, uniform_damping_constant: np.float64, time_step: np.float64
    ) -> DampenType:
        self._translational_damping_coefficient = (
            self._rotational_damping_coefficient
        ) = np.exp(-uniform_damping_constant * time_step)

        def dampen_rates_protocol(rod: RodType) -> None:
            rod.velocity_collection *= self._translational_damping_coefficient
            rod.omega_collection *= self._rotational_damping_coefficient

        return dampen_rates_protocol

    def _physical_damping_protocol(
        self,
        translational_damping_constant: np.float64,
        rotational_damping_constant: np.float64,
        time_step: np.float64,
    ) -> DampenType:
        nodal_mass = self._system.mass.view()
        self._translational_damping_coefficient = np.exp(
            -translational_damping_constant / nodal_mass * time_step
        )

        inv_moi = np.diagonal(self._system.inv_mass_second_moment_of_inertia).T
        self._rotational_damping_coefficient = np.exp(
            -rotational_damping_constant * inv_moi * time_step
        )

        def dampen_rates_protocol(rod: RodType) -> None:
            rod.velocity_collection *= self._translational_damping_coefficient
            rod.omega_collection *= np.power(
                self._rotational_damping_coefficient, rod.dilatation
            )

        return dampen_rates_protocol

    def dampen_rates(self, rod: RodType, time: np.float64) -> None:
        self._dampen_rates_protocol(rod)
