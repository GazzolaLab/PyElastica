__doc__ = """Analytical linear damper implementation"""
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

        m \\frac{\\partial \\mathbf{v}}{\\partial t} = -\\gamma_t \\mathbf{v}

        \\frac{\\mathbf{J}}{e} \\frac{\\partial \\pmb{\\omega}}{\\partial t} = -\\gamma_r \\pmb{\\omega}

    Examples
    --------
    The current AnalyticalLinearDamper class supports three types of protocols:

    1.  Uniform damping constant: the user provides the keyword argument `uniform_damping_constant`
        of dimension (1/T). This leads to an exponential damping constant that is used for both
        translation and rotational velocities.

    >>> simulator.dampen(rod).using(
    ...     AnalyticalLinearDamper,
    ...     uniform_damping_constant=0.1,
    ...     time_step = 1E-4,   # Simulation time-step
    ... )

    2.  Physical damping constant: separate exponential coefficients are computed for the
        translational and rotational velocities, based on user-defined
        `translational_damping_constant` and `rotational_damping_constant`.

    >>> simulator.dampen(rod).using(
    ...     AnalyticalLinearDamper,
    ...     translational_damping_constant=0.1,
    ...     rotational_damping_constant=0.05,
    ...     time_step = 1E-4,   # Simulation time-step
    ... )

    3.  Damping constant: this protocol follows the original algorithm where the damping
        constants for translational and rotational velocities are assumed to be numerically
        identical. This leads to dimensional inconsistencies (see
        https://github.com/GazzolaLab/PyElastica/issues/354). Hence, this option will be deprecated
        in version 0.4.0.

    >>> simulator.dampen(rod).using(
    ...     AnalyticalLinearDamper,
    ...     damping_constant=0.1,   # To be deprecated in 0.4.0
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
