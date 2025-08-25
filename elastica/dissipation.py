__doc__ = """
(added in version 0.3.0)

Built in damper module implementations
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, TypeAlias, Callable

from elastica.typing import RodType

from numba import njit

import numpy as np
from numpy.typing import NDArray


T = TypeVar("T")


class DamperBase(Generic[T], ABC):
    """Base class for damping module implementations.

    Notes
    -----
    All damper classes must inherit DamperBase class.


    Attributes
    ----------
    system : RodBase

    """

    _system: T

    # TODO typing can be made better
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize damping module"""
        try:
            self._system = kwargs["_system"]
        except KeyError:
            raise KeyError(
                "Please use simulator.dampen(...).using(...) syntax to establish "
                "damping."
            )

    @property
    def system(self) -> T:
        """
        get system (rod or rigid body) reference

        Returns
        -------
        SystemType

        """
        return self._system

    @abstractmethod
    def dampen_rates(self, system: T, time: np.float64) -> None:
        # TODO: In the future, we can remove rod and use self.system
        """
        Dampen rates (velocity and/or omega) of a rod object.

        Parameters
        ----------
        system : SystemType
            System (rod or rigid-body) object.
        time : float
            The time of simulation.

        """
        pass


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
        nodal_mass = self._system.mass
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

    def dampen_rates(self, system: RodType, time: np.float64) -> None:
        self._dampen_rates_protocol(system)


class LaplaceDissipationFilter(DamperBase):
    """
    Laplace Dissipation Filter class. This class corresponds qualitatively to a
    low-pass filter generated via the 1D Laplacian operator. It is applied to the
    translational and rotational velocities, where it filters out the high
    frequency (noise) modes, while having negligible effect on the low frequency
    smooth modes.

    Examples
    --------
    How to set Laplace dissipation filter for rod:

    >>> simulator.dampen(rod).using(
    ...     LaplaceDissipationFilter,
    ...     filter_order=3,   # order of the filter
    ... )

    Notes
    -----
    The extent of filtering can be controlled by the `filter_order`, which refers
    to the number of times the Laplacian operator is applied. Small
    integer values (1, 2, etc.) result in aggressive filtering, and can lead to
    the "physics" being filtered out. While high values (9, 10, etc.) imply
    minimal filtering, and thus negligible effect on the velocities.
    Values in the range of 3-7 are usually recommended.

    For details regarding the numerics behind the filtering, refer to [1]_, [2]_.

    .. [1] Jeanmart, H., & Winckelmans, G. (2007). Investigation of eddy-viscosity
       models modified using discrete filters: a simplified “regularized variational
       multiscale model” and an “enhanced field model”. Physics of fluids, 19(5), 055110.
    .. [2] Lorieul, G. (2018). Development and validation of a 2D Vortex Particle-Mesh
       method for incompressible multiphase flows (Doctoral dissertation,
       Université Catholique de Louvain).

    Attributes
    ----------
    filter_order : int
        Filter order, which corresponds to the number of times the Laplacian
        operator is applied. Increasing `filter_order` implies higher-order/weaker
        filtering.
    velocity_filter_term: numpy.ndarray
        2D array containing data with 'float' type.
        Filter term that modifies rod translational velocity.
    omega_filter_term: numpy.ndarray
        2D array containing data with 'float' type.
        Filter term that modifies rod rotational velocity.
    """

    def __init__(self, filter_order: int, **kwargs: Any) -> None:
        """
        Filter damper initializer

        Parameters
        ----------
        filter_order : int
            Filter order, which corresponds to the number of times the Laplacian
            operator is applied. Increasing `filter_order` implies higher-order/weaker
            filtering.
        """
        super().__init__(**kwargs)
        if not (filter_order > 0 and isinstance(filter_order, int)):
            raise ValueError(
                "Invalid filter order! Filter order must be a positive integer."
            )
        self.filter_order = filter_order

        if self._system.ring_rod_flag:
            # There are two periodic boundaries
            blocksize = self._system.n_elems + 2
            self.velocity_filter_term = np.zeros((3, blocksize))
            self.omega_filter_term = np.zeros((3, blocksize))
            self.filter_function = _filter_function_periodic_condition_ring_rod

        else:
            self.velocity_filter_term = np.zeros_like(self._system.velocity_collection)
            self.omega_filter_term = np.zeros_like(self._system.omega_collection)
            self.filter_function = _filter_function_periodic_condition

    def dampen_rates(self, system: RodType, time: np.float64) -> None:

        self.filter_function(
            system.velocity_collection,
            self.velocity_filter_term,
            system.omega_collection,
            self.omega_filter_term,
            self.filter_order,
        )


@njit(cache=True)  # type: ignore
def _filter_function_periodic_condition_ring_rod(
    velocity_collection: NDArray[np.float64],
    velocity_filter_term: NDArray[np.float64],
    omega_collection: NDArray[np.float64],
    omega_filter_term: NDArray[np.float64],
    filter_order: int,
) -> None:
    blocksize = velocity_filter_term.shape[1]

    # Transfer velocity to an array which has periodic boundaries and synchornize boundaries
    velocity_collection_with_periodic_bc = np.empty((3, blocksize))
    velocity_collection_with_periodic_bc[:, 1:-1] = velocity_collection[:]
    velocity_collection_with_periodic_bc[:, 0] = velocity_collection[:, -1]
    velocity_collection_with_periodic_bc[:, -1] = velocity_collection[:, 0]

    # Transfer omega to an array which has periodic boundaries and synchornize boundaries
    omega_collection_with_periodic_bc = np.empty((3, blocksize))
    omega_collection_with_periodic_bc[:, 1:-1] = omega_collection[:]
    omega_collection_with_periodic_bc[:, 0] = omega_collection[:, -1]
    omega_collection_with_periodic_bc[:, -1] = omega_collection[:, 0]

    nb_filter_rate(
        rate_collection=velocity_collection_with_periodic_bc,
        filter_term=velocity_filter_term,
        filter_order=filter_order,
    )
    nb_filter_rate(
        rate_collection=omega_collection_with_periodic_bc,
        filter_term=omega_filter_term,
        filter_order=filter_order,
    )

    # Transfer filtered velocity back
    velocity_collection[:] = velocity_collection_with_periodic_bc[:, 1:-1]
    omega_collection[:] = omega_collection_with_periodic_bc[:, 1:-1]


@njit(cache=True)  # type: ignore
def _filter_function_periodic_condition(
    velocity_collection: NDArray[np.float64],
    velocity_filter_term: NDArray[np.float64],
    omega_collection: NDArray[np.float64],
    omega_filter_term: NDArray[np.float64],
    filter_order: int,
) -> None:
    nb_filter_rate(
        rate_collection=velocity_collection,
        filter_term=velocity_filter_term,
        filter_order=filter_order,
    )
    nb_filter_rate(
        rate_collection=omega_collection,
        filter_term=omega_filter_term,
        filter_order=filter_order,
    )


@njit(cache=True)  # type: ignore
def nb_filter_rate(
    rate_collection: NDArray[np.float64],
    filter_term: NDArray[np.float64],
    filter_order: int,
) -> None:
    """
    Filters the rod rates (velocities) in numba njit decorator

    Parameters
    ----------
    rate_collection : numpy.ndarray
        2D array containing data with 'float' type.
        Array containing rod rates (velocities).
    filter_term: numpy.ndarray
        2D array containing data with 'float' type.
        Filter term that modifies rod rates (velocities).
    filter_order : int
        Filter order, which corresponds to the number of times the Laplacian
        operator is applied. Increasing `filter_order` implies higher order/weaker
        filtering.

    Notes
    -----
    For details regarding the numerics behind the filtering, refer to:

    .. [1] Jeanmart, H., & Winckelmans, G. (2007). Investigation of eddy-viscosity
       models modified using discrete filters: a simplified “regularized variational
       multiscale model” and an “enhanced field model”. Physics of fluids, 19(5), 055110.
    .. [2] Lorieul, G. (2018). Development and validation of a 2D Vortex Particle-Mesh
       method for incompressible multiphase flows (Doctoral dissertation,
       Université Catholique de Louvain).
    """

    filter_term[...] = rate_collection
    for i in range(filter_order):
        filter_term[..., 1:-1] = (
            -filter_term[..., 2:] - filter_term[..., :-2] + 2.0 * filter_term[..., 1:-1]
        ) / 4.0
        # dont touch boundary values
        filter_term[..., 0] = 0.0
        filter_term[..., -1] = 0.0
    rate_collection[...] = rate_collection - filter_term
