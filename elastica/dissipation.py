__doc__ = """
(added in version 0.3.0)

Built in damper module implementations
"""

from abc import ABC, abstractmethod

from elastica.typing import RodType, SystemType

from numba import njit

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
    def system(self):  # -> SystemType: (Return type is not parsed with sphinx book.)
        """
        get system (rod or rigid body) reference

        Returns
        -------
        SystemType

        """
        return self._system

    @abstractmethod
    def dampen_rates(self, system: SystemType, time: float):
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

    def __init__(self, damping_constant, time_step, **kwargs):
        """
        Analytical linear damper initializer

        Parameters
        ----------
        damping_constant : float
            Damping constant for the analytical linear damper.
        time_step : float
            Time-step of simulation
        """
        super().__init__(**kwargs)
        # Compute the damping coefficient for translational velocity
        nodal_mass = self._system.mass
        self.translational_damping_coefficient = np.exp(-damping_constant * time_step)

        # Compute the damping coefficient for exponential velocity
        if self._system.ring_rod_flag:
            element_mass = nodal_mass
        else:
            element_mass = 0.5 * (nodal_mass[1:] + nodal_mass[:-1])
            element_mass[0] += 0.5 * nodal_mass[0]
            element_mass[-1] += 0.5 * nodal_mass[-1]
        self.rotational_damping_coefficient = np.exp(
            -damping_constant
            * time_step
            * element_mass
            * np.diagonal(self._system.inv_mass_second_moment_of_inertia).T
        )

    def dampen_rates(self, rod: RodType, time: float):
        rod.velocity_collection[:] = (
            rod.velocity_collection * self.translational_damping_coefficient
        )

        rod.omega_collection[:] = rod.omega_collection * np.power(
            self.rotational_damping_coefficient, rod.dilatation
        )


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

    def __init__(self, filter_order: int, **kwargs):
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

    def dampen_rates(self, rod: RodType, time: float) -> None:

        self.filter_function(
            rod.velocity_collection,
            self.velocity_filter_term,
            rod.omega_collection,
            self.omega_filter_term,
            self.filter_order,
        )


@njit(cache=True)
def _filter_function_periodic_condition_ring_rod(
    velocity_collection,
    velocity_filter_term,
    omega_collection,
    omega_filter_term,
    filter_order,
):
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


@njit(cache=True)
def _filter_function_periodic_condition(
    velocity_collection,
    velocity_filter_term,
    omega_collection,
    omega_filter_term,
    filter_order,
):
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


@njit(cache=True)
def nb_filter_rate(
    rate_collection: np.ndarray, filter_term: np.ndarray, filter_order: int
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
