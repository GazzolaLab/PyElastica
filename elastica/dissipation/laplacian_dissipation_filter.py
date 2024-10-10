__doc__ = """Laplacian dissipation filter implementation"""
from typing import Any

import numpy as np
from numpy.typing import NDArray
from numba import njit

from elastica.typing import RodType
from elastica.dissipation.damper_base import DamperBase


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
