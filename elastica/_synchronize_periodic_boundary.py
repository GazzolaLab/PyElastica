__doc__ = (
    """These functions are used to synchronize periodic boundaries for ring rods.  """
)

from typing import Any
from numba import njit
import numpy as np
from numpy.typing import NDArray
from elastica.boundary_conditions import ConstraintBase
from elastica.typing import RodType


@njit(cache=True)  # type: ignore
def _synchronize_periodic_boundary_of_vector_collection(
    input_array: NDArray[np.float64], periodic_idx: NDArray[np.float64]
) -> None:
    """
    This function synchronizes the periodic boundaries of a vector collection.
    Parameters
    ----------
    input_array : numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type. Vector that is going to be synched.
    periodic_idx : numpy.ndarray
        2D (2, n_periodic_boundary) array containing data with 'float' type. Vector containing periodic boundary
        index. First dimension is the periodic boundary index, second dimension is the referenced cell index.

    Returns
    -------

    """
    for i in range(3):
        for k in range(periodic_idx.shape[1]):
            input_array[i, periodic_idx[0, k]] = input_array[i, periodic_idx[1, k]]


@njit(cache=True)  # type: ignore
def _synchronize_periodic_boundary_of_matrix_collection(
    input_array: NDArray[np.float64], periodic_idx: NDArray[np.float64]
) -> None:
    """
    This function synchronizes the periodic boundaries of a matrix collection.
    Parameters
    ----------
    input_array : numpy.ndarray
        2D (dim, dim, blocksize) array containing data with 'float' type. Matrix collection that is going to be synched.
    periodic_idx : numpy.ndarray
        2D (2, n_periodic_boundary) array containing data with 'float' type. Vector containing periodic boundary
        index. First dimension is the periodic boundary index, second dimension is the referenced cell index.

    Returns
    -------

    """
    for i in range(3):
        for j in range(3):
            for k in range(periodic_idx.shape[1]):
                input_array[i, j, periodic_idx[0, k]] = input_array[
                    i, j, periodic_idx[1, k]
                ]


@njit(cache=True)  # type: ignore
def _synchronize_periodic_boundary_of_scalar_collection(
    input_array: NDArray[np.float64], periodic_idx: NDArray[np.float64]
) -> None:
    """
    This function synchronizes the periodic boundaries of a scalar collection.

    Parameters
    ----------
    input_array : numpy.ndarray
        2D (dim, dim, blocksize) array containing data with 'float' type. Scalar collection that is going to be synched.
    periodic_idx : numpy.ndarray
        2D (2, n_periodic_boundary) array containing data with 'float' type. Vector containing periodic boundary
        index. First dimension is the periodic boundary index, second dimension is the referenced cell index.

    Returns
    -------

    """
    for k in range(periodic_idx.shape[1]):
        input_array[periodic_idx[0, k]] = input_array[periodic_idx[1, k]]


class _ConstrainPeriodicBoundaries(ConstraintBase):
    """
    This class is used only when ring rods are present in the simulation. This class is a wrapper and its purpose
    is to synchronize periodic boundaries of ring rod.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def constrain_values(self, system: RodType, time: np.float64) -> None:
        _synchronize_periodic_boundary_of_vector_collection(
            system.position_collection, system.periodic_boundary_nodes_idx
        )
        _synchronize_periodic_boundary_of_matrix_collection(
            system.director_collection, system.periodic_boundary_elems_idx
        )

    def constrain_rates(self, system: RodType, time: np.float64) -> None:
        _synchronize_periodic_boundary_of_vector_collection(
            system.velocity_collection, system.periodic_boundary_nodes_idx
        )
        _synchronize_periodic_boundary_of_vector_collection(
            system.omega_collection, system.periodic_boundary_elems_idx
        )
