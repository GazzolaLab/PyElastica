__doc__ = """Base class for rods"""

from typing import Type
import numpy as np
from numpy.typing import NDArray

from elastica.rod.energy import RodEnergy
from elastica.rod.knot_theory import KnotTheory


class RodBase(RodEnergy, KnotTheory):
    """
    Base class for all rods.

    Notes
    -----
    All new rod classes must be derived from this RodBase class.

    """

    REQUISITE_MODULES: list[Type] = []

    # Geometry
    n_elems: int
    n_nodes: int

    # State arrays
    position_collection: NDArray[np.float64]
    velocity_collection: NDArray[np.float64]
    acceleration_collection: NDArray[np.float64]
    director_collection: NDArray[np.float64]
    omega_collection: NDArray[np.float64]
    alpha_collection: NDArray[np.float64]

    # External forces/torques
    external_forces: NDArray[np.float64]
    external_torques: NDArray[np.float64]

    # Internal forces/torques
    internal_forces: NDArray[np.float64]
    internal_torques: NDArray[np.float64]

    # Rod-specific properties
    mass: NDArray[np.float64]
    volume: NDArray[np.float64]
    radius: NDArray[np.float64]
    tangents: NDArray[np.float64]
    lengths: NDArray[np.float64]
    rest_lengths: NDArray[np.float64]
    rest_voronoi_lengths: NDArray[np.float64]
    kappa: NDArray[np.float64]
    sigma: NDArray[np.float64]
    rest_kappa: NDArray[np.float64]
    rest_sigma: NDArray[np.float64]

    internal_stress: NDArray[np.float64]
    internal_couple: NDArray[np.float64]
    dilatation: NDArray[np.float64]
    dilatation_rate: NDArray[np.float64]
    voronoi_dilatation: NDArray[np.float64]

    bend_matrix: NDArray[np.float64]
    shear_matrix: NDArray[np.float64]

    mass_second_moment_of_inertia: NDArray[np.float64]
    inv_mass_second_moment_of_inertia: NDArray[np.float64]

    # Ring rod / periodic boundary
    ring_rod_flag: bool
    ghost_voronoi_idx: NDArray[np.int32]
    ghost_elems_idx: NDArray[np.int32]
    periodic_boundary_nodes_idx: NDArray[np.int32]
    periodic_boundary_elems_idx: NDArray[np.int32]
    periodic_boundary_voronoi_idx: NDArray[np.int32]

    # Symplectic stepper state
    v_w_collection: NDArray[np.float64]
    dvdt_dwdt_collection: NDArray[np.float64]
