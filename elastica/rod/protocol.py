from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from elastica.systems.protocol import SystemProtocol, SlenderBodyGeometryProtocol


class _RodEnergy(Protocol):
    def compute_bending_energy(self) -> NDArray[np.float64]: ...

    def compute_shear_energy(self) -> NDArray[np.float64]: ...


class CosseratRodProtocol(
    SystemProtocol, SlenderBodyGeometryProtocol, _RodEnergy, Protocol
):

    mass: NDArray[np.floating]
    volume: NDArray[np.floating]
    radius: NDArray[np.floating]
    tangents: NDArray[np.floating]
    lengths: NDArray[np.floating]
    rest_lengths: NDArray[np.floating]
    rest_voronoi_lengths: NDArray[np.floating]
    kappa: NDArray[np.floating]
    sigma: NDArray[np.floating]
    rest_kappa: NDArray[np.floating]
    rest_sigma: NDArray[np.floating]

    internal_stress: NDArray[np.floating]
    internal_couple: NDArray[np.floating]
    dilatation: NDArray[np.floating]
    dilatation_rate: NDArray[np.floating]
    voronoi_dilatation: NDArray[np.floating]

    bend_matrix: NDArray[np.floating]
    shear_matrix: NDArray[np.floating]

    mass_second_moment_of_inertia: NDArray[np.floating]
    inv_mass_second_moment_of_inertia: NDArray[np.floating]

    ghost_voronoi_idx: NDArray[np.integer]
    ghost_elems_idx: NDArray[np.integer]

    ring_rod_flag: bool
    periodic_boundary_nodes_idx: NDArray[np.integer]
    periodic_boundary_elems_idx: NDArray[np.integer]
    periodic_boundary_voronoi_idx: NDArray[np.integer]
