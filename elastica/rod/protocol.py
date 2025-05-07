from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from elastica.systems.protocol import SystemProtocol, SlenderBodyGeometryProtocol


class _CosseratRodEnergy(Protocol):
    def compute_bending_energy(self) -> NDArray[np.float64]: ...

    def compute_shear_energy(self) -> NDArray[np.float64]: ...

    def compute_translational_energy(self) -> NDArray[np.float64]: ...

    def compute_rotational_energy(self) -> NDArray[np.float64]: ...


class CosseratRodProtocol(
    SystemProtocol, SlenderBodyGeometryProtocol, _CosseratRodEnergy, Protocol
):

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

    ghost_voronoi_idx: NDArray[np.int32]
    ghost_elems_idx: NDArray[np.int32]

    ring_rod_flag: bool
    periodic_boundary_nodes_idx: NDArray[np.int32]
    periodic_boundary_elems_idx: NDArray[np.int32]
    periodic_boundary_voronoi_idx: NDArray[np.int32]
