__doc__ = """ Rod classes and implementation details """
from typing import TYPE_CHECKING, Any, Optional, Type
from typing_extensions import Self

from elastica.typing import RodType
from .protocol import CosseratRodProtocol

from numpy.typing import NDArray

import numpy as np
import functools
import numba
from elastica.rod import RodBase
from elastica._linalg import (
    _batch_cross,
    _batch_norm,
    _batch_dot,
    _batch_matvec,
)
from elastica._rotations import _inv_rotate
from elastica._calculus import (
    quadrature_kernel_for_block_structure,
    difference_kernel_for_block_structure,
    _difference,
    _average,
)
from .factory_function import allocate
from .knot_theory import KnotTheory

position_difference_kernel = _difference
position_average = _average


@functools.lru_cache(maxsize=1)
def _get_z_vector() -> NDArray[np.float64]:
    return np.array([0.0, 0.0, 1.0]).reshape(3, -1)


def _compute_sigma_kappa_for_blockstructure(memory_block: RodType) -> None:
    """
    This function is a wrapper to call functions which computes shear stretch, strain and bending twist and strain.

    Parameters
    ----------
    memory_block : object

    Returns
    -------

    """
    _compute_shear_stretch_strains(
        memory_block.position_collection,
        memory_block.volume,
        memory_block.lengths,
        memory_block.tangents,
        memory_block.radius,
        memory_block.rest_lengths,
        memory_block.rest_voronoi_lengths,
        memory_block.dilatation,
        memory_block.voronoi_dilatation,
        memory_block.director_collection,
        memory_block.sigma,
    )

    # Compute bending twist strains for the block
    _compute_bending_twist_strains(
        memory_block.director_collection,
        memory_block.rest_voronoi_lengths,
        memory_block.kappa,
    )


class CosseratRod(RodBase, KnotTheory):
    """
    Cosserat Rod class. This is the preferred class for rods because it is derived from some
    of the essential base classes.

        Attributes
        ----------
        n_elems: int
            The number of elements of the rod.
        position_collection: NDArray[np.float64]
            2D (dim, n_nodes) array containing data with 'float' type.
            Array containing node position vectors.
        velocity_collection: NDArray[np.float64]
            2D (dim, n_nodes) array containing data with 'float' type.
            Array containing node velocity vectors.
        acceleration_collection: NDArray[np.float64]
            2D (dim, n_nodes) array containing data with 'float' type.
            Array containing node acceleration vectors.
        omega_collection: NDArray[np.float64]
            2D (dim, n_elems) array containing data with 'float' type.
            Array containing element angular velocity vectors.
        alpha_collection: NDArray[np.float64]
            2D (dim, n_elems) array containing data with 'float' type.
            Array contining element angular acceleration vectors.
        director_collection: NDArray[np.float64]
            3D (dim, dim, n_elems) array containing data with 'float' type.
            Array containing element director matrices.
        rest_lengths: NDArray[np.float64]
            1D (n_elems) array containing data with 'float' type.
            Rod element lengths at rest configuration.
        density: NDArray[np.float64]
            1D (n_elems) array containing data with 'float' type.
            Rod elements densities.
        volume: NDArray[np.float64]
            1D (n_elems) array containing data with 'float' type.
            Rod element volumes.
        mass: NDArray[np.float64]
            1D (n_nodes) array containing data with 'float' type.
            Rod node masses. Note that masses are stored on the nodes, not on elements.
        mass_second_moment_of_inertia: NDArray[np.float64]
            3D (dim, dim, n_elems) array containing data with 'float' type.
            Rod element mass second moment of interia.
        inv_mass_second_moment_of_inertia: NDArray[np.float64]
            3D (dim, dim, n_elems) array containing data with 'float' type.
            Rod element inverse mass moment of inertia.
        rest_voronoi_lengths: NDArray[np.float64]
            1D (n_voronoi) array containing data with 'float' type.
            Rod lengths on the voronoi domain at the rest configuration.
        internal_forces: NDArray[np.float64]
            2D (dim, n_nodes) array containing data with 'float' type.
            Rod node internal forces. Note that internal forces are stored on the node, not on elements.
        internal_torques: NDArray[np.float64]
            2D (dim, n_elems) array containing data with 'float' type.
            Rod element internal torques.
        external_forces: NDArray[np.float64]
            2D (dim, n_nodes) array containing data with 'float' type.
            External forces acting on rod nodes.
        external_torques: NDArray[np.float64]
            2D (dim, n_elems) array containing data with 'float' type.
            External torques acting on rod elements.
        lengths: NDArray[np.float64]
            1D (n_elems) array containing data with 'float' type.
            Rod element lengths.
        tangents: NDArray[np.float64]
            2D (dim, n_elems) array containing data with 'float' type.
            Rod element tangent vectors.
        radius: NDArray[np.float64]
            1D (n_elems) array containing data with 'float' type.
            Rod element radius.
        dilatation: NDArray[np.float64]
            1D (n_elems) array containing data with 'float' type.
            Rod element dilatation.
        voronoi_dilatation: NDArray[np.float64]
            1D (n_voronoi) array containing data with 'float' type.
            Rod dilatation on voronoi domain.
        dilatation_rate: NDArray[np.float64]
            1D (n_elems) array containing data with 'float' type.
            Rod element dilatation rates.
    """

    REQUISITE_MODULES: list[Type] = []

    def __init__(
        self: CosseratRodProtocol,
        n_elements: int,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64],
        omega: NDArray[np.float64],
        acceleration: NDArray[np.float64],
        angular_acceleration: NDArray[np.float64],
        directors: NDArray[np.float64],
        radius: NDArray[np.float64],
        mass_second_moment_of_inertia: NDArray[np.float64],
        inv_mass_second_moment_of_inertia: NDArray[np.float64],
        shear_matrix: NDArray[np.float64],
        bend_matrix: NDArray[np.float64],
        density_array: NDArray[np.float64],
        volume: NDArray[np.float64],
        mass: NDArray[np.float64],
        internal_forces: NDArray[np.float64],
        internal_torques: NDArray[np.float64],
        external_forces: NDArray[np.float64],
        external_torques: NDArray[np.float64],
        lengths: NDArray[np.float64],
        rest_lengths: NDArray[np.float64],
        tangents: NDArray[np.float64],
        dilatation: NDArray[np.float64],
        dilatation_rate: NDArray[np.float64],
        voronoi_dilatation: NDArray[np.float64],
        rest_voronoi_lengths: NDArray[np.float64],
        sigma: NDArray[np.float64],
        kappa: NDArray[np.float64],
        rest_sigma: NDArray[np.float64],
        rest_kappa: NDArray[np.float64],
        internal_stress: NDArray[np.float64],
        internal_couple: NDArray[np.float64],
        ring_rod_flag: bool,
    ) -> None:
        self.n_nodes = n_elements + 1 if not ring_rod_flag else n_elements
        self.n_elems = n_elements
        self.position_collection = position
        self.velocity_collection = velocity
        self.omega_collection = omega
        self.acceleration_collection = acceleration
        self.alpha_collection = angular_acceleration
        self.director_collection = directors
        self.radius = radius
        self.mass_second_moment_of_inertia = mass_second_moment_of_inertia
        self.inv_mass_second_moment_of_inertia = inv_mass_second_moment_of_inertia
        self.shear_matrix = shear_matrix
        self.bend_matrix = bend_matrix
        self.density = density_array
        self.volume = volume
        self.mass = mass
        self.internal_forces = internal_forces
        self.internal_torques = internal_torques
        self.external_forces = external_forces
        self.external_torques = external_torques
        self.lengths = lengths
        self.rest_lengths = rest_lengths
        self.tangents = tangents
        self.dilatation = dilatation
        self.dilatation_rate = dilatation_rate
        self.voronoi_dilatation = voronoi_dilatation
        self.rest_voronoi_lengths = rest_voronoi_lengths
        self.sigma = sigma
        self.kappa = kappa
        self.rest_sigma = rest_sigma
        self.rest_kappa = rest_kappa
        self.internal_stress = internal_stress
        self.internal_couple = internal_couple
        self.ring_rod_flag = ring_rod_flag

        if not self.ring_rod_flag:
            # For ring rod there are no periodic elements so below code won't run.
            # We add periodic elements at the memory block construction.
            # Compute shear stretch and strains.
            _compute_shear_stretch_strains(
                self.position_collection,
                self.volume,
                self.lengths,
                self.tangents,
                self.radius,
                self.rest_lengths,
                self.rest_voronoi_lengths,
                self.dilatation,
                self.voronoi_dilatation,
                self.director_collection,
                self.sigma,
            )

            # Compute bending twist strains
            _compute_bending_twist_strains(
                self.director_collection, self.rest_voronoi_lengths, self.kappa
            )

    @classmethod
    def straight_rod(
        cls,
        n_elements: int,
        start: NDArray[np.float64],
        direction: NDArray[np.float64],
        normal: NDArray[np.float64],
        base_length: float,
        base_radius: float,
        density: float,
        *,
        nu: Optional[np.float64] = None,
        youngs_modulus: float,
        **kwargs: Any,
    ) -> Self:
        """
        Cosserat rod constructor for straight-rod geometry.


        Notes
        -----
        Since we expect the Cosserat Rod to simulate soft rod, Poisson's ratio is set to 0.5 by default.
        It is possible to give additional argument "shear_modulus" or "poisson_ratio" to specify extra modulus.


        Parameters
        ----------
        n_elements : int
            Number of element. Must be greater than 3.
            Generally recommended to start with 40-50, and adjust the resolution.
        start : NDArray[np.float64]
            Starting coordinate in 3D
        direction : NDArray[np.float64]
            Direction of the rod in 3D
        normal : NDArray[np.float64]
            Normal vector of the rod in 3D
        base_length : float
            Total length of the rod
        base_radius : float
            Uniform radius of the rod
        density : float
            Density of the rod
        nu : float
            Damping coefficient for Rayleigh damping
        youngs_modulus : float
            Young's modulus
        **kwargs : dict, optional
            The "position" and/or "directors" can be overrided by passing "position" and "directors" argument. Remember, the shape of the "position" is (3,n_elements+1) and the shape of the "directors" is (3,3,n_elements).

        Returns
        -------
        CosseratRod

        """

        if nu is not None:
            raise ValueError(
                # Remove the option to set internal nu inside, beyond v0.4.0
                "The option to set damping coefficient (nu) for the rod during rod\n"
                "initialisation is now deprecated. Instead, for adding damping to rods,\n"
                "please derive your simulation class from the add-on Damping mixin class.\n"
                "For reference see the class elastica.dissipation.AnalyticalLinearDamper(),\n"
                "and for usage check examples/axial_stretching.py"
            )
        # Straight rod is not ring rod set flag to false
        ring_rod_flag = False
        (
            n_elements,
            position,
            velocities,
            omegas,
            accelerations,
            angular_accelerations,
            directors,
            radius,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            shear_matrix,
            bend_matrix,
            density_array,
            volume,
            mass,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
        ) = allocate(
            n_elements,
            direction,
            normal,
            np.float64(base_length),
            np.float64(base_radius),
            np.float64(density),
            np.float64(youngs_modulus),
            rod_origin_position=start,
            ring_rod_flag=ring_rod_flag,
            **kwargs,
        )

        return cls(
            n_elements,
            position,
            velocities,
            omegas,
            accelerations,
            angular_accelerations,
            directors,
            radius,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            shear_matrix,
            bend_matrix,
            density_array,
            volume,
            mass,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
            ring_rod_flag,
        )

    @classmethod
    def ring_rod(
        cls,
        n_elements: int,
        ring_center_position: NDArray[np.float64],
        direction: NDArray[np.float64],
        normal: NDArray[np.float64],
        base_length: float,
        base_radius: float,
        density: float,
        *,
        nu: Optional[float] = None,
        youngs_modulus: float,
        **kwargs: Any,
    ) -> Self:
        """
        Cosserat rod constructor for straight-rod geometry.


        Notes
        -----
        Since we expect the Cosserat Rod to simulate soft rod, Poisson's ratio is set to 0.5 by default.
        It is possible to give additional argument "shear_modulus" or "poisson_ratio" to specify extra modulus.


        Parameters
        ----------
        n_elements : int
            Number of element. Must be greater than 3. Generarally recommended to start with 40-50, and adjust the resolution.
        ring_center_position : NDArray[np.float64]
            Center coordinate for ring rod in 3D
        direction : NDArray[np.float64]
            Direction of the rod in 3D
        normal : NDArray[np.float64]
            Normal vector of the rod in 3D
        base_length : float
            Total length of the rod
        base_radius : float
            Uniform radius of the rod
        density : float
            Density of the rod
        nu : float | None
            Damping coefficient for Rayleigh damping
        youngs_modulus : float
            Young's modulus
        **kwargs : dict, optional
            The "position" and/or "directors" can be overrided by passing "position" and "directors" argument. Remember, the shape of the "position" is (3,n_elements+1) and the shape of the "directors" is (3,3,n_elements).

        Returns
        -------
        CosseratRod

        """
        from elastica.modules.constraints import Constraints

        if nu is not None:
            raise ValueError(
                # Remove the option to set internal nu inside, beyond v0.4.0
                "The option to set damping coefficient (nu) for the rod during rod\n"
                "initialisation is now deprecated. Instead, for adding damping to rods,\n"
                "please derive your simulation class from the add-on Damping mixin class.\n"
                "For reference see the class elastica.dissipation.AnalyticalLinearDamper(),\n"
                "and for usage check examples/axial_stretching.py"
            )
        # Straight rod is not ring rod set flag to false
        ring_rod_flag = True
        (
            n_elements,
            position,
            velocities,
            omegas,
            accelerations,
            angular_accelerations,
            directors,
            radius,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            shear_matrix,
            bend_matrix,
            density_array,
            volume,
            mass,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
        ) = allocate(
            n_elements,
            direction,
            normal,
            np.float64(base_length),
            np.float64(base_radius),
            np.float64(density),
            np.float64(youngs_modulus),
            rod_origin_position=ring_center_position,
            ring_rod_flag=ring_rod_flag,
            **kwargs,
        )

        rod = cls(
            n_elements,
            position,
            velocities,
            omegas,
            accelerations,
            angular_accelerations,
            directors,
            radius,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            shear_matrix,
            bend_matrix,
            density_array,
            volume,
            mass,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
            ring_rod_flag,
        )
        rod.REQUISITE_MODULES.append(Constraints)
        return rod

    def compute_internal_forces_and_torques(
        self: CosseratRodProtocol, time: np.float64
    ) -> None:
        """
        Compute internal forces and torques. We need to compute internal forces and torques before the acceleration because
        they are used in interaction. Thus in order to speed up simulation, we will compute internal forces and torques
        one time and use them. Previously, we were computing internal forces and torques multiple times in interaction.
        Saving internal forces and torques in a variable take some memory, but we will gain speed up.

        Parameters
        ----------
        time: np.float64
            current time

        """
        _compute_internal_forces(
            self.position_collection,
            self.volume,
            self.lengths,
            self.tangents,
            self.radius,
            self.rest_lengths,
            self.rest_voronoi_lengths,
            self.dilatation,
            self.voronoi_dilatation,
            self.director_collection,
            self.sigma,
            self.rest_sigma,
            self.shear_matrix,
            self.internal_stress,
            self.internal_forces,
            self.ghost_elems_idx,
        )

        _compute_internal_torques(
            self.position_collection,
            self.velocity_collection,
            self.tangents,
            self.lengths,
            self.rest_lengths,
            self.director_collection,
            self.rest_voronoi_lengths,
            self.bend_matrix,
            self.rest_kappa,
            self.kappa,
            self.voronoi_dilatation,
            self.mass_second_moment_of_inertia,
            self.omega_collection,
            self.internal_stress,
            self.internal_couple,
            self.dilatation,
            self.dilatation_rate,
            self.internal_torques,
            self.ghost_voronoi_idx,
        )

    # Interface to time-stepper mixins (Symplectic, Explicit), which calls this method
    def update_accelerations(self: CosseratRodProtocol, time: np.float64) -> None:
        """
        Updates the acceleration variables

        Parameters
        ----------
        time: np.float64
            current time

        """
        _update_accelerations(
            self.acceleration_collection,
            self.internal_forces,
            self.external_forces,
            self.mass,
            self.alpha_collection,
            self.inv_mass_second_moment_of_inertia,
            self.internal_torques,
            self.external_torques,
            self.dilatation,
        )

    def zeroed_out_external_forces_and_torques(
        self: CosseratRodProtocol, time: np.float64
    ) -> None:
        _zeroed_out_external_forces_and_torques(
            self.external_forces, self.external_torques
        )

    def compute_translational_energy(self: CosseratRodProtocol) -> NDArray[np.float64]:
        """
        Compute total translational energy of the rod at the instance.
        """
        return (
            0.5
            * (
                self.mass
                * np.einsum(
                    "ij, ij-> j", self.velocity_collection, self.velocity_collection
                )
            ).sum()
        )

    def compute_rotational_energy(self: CosseratRodProtocol) -> NDArray[np.float64]:
        """
        Compute total rotational energy of the rod at the instance.
        """
        J_omega_upon_e = (
            _batch_matvec(self.mass_second_moment_of_inertia, self.omega_collection)
            / self.dilatation
        )
        return 0.5 * np.einsum("ik,ik->k", self.omega_collection, J_omega_upon_e).sum()

    def compute_velocity_center_of_mass(
        self: CosseratRodProtocol,
    ) -> NDArray[np.float64]:
        """
        Compute velocity center of mass of the rod at the instance.
        """
        mass_times_velocity = np.einsum("j,ij->ij", self.mass, self.velocity_collection)
        sum_mass_times_velocity = np.einsum("ij->i", mass_times_velocity)

        return sum_mass_times_velocity / self.mass.sum()

    def compute_position_center_of_mass(
        self: CosseratRodProtocol,
    ) -> NDArray[np.float64]:
        """
        Compute position center of mass of the rod at the instance.
        """
        mass_times_position = np.einsum("j,ij->ij", self.mass, self.position_collection)
        sum_mass_times_position = np.einsum("ij->i", mass_times_position)

        return sum_mass_times_position / self.mass.sum()

    def compute_bending_energy(self: CosseratRodProtocol) -> NDArray[np.float64]:
        """
        Compute total bending energy of the rod at the instance.
        """

        kappa_diff = self.kappa - self.rest_kappa
        bending_internal_torques = _batch_matvec(self.bend_matrix, kappa_diff)

        return (
            0.5
            * (
                _batch_dot(kappa_diff, bending_internal_torques)
                * self.rest_voronoi_lengths
            ).sum()
        )

    def compute_shear_energy(self: CosseratRodProtocol) -> NDArray[np.float64]:
        """
        Compute total shear energy of the rod at the instance.
        """

        sigma_diff = self.sigma - self.rest_sigma
        shear_internal_forces = _batch_matvec(self.shear_matrix, sigma_diff)

        return (
            0.5
            * (_batch_dot(sigma_diff, shear_internal_forces) * self.rest_lengths).sum()
        )


# Below is the numba-implementation of Cosserat Rod equations. They don't need to be visible by users.


@numba.njit(cache=True)  # type: ignore
def _compute_geometry_from_state(
    position_collection: NDArray[np.float64],
    volume: NDArray[np.float64],
    lengths: NDArray[np.float64],
    tangents: NDArray[np.float64],
    radius: NDArray[np.float64],
) -> None:
    """
    Update <length, tangents, and radius> given <position and volume>.
    """
    # Compute eq (3.3) from 2018 RSOS paper

    # Note : we can use the two-point difference kernel, but it needs unnecessary padding
    # and hence will always be slower
    position_diff = position_difference_kernel(position_collection)
    # FIXME: Here 1E-14 is added to fix ghost lengths, which is 0, and causes division by zero error!
    lengths[:] = _batch_norm(position_diff) + 1e-14
    # _reset_scalar_ghost(lengths, ghost_elems_idx, 1.0)

    for k in range(lengths.shape[0]):
        tangents[0, k] = position_diff[0, k] / lengths[k]
        tangents[1, k] = position_diff[1, k] / lengths[k]
        tangents[2, k] = position_diff[2, k] / lengths[k]
        # resize based on volume conservation
        radius[k] = np.sqrt(volume[k] / lengths[k] / np.pi)


@numba.njit(cache=True)  # type: ignore
def _compute_all_dilatations(
    position_collection: NDArray[np.float64],
    volume: NDArray[np.float64],
    lengths: NDArray[np.float64],
    tangents: NDArray[np.float64],
    radius: NDArray[np.float64],
    dilatation: NDArray[np.float64],
    rest_lengths: NDArray[np.float64],
    rest_voronoi_lengths: NDArray[np.float64],
    voronoi_dilatation: NDArray[np.float64],
) -> None:
    """
    Update <dilatation and voronoi_dilatation>
    """
    _compute_geometry_from_state(position_collection, volume, lengths, tangents, radius)
    # Caveat : Needs already set rest_lengths and rest voronoi domain lengths
    # Put in initialization
    for k in range(lengths.shape[0]):
        dilatation[k] = lengths[k] / rest_lengths[k]

    # Cmopute eq (3.4) from 2018 RSOS paper
    # Note : we can use trapezoidal kernel, but it has padding and will be slower
    voronoi_lengths = position_average(lengths)

    # Cmopute eq (3.45 from 2018 RSOS paper
    for k in range(voronoi_lengths.shape[0]):
        voronoi_dilatation[k] = voronoi_lengths[k] / rest_voronoi_lengths[k]


@numba.njit(cache=True)  # type: ignore
def _compute_dilatation_rate(
    position_collection: NDArray[np.float64],
    velocity_collection: NDArray[np.float64],
    lengths: NDArray[np.float64],
    rest_lengths: NDArray[np.float64],
    dilatation_rate: NDArray[np.float64],
) -> None:
    """
    Update dilatation_rate given position, velocity, length, and rest_length
    """
    # TODO Use the vector formula rather than separating it out
    # self.lengths = l_i = |r^{i+1} - r^{i}|
    r_dot_v = _batch_dot(position_collection, velocity_collection)
    r_plus_one_dot_v = _batch_dot(
        position_collection[..., 1:], velocity_collection[..., :-1]
    )
    r_dot_v_plus_one = _batch_dot(
        position_collection[..., :-1], velocity_collection[..., 1:]
    )

    blocksize = lengths.shape[0]

    for k in range(blocksize):
        dilatation_rate[k] = (
            (r_dot_v[k] + r_dot_v[k + 1] - r_dot_v_plus_one[k] - r_plus_one_dot_v[k])
            / lengths[k]
            / rest_lengths[k]
        )


@numba.njit(cache=True)  # type: ignore
def _compute_shear_stretch_strains(
    position_collection: NDArray[np.float64],
    volume: NDArray[np.float64],
    lengths: NDArray[np.float64],
    tangents: NDArray[np.float64],
    radius: NDArray[np.float64],
    rest_lengths: NDArray[np.float64],
    rest_voronoi_lengths: NDArray[np.float64],
    dilatation: NDArray[np.float64],
    voronoi_dilatation: NDArray[np.float64],
    director_collection: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> None:
    """
    Update <shear/stretch(sigma)> given <dilatation, director, and tangent>.
    """

    # Quick trick : Instead of evaliation Q(et-d^3), use property that Q*d3 = (0,0,1), a constant
    _compute_all_dilatations(
        position_collection,
        volume,
        lengths,
        tangents,
        radius,
        dilatation,
        rest_lengths,
        rest_voronoi_lengths,
        voronoi_dilatation,
    )

    z_vector = np.array([0.0, 0.0, 1.0]).reshape(3, -1)
    sigma[:] = dilatation * _batch_matvec(director_collection, tangents) - z_vector


@numba.njit(cache=True)  # type: ignore
def _compute_internal_shear_stretch_stresses_from_model(
    position_collection: NDArray[np.float64],
    volume: NDArray[np.float64],
    lengths: NDArray[np.float64],
    tangents: NDArray[np.float64],
    radius: NDArray[np.float64],
    rest_lengths: NDArray[np.float64],
    rest_voronoi_lengths: NDArray[np.float64],
    dilatation: NDArray[np.float64],
    voronoi_dilatation: NDArray[np.float64],
    director_collection: NDArray[np.float64],
    sigma: NDArray[np.float64],
    rest_sigma: NDArray[np.float64],
    shear_matrix: NDArray[np.float64],
    internal_stress: NDArray[np.float64],
) -> None:
    """
    Update <internal stress> given <shear matrix, sigma, and rest_sigma>.

    Linear force functional
    Operates on
    S : (3,3,n) tensor and sigma (3,n)
    """
    _compute_shear_stretch_strains(
        position_collection,
        volume,
        lengths,
        tangents,
        radius,
        rest_lengths,
        rest_voronoi_lengths,
        dilatation,
        voronoi_dilatation,
        director_collection,
        sigma,
    )
    internal_stress[:] = _batch_matvec(shear_matrix, sigma - rest_sigma)


@numba.njit(cache=True)  # type: ignore
def _compute_bending_twist_strains(
    director_collection: NDArray[np.float64],
    rest_voronoi_lengths: NDArray[np.float64],
    kappa: NDArray[np.float64],
) -> None:
    """
    Update <curvature/twist (kappa)> given <director and rest_voronoi_length>.
    """
    temp = _inv_rotate(director_collection)
    blocksize = rest_voronoi_lengths.shape[0]
    for k in range(blocksize):
        kappa[0, k] = temp[0, k] / rest_voronoi_lengths[k]
        kappa[1, k] = temp[1, k] / rest_voronoi_lengths[k]
        kappa[2, k] = temp[2, k] / rest_voronoi_lengths[k]


@numba.njit(cache=True)  # type: ignore
def _compute_internal_bending_twist_stresses_from_model(
    director_collection: NDArray[np.float64],
    rest_voronoi_lengths: NDArray[np.float64],
    internal_couple: NDArray[np.float64],
    bend_matrix: NDArray[np.float64],
    kappa: NDArray[np.float64],
    rest_kappa: NDArray[np.float64],
) -> None:
    """
    Upate <internal couple> given <curvature(kappa) and bend_matrix>.

    Linear force functional
    Operates on
    B : (3,3,n) tensor and curvature kappa (3,n)
    """
    _compute_bending_twist_strains(
        director_collection, rest_voronoi_lengths, kappa
    )  # concept : needs to compute kappa

    blocksize = kappa.shape[1]
    temp = np.empty((3, blocksize))
    for i in range(3):
        for k in range(blocksize):
            temp[i, k] = kappa[i, k] - rest_kappa[i, k]

    internal_couple[:] = _batch_matvec(bend_matrix, temp)


@numba.njit(cache=True)  # type: ignore
def _compute_internal_forces(
    position_collection: NDArray[np.float64],
    volume: NDArray[np.float64],
    lengths: NDArray[np.float64],
    tangents: NDArray[np.float64],
    radius: NDArray[np.float64],
    rest_lengths: NDArray[np.float64],
    rest_voronoi_lengths: NDArray[np.float64],
    dilatation: NDArray[np.float64],
    voronoi_dilatation: NDArray[np.float64],
    director_collection: NDArray[np.float64],
    sigma: NDArray[np.float64],
    rest_sigma: NDArray[np.float64],
    shear_matrix: NDArray[np.float64],
    internal_stress: NDArray[np.float64],
    internal_forces: NDArray[np.float64],
    ghost_elems_idx: NDArray[np.float64],
) -> None:
    """
    Update <internal force> given <director, internal_stress and velocity>.
    """

    # Compute n_l and cache it using internal_stress
    # Be careful about usage though
    _compute_internal_shear_stretch_stresses_from_model(
        position_collection,
        volume,
        lengths,
        tangents,
        radius,
        rest_lengths,
        rest_voronoi_lengths,
        dilatation,
        voronoi_dilatation,
        director_collection,
        sigma,
        rest_sigma,
        shear_matrix,
        internal_stress,
    )

    # Signifies Q^T n_L / e
    # Not using batch matvec as I don't want to take directors.T here

    blocksize = internal_stress.shape[1]
    cosserat_internal_stress = np.zeros((3, blocksize))

    for i in range(3):
        for j in range(3):
            for k in range(blocksize):
                cosserat_internal_stress[i, k] += (
                    director_collection[j, i, k] * internal_stress[j, k]
                )

    cosserat_internal_stress /= dilatation
    internal_forces[:] = difference_kernel_for_block_structure(
        cosserat_internal_stress, ghost_elems_idx
    )


@numba.njit(cache=True)  # type: ignore
def _compute_internal_torques(
    position_collection: NDArray[np.float64],
    velocity_collection: NDArray[np.float64],
    tangents: NDArray[np.float64],
    lengths: NDArray[np.float64],
    rest_lengths: NDArray[np.float64],
    director_collection: NDArray[np.float64],
    rest_voronoi_lengths: NDArray[np.float64],
    bend_matrix: NDArray[np.float64],
    rest_kappa: NDArray[np.float64],
    kappa: NDArray[np.float64],
    voronoi_dilatation: NDArray[np.float64],
    mass_second_moment_of_inertia: NDArray[np.float64],
    omega_collection: NDArray[np.float64],
    internal_stress: NDArray[np.float64],
    internal_couple: NDArray[np.float64],
    dilatation: NDArray[np.float64],
    dilatation_rate: NDArray[np.float64],
    internal_torques: NDArray[np.float64],
    ghost_voronoi_idx: NDArray[np.int32],
) -> None:
    """
    Update <internal torque>.
    """
    # Compute \tau_l and cache it using internal_couple
    # Be careful about usage though
    _compute_internal_bending_twist_stresses_from_model(
        director_collection,
        rest_voronoi_lengths,
        internal_couple,
        bend_matrix,
        kappa,
        rest_kappa,
    )
    # Compute dilatation rate when needed, dilatation itself is done before
    # in internal_stresses
    _compute_dilatation_rate(
        position_collection, velocity_collection, lengths, rest_lengths, dilatation_rate
    )

    # FIXME: change memory overload instead for the below calls!
    voronoi_dilatation_inv_cube_cached = 1.0 / voronoi_dilatation**3
    # Delta(\tau_L / \Epsilon^3)
    bend_twist_couple_2D = difference_kernel_for_block_structure(
        internal_couple * voronoi_dilatation_inv_cube_cached, ghost_voronoi_idx
    )
    # \mathcal{A}[ (\kappa x \tau_L ) * \hat{D} / \Epsilon^3 ]
    bend_twist_couple_3D = quadrature_kernel_for_block_structure(
        _batch_cross(kappa, internal_couple)
        * rest_voronoi_lengths
        * voronoi_dilatation_inv_cube_cached,
        ghost_voronoi_idx,
    )
    # (Qt x n_L) * \hat{l}
    shear_stretch_couple = (
        _batch_cross(_batch_matvec(director_collection, tangents), internal_stress)
        * rest_lengths
    )

    # I apply common sub expression elimination here, as J w / e is used in both the lagrangian and dilatation
    # terms
    # TODO : the _batch_matvec kernel needs to depend on the representation of J, and should be coded as such
    J_omega_upon_e = (
        _batch_matvec(mass_second_moment_of_inertia, omega_collection) / dilatation
    )

    # (J \omega_L / e) x \omega_L
    # Warning : Do not do micro-optimization here : you can ignore dividing by dilatation as we later multiply by it
    # but this causes confusion and violates SRP
    lagrangian_transport = _batch_cross(J_omega_upon_e, omega_collection)

    # Note : in the computation of dilatation_rate, there is an optimization opportunity as dilatation rate has
    # a dilatation-like term in the numerator, which we cancel here
    # (J \omega_L / e^2) . (de/dt)
    unsteady_dilatation = J_omega_upon_e * dilatation_rate / dilatation

    blocksize = internal_torques.shape[1]
    for i in range(3):
        for k in range(blocksize):
            internal_torques[i, k] = (
                bend_twist_couple_2D[i, k]
                + bend_twist_couple_3D[i, k]
                + shear_stretch_couple[i, k]
                + lagrangian_transport[i, k]
                + unsteady_dilatation[i, k]
            )


@numba.njit(cache=True)  # type: ignore
def _update_accelerations(
    acceleration_collection: NDArray[np.float64],
    internal_forces: NDArray[np.float64],
    external_forces: NDArray[np.float64],
    mass: NDArray[np.float64],
    alpha_collection: NDArray[np.float64],
    inv_mass_second_moment_of_inertia: NDArray[np.float64],
    internal_torques: NDArray[np.float64],
    external_torques: NDArray[np.float64],
    dilatation: NDArray[np.float64],
) -> None:
    """
    Update <acceleration and angular acceleration> given <internal force/torque and external force/torque>.
    """

    blocksize_acc = internal_forces.shape[1]
    blocksize_alpha = internal_torques.shape[1]

    for i in range(3):
        for k in range(blocksize_acc):
            acceleration_collection[i, k] = (
                internal_forces[i, k] + external_forces[i, k]
            ) / mass[k]

    alpha_collection *= 0.0
    for i in range(3):
        for j in range(3):
            for k in range(blocksize_alpha):
                alpha_collection[i, k] += (
                    inv_mass_second_moment_of_inertia[i, j, k]
                    * (internal_torques[j, k] + external_torques[j, k])
                ) * dilatation[k]


@numba.njit(cache=True)  # type: ignore
def _zeroed_out_external_forces_and_torques(
    external_forces: NDArray[np.float64], external_torques: NDArray[np.float64]
) -> None:
    """
    This function is to zeroed out external forces and torques.

    Notes
    -----
    Microbenchmark results 100 elements
    python version: 3.32 µs ± 44.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    this version: 583 ns ± 1.94 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    """
    n_nodes = external_forces.shape[1]
    n_elems = external_torques.shape[1]

    for i in range(3):
        for k in range(n_nodes):
            external_forces[i, k] = 0.0

    for i in range(3):
        for k in range(n_elems):
            external_torques[i, k] = 0.0


if TYPE_CHECKING:
    _: CosseratRodProtocol = CosseratRod.straight_rod(
        3,
        np.zeros(3),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        1.0,
        0.1,
        1.0,
        youngs_modulus=1.0,
    )
    _: CosseratRodProtocol = CosseratRod.ring_rod(  # type: ignore[no-redef]
        3,
        np.zeros(3),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        1.0,
        0.1,
        1.0,
        youngs_modulus=1.0,
    )
