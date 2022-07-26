__doc__ = """ Rod classes and implementation details """
__all__ = ["CosseratRod"]

import typing

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
from elastica.rod.factory_function import allocate
from elastica.rod.knot_theory import KnotTheory
from elastica._calculus import (
    quadrature_kernel_for_block_structure,
    difference_kernel_for_block_structure,
    _difference,
    _average,
)

position_difference_kernel = _difference
position_average = _average


@functools.lru_cache(maxsize=1)
def _get_z_vector():
    return np.array([0.0, 0.0, 1.0]).reshape(3, -1)


class CosseratRod(RodBase, KnotTheory):
    """
    Cosserat Rod class. This is the preferred class for rods because it is derived from some
    of the essential base classes.

        Attributes
        ----------
        n_elems: int
            The number of elements of the rod.
        position_collection: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            Array containing node position vectors.
        velocity_collection: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            Array containing node velocity vectors.
        acceleration_collection: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            Array containing node acceleration vectors.
        omega_collection: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            Array containing element angular velocity vectors.
        alpha_collection: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            Array contining element angular acceleration vectors.
        director_collection: numpy.ndarray
            3D (dim, dim, n_elems) array containing data with 'float' type.
            Array containing element director matrices.
        rest_lengths: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element lengths at rest configuration.
        density: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod elements densities.
        volume: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element volumes.
        mass: numpy.ndarray
            1D (n_nodes) array containing data with 'float' type.
            Rod node masses. Note that masses are stored on the nodes, not on elements.
        mass_second_moment_of_inertia: numpy.ndarray
            3D (dim, dim, n_elems) array containing data with 'float' type.
            Rod element mass second moment of interia.
        inv_mass_second_moment_of_inertia: numpy.ndarray
            3D (dim, dim, n_elems) array containing data with 'float' type.
            Rod element inverse mass moment of inertia.
        dissipation_constant_for_forces: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element dissipation coefficient (nu).
        dissipation_constant_for_torques: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element dissipation (nu).
            Can be customized by passing 'nu_for_torques'.
        rest_voronoi_lengths: numpy.ndarray
            1D (n_voronoi) array containing data with 'float' type.
            Rod lengths on the voronoi domain at the rest configuration.
        internal_forces: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            Rod node internal forces. Note that internal forces are stored on the node, not on elements.
        internal_torques: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            Rod element internal torques.
        external_forces: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            External forces acting on rod nodes.
        external_torques: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            External torques acting on rod elements.
        lengths: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element lengths.
        tangents: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            Rod element tangent vectors.
        radius: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element radius.
        dilatation: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element dilatation.
        voronoi_dilatation: numpy.ndarray
            1D (n_voronoi) array containing data with 'float' type.
            Rod dilatation on voronoi domain.
        dilatation_rate: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element dilatation rates.
    """

    def __init__(
        self,
        n_elements,
        position,
        velocity,
        omega,
        acceleration,
        angular_acceleration,
        directors,
        radius,
        mass_second_moment_of_inertia,
        inv_mass_second_moment_of_inertia,
        shear_matrix,
        bend_matrix,
        density,
        volume,
        mass,
        dissipation_constant_for_forces,
        dissipation_constant_for_torques,
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
        damping_forces,
        damping_torques,
    ):
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
        self.density = density
        self.volume = volume
        self.mass = mass
        self.dissipation_constant_for_forces = dissipation_constant_for_forces
        self.dissipation_constant_for_torques = dissipation_constant_for_torques
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
        self.damping_forces = damping_forces
        self.damping_torques = damping_torques

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
        start: np.ndarray,
        direction: np.ndarray,
        normal: np.ndarray,
        base_length: float,
        base_radius: float,
        density: float,
        nu: float,
        youngs_modulus: float,
        *args,
        **kwargs,
    ):
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
        start : NDArray[3, float]
            Starting coordinate in 3D
        direction : NDArray[3, float]
            Direction of the rod in 3D
        normal : NDArray[3, float]
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
        *args : tuple
            Additional arguments should be passed as keyward arguments.
            (e.g. shear_modulus, poisson_ratio)
        **kwargs : dict, optional
            The "position" and/or "directors" can be overrided by passing "position" and "directors" argument. Remember, the shape of the "position" is (3,n_elements+1) and the shape of the "directors" is (3,3,n_elements).

        Returns
        -------
        CosseratRod

        """

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
            density,
            volume,
            mass,
            dissipation_constant_for_forces,
            dissipation_constant_for_torques,
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
            damping_forces,
            damping_torques,
        ) = allocate(
            n_elements,
            start,
            direction,
            normal,
            base_length,
            base_radius,
            density,
            nu,
            youngs_modulus,
            *args,
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
            density,
            volume,
            mass,
            dissipation_constant_for_forces,
            dissipation_constant_for_torques,
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
            damping_forces,
            damping_torques,
        )

    def compute_internal_forces_and_torques(self, time):
        """
        Compute internal forces and torques. We need to compute internal forces and torques before the acceleration because
        they are used in interaction. Thus in order to speed up simulation, we will compute internal forces and torques
        one time and use them. Previously, we were computing internal forces and torques multiple times in interaction.
        Saving internal forces and torques in a variable take some memory, but we will gain speed up.

        Parameters
        ----------
        time: float
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
            self.velocity_collection,
            self.dissipation_constant_for_forces,
            self.damping_forces,
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
            self.dissipation_constant_for_torques,
            self.damping_torques,
            self.internal_torques,
            self.ghost_voronoi_idx,
        )

    # Interface to time-stepper mixins (Symplectic, Explicit), which calls this method
    def update_accelerations(self, time):
        """
        Updates the acceleration variables

        Parameters
        ----------
        time: float
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

    def zeroed_out_external_forces_and_torques(self, time):
        _zeroed_out_external_forces_and_torques(
            self.external_forces, self.external_torques
        )

    def compute_translational_energy(self):
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

    def compute_rotational_energy(self):
        """
        Compute total rotational energy of the rod at the instance.
        """
        J_omega_upon_e = (
            _batch_matvec(self.mass_second_moment_of_inertia, self.omega_collection)
            / self.dilatation
        )
        return 0.5 * np.einsum("ik,ik->k", self.omega_collection, J_omega_upon_e).sum()

    def compute_velocity_center_of_mass(self):
        """
        Compute velocity center of mass of the rod at the instance.
        """
        mass_times_velocity = np.einsum("j,ij->ij", self.mass, self.velocity_collection)
        sum_mass_times_velocity = np.einsum("ij->i", mass_times_velocity)

        return sum_mass_times_velocity / self.mass.sum()

    def compute_position_center_of_mass(self):
        """
        Compute position center of mass of the rod at the instance.
        """
        mass_times_position = np.einsum("j,ij->ij", self.mass, self.position_collection)
        sum_mass_times_position = np.einsum("ij->i", mass_times_position)

        return sum_mass_times_position / self.mass.sum()

    def compute_bending_energy(self):
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

    def compute_shear_energy(self):
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


@numba.njit(cache=True)
def _compute_geometry_from_state(
    position_collection, volume, lengths, tangents, radius
):
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


@numba.njit(cache=True)
def _compute_all_dilatations(
    position_collection,
    volume,
    lengths,
    tangents,
    radius,
    dilatation,
    rest_lengths,
    rest_voronoi_lengths,
    voronoi_dilatation,
):
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


@numba.njit(cache=True)
def _compute_dilatation_rate(
    position_collection, velocity_collection, lengths, rest_lengths, dilatation_rate
):
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


@numba.njit(cache=True)
def _compute_shear_stretch_strains(
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
):
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


@numba.njit(cache=True)
def _compute_internal_shear_stretch_stresses_from_model(
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
):
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


@numba.njit(cache=True)
def _compute_bending_twist_strains(director_collection, rest_voronoi_lengths, kappa):
    """
    Update <curvature/twist (kappa)> given <director and rest_voronoi_length>.
    """
    temp = _inv_rotate(director_collection)
    blocksize = rest_voronoi_lengths.shape[0]
    for k in range(blocksize):
        kappa[0, k] = temp[0, k] / rest_voronoi_lengths[k]
        kappa[1, k] = temp[1, k] / rest_voronoi_lengths[k]
        kappa[2, k] = temp[2, k] / rest_voronoi_lengths[k]


@numba.njit(cache=True)
def _compute_internal_bending_twist_stresses_from_model(
    director_collection,
    rest_voronoi_lengths,
    internal_couple,
    bend_matrix,
    kappa,
    rest_kappa,
):
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


@numba.njit(cache=True)
def _compute_damping_forces(
    damping_forces,
    velocity_collection,
    dissipation_constant_for_forces,
):
    """
    Update <damping force> given <velocity>
    """

    # Internal damping forces.
    blocksize = velocity_collection.shape[1]

    for i in range(3):
        for k in range(blocksize):
            damping_forces[i, k] = (
                dissipation_constant_for_forces[k] * velocity_collection[i, k]
            )


@numba.njit(cache=True)
def _compute_internal_forces(
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
    velocity_collection,
    dissipation_constant_for_forces,
    damping_forces,
    internal_forces,
    ghost_elems_idx,
):
    """
    Update <internal force> given <director, internal_stress, velocity and damping force>.
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

    _compute_damping_forces(
        damping_forces,
        velocity_collection,
        dissipation_constant_for_forces,
    )

    internal_forces[:] = (
        difference_kernel_for_block_structure(cosserat_internal_stress, ghost_elems_idx)
        - damping_forces
    )


@numba.njit(cache=True)
def _compute_damping_torques(
    damping_torques, omega_collection, dissipation_constant_for_torques
):
    """
    Update <damping torque> given <angular velocity>.
    """
    blocksize = omega_collection.shape[1]
    for i in range(3):
        for k in range(blocksize):
            damping_torques[i, k] = (
                dissipation_constant_for_torques[k] * omega_collection[i, k]
            )


@numba.njit(cache=True)
def _compute_internal_torques(
    position_collection,
    velocity_collection,
    tangents,
    lengths,
    rest_lengths,
    director_collection,
    rest_voronoi_lengths,
    bend_matrix,
    rest_kappa,
    kappa,
    voronoi_dilatation,
    mass_second_moment_of_inertia,
    omega_collection,
    internal_stress,
    internal_couple,
    dilatation,
    dilatation_rate,
    dissipation_constant_for_torques,
    damping_torques,
    internal_torques,
    ghost_voronoi_idx,
):
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
    voronoi_dilatation_inv_cube_cached = 1.0 / voronoi_dilatation ** 3
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

    _compute_damping_torques(
        damping_torques, omega_collection, dissipation_constant_for_torques
    )

    blocksize = internal_torques.shape[1]
    for i in range(3):
        for k in range(blocksize):
            internal_torques[i, k] = (
                bend_twist_couple_2D[i, k]
                + bend_twist_couple_3D[i, k]
                + shear_stretch_couple[i, k]
                + lagrangian_transport[i, k]
                + unsteady_dilatation[i, k]
                - damping_torques[i, k]
            )


@numba.njit(cache=True)
def _update_accelerations(
    acceleration_collection,
    internal_forces,
    external_forces,
    mass,
    alpha_collection,
    inv_mass_second_moment_of_inertia,
    internal_torques,
    external_torques,
    dilatation,
):
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


@numba.njit(cache=True)
def _zeroed_out_external_forces_and_torques(external_forces, external_torques):
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
