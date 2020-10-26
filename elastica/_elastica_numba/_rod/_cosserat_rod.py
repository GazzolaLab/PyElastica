__doc__ = """ Cosserat rod equations implementation for Elactica Numba Implementation"""
__all__ = ["CosseratRod"]
import numpy as np
import functools
import numba
from elastica.rod import RodBase
from elastica._elastica_numba._rod._data_structures import _RodSymplecticStepperMixin
from elastica._elastica_numba._linalg import (
    _batch_cross,
    _batch_norm,
    _batch_dot,
    _batch_matvec,
)
from elastica._elastica_numba._rotations import _inv_rotate
from elastica.rod.factory_function import allocate
from elastica._calculus import (
    quadrature_kernel,
    difference_kernel,
    _difference,
    _average,
)
from elastica._elastica_numba._interaction import node_to_element_pos_or_vel

position_difference_kernel = _difference
position_average = _average


@functools.lru_cache(maxsize=1)
def _get_z_vector():
    return np.array([0.0, 0.0, 1.0]).reshape(3, -1)


class CosseratRod(RodBase, _RodSymplecticStepperMixin):
    def __init__(
        self,
        n_elements,
        _vector_states,
        _matrix_states,
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
        self._vector_states = _vector_states
        self._matrix_states = _matrix_states
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

        _RodSymplecticStepperMixin.__init__(self)

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
        n_elements,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        poisson_ratio,
        alpha_c=4.0 / 3.0,
        *args,
        **kwargs
    ):

        (
            n_elements,
            _vector_states,
            _matrix_states,
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
            poisson_ratio,
            alpha_c=4.0 / 3.0,
            *args,
            **kwargs
        )

        return cls(
            n_elements,
            _vector_states,
            _matrix_states,
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

    def _compute_internal_forces_and_torques(self, time):
        """
        Compute internal forces and torques. We need to compute internal forces and torques before the acceleration because
        they are used in interaction. Thus in order to speed up simulation, we will compute internal forces and torques
        one time and use them. Previously, we were computing internal forces and torques multiple times in interaction.
        Saving internal forces and torques in a variable take some memory, but we will gain speed up.
        Parameters
        ----------
        time

        Returns
        -------

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
        )

    # Interface to time-stepper mixins (Symplectic, Explicit), which calls this method
    def update_accelerations(self, time):
        """
        This class method function is only a wrapper to call Numba njit function, which
        updates the acceleration

        Parameters
        ----------
        time

        Returns
        -------

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

    def compute_translational_energy(self):
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
        J_omega_upon_e = (
            _batch_matvec(self.mass_second_moment_of_inertia, self.omega_collection)
            / self.dilatation
        )
        return 0.5 * np.einsum("ik,ik->k", self.omega_collection, J_omega_upon_e).sum()

    def compute_velocity_center_of_mass(self):
        mass_times_velocity = np.einsum("j,ij->ij", self.mass, self.velocity_collection)
        sum_mass_times_velocity = np.einsum("ij->i", mass_times_velocity)

        return sum_mass_times_velocity / self.mass.sum()

    def compute_position_center_of_mass(self):
        mass_times_position = np.einsum("j,ij->ij", self.mass, self.position_collection)
        sum_mass_times_position = np.einsum("ij->i", mass_times_position)

        return sum_mass_times_position / self.mass.sum()

    def compute_bending_energy(self):

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

        sigma_diff = self.sigma - self.rest_sigma
        shear_internal_torques = _batch_matvec(self.shear_matrix, sigma_diff)

        return (
            0.5
            * (_batch_dot(sigma_diff, shear_internal_torques) * self.rest_lengths).sum()
        )


@numba.njit(cache=True)
def _compute_geometry_from_state(
    position_collection, volume, lengths, tangents, radius
):
    """
    Returns
    -------

    """
    # Compute eq (3.3) from 2018 RSOS paper

    # Note : we can use the two-point difference kernel, but it needs unnecessary padding
    # and hence will always be slower
    # FIXME: change memory overload instead for the below calls!
    position_diff = position_difference_kernel(position_collection)
    lengths[:] = _batch_norm(position_diff)
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
    Compute element and Voronoi region dilatations
    Returns
    -------

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

    Returns
    -------

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
    Linear force functional
    Operates on
    S : (3,3,n) tensor and sigma (3,n)

    Returns
    -------

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
    Linear force functional
    Operates on
    B : (3,3,n) tensor and curvature kappa (3,n)

    Returns
    -------

    """
    _compute_bending_twist_strains(
        director_collection, rest_voronoi_lengths, kappa
    )  # concept : needs to compute kappa
    internal_couple[:] = _batch_matvec(bend_matrix, kappa - rest_kappa)


@numba.njit(cache=True)
def _compute_damping_forces(
    damping_forces, velocity_collection, dissipation_constant_for_forces, lengths
):
    # Internal damping foces.
    elemental_velocities = node_to_element_pos_or_vel(velocity_collection)

    blocksize = elemental_velocities.shape[1]
    elemental_damping_forces = np.zeros((3, blocksize))

    for i in range(3):
        for k in range(blocksize):
            elemental_damping_forces[i, k] = (
                dissipation_constant_for_forces[k]
                * elemental_velocities[i, k]
                * lengths[k]
            )

    damping_forces[:] = quadrature_kernel(elemental_damping_forces)


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
):
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
        damping_forces, velocity_collection, dissipation_constant_for_forces, lengths
    )

    internal_forces[:] = difference_kernel(cosserat_internal_stress) - damping_forces


@numba.njit(cache=True)
def _compute_damping_torques(
    damping_torques, omega_collection, dissipation_constant_for_torques, lengths
):
    blocksize = damping_torques.shape[1]
    for i in range(3):
        for k in range(blocksize):
            damping_torques[i, k] = (
                dissipation_constant_for_torques[k]
                * omega_collection[i, k]
                * lengths[k]
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
):
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
    bend_twist_couple_2D = difference_kernel(
        internal_couple * voronoi_dilatation_inv_cube_cached
    )
    # \mathcal{A}[ (\kappa x \tau_L ) * \hat{D} / \Epsilon^3 ]
    bend_twist_couple_3D = quadrature_kernel(
        _batch_cross(kappa, internal_couple)
        * rest_voronoi_lengths
        * voronoi_dilatation_inv_cube_cached
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
        damping_torques, omega_collection, dissipation_constant_for_torques, lengths
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

    blocksize_acc = internal_forces.shape[1]
    blocksize_alpha = internal_torques.shape[1]

    for i in range(3):
        for k in range(blocksize_acc):
            acceleration_collection[i, k] = (
                internal_forces[i, k] + external_forces[i, k]
            ) / mass[k]
            external_forces[i, k] = 0.0

    alpha_collection *= 0.0
    for i in range(3):
        for j in range(3):
            for k in range(blocksize_alpha):
                alpha_collection[i, k] += (
                    inv_mass_second_moment_of_inertia[i, j, k]
                    * (internal_torques[j, k] + external_torques[j, k])
                ) * dilatation[k]

    # Reset torques
    external_torques *= 0.0
