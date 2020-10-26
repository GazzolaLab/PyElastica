__doc__ = """ Cosserat rod equations implementation for Elactica Numpy Implementation"""
__all__ = ["CosseratRod"]
import numpy as np
import functools
from elastica.rod import RodBase
from elastica._elastica_numpy._rod._data_structures import _RodSymplecticStepperMixin
from elastica._elastica_numpy._linalg import (
    _batch_cross,
    _batch_norm,
    _batch_dot,
    _batch_matvec,
)
from elastica._elastica_numpy._rotations import _inv_rotate
from elastica.rod.factory_function import allocate
from elastica._calculus import quadrature_kernel, difference_kernel


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

        self._compute_shear_stretch_strains()
        self._compute_bending_twist_strains()

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

    def _compute_geometry_from_state(self):
        """
        Returns
        -------

        """
        # Compute eq (3.3) from 2018 RSOS paper

        # Note : we can use the two-point difference kernel, but it needs unnecessary padding
        # and hence will always be slower
        position_diff = (
            self.position_collection[..., 1:] - self.position_collection[..., :-1]
        )
        self.lengths = _batch_norm(position_diff)
        self.tangents = position_diff / self.lengths
        # resize based on volume conservation
        self.radius = np.sqrt(self.volume / self.lengths / np.pi)

    def _compute_all_dilatations(self):
        """
        Compute element and Voronoi region dilatations
        Returns
        -------

        """
        # compute_geometry_from_state(self.position_collection, self.volume, self.lengths, self.tangents, self.radius)
        self._compute_geometry_from_state()
        # Caveat : Needs already set rest_lengths and rest voronoi domain lengths
        # Put in initialization
        self.dilatation = self.lengths / self.rest_lengths

        # Cmopute eq (3.4) from 2018 RSOS paper

        # Note : we can use trapezoidal kernel, but it has padding and will be slower
        voronoi_lengths = 0.5 * (self.lengths[1:] + self.lengths[:-1])

        # Cmopute eq (3.45 from 2018 RSOS paper
        self.voronoi_dilatation = voronoi_lengths / self.rest_voronoi_lengths

    def _compute_dilatation_rate(self):
        """

        Returns
        -------

        """
        r_dot_v = _batch_dot(self.position_collection, self.velocity_collection)
        r_plus_one_dot_v = _batch_dot(
            self.position_collection[..., 1:], self.velocity_collection[..., :-1]
        )
        r_dot_v_plus_one = _batch_dot(
            self.position_collection[..., :-1], self.velocity_collection[..., 1:]
        )

        self.dilatation_rate = (
            (r_dot_v[:-1] + r_dot_v[1:] - r_dot_v_plus_one - r_plus_one_dot_v)
            / self.lengths
            / self.rest_lengths
        )

    def _compute_shear_stretch_strains(self):
        # Quick trick : Instead of evaliation Q(et-d^3), use property that Q*d3 = (0,0,1), a constant
        self._compute_all_dilatations()
        # FIXME: change memory overload instead for the below calls!
        self.sigma = (
            self.dilatation * _batch_matvec(self.director_collection, self.tangents)
            - _get_z_vector()
        )

    def _compute_internal_shear_stretch_stresses_from_model(self):
        """
        Linear force functional
        Operates on
        S : (3,3,n) tensor and sigma (3,n)

        Returns
        -------

        """
        self._compute_shear_stretch_strains()  # concept : needs to compute sigma
        self.internal_stress = _batch_matvec(
            self.shear_matrix, self.sigma - self.rest_sigma
        )

    def _compute_bending_twist_strains(self):
        # Note: dilatations are computed previously inside ` _compute_all_dilatations `
        self.kappa = _inv_rotate(self.director_collection) / self.rest_voronoi_lengths

    def _compute_internal_bending_twist_stresses_from_model(self):
        """
        Linear force functional
        Operates on
        B : (3,3,n) tensor and curvature kappa (3,n)

        Returns
        -------

        """
        self._compute_bending_twist_strains()  # concept : needs to compute kappa
        self.internal_couple = _batch_matvec(
            self.bend_matrix, self.kappa - self.rest_kappa
        )

    def _compute_damping_forces(self):
        # Internal damping forces.
        elemental_velocities = 0.5 * (
            self.velocity_collection[..., :-1] + self.velocity_collection[..., 1:]
        )
        elemental_damping_forces = (
            self.dissipation_constant_for_forces * elemental_velocities * self.lengths
        )
        self.damping_forces = quadrature_kernel(elemental_damping_forces)

    def _compute_internal_forces(self):
        # Compute n_l and cache it using internal_stress
        # Be careful about usage though
        self._compute_internal_shear_stretch_stresses_from_model()

        # Signifies Q^T n_L / e
        # Not using batch matvec as I don't want to take directors.T here
        # FIXME: change memory overload instead for the below calls!
        cosserat_internal_stress = (
            np.einsum("jik, jk->ik", self.director_collection, self.internal_stress)
            / self.dilatation  # computed in comp_dilatation <- compute_strain <- compute_stress
        )
        # self._compute_internal_forces()
        self._compute_damping_forces()

        return difference_kernel(cosserat_internal_stress) - self.damping_forces

    def _compute_damping_torques(self):
        # Internal damping torques
        self.damping_torques = (
            self.dissipation_constant_for_torques * self.omega_collection * self.lengths
        )

    def _compute_internal_torques(self):
        # Compute \tau_l and cache it using internal_couple
        # Be careful about usage though
        self._compute_internal_bending_twist_stresses_from_model()

        # Compute dilatation rate when needed, dilatation itself is done before
        # in internal_stresses
        self._compute_dilatation_rate()

        # FIXME: change memory overload instead for the below calls!
        voronoi_dilatation_inv_cube_cached = 1.0 / self.voronoi_dilatation ** 3
        # Delta(\tau_L / \Epsilon^3)
        bend_twist_couple_2D = difference_kernel(
            self.internal_couple * voronoi_dilatation_inv_cube_cached
        )
        # \mathcal{A}[ (\kappa x \tau_L ) * \hat{D} / \Epsilon^3 ]
        bend_twist_couple_3D = quadrature_kernel(
            _batch_cross(self.kappa, self.internal_couple)
            * self.rest_voronoi_lengths
            * voronoi_dilatation_inv_cube_cached
        )
        # (Qt x n_L) * \hat{l}
        shear_stretch_couple = (
            _batch_cross(
                _batch_matvec(self.director_collection, self.tangents),
                self.internal_stress,
            )
            * self.rest_lengths
        )

        # I apply common sub expression elimination here, as J w / e is used in both the lagrangian and dilatation
        # terms
        # TODO : the _batch_matvec kernel needs to depend on the representation of J, and should be coded as such
        J_omega_upon_e = (
            _batch_matvec(self.mass_second_moment_of_inertia, self.omega_collection)
            / self.dilatation
        )

        # (J \omega_L / e) x \omega_L
        # Warning : Do not do micro-optimization here : you can ignore dividing by dilatation as we later multiply by it
        # but this causes confusion and violates SRP
        lagrangian_transport = _batch_cross(J_omega_upon_e, self.omega_collection)

        # Note : in the computation of dilatation_rate, there is an optimization opportunity as dilatation rate has
        # a dilatation-like term in the numerator, which we cancel here
        # (J \omega_L / e^2) . (de/dt)
        unsteady_dilatation = J_omega_upon_e * self.dilatation_rate / self.dilatation

        # Compute damping torques
        self._compute_damping_torques()

        return (
            bend_twist_couple_2D
            + bend_twist_couple_3D
            + shear_stretch_couple
            + lagrangian_transport
            + unsteady_dilatation
            - self.damping_torques
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
        # FIXME: change memory overload instead for the below calls!
        self.internal_forces = self._compute_internal_forces()

        self.internal_torques = self._compute_internal_torques()

    # Interface to time-stepper mixins (Symplectic, Explicit), which calls this method
    def update_accelerations(self, time):
        """TODO Do we need to make the collection members abstract?

        Parameters
        ----------
        time

        Returns
        -------

        """
        np.copyto(
            self.acceleration_collection,
            (self.internal_forces + self.external_forces) / self.mass,
        )
        np.copyto(
            self.alpha_collection,
            _batch_matvec(
                self.inv_mass_second_moment_of_inertia,
                (self.internal_torques + self.external_torques),
            )
            * self.dilatation,
        )

        # Reset forces and torques
        self.external_forces *= 0.0
        self.external_torques *= 0.0

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
