__doc__ = """ Rod base classes and implementation details that need to be hidden from the user"""
__all__ = ["CosseratRod"]
# import numpy as np
# import functools
#
# from elastica._linalg import _batch_matvec, _batch_cross, _batch_norm, _batch_dot
#
# # FIXME: when cosserat rod for numba and numpy written do import from correct places
# from elastica._elastica_numba._calculus import _difference, _average
#
# from elastica._calculus import (
#     quadrature_kernel,
#     difference_kernel,
# )
# from elastica._rotations import _inv_rotate
# from elastica.utils import MaxDimension, Tolerance
#
# from elastica.rod import RodBase
# from elastica.rod.constitutive_model import _LinearConstitutiveModelMixin
# from elastica.rod.data_structures import _RodSymplecticStepperMixin
#
# # from ..interaction import node_to_element_velocity
# from elastica._elastica_numba._interaction import node_to_element_pos_or_vel
#
# # TODO Add documentation for all functions
# import numba
# from numpy import pi

# position_difference_kernel = _difference
# position_average = _average

from elastica import IMPORT_NUMBA

if IMPORT_NUMBA:
    from elastica._elastica_numba._rod._cosserat_rod import CosseratRod
else:
    from elastica._elastica_numpy._rod._cosserat_rod import CosseratRod


# @functools.lru_cache(maxsize=1)
# def _get_z_vector():
#     return np.array([0.0, 0.0, 1.0]).reshape(3, -1)
#
#
# class _CosseratRodBase(RodBase):
#     # I'm assuming number of elements can be deduced from the size of the inputs
#     def __init__(
#         self,
#         n_elements,
#         position,
#         directors,
#         rest_lengths,
#         density,
#         volume,
#         mass_second_moment_of_inertia,
#         nu,
#         *args,
#         **kwargs
#     ):
#         velocities = np.zeros((MaxDimension.value(), n_elements + 1))
#         omegas = np.zeros((MaxDimension.value(), n_elements))  # + 1e-16
#         accelerations = 0.0 * velocities
#         angular_accelerations = 0.0 * omegas
#         self.n_elems = n_elements
#         self._vector_states = np.hstack(
#             (position, velocities, omegas, accelerations, angular_accelerations)
#         )
#         self._matrix_states = directors.copy()
#         # initial set to zero; if coming through kwargs then modify
#         self.rest_lengths = rest_lengths
#         self.density = density
#         self.volume = volume
#
#         self.mass = np.zeros(n_elements + 1)
#         self.mass[:-1] += 0.5 * self.density * self.volume
#         self.mass[1:] += 0.5 * self.density * self.volume
#
#         self.mass_second_moment_of_inertia = mass_second_moment_of_inertia
#
#         self.inv_mass_second_moment_of_inertia = np.zeros(
#             (MaxDimension.value(), MaxDimension.value(), n_elements)
#         )
#         for i in range(n_elements):
#             # Check rank of mass moment of inertia matrix to see if it is invertible
#             assert (
#                 np.linalg.matrix_rank(mass_second_moment_of_inertia[..., i])
#                 == MaxDimension.value()
#             )
#             self.inv_mass_second_moment_of_inertia[..., i] = np.linalg.inv(
#                 mass_second_moment_of_inertia[..., i]
#             )
#
#         self.nu = nu
#         self.rest_voronoi_lengths = 0.5 * (
#             self.rest_lengths[1:] + self.rest_lengths[:-1]
#         )
#         # calculated in `_compute_internal_forces_and_torques`
#         self.internal_forces = 0 * accelerations
#         self.internal_torques = 0 * angular_accelerations
#
#         # will apply external force and torques externally
#         self.external_forces = 0 * accelerations
#         self.external_torques = 0 * angular_accelerations
#
#         # calculated in `compute_geometry_from_state`
#         # self.lengths = NotImplemented
#         # self.tangents = NotImplemented
#         # self.radius = NotImplemented
#
#         self.lengths = np.zeros((n_elements))
#         self.tangents = np.zeros((3, n_elements))
#         self.radius = np.zeros((n_elements))
#
#         # calculated in `compute_all_dilatatation`
#         # self.dilatation = NotImplemented
#         # self.voronoi_dilatation = NotImplemented
#         # self.dilatation_rate = NotImplemented
#
#         self.dilatation = np.zeros((n_elements))
#         self.voronoi_dilatation = np.zeros((n_elements - 1))
#         self.dilatation_rate = np.zeros((n_elements))
#
#         self.sigma = np.zeros((3, n_elements))
#         self.internal_stress = np.zeros((3, n_elements))
#         self.kappa = np.zeros((3, n_elements - 1))
#
#         self.internal_couple = np.zeros((3, n_elements - 1))
#
#         self.damping_forces = np.zeros((3, n_elements + 1))
#         self.damping_torques = np.zeros((3, n_elements))
#
#         self.internal_torques = np.zeros((3, n_elements))
#
#         self.compute_shear_stretch_strains_numba = compute_shear_stretch_strains
#         self.compute_bending_twist_strains_numba = compute_bending_twist_strains
#
#     @classmethod
#     def straight_rod(
#         cls,
#         n_elements,
#         start,
#         direction,
#         normal,
#         base_length,
#         base_radius,
#         density,
#         nu,
#         mass_second_moment_of_inertia,
#         *args,
#         **kwargs
#     ):
#         # sanity checks here
#         assert n_elements > 1
#         assert base_length > Tolerance.atol()
#         assert base_radius > Tolerance.atol()
#         assert density > Tolerance.atol()
#         assert nu >= 0.0
#         assert np.sqrt(np.dot(normal, normal)) > Tolerance.atol()
#         assert np.sqrt(np.dot(direction, direction)) > Tolerance.atol()
#         for i in range(0, MaxDimension.value()):
#             assert mass_second_moment_of_inertia[i, i] > Tolerance.atol()
#
#         end = start + direction * base_length
#         position = np.zeros((MaxDimension.value(), n_elements + 1))
#         for i in range(0, MaxDimension.value()):
#             position[i, ...] = np.linspace(start[i], end[i], num=n_elements + 1)
#
#         # compute rest lengths and tangents
#         position_diff = position[..., 1:] - position[..., :-1]
#         rest_lengths = np.sqrt(np.einsum("ij,ij->j", position_diff, position_diff))
#         tangents = position_diff / rest_lengths
#         normal /= np.sqrt(np.dot(normal, normal))
#
#         # set directors
#         # check this order once
#         directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elements))
#         normal_collection = np.repeat(normal[:, np.newaxis], n_elements, axis=1)
#         directors[0, ...] = normal_collection
#         directors[1, ...] = _batch_cross(tangents, normal_collection)
#         directors[2, ...] = tangents
#
#         volume = np.pi * base_radius ** 2 * rest_lengths
#
#         inertia_collection = np.repeat(
#             mass_second_moment_of_inertia[:, :, np.newaxis], n_elements, axis=2
#         )
#
#         # create rod
#         return cls(
#             n_elements,
#             position,
#             directors,
#             rest_lengths,
#             density,
#             volume,
#             inertia_collection,
#             nu,
#             *args,
#             **kwargs
#         )
#
#     def _compute_geometry_from_state(self):
#         """
#         Returns
#         -------
#
#         """
#         # Compute eq (3.3) from 2018 RSOS paper
#
#         # Note : we can use the two-point difference kernel, but it needs unnecessary padding
#         # and hence will always be slower
#         # position_diff = (
#         #     self.position_collection[..., 1:] - self.position_collection[..., :-1]
#         # )
#         # FIXME: change memory overload instead for the below calls!
#         position_diff = position_difference_kernel(self.position_collection)
#         # self.lengths = np.sqrt(np.einsum("ij,ij->j", position_diff, position_diff))
#         self.lengths = _batch_norm(position_diff)
#         self.tangents = position_diff / self.lengths
#         # resize based on volume conservation
#         self.radius = np.sqrt(self.volume / self.lengths / np.pi)
#
#     def _compute_all_dilatations(self):
#         """
#         Compute element and Voronoi region dilatations
#         Returns
#         -------
#
#         """
#         # compute_geometry_from_state(self.position_collection, self.volume, self.lengths, self.tangents, self.radius)
#         self._compute_geometry_from_state()
#         # Caveat : Needs already set rest_lengths and rest voronoi domain lengths
#         # Put in initialization
#         # FIXME: change memory overload instead for the below calls!
#         self.dilatation = self.lengths / self.rest_lengths
#
#         # Cmopute eq (3.4) from 2018 RSOS paper
#
#         # Note : we can use trapezoidal kernel, but it has padding and will be slower
#         # voronoi_lengths = 0.5 * (self.lengths[1:] + self.lengths[:-1])
#         voronoi_lengths = position_average(self.lengths)
#
#         # Cmopute eq (3.45 from 2018 RSOS paper
#         self.voronoi_dilatation = voronoi_lengths / self.rest_voronoi_lengths
#
#     def _compute_dilatation_rate(self):
#         """
#
#         Returns
#         -------
#
#         """
#         # TODO Use the vector formula rather than separating it out
#         # self.lengths = l_i = |r^{i+1} - r^{i}|
#         # r_dot_v = np.einsum(
#         #     "ij,ij->j", self.position_collection, self.velocity_collection
#         # )
#         # r_plus_one_dot_v = np.einsum(
#         #     "ij, ij->j",
#         #     self.position_collection[..., 1:],
#         #     self.velocity_collection[..., :-1],
#         # )
#         # r_dot_v_plus_one = np.einsum(
#         #     "ij, ij->j",
#         #     self.position_collection[..., :-1],
#         #     self.velocity_collection[..., 1:],
#         # )
#         # FIXME: change memory overload instead for the below calls!
#         r_dot_v = _batch_dot(self.position_collection, self.velocity_collection)
#         r_plus_one_dot_v = _batch_dot(
#             self.position_collection[..., 1:], self.velocity_collection[..., :-1]
#         )
#         r_dot_v_plus_one = _batch_dot(
#             self.position_collection[..., :-1], self.velocity_collection[..., 1:]
#         )
#
#         self.dilatation_rate = (
#             (r_dot_v[:-1] + r_dot_v[1:] - r_dot_v_plus_one - r_plus_one_dot_v)
#             / self.lengths
#             / self.rest_lengths
#         )
#
#     def _compute_shear_stretch_strains(self):
#         # Quick trick : Instead of evaliation Q(et-d^3), use property that Q*d3 = (0,0,1), a constant
#         self._compute_all_dilatations()
#         # compute_all_dilatations(
#         #     self.position_collection,
#         #     self.volume,
#         #     self.lengths,
#         #     self.tangents,
#         #     self.radius,
#         #     self.dilatation,
#         #     self.rest_lengths,
#         #     self.rest_voronoi_lengths,
#         #     self.voronoi_dilatation,
#         # )
#         # FIXME: change memory overload instead for the below calls!
#         self.sigma = (
#             self.dilatation * _batch_matvec(self.director_collection, self.tangents)
#             - _get_z_vector()
#         )
#
#     def _compute_bending_twist_strains(self):
#         # Note: dilatations are computed previously inside ` _compute_all_dilatations `
#         self.kappa = _inv_rotate(self.director_collection) / self.rest_voronoi_lengths
#
#     # @profile
#     def _compute_damping_forces(self):
#         # Internal damping foces.
#         # damping_forces = self.nu * self.velocity_collection
#         # damping_forces[..., 0] *= 0.5  # first and last nodes have half mass
#         # damping_forces[..., -1] *= 0.5  # first and last nodes have half mass
#         #
#         # return damping_forces
#
#         # elemental_velocities = 0.5 * (
#         #     self.velocity_collection[..., :-1] + self.velocity_collection[..., 1:]
#         # )
#         # # FIXME: change memory overload instead for the below calls!
#         # elemental_velocities = node_to_element_velocity(self.velocity_collection)
#         # elemental_damping_forces = self.nu * elemental_velocities * self.lengths
#         #
#         # # nodal_damping_forces = quadrature_kernel(elemental_damping_forces)
#         # # return nodal_damping_forces
#         # return quadrature_kernel(elemental_damping_forces)
#         return compute_damping_forces(self.velocity_collection, self.nu, self.lengths)
#
#     def _compute_internal_forces(self):
#         # Compute n_l and cache it using internal_stress
#         # Be careful about usage though
#         # self._compute_internal_shear_stretch_stresses_from_model()
#
#         compute_internal_shear_stretch_stresses_from_model(
#             self.position_collection,
#             self.volume,
#             self.lengths,
#             self.tangents,
#             self.radius,
#             self.rest_lengths,
#             self.rest_voronoi_lengths,
#             self.dilatation,
#             self.voronoi_dilatation,
#             self.director_collection,
#             self.sigma,
#             self.rest_sigma,
#             self.shear_matrix,
#             self.internal_stress,
#         )
#
#         # Signifies Q^T n_L / e
#         # Not using batch matvec as I don't want to take directors.T here
#         # FIXME: change memory overload instead for the below calls!
#         cosserat_internal_stress = (
#             np.einsum("jik, jk->ik", self.director_collection, self.internal_stress)
#             / self.dilatation  # computed in comp_dilatation <- compute_strain <- compute_stress
#         )
#         return difference_kernel(cosserat_internal_stress) - compute_damping_forces(
#             self.velocity_collection, self.nu, self.lengths
#         )  # self._compute_damping_forces()
#
#     def _compute_damping_torques(self):
#         # Internal damping torques
#         # damping_torques = self.nu * self.omega_collection * self.lengths
#         # return damping_torques
#         return self.nu * self.omega_collection * self.lengths
#
#     # @profile
#     def _compute_internal_torques(self):
#         # Compute \tau_l and cache it using internal_couple
#         # Be careful about usage though
#         # self._compute_internal_bending_twist_stresses_from_model()
#         compute_internal_bending_twist_stresses_from_model(
#             self.director_collection,
#             self.rest_voronoi_lengths,
#             self.internal_couple,
#             self.bend_matrix,
#             self.kappa,
#             self.rest_kappa,
#         )
#         # Compute dilatation rate when needed, dilatation itself is done before
#         # in internal_stresses
#         # self._compute_dilatation_rate()
#         compute_dilatation_rate(
#             self.position_collection,
#             self.velocity_collection,
#             self.lengths,
#             self.rest_lengths,
#             self.dilatation_rate,
#         )
#
#         # FIXME: change memory overload instead for the below calls!
#         voronoi_dilatation_inv_cube_cached = 1.0 / self.voronoi_dilatation ** 3
#         # Delta(\tau_L / \Epsilon^3)
#         bend_twist_couple_2D = difference_kernel(
#             self.internal_couple * voronoi_dilatation_inv_cube_cached
#         )
#         # \mathcal{A}[ (\kappa x \tau_L ) * \hat{D} / \Epsilon^3 ]
#         bend_twist_couple_3D = quadrature_kernel(
#             _batch_cross(self.kappa, self.internal_couple)
#             * self.rest_voronoi_lengths
#             * voronoi_dilatation_inv_cube_cached
#         )
#         # (Qt x n_L) * \hat{l}
#         shear_stretch_couple = (
#             _batch_cross(
#                 _batch_matvec(self.director_collection, self.tangents),
#                 self.internal_stress,
#             )
#             * self.rest_lengths
#         )
#
#         # I apply common sub expression elimination here, as J w / e is used in both the lagrangian and dilatation
#         # terms
#         # TODO : the _batch_matvec kernel needs to depend on the representation of J, and should be coded as such
#         J_omega_upon_e = (
#             _batch_matvec(self.mass_second_moment_of_inertia, self.omega_collection)
#             / self.dilatation
#         )
#
#         # (J \omega_L / e) x \omega_L
#         # Warning : Do not do micro-optimization here : you can ignore dividing by dilatation as we later multiply by it
#         # but this causes confusion and violates SRP
#         lagrangian_transport = _batch_cross(J_omega_upon_e, self.omega_collection)
#
#         # Note : in the computation of dilatation_rate, there is an optimization opportunity as dilatation rate has
#         # a dilatation-like term in the numerator, which we cancel here
#         # (J \omega_L / e^2) . (de/dt)
#         unsteady_dilatation = J_omega_upon_e * self.dilatation_rate / self.dilatation
#
#         compute_damping_torques(
#             self.nu, self.omega_collection, self.lengths, self.damping_torques
#         )
#
#         return (
#             bend_twist_couple_2D
#             + bend_twist_couple_3D
#             + shear_stretch_couple
#             + lagrangian_transport
#             + unsteady_dilatation
#             - self.damping_torques  # self._compute_damping_torques()
#         )
#
#     # @profile
#     def _compute_internal_forces_and_torques(self, time):
#         """
#         Compute internal forces and torques. We need to compute internal forces and torques before the acceleration because
#         they are used in interaction. Thus in order to speed up simulation, we will compute internal forces and torques
#         one time and use them. Previously, we were computing internal forces and torques multiple times in interaction.
#         Saving internal forces and torques in a variable take some memory, but we will gain speed up.
#         Parameters
#         ----------
#         time
#
#         Returns
#         -------
#
#         """
#         # FIXME: change memory overload instead for the below calls!
#         # self.internal_forces = self._compute_internal_forces()
#         self.internal_forces = compute_internal_forces(
#             self.position_collection,
#             self.volume,
#             self.lengths,
#             self.tangents,
#             self.radius,
#             self.rest_lengths,
#             self.rest_voronoi_lengths,
#             self.dilatation,
#             self.voronoi_dilatation,
#             self.director_collection,
#             self.sigma,
#             self.rest_sigma,
#             self.shear_matrix,
#             self.internal_stress,
#             self.velocity_collection,
#             self.nu,
#         )
#         # self.internal_torques = \
#         #     self._compute_internal_torques()
#
#         compute_internal_torques(
#             self.position_collection,
#             self.velocity_collection,
#             self.tangents,
#             self.lengths,
#             self.rest_lengths,
#             self.director_collection,
#             self.rest_voronoi_lengths,
#             self.bend_matrix,
#             self.rest_kappa,
#             self.kappa,
#             self.voronoi_dilatation,
#             self.mass_second_moment_of_inertia,
#             self.omega_collection,
#             self.internal_stress,
#             self.internal_couple,
#             self.dilatation,
#             self.dilatation_rate,
#             self.nu,
#             self.damping_torques,
#             self.internal_torques,
#         )
#
#     # @profile
#     # Interface to time-stepper mixins (Symplectic, Explicit), which calls this method
#     def update_accelerations(self, time):
#         """ TODO Do we need to make the collection members abstract?
#
#         Parameters
#         ----------
#         time
#
#         Returns
#         -------
#
#         """
#         # np.copyto(
#         #     self.acceleration_collection,
#         #     (self.internal_forces + self.external_forces) / self.mass,
#         # )
#         # np.copyto(
#         #     self.alpha_collection,
#         #     _batch_matvec(
#         #         self.inv_mass_second_moment_of_inertia,
#         #         (self.internal_torques + self.external_torques),
#         #     )
#         #     * self.dilatation,
#         # )
#         #
#         # # Reset forces and torques
#         # self.external_forces *= 0.0
#         # self.external_torques *= 0.0
#
#         _update_accelerations(
#             self.acceleration_collection,
#             self.internal_forces,
#             self.external_forces,
#             self.mass,
#             self.alpha_collection,
#             self.inv_mass_second_moment_of_inertia,
#             self.internal_torques,
#             self.external_torques,
#             self.dilatation,
#         )
#
#     def compute_translational_energy(self):
#         return (
#             0.5
#             * (
#                 self.mass
#                 * np.einsum(
#                     "ij, ij-> j", self.velocity_collection, self.velocity_collection
#                 )
#             ).sum()
#         )
#
#     def compute_rotational_energy(self):
#         J_omega_upon_e = (
#             _batch_matvec(self.mass_second_moment_of_inertia, self.omega_collection)
#             / self.dilatation
#         )
#         return 0.5 * np.einsum("ik,ik->k", self.omega_collection, J_omega_upon_e).sum()
#
#     def compute_velocity_center_of_mass(self):
#         mass_times_velocity = np.einsum("j,ij->ij", self.mass, self.velocity_collection)
#         sum_mass_times_velocity = np.einsum("ij->i", mass_times_velocity)
#
#         return sum_mass_times_velocity / self.mass.sum()
#
#     def compute_position_center_of_mass(self):
#         mass_times_position = np.einsum("j,ij->ij", self.mass, self.position_collection)
#         sum_mass_times_position = np.einsum("ij->i", mass_times_position)
#
#         return sum_mass_times_position / self.mass.sum()
#
#
# ################################ NUMBA FUNCTIONS ###########################################
# ############################################################################################
# @numba.njit(cache=True)
# def _update_accelerations(
#     acceleration_collection,
#     internal_forces,
#     external_forces,
#     mass,
#     alpha_collection,
#     inv_mass_second_moment_of_inertia,
#     internal_torques,
#     external_torques,
#     dilatation,
# ):
#
#     blocksize_acc = internal_forces.shape[1]
#     blocksize_alpha = internal_torques.shape[1]
#
#     for i in range(3):
#         for k in range(blocksize_acc):
#             acceleration_collection[i, k] = (
#                 internal_forces[i, k] + external_forces[i, k]
#             ) / mass[k]
#             external_forces[i, k] = 0.0
#
#     alpha_collection *= 0.0
#     for i in range(3):
#         for j in range(3):
#             for k in range(blocksize_alpha):
#                 alpha_collection[i, k] += (
#                     inv_mass_second_moment_of_inertia[i, j, k]
#                     * (internal_torques[j, k] + external_torques[j, k])
#                 ) * dilatation[k]
#
#     # Reset torques
#     external_torques *= 0.0
#
#
# @numba.njit(cache=True)
# def compute_damping_forces(velocity_collection, nu, lengths):
#     # Internal damping foces.
#     elemental_velocities = node_to_element_pos_or_vel(velocity_collection)
#
#     blocksize = elemental_velocities.shape[1]
#     elemental_damping_forces = np.zeros((3, blocksize))
#
#     for i in range(3):
#         for k in range(blocksize):
#             elemental_damping_forces[i, k] = (
#                 nu * elemental_velocities[i, k] * lengths[k]
#             )
#
#     # nodal_damping_forces = quadrature_kernel(elemental_damping_forces)
#     # return nodal_damping_forces
#     return quadrature_kernel(elemental_damping_forces)
#
#
# @numba.njit(cache=True)
# def compute_geometry_from_state(position_collection, volume, lengths, tangents, radius):
#     """
#     Returns
#     -------
#
#     """
#     # Compute eq (3.3) from 2018 RSOS paper
#
#     # Note : we can use the two-point difference kernel, but it needs unnecessary padding
#     # and hence will always be slower
#     # position_diff = (
#     #     self.position_collection[..., 1:] - self.position_collection[..., :-1]
#     # )
#     # FIXME: change memory overload instead for the below calls!
#     position_diff = position_difference_kernel(position_collection)
#     # self.lengths = np.sqrt(np.einsum("ij,ij->j", position_diff, position_diff))
#     lengths[:] = _batch_norm(position_diff)
#     for k in range(lengths.shape[0]):
#         tangents[0, k] = position_diff[0, k] / lengths[k]
#         tangents[1, k] = position_diff[1, k] / lengths[k]
#         tangents[2, k] = position_diff[2, k] / lengths[k]
#         # resize based on volume conservation
#         radius[k] = np.sqrt(volume[k] / lengths[k] / pi)
#
#
# @numba.njit(cache=True)
# def compute_all_dilatations(
#     position_collection,
#     volume,
#     lengths,
#     tangents,
#     radius,
#     dilatation,
#     rest_lengths,
#     rest_voronoi_lengths,
#     voronoi_dilatation,
# ):
#     """
#     Compute element and Voronoi region dilatations
#     Returns
#     -------
#
#     """
#     compute_geometry_from_state(position_collection, volume, lengths, tangents, radius)
#     # Caveat : Needs already set rest_lengths and rest voronoi domain lengths
#     # Put in initialization
#     # FIXME: change memory overload instead for the below calls!
#     for k in range(lengths.shape[0]):
#         dilatation[k] = lengths[k] / rest_lengths[k]
#
#     # Cmopute eq (3.4) from 2018 RSOS paper
#
#     # Note : we can use trapezoidal kernel, but it has padding and will be slower
#     # voronoi_lengths = 0.5 * (self.lengths[1:] + self.lengths[:-1])
#
#     voronoi_lengths = position_average(lengths)
#
#     # Cmopute eq (3.45 from 2018 RSOS paper
#     for k in range(voronoi_lengths.shape[0]):
#         voronoi_dilatation[k] = voronoi_lengths[k] / rest_voronoi_lengths[k]
#
#
# @numba.njit(cache=True)
# def compute_shear_stretch_strains(
#     position_collection,
#     volume,
#     lengths,
#     tangents,
#     radius,
#     rest_lengths,
#     rest_voronoi_lengths,
#     dilatation,
#     voronoi_dilatation,
#     director_collection,
#     sigma,
# ):
#     # Quick trick : Instead of evaliation Q(et-d^3), use property that Q*d3 = (0,0,1), a constant
#
#     compute_all_dilatations(
#         position_collection,
#         volume,
#         lengths,
#         tangents,
#         radius,
#         dilatation,
#         rest_lengths,
#         rest_voronoi_lengths,
#         voronoi_dilatation,
#     )
#
#     # FIXME: change memory overload instead for the below calls!
#     z_vector = np.array([0.0, 0.0, 1.0]).reshape(3, -1)
#     sigma[:] = dilatation * _batch_matvec(director_collection, tangents) - z_vector
#
#
# @numba.njit(cache=True)
# def compute_internal_shear_stretch_stresses_from_model(
#     position_collection,
#     volume,
#     lengths,
#     tangents,
#     radius,
#     rest_lengths,
#     rest_voronoi_lengths,
#     dilatation,
#     voronoi_dilatation,
#     director_collection,
#     sigma,
#     rest_sigma,
#     shear_matrix,
#     internal_stress,
# ):
#     """
#     Linear force functional
#     Operates on
#     S : (3,3,n) tensor and sigma (3,n)
#
#     Returns
#     -------
#
#     """
#     compute_shear_stretch_strains(
#         position_collection,
#         volume,
#         lengths,
#         tangents,
#         radius,
#         rest_lengths,
#         rest_voronoi_lengths,
#         dilatation,
#         voronoi_dilatation,
#         director_collection,
#         sigma,
#     )
#     # TODO : the _batch_matvec kernel needs to depend on the representation of Shearmatrix
#     # FIXME: change memory overload instead for the below calls!
#     internal_stress[:] = _batch_matvec(shear_matrix, sigma - rest_sigma)
#
#
# @numba.njit(cache=True)
# def compute_internal_forces(
#     position_collection,
#     volume,
#     lengths,
#     tangents,
#     radius,
#     rest_lengths,
#     rest_voronoi_lengths,
#     dilatation,
#     voronoi_dilatation,
#     director_collection,
#     sigma,
#     rest_sigma,
#     shear_matrix,
#     internal_stress,
#     velocity_collection,
#     nu,
# ):
#     # Compute n_l and cache it using internal_stress
#     # Be careful about usage though
#     compute_internal_shear_stretch_stresses_from_model(
#         position_collection,
#         volume,
#         lengths,
#         tangents,
#         radius,
#         rest_lengths,
#         rest_voronoi_lengths,
#         dilatation,
#         voronoi_dilatation,
#         director_collection,
#         sigma,
#         rest_sigma,
#         shear_matrix,
#         internal_stress,
#     )
#
#     # Signifies Q^T n_L / e
#     # Not using batch matvec as I don't want to take directors.T here
#     # FIXME: change memory overload instead for the below calls!
#
#     blocksize = internal_stress.shape[1]
#     cosserat_internal_stress = np.zeros((3, blocksize))
#
#     for i in range(3):
#         for j in range(3):
#             for k in range(blocksize):
#                 cosserat_internal_stress[i, k] += (
#                     director_collection[j, i, k] * internal_stress[j, k]
#                 )
#
#     cosserat_internal_stress /= dilatation
#
#     # cosserat_internal_stress = (
#     #     np.einsum("jik, jk->ik", director_collection, internal_stress)
#     #     / dilatation  # computed in comp_dilatation <- compute_strain <- compute_stress
#     # )
#     return difference_kernel(cosserat_internal_stress) - compute_damping_forces(
#         velocity_collection, nu, lengths
#     )
#
#
# @numba.njit(cache=True)
# def compute_dilatation_rate(
#     position_collection, velocity_collection, lengths, rest_lengths, dilatation_rate
# ):
#     """
#
#     Returns
#     -------
#
#     """
#     # TODO Use the vector formula rather than separating it out
#     # self.lengths = l_i = |r^{i+1} - r^{i}|
#     r_dot_v = _batch_dot(position_collection, velocity_collection)
#     r_plus_one_dot_v = _batch_dot(
#         position_collection[..., 1:], velocity_collection[..., :-1]
#     )
#     r_dot_v_plus_one = _batch_dot(
#         position_collection[..., :-1], velocity_collection[..., 1:]
#     )
#
#     blocksize = lengths.shape[0]
#
#     for k in range(blocksize):
#         dilatation_rate[k] = (
#             (r_dot_v[k] + r_dot_v[k + 1] - r_dot_v_plus_one[k] - r_plus_one_dot_v[k])
#             / lengths[k]
#             / rest_lengths[k]
#         )
#
#
# @numba.njit(cache=True)
# def compute_bending_twist_strains(director_collection, rest_voronoi_lengths, kappa):
#     temp = _inv_rotate(director_collection)
#     blocksize = rest_voronoi_lengths.shape[0]
#     for k in range(blocksize):
#         kappa[0, k] = temp[0, k] / rest_voronoi_lengths[k]
#         kappa[1, k] = temp[1, k] / rest_voronoi_lengths[k]
#         kappa[2, k] = temp[2, k] / rest_voronoi_lengths[k]
#
#
# @numba.njit(cache=True)
# def compute_internal_bending_twist_stresses_from_model(
#     director_collection,
#     rest_voronoi_lengths,
#     internal_couple,
#     bend_matrix,
#     kappa,
#     rest_kappa,
# ):
#     """
#     Linear force functional
#     Operates on
#     B : (3,3,n) tensor and curvature kappa (3,n)
#
#     Returns
#     -------
#
#     """
#     # _compute_bending_twist_strains(director_collection, rest_voronoi_lengths)  # concept : needs to compute kappa
#     compute_bending_twist_strains(director_collection, rest_voronoi_lengths, kappa)
#     # TODO : the _batch_matvec kernel needs to depend on the representation of Bendmatrix
#     internal_couple[:] = _batch_matvec(bend_matrix, kappa - rest_kappa)
#
#
# @numba.njit(cache=True)
# def compute_damping_torques(nu, omega_collection, lengths, damping_torques):
#     blocksize = damping_torques.shape[1]
#     for i in range(3):
#         for k in range(blocksize):
#             damping_torques[i, k] = nu * omega_collection[i, k] * lengths[k]
#
#
# @numba.njit(cache=True)
# def compute_internal_torques(
#     position_collection,
#     velocity_collection,
#     tangents,
#     lengths,
#     rest_lengths,
#     director_collection,
#     rest_voronoi_lengths,
#     bend_matrix,
#     rest_kappa,
#     kappa,
#     voronoi_dilatation,
#     mass_second_moment_of_inertia,
#     omega_collection,
#     internal_stress,
#     internal_couple,
#     dilatation,
#     dilatation_rate,
#     nu,
#     damping_torques,
#     internal_torques,
# ):
#     # Compute \tau_l and cache it using internal_couple
#     # Be careful about usage though
#     # internal_couple = _compute_internal_bending_twist_stresses_from_model(director_collection, rest_voronoi_lengths,bend_matrix, rest_kappa, kappa)
#     compute_internal_bending_twist_stresses_from_model(
#         director_collection,
#         rest_voronoi_lengths,
#         internal_couple,
#         bend_matrix,
#         kappa,
#         rest_kappa,
#     )
#     # Compute dilatation rate when needed, dilatation itself is done before
#     # in internal_stresses
#     # self._compute_dilatation_rate()
#     # _compute_dilatation_rate(position_collection, velocity_collection, lengths, rest_lengths)
#     compute_dilatation_rate(
#         position_collection, velocity_collection, lengths, rest_lengths, dilatation_rate
#     )
#
#     # FIXME: change memory overload instead for the below calls!
#     voronoi_dilatation_inv_cube_cached = 1.0 / voronoi_dilatation ** 3
#     # Delta(\tau_L / \Epsilon^3)
#     bend_twist_couple_2D = difference_kernel(
#         internal_couple * voronoi_dilatation_inv_cube_cached
#     )
#     # \mathcal{A}[ (\kappa x \tau_L ) * \hat{D} / \Epsilon^3 ]
#     bend_twist_couple_3D = quadrature_kernel(
#         _batch_cross(kappa, internal_couple)
#         * rest_voronoi_lengths
#         * voronoi_dilatation_inv_cube_cached
#     )
#     # (Qt x n_L) * \hat{l}
#     shear_stretch_couple = (
#         _batch_cross(_batch_matvec(director_collection, tangents), internal_stress,)
#         * rest_lengths
#     )
#
#     # I apply common sub expression elimination here, as J w / e is used in both the lagrangian and dilatation
#     # terms
#     # TODO : the _batch_matvec kernel needs to depend on the representation of J, and should be coded as such
#     J_omega_upon_e = (
#         _batch_matvec(mass_second_moment_of_inertia, omega_collection) / dilatation
#     )
#
#     # (J \omega_L / e) x \omega_L
#     # Warning : Do not do micro-optimization here : you can ignore dividing by dilatation as we later multiply by it
#     # but this causes confusion and violates SRP
#     lagrangian_transport = _batch_cross(J_omega_upon_e, omega_collection)
#
#     # Note : in the computation of dilatation_rate, there is an optimization opportunity as dilatation rate has
#     # a dilatation-like term in the numerator, which we cancel here
#     # (J \omega_L / e^2) . (de/dt)
#     unsteady_dilatation = J_omega_upon_e * dilatation_rate / dilatation
#
#     compute_damping_torques(nu, omega_collection, lengths, damping_torques)
#
#     blocksize = internal_torques.shape[1]
#     for i in range(3):
#         for k in range(blocksize):
#             internal_torques[i, k] = (
#                 bend_twist_couple_2D[i, k]
#                 + bend_twist_couple_3D[i, k]
#                 + shear_stretch_couple[i, k]
#                 + lagrangian_transport[i, k]
#                 + unsteady_dilatation[i, k]
#                 - damping_torques[i, k]
#             )
#
#
# ################################ NUMBA FUNCTIONS ###########################################
# ############################################################################################
#
# # TODO Fix this classmethod weirdness to a more scalable and maintainable solution
# # TODO Fix the SymplecticStepperMixin interface class as it does not belong here
# class CosseratRod(
#     _LinearConstitutiveModelMixin, _CosseratRodBase, _RodSymplecticStepperMixin
# ):
#     def __init__(self, n_elements, shear_matrix, bend_matrix, rod, *args, **kwargs):
#         _LinearConstitutiveModelMixin.__init__(
#             self,
#             n_elements,
#             shear_matrix,
#             bend_matrix,
#             rod.rest_lengths,
#             *args,
#             **kwargs
#         )
#         _CosseratRodBase.__init__(
#             self,
#             n_elements,
#             rod._vector_states.copy()[..., : n_elements + 1],
#             rod._matrix_states.copy(),
#             rod.rest_lengths,
#             rod.density,
#             rod.volume,
#             rod.mass_second_moment_of_inertia,
#             rod.nu,
#             *args,
#             **kwargs
#         )
#         _RodSymplecticStepperMixin.__init__(self)
#         del rod
#
#         # This below two lines are for initializing sigma and kappa
#         # TODO: Change them after Numba implementation
#         self._compute_shear_stretch_strains()
#         self._compute_bending_twist_strains()
#
#     @classmethod
#     def straight_rod(
#         cls,
#         n_elements,
#         start,
#         direction,
#         normal,
#         base_length,
#         base_radius,
#         density,
#         nu,
#         youngs_modulus,
#         poisson_ratio,
#         alpha_c=4.0 / 3.0,
#         *args,
#         **kwargs
#     ):
#         # FIXME: Make sure G=E/(poisson_ratio+1.0) in wikipedia it is different
#         # Shear Modulus
#         shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
#
#         # Second moment of inertia
#         A0 = np.pi * base_radius * base_radius
#         I0_1 = A0 * A0 / (4.0 * np.pi)
#         I0_2 = I0_1
#         I0_3 = 2.0 * I0_2
#         I0 = np.array([I0_1, I0_2, I0_3])
#
#         # Mass second moment of inertia for disk cross-section
#         mass_second_moment_of_inertia = np.zeros(
#             (MaxDimension.value(), MaxDimension.value()), np.float64
#         )
#         np.fill_diagonal(
#             mass_second_moment_of_inertia, I0 * density * base_length / n_elements
#         )
#
#         # Shear/Stretch matrix
#         shear_matrix = np.zeros(
#             (MaxDimension.value(), MaxDimension.value()), np.float64
#         )
#         np.fill_diagonal(
#             shear_matrix,
#             [
#                 alpha_c * shear_modulus * A0,
#                 alpha_c * shear_modulus * A0,
#                 youngs_modulus * A0,
#             ],
#         )
#
#         # Bend/Twist matrix
#         bend_matrix = np.zeros((MaxDimension.value(), MaxDimension.value()), np.float64)
#         np.fill_diagonal(
#             bend_matrix,
#             [youngs_modulus * I0_1, youngs_modulus * I0_2, shear_modulus * I0_3],
#         )
#
#         rod = _CosseratRodBase.straight_rod(
#             n_elements,
#             start,
#             direction,
#             normal,
#             base_length,
#             base_radius,
#             density,
#             nu,
#             mass_second_moment_of_inertia,
#             *args,
#             **kwargs
#         )
#         return cls(n_elements, shear_matrix, bend_matrix, rod, *args, **kwargs)
