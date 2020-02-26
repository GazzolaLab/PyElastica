import numpy as np
from numpy.testing import assert_allclose
import numba
from numba import njit, prange, guvectorize, float64, jitclass, int32, uint16
from elastica.utils import Tolerance
from numpy import zeros, empty, sqrt, arccos, sin
from elastica.utils import MaxDimension, Tolerance

import numpy as np
import functools

from elastica._linalg import _batch_matvec, _batch_cross, _batch_norm, _batch_dot
from elastica._calculus import (
    quadrature_kernel,
    difference_kernel,
    position_difference_kernel,
    position_average,
)
from elastica._rotations import _inv_rotate
from elastica.utils import MaxDimension, Tolerance

from elastica.rod import RodBase
from elastica.rod.constitutive_model import _LinearConstitutiveModelMixin
from elastica.rod.data_structures import _RodSymplecticStepperMixin

from elastica.rod.cosserat_rod import (
    compute_internal_forces,
    compute_internal_torques,
    _update_accelerations,
)

# from ..interaction import node_to_element_velocity
from elastica.interaction import node_to_element_velocity
from elastica.rod.data_structures import _KinematicState, _DynamicState
from elastica.rod.data_structures import _bootstrap_from_data


class FactoryClass:
    def __init__(self):

        pass

    def allocate(
        self,
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
        stepper_type="symplectic",
        *args,
        **kwargs
    ):

        # sanity checks here
        assert n_elements > 1
        assert base_length > Tolerance.atol()
        assert base_radius > Tolerance.atol()
        assert density > Tolerance.atol()
        assert nu >= 0.0
        assert np.sqrt(np.dot(normal, normal)) > Tolerance.atol()
        assert np.sqrt(np.dot(direction, direction)) > Tolerance.atol()

        end = start + direction * base_length
        position = np.zeros((3, n_elements + 1))
        for i in range(0, 3):
            position[i, ...] = np.linspace(start[i], end[i], n_elements + 1)

        # compute rest lengths and tangents
        position_diff = position[..., 1:] - position[..., :-1]
        rest_lengths = np.sqrt(np.einsum("ij,ij->j", position_diff, position_diff))
        tangents = position_diff / rest_lengths
        normal /= np.sqrt(np.dot(normal, normal))

        # set directors
        # check this order once
        directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elements))
        normal_collection = np.repeat(normal[:, np.newaxis], n_elements, axis=1)
        directors[0, ...] = normal_collection
        directors[1, ...] = _batch_cross(tangents, normal_collection)
        directors[2, ...] = tangents

        # check dimension of radius
        if np.array(base_radius).ndim == 0:
            base_radius = np.repeat(
                np.array(base_radius)[np.newaxis], n_elements, axis=0
            )

        # Second moment of inertia
        A0 = np.pi * base_radius * base_radius
        I0_1 = A0 * A0 / (4.0 * np.pi)
        I0_2 = I0_1
        I0_3 = 2.0 * I0_2
        I0 = np.array([I0_1, I0_2, I0_3]).transpose()
        # Mass second moment of inertia for disk cross-section
        mass_second_moment_of_inertia = np.zeros(
            (MaxDimension.value(), MaxDimension.value(), n_elements), np.float64
        )
        mass_second_moment_of_inertia_temp = I0 * density * base_length / n_elements
        for i in range(n_elements):
            np.fill_diagonal(
                mass_second_moment_of_inertia[..., i],
                mass_second_moment_of_inertia_temp[i, :],
            )
        # sanity check of mass second moment of inertia
        for k in range(n_elements):
            for i in range(0, MaxDimension.value()):
                assert mass_second_moment_of_inertia[i, i, k] > Tolerance.atol()

        # Inverse of second moment of inertia
        inv_mass_second_moment_of_inertia = np.zeros(
            (MaxDimension.value(), MaxDimension.value(), n_elements)
        )
        for i in range(n_elements):
            # Check rank of mass moment of inertia matrix to see if it is invertible
            assert (
                np.linalg.matrix_rank(mass_second_moment_of_inertia[..., i])
                == MaxDimension.value()
            )
            inv_mass_second_moment_of_inertia[..., i] = np.linalg.inv(
                mass_second_moment_of_inertia[..., i]
            )

        # Shear/Stretch matrix
        shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
        shear_matrix = np.zeros(
            (MaxDimension.value(), MaxDimension.value(), n_elements), np.float64
        )
        for i in range(n_elements):
            np.fill_diagonal(
                shear_matrix[..., i],
                [
                    alpha_c * shear_modulus * A0[i],
                    alpha_c * shear_modulus * A0[i],
                    youngs_modulus * A0[i],
                ],
            )
        for k in range(n_elements):
            for i in range(0, MaxDimension.value()):
                assert shear_matrix[i, i, k] > Tolerance.atol()

        # Bend/Twist matrix
        bend_matrix = np.zeros(
            (MaxDimension.value(), MaxDimension.value(), n_elements), np.float64
        )
        for i in range(n_elements):
            np.fill_diagonal(
                bend_matrix[..., i],
                [
                    youngs_modulus * I0_1[i],
                    youngs_modulus * I0_2[i],
                    shear_modulus * I0_3[i],
                ],
            )
        for k in range(n_elements):
            for i in range(0, MaxDimension.value()):
                assert bend_matrix[i, i, k] > Tolerance.atol()
        # Compute bend matrix in Voronoi Domain
        bend_matrix = (
            bend_matrix[..., 1:] * rest_lengths[1:]
            + bend_matrix[..., :-1] * rest_lengths[0:-1]
        ) / (rest_lengths[1:] + rest_lengths[:-1])

        # Compute volume of elements
        volume = np.pi * base_radius ** 2 * rest_lengths

        velocities = np.zeros((MaxDimension.value(), n_elements + 1))
        omegas = np.zeros((MaxDimension.value(), n_elements))  # + 1e-16
        accelerations = 0.0 * velocities
        angular_accelerations = 0.0 * omegas
        n_elems = n_elements
        _vector_states = np.hstack(
            (position, velocities, omegas, accelerations, angular_accelerations)
        )
        _matrix_states = directors.copy()
        # initial set to zero; if coming through kwargs then modify
        # rest_lengths = rest_lengths
        # density = density
        # volume = volume

        mass = np.zeros(n_elems + 1)
        mass[:-1] += 0.5 * density * volume
        mass[1:] += 0.5 * density * volume

        # mass_second_moment_of_inertia = mass_second_moment_of_inertia
        # inv_mass_second_moment_of_inertia = inv_mass_second_moment_of_inertia
        # shear_matrix = shear_matrix
        # bend_matrix = bend_matrix

        nu = nu
        rest_voronoi_lengths = 0.5 * (rest_lengths[1:] + rest_lengths[:-1])
        # calculated in `_compute_internal_forces_and_torques`
        internal_forces = 0 * accelerations
        internal_torques = 0 * angular_accelerations

        # will apply external force and torques externally
        external_forces = 0 * accelerations
        external_torques = 0 * angular_accelerations

        lengths = np.zeros((n_elements))
        tangents = np.zeros((MaxDimension.value(), n_elements))
        radius = np.zeros((n_elements))

        dilatation = np.zeros((n_elements))
        voronoi_dilatation = np.zeros((n_elements - 1))
        dilatation_rate = np.zeros((n_elements))

        # set rest strains and curvature to be  zero at start
        # if found in kwargs modify (say for curved rod)
        rest_sigma = np.zeros((MaxDimension.value(), n_elements))
        if kwargs.__contains__("rest_sigma"):
            temp_rest_sigma = kwargs["rest_sigma"]
            if temp_rest_sigma.shape == (MaxDimension.value(), n_elements):
                rest_sigma = temp_rest_sigma
            else:
                raise TypeError(
                    "Input rest sigma shape is not correct "
                    + str(temp_rest_sigma.shape)
                    + " It should be "
                    + str(rest_sigma.shape)
                )

        rest_kappa = np.zeros((MaxDimension.value(), n_elements - 1))
        if kwargs.__contains__("rest_kappa"):
            temp_rest_kappa = kwargs["rest_kappa"]
            if temp_rest_kappa.shape == (MaxDimension.value(), n_elements):
                rest_kappa = temp_rest_kappa
            else:
                raise TypeError(
                    "Input rest kappa shape is not correct "
                    + str(temp_rest_kappa.shape)
                    + " It should be "
                    + str(rest_kappa.shape)
                )

        sigma = np.zeros((MaxDimension.value(), n_elements))
        kappa = np.zeros((MaxDimension.value(), n_elements - 1))

        damping_forces = np.zeros((MaxDimension.value(), n_elements + 1))
        damping_torques = np.zeros((MaxDimension.value(), n_elements))

        internal_stress = np.zeros((MaxDimension.value(), n_elements))
        internal_couple = np.zeros((MaxDimension.value(), n_elements - 1))

        (
            kinematic_states,
            dynamic_states,
            position_collection,
            director_collection,
            velocity_collection,
            omega_collection,
            acceleration_collection,
            alpha_collection,  # angular acceleration
        ) = _bootstrap_from_data("symplectic", n_elems, _vector_states, _matrix_states)

        # # TODO: Remove below part and find a better solution
        # n_nodes = n_elems + 1
        # position_collection = np.ndarray.view(_vector_states[..., :n_nodes])
        # directors = np.ndarray.view(_matrix_states)
        #
        # # Set the states depending on the stepper type
        # if stepper_type == "explicit":
        #     v_w_states = np.ndarray.view(_vector_states[..., n_nodes : 3 * n_nodes - 1])
        #     v_w_dvdt_dwdt = np.ndarray.view(_vector_states[..., n_nodes:])
        # elif stepper_type == "symplectic":
        #     kinematic_rate = np.ndarray.view(
        #         _vector_states[..., n_nodes : 3 * n_nodes - 1]
        #     )
        #     dynamic_rate = np.ndarray.view(_vector_states[..., 3 * n_nodes :])
        # else:
        #     return
        #
        # n_velocity_end = n_nodes + n_nodes
        # velocity = np.ndarray.view(_vector_states[..., n_nodes:n_velocity_end])
        #
        # n_omega_end = n_velocity_end + n_elems
        # omega = np.ndarray.view(_vector_states[..., n_velocity_end:n_omega_end])
        #
        # n_acceleration_end = n_omega_end + n_nodes
        # acceleration = np.ndarray.view(
        #     _vector_states[..., n_omega_end:n_acceleration_end]
        # )
        #
        # n_alpha_end = n_acceleration_end + n_elems
        # alpha = np.ndarray.view(_vector_states[..., n_acceleration_end:n_alpha_end])

        cosserat_rod_spec = [
            ("n_elems", int32),
            ("_vector_states", float64[:, :]),
            ("_matrix_states", float64[:, :, :]),
            ("rest_lengths", float64[:]),
            ("density", float64),
            ("volume", float64[:]),
            ("mass", float64[:]),
            ("mass_second_moment_of_inertia", float64[:, :, :]),
            ("inv_mass_second_moment_of_inertia", float64[:, :, :]),
            ("shear_matrix", float64[:, :, :]),
            ("bend_matrix", float64[:, :, :]),
            ("nu", float64),
            ("rest_voronoi_lengths", float64[:]),
            ("internal_forces", float64[:, :]),
            ("internal_torques", float64[:, :]),
            ("external_forces", float64[:, :]),
            ("external_torques", float64[:, :]),
            ("lengths", float64[:]),
            ("tangents", float64[:, :]),
            ("radius", float64[:]),
            ("dilatation", float64[:]),
            ("voronoi_dilatation", float64[:]),
            ("dilatation_rate", float64[:]),
            ("rest_sigma", float64[:, :]),
            ("rest_kappa", float64[:, :]),
            ("sigma", float64[:, :]),
            ("kappa", float64[:, :]),
            ("damping_forces", float64[:, :]),
            ("damping_torques", float64[:, :]),
            ("internal_stress", float64[:, :]),
            ("internal_couple", float64[:, :]),
            ("position_collection", float64[:, :]),
            ("director_collection", float64[:, :, :]),
            ("velocity_collection", float64[:, :]),
            ("omega_collection", float64[:, :]),
            ("acceleration_collection", float64[:, :]),
            ("alpha_collection", float64[:, :]),
            # ("v_w_states", float64[:, :]),
            # ("v_w_dvdt_dwdt", float64[:, :]),
            # ("kinematic_rates", float64[:, :]),
            ("kinematic_states", _KinematicState.class_type.instance_type),
            ("dynamic_states", _DynamicState.class_type.instance_type),
            # ("dynamic_rate", float64[:, :]),
            # ("stepper_condition", int32),
        ]

        # Depending on the time stepper, set the arrays for CosseratRod JitClass
        # stepper_condition = 0
        # if stepper_type == "explicit":
        #     stepper_condition = 0
        #     rodclass = jitclass(cosserat_rod_spec)(CosseratRodJIT)
        #     v_w_states_or_kinematic_rate = v_w_states
        #     v_w_dvdt_dwdt_or_dynamic_rate = v_w_dvdt_dwdt
        #
        # elif stepper_type == "symplectic":
        #     stepper_condition = 1
        #     rodclass = jitclass(cosserat_rod_spec)(CosseratRodJIT)
        #     v_w_states_or_kinematic_rate = kinematic_rate
        #     v_w_dvdt_dwdt_or_dynamic_rate = dynamic_rate
        rodclass = jitclass(cosserat_rod_spec)(CosseratRodJIT)
        # rodclass = CosseratRodJIT
        return rodclass(
            n_elems,
            _vector_states,
            _matrix_states,
            rest_lengths,
            density,
            volume,
            mass,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            shear_matrix,
            bend_matrix,
            nu,
            rest_voronoi_lengths,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            tangents,
            radius,
            dilatation,
            voronoi_dilatation,
            dilatation_rate,
            rest_sigma,
            rest_kappa,
            sigma,
            kappa,
            damping_forces,
            damping_torques,
            internal_stress,
            internal_couple,
            position_collection,
            director_collection,
            velocity_collection,
            omega_collection,
            acceleration_collection,
            alpha_collection,
            kinematic_states,
            dynamic_states,
            # stepper_condition,
            # v_w_states_or_kinematic_rate,
            # v_w_dvdt_dwdt_or_dynamic_rate,
        )


class CosseratRodJIT:
    def __init__(
        self,
        n_elems,
        _vector_states,
        _matrix_states,
        rest_lengths,
        density,
        volume,
        mass,
        mass_second_moment_of_inertia,
        inv_mass_second_moment_of_inertia,
        shear_matrix,
        bend_matrix,
        nu,
        rest_voronoi_lengths,
        internal_forces,
        internal_torques,
        external_forces,
        external_torques,
        lengths,
        tangents,
        radius,
        dilatation,
        voronoi_dilatation,
        dilatation_rate,
        rest_sigma,
        rest_kappa,
        sigma,
        kappa,
        damping_forces,
        damping_torques,
        internal_stress,
        internal_couple,
        position_collection,
        director_collection,
        velocity_collection,
        omega_collection,
        acceleration_collection,
        alpha_collection,
        kinematic_states,
        dynamic_states,
        # stepper_condition,
        # v_w_states_or_kinematic_rate,
        # v_w_dvdt_dwdt_or_dynamic_rate,
    ):

        self.n_elems = n_elems
        self._vector_states = _vector_states
        self._matrix_states = _matrix_states
        self.rest_lengths = rest_lengths
        self.density = density
        self.volume = volume
        self.mass = mass
        self.mass_second_moment_of_inertia = mass_second_moment_of_inertia
        self.inv_mass_second_moment_of_inertia = inv_mass_second_moment_of_inertia
        self.shear_matrix = shear_matrix
        self.bend_matrix = bend_matrix
        self.nu = nu
        self.rest_voronoi_lengths = rest_voronoi_lengths
        self.internal_forces = internal_forces
        self.internal_torques = internal_torques
        self.external_forces = external_forces
        self.external_torques = external_torques
        self.lengths = lengths
        self.tangents = tangents
        self.radius = radius
        self.dilatation = dilatation
        self.voronoi_dilatation = voronoi_dilatation
        self.dilatation_rate = dilatation_rate
        self.rest_sigma = rest_sigma
        self.rest_kappa = rest_kappa
        self.sigma = sigma
        self.kappa = kappa
        self.damping_forces = damping_forces
        self.damping_torques = damping_torques
        self.internal_stress = internal_stress
        self.internal_couple = internal_couple

        # self.position_collection = position_collection
        # self.director_collection = director_collection
        # self.velocity_collection = velocity_collection
        # self.omega_collection = omega_collection
        # self.acceleration_collection = acceleration_collection
        # self.alpha_collection = alpha_collection

        self.kinematic_states = kinematic_states
        self.dynamic_states = dynamic_states
        self.position_collection = position_collection
        self.director_collection = director_collection
        self.velocity_collection = velocity_collection
        self.omega_collection = omega_collection
        self.acceleration_collection = acceleration_collection
        self.alpha_collection = alpha_collection

        # self.stepper_condition = stepper_condition
        # if self.stepper_condition == 0:
        #
        #     self.v_w_states = v_w_states_or_kinematic_rate
        #     self.v_w_dvdt_dwdt = v_w_dvdt_dwdt_or_dynamic_rate
        #
        # elif self.stepper_condition == 1:
        #
        #     self.kinematic_rates = v_w_states_or_kinematic_rate
        #     self.dynamic_rate = v_w_dvdt_dwdt_or_dynamic_rate

    def kinematic_rates(self, time):
        return self.dynamic_states.kinematic_rates(time)

    def dynamic_rates(self, time):
        self.update_accelerations()

        """
        The following commented block of code is a test to ensure that
        the time-integrator always updates the view of the
        collection variables, and not an independent variable
        (aka no copy is made). It exists only for legacy
        purposes and will be either refactored or removed once
        testing is done.
        """
        # def shmem(x):
        #     if np.shares_memory(
        #             self.dynamic_states.rate_collection, x
        #     ) : print("Shares memory")
        #     else :
        #         print("Explicit states does not share memory")
        # shmem(self.velocity_collection)
        # shmem(self.acceleration_collection)
        # shmem(self.omega_collection)
        # shmem(self.alpha_collection)
        return self.dynamic_states.dynamic_rates(time)

    def _compute_geometry_from_state(self):
        """
        Returns
        -------

        """
        # Compute eq (3.3) from 2018 RSOS paper

        position_diff = position_difference_kernel(self.position_collection)
        # self.lengths = np.sqrt(np.einsum("ij,ij->j", position_diff, position_diff))
        self.lengths[:] = _batch_norm(position_diff)
        for k in range(self.lengths.shape[0]):
            self.tangents[0, k] = position_diff[0, k] / self.lengths[k]
            self.tangents[1, k] = position_diff[1, k] / self.lengths[k]
            self.tangents[2, k] = position_diff[2, k] / self.lengths[k]
            # resize based on volume conservation
            self.radius[k] = np.sqrt(self.volume[k] / self.lengths[k] / np.pi)

    def _compute_all_dilatations(self):
        """
        Compute element and Voronoi region dilatations
        Returns
        -------

        """
        self._compute_geometry_from_state()
        # Caveat : Needs already set rest_lengths and rest voronoi domain lengths
        # Put in initialization
        for k in range(self.lengths.shape[0]):
            self.dilatation[k] = self.lengths[k] / self.rest_lengths[k]

        # Compute eq (3.4) from 2018 RSOS paper
        # Note : we can use trapezoidal kernel, but it has padding and will be slower
        voronoi_lengths = position_average(self.lengths)

        # Compute eq (3.45 from 2018 RSOS paper
        for k in range(voronoi_lengths.shape[0]):
            self.voronoi_dilatation[k] = (
                voronoi_lengths[k] / self.rest_voronoi_lengths[k]
            )

    def _compute_dilatation_rate(self):
        """

        Returns
        -------

        """
        # self.lengths = l_i = |r^{i+1} - r^{i}|
        r_dot_v = _batch_dot(self.position_collection, self.velocity_collection)
        r_plus_one_dot_v = _batch_dot(
            self.position_collection[..., 1:], self.velocity_collection[..., :-1]
        )
        r_dot_v_plus_one = _batch_dot(
            self.position_collection[..., :-1], self.velocity_collection[..., 1:]
        )

        for k in range(self.lengths.shape[0]):
            self.dilatation_rate[k] = (
                (
                    r_dot_v[k]
                    + r_dot_v[k + 1]
                    - r_dot_v_plus_one[k]
                    - r_plus_one_dot_v[k]
                )
                / self.lengths[k]
                / self.rest_lengths[k]
            )

    def _compute_shear_stretch_strains(self):
        # Quick trick : Instead of evaliation Q(et-d^3), use property that Q*d3 = (0,0,1), a constant
        self._compute_all_dilatations()

        z_vector = np.array([0.0, 0.0, 1.0]).reshape(3, -1)
        self.sigma[:] = (
            self.dilatation * _batch_matvec(self.director_collection, self.tangents)
            - z_vector
        )

    def _compute_bending_twist_strains(self):
        # Note: dilatations are computed previously inside ` _compute_all_dilatations `
        temp = _inv_rotate(self.director_collection)
        for k in range(self.rest_voronoi_lengths.shape[0]):
            self.kappa[0, k] = temp[0, k] / self.rest_voronoi_lengths[k]
            self.kappa[1, k] = temp[1, k] / self.rest_voronoi_lengths[k]
            self.kappa[2, k] = temp[2, k] / self.rest_voronoi_lengths[k]

    def _compute_damping_forces(self):
        # Internal damping foces.
        elemental_velocities = node_to_element_velocity(self.velocity_collection)
        blocksize = elemental_velocities.shape[1]
        elemental_damping_forces = np.zeros((3, blocksize))

        for i in range(3):
            for k in range(blocksize):
                elemental_damping_forces[i, k] = (
                    self.nu * elemental_velocities[i, k] * self.lengths[k]
                )
        self.damping_forces[:] = quadrature_kernel(elemental_damping_forces)

    # TODO: May be we can move model in Constitutive Model Class
    def _compute_internal_shear_stretch_stresses_from_model(self):
        """
        Linear force functional
        Operates on
        S : (3,3,n) tensor and sigma (3,n)

        Returns
        -------

        """
        self._compute_shear_stretch_strains()
        self.internal_stress[:] = _batch_matvec(
            self.shear_matrix, self.sigma - self.rest_sigma
        )

    def _compute_internal_forces(self):
        # Compute n_l and cache it using internal_stress
        # Be careful about usage though
        self._compute_internal_shear_stretch_stresses_from_model()

        # Signifies Q^T n_L / e
        # Not using batch matvec as I don't want to take directors.T here
        # FIXME: change memory overload instead for the below calls!
        blocksize = self.internal_stress.shape[1]
        cosserat_internal_stress = np.zeros((3, blocksize))
        for i in range(3):
            for j in range(3):
                for k in range(blocksize):
                    cosserat_internal_stress[i, k] += (
                        self.director_collection[j, i, k] * self.internal_stress[j, k]
                    )
        cosserat_internal_stress /= self.dilatation

        # Compute damping forces
        self._compute_damping_forces()

        # Compute internal forces
        self.internal_forces[:] = (
            difference_kernel(cosserat_internal_stress) - self.damping_forces
        )

    def _compute_damping_torques(self):
        # Internal damping torques
        blocksize = self.damping_torques.shape[1]
        for i in range(3):
            for k in range(blocksize):
                self.damping_torques[i, k] = (
                    self.nu * self.omega_collection[i, k] * self.lengths[k]
                )

    def _compute_internal_bending_twist_stresses_from_model(self):
        """
        Linear force functional
        Operates on
        B : (3,3,n) tensor and curvature kappa (3,n)

        Returns
        -------

        """
        self._compute_bending_twist_strains()  # concept : needs to compute kappa
        self.internal_couple[:] = _batch_matvec(
            self.bend_matrix, self.kappa - self.rest_kappa
        )

    def _compute_internal_torques(self):
        # Compute \tau_l and cache it using internal_couple
        # Be careful about usage though
        self._compute_internal_bending_twist_stresses_from_model()
        # Compute dilatation rate when needed, dilatation itself is done before
        # in internal_stresses
        self._compute_dilatation_rate()

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

        # Compute internal torques
        blocksize = self.internal_torques.shape[1]
        for i in range(3):
            for k in range(blocksize):
                self.internal_torques[i, k] = (
                    bend_twist_couple_2D[i, k]
                    + bend_twist_couple_3D[i, k]
                    + shear_stretch_couple[i, k]
                    + lagrangian_transport[i, k]
                    + unsteady_dilatation[i, k]
                    - self.damping_torques[i, k]
                )

    def compute_translational_energy(self):
        return (
            0.5
            * (
                self.mass
                * _batch_dot(self.velocity_collection, self.velocity_collection)
            ).sum()
        )

    def compute_rotational_energy(self):
        J_omega_upon_e = (
            _batch_matvec(self.mass_second_moment_of_inertia, self.omega_collection)
            / self.dilatation
        )

        return 0.5 * _batch_dot(self.omega_collection, J_omega_upon_e).sum()

    def compute_velocity_center_of_mass(self):
        sum_mass_times_velocity = np.zeros((3))
        blocksize = self.mass.shape[0]
        for i in range(3):
            for k in range(blocksize):
                sum_mass_times_velocity[i] += (
                    self.mass[k] * self.velocity_collection[i, k]
                )

        return sum_mass_times_velocity / self.mass.sum()

    def compute_position_center_of_mass(self):
        sum_mass_times_position = np.zeros((3))
        blocksize = self.mass.shape[0]
        for i in range(3):
            for k in range(blocksize):
                sum_mass_times_position[i] += (
                    self.mass[k] * self.position_collection[i, k]
                )

        return sum_mass_times_position / self.mass.sum()

    def _compute_internal_forces_and_torques(self):
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

        self._compute_internal_forces()
        self._compute_internal_torques()

        # self.internal_forces = compute_internal_forces(
        #     self.position_collection,
        #     self.volume,
        #     self.lengths,
        #     self.tangents,
        #     self.radius,
        #     self.rest_lengths,
        #     self.rest_voronoi_lengths,
        #     self.dilatation,
        #     self.voronoi_dilatation,
        #     self.director_collection,
        #     self.sigma,
        #     self.rest_sigma,
        #     self.shear_matrix,
        #     self.internal_stress,
        #     self.velocity_collection,
        #     self.nu,
        # )
        #
        # compute_internal_torques(
        #     self.position_collection,
        #     self.velocity_collection,
        #     self.tangents,
        #     self.lengths,
        #     self.rest_lengths,
        #     self.director_collection,
        #     self.rest_voronoi_lengths,
        #     self.bend_matrix,
        #     self.rest_kappa,
        #     self.kappa,
        #     self.voronoi_dilatation,
        #     self.mass_second_moment_of_inertia,
        #     self.omega_collection,
        #     self.internal_stress,
        #     self.internal_couple,
        #     self.dilatation,
        #     self.dilatation_rate,
        #     self.nu,
        #     self.damping_torques,
        #     self.internal_torques,
        # )

    # TODO: find better way and place to compute internal forces and torques
    def update_internal_forces_and_torques(self):
        self._compute_internal_forces_and_torques()

    def update_accelerations(self):
        """ TODO Do we need to make the collection members abstract?

        Parameters
        ----------
        time

        Returns
        -------

        """
        blocksize_acc = self.internal_forces.shape[1]
        blocksize_alpha = self.internal_torques.shape[1]

        for i in range(3):
            for k in range(blocksize_acc):
                self.acceleration_collection[i, k] = (
                    self.internal_forces[i, k] + self.external_forces[i, k]
                ) / self.mass[k]
                self.external_forces[i, k] = 0.0

        self.alpha_collection *= 0.0
        for i in range(3):
            for j in range(3):
                for k in range(blocksize_alpha):
                    self.alpha_collection[i, k] += (
                        self.inv_mass_second_moment_of_inertia[i, j, k]
                        * (self.internal_torques[j, k] + self.external_torques[j, k])
                    ) * self.dilatation[k]

        # Reset torques
        self.external_torques *= 0.0

        # _update_accelerations(
        #     self.acceleration_collection,
        #     self.internal_forces,
        #     self.external_forces,
        #     self.mass,
        #     self.alpha_collection,
        #     self.inv_mass_second_moment_of_inertia,
        #     self.internal_torques,
        #     self.external_torques,
        #     self.dilatation,
        # )


# n_elements = 100
# start = np.zeros((3,))
# direction = np.array([0.0, 0.0, 1.0])
# normal = np.array([0.0, 1.0, 0.0])
# base_length = 1.0
# base_radius = 0.25
# density = 500.0
# nu = 0.1
# youngs_modulus = 1e3
# poisson_ratio = 0.2

# factory_function = FactoryClass()
# rodjit = factory_function.allocate(
#     n_elements,
#     start,
#     direction,
#     normal,
#     base_length,
#     base_radius,
#     density,
#     nu,
#     youngs_modulus,
#     poisson_ratio,
# )
