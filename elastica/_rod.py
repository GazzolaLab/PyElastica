__doc__ = """ Rod base classes and implementation details that need to be hidden from the user"""
import numpy as np
import functools

from ._linalg import _batch_matmul, _batch_matvec, _batch_cross
from ._calculus import quadrature_kernel, difference_kernel
from ._rotations import _inv_rotate
from .utils import MaxDimension, Tolerance

# TODO Add documentation for all functions


@functools.lru_cache(maxsize=1)
def _get_z_vector():
    return np.array([0.0, 0.0, 1.0]).reshape(3, -1)


# First the constitutive laws, only simple linear for now
class _LinearConstitutiveModel:

    # Needs
    # kappa, kappa0, strain (sigma), sigma0, B, S in specified formats
    # maybe use __init__ to initialize if not found?
    def __init__(
        self, n_elements, shear_matrix, bend_matrix, rest_lengths, *args, **kwargs
    ):
        # set rest strains and curvature to be  zero at start
        # if found in kwargs modify (say for curved rod)
        self.rest_sigma = np.zeros((MaxDimension.value(), n_elements))
        self.rest_kappa = np.zeros((MaxDimension.value(), n_elements - 1))
        # sanity checks here
        # NOTE: assuming matrices to be diagonal here
        for i in range(0, MaxDimension.value()):
            assert shear_matrix[i, i] > Tolerance.atol()
            assert bend_matrix[i, i] > Tolerance.atol()

        self.shear_matrix = np.repeat(
            shear_matrix[:, :, np.newaxis], n_elements, axis=2
        )
        self.bend_matrix = np.repeat(bend_matrix[:, :, np.newaxis], n_elements, axis=2)

        # Compute bend matrix in Voronoi Domain
        self.bend_matrix = (
            self.bend_matrix[..., 1:] * rest_lengths[1:]
            + self.bend_matrix[..., :-1] * rest_lengths[0:-1]
        ) / (rest_lengths[1:] + rest_lengths[:-1])

    def _compute_internal_shear_stretch_stresses_from_model(self):
        """
        Linear force functional
        Operates on
        S : (3,3,n) tensor and sigma (3,n)

        Returns
        -------

        """
        self._compute_shear_stretch_strains()  # concept : needs to compute sigma
        # TODO : the _batch_matvec kernel needs to depend on the representation of Shearmatrix
        self.internal_stress = _batch_matvec(
            self.shear_matrix, self.sigma - self.rest_sigma
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
        # TODO : the _batch_matvec kernel needs to depend on the representation of Bendmatrix
        self.internal_couple = _batch_matvec(
            self.bend_matrix, self.kappa - self.rest_kappa
        )


class _LinearConstitutiveModelWithStrainRate(_LinearConstitutiveModel):
    def __init__(
        self, n_elements, shear_matrix, bend_matrix, rest_lengths, *args, **kwargs
    ):
        _LinearConstitutiveModel.__init__(
            self, n_elements, shear_matrix, bend_matrix, rest_lengths, *args, **kwargs
        )
        if "shear_rate_matrix" in kwargs.keys():
            self.shear_rate_matrix = np.repeat(
                kwargs["shear_rate_matrix"][:, :, np.newaxis], n_elements, axis=2
            )
        else:
            raise ValueError("shear rate matrix value missing!")
        if "bend_rate_matrix" in kwargs.keys():
            self.bend_rate_matrix = np.repeat(
                kwargs["bend_rate_matrix"][:, :, np.newaxis], n_elements, axis=2
            )
            # Compute bend rate matrix in Voronoi Domain
            self.bend_rate_matrix = (
                self.bend_rate_matrix[..., 1:] * rest_lengths[1:]
                + self.bend_rate_matrix[..., :-1] * rest_lengths[0:-1]
            ) / (rest_lengths[1:] + rest_lengths[:-1])
        else:
            raise ValueError("bend rate matrix value missing!")

    def _compute_internal_shear_stretch_stresses_from_model(self):
        """
        Linear force functional
        Operates on
        S : (3,3,n) tensor and sigma (3,n)

        Returns
        -------

        """
        # TODO : test this function
        # Calculates stress based purely on strain component
        super(
            _LinearConstitutiveModelWithStrainRate, self
        )._compute_internal_shear_stretch_stresses_from_model()
        self._compute_shear_stretch_strains_rates()  # concept : needs to compute sigma_dot
        # TODO : the _batch_matvec kernel needs to depend on the representation of ShearStrainmatrix
        self.internal_stress += _batch_matvec(self.shear_rate_matrix, self.sigma_dot)

    def _compute_internal_bending_twist_stresses_from_model(self):
        """
        Linear force functional
        Operates on
        B : (3,3,n) tensor and curvature kappa (3,n)

        Returns
        -------

        """
        # TODO : test this function
        # Calculates stress based purely on strain component
        super(
            _LinearConstitutiveModelWithStrainRate, self
        )._compute_internal_bending_twist_stresses_from_model()
        self._compute_bending_twist_strain_rates()  # concept : needs to compute kappa rate
        # TODO : the _batch_matvec kernel needs to depend on the representation of Bendmatrix
        self.internal_couple += _batch_matvec(self.bend_rate_matrix, self.kappa_dot)


# The interface class, as seen from global scope
# Can be made common to all entities in the code
class RodBase:
    """
    Base class for all rods
    # TODO : What needs to be ported here?
    """

    def __init__(self):
        pass

    def get_velocity(self):
        return self.velocity

    def get_angular_velocity(self):
        return self.omega

    def get_acceleration(self):
        return (self._compute_internal_forces() + self.external_forces) / self.mass

    def get_angular_acceleration(self):
        return (
            _batch_matvec(
                self.inv_mass_second_moment_of_inertia,
                (self._compute_internal_torques() + self.external_torques),
            )
            * self.dilatation
        )


class _CosseratRodBase(RodBase):
    # I'm assuming number of elements can be deduced from the size of the inputs
    def __init__(
        self,
        n_elements,
        position,
        directors,
        rest_lengths,
        mass,
        density,
        volume,
        mass_second_moment_of_inertia,
        inv_mass_second_moment_of_inertia,
        nu,
        rest_voronoi_lengths,
        *args,
        **kwargs
    ):
        self.position = position
        self.directors = directors
        # initial set to zero; if coming through kwargs then modify
        self.velocity = np.zeros((MaxDimension.value(), n_elements + 1))
        self.omega = np.zeros((MaxDimension.value(), n_elements))
        self.rest_lengths = rest_lengths
        self.mass = mass
        self.density = density
        self.volume = volume
        self.mass_second_moment_of_inertia = mass_second_moment_of_inertia
        self.inv_mass_second_moment_of_inertia = inv_mass_second_moment_of_inertia
        self.nu = nu
        self.rest_voronoi_lengths = rest_voronoi_lengths
        # will apply external force and torques externally
        self.external_forces = 0 * self.position
        self.external_torques = 0 * self.omega

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
        mass_second_moment_of_inertia,
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
        for i in range(0, MaxDimension.value()):
            assert mass_second_moment_of_inertia[i, i] > Tolerance.atol()

        end = start + direction * base_length
        position = np.zeros((MaxDimension.value(), n_elements + 1))
        for i in range(0, MaxDimension.value()):
            position[i, ...] = np.linspace(start[i], end[i], num=n_elements + 1)

        # compute rest lengths and tangents
        position_diff = position[..., 1:] - position[..., :-1]
        rest_lengths = np.sqrt(np.einsum("ij,ij->j", position_diff, position_diff))
        tangents = position_diff / rest_lengths
        normal /= np.sqrt(np.dot(normal, normal))

        # Note : we can use trapezoidal kernel, but it has padding and will be slower
        rest_voronoi_lengths = 0.5 * (rest_lengths[1:] + rest_lengths[:-1])

        # set directors
        # check this order once
        directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elements))
        normal_collection = np.repeat(normal[:, np.newaxis], n_elements, axis=1)
        directors[0, ...] = normal_collection
        directors[1, ...] = _batch_cross(tangents, normal_collection)
        directors[2, ...] = tangents

        volume = np.pi * base_radius ** 2 * rest_lengths
        mass = np.zeros(n_elements + 1)
        mass[0:-1] = density * volume
        mass[0] *= 0.5
        mass[-1] = 0.5 * mass[-2]
        inertia_collection = np.repeat(
            mass_second_moment_of_inertia[:, :, np.newaxis], n_elements, axis=2
        )
        inv_mass_second_moment_of_inertia = np.linalg.inv(mass_second_moment_of_inertia)
        inv_inertia_collection = np.repeat(
            inv_mass_second_moment_of_inertia[:, :, np.newaxis], n_elements, axis=2
        )

        # create rod
        return cls(
            n_elements,
            position,
            directors,
            rest_lengths,
            mass,
            density,
            volume,
            inertia_collection,
            inv_inertia_collection,
            nu,
            rest_voronoi_lengths,
            *args,
            **kwargs
        )

    def _compute_geometry_from_state(self):
        """
        Returns
        -------

        """
        # Compute eq (3.3) from 2018 RSOS paper

        # Note : we can use the two-point difference kernel, but it needs unnecessary padding
        # and hence will always be slower
        position_diff = self.position[..., 1:] - self.position[..., :-1]
        self.lengths = np.sqrt(np.einsum("ij,ij->j", position_diff, position_diff))
        self.tangents = position_diff / self.lengths
        # resize based on volume conservation
        self.radius = np.sqrt(self.volume / self.lengths / np.pi)

    def _compute_all_dilatations(self):
        """
        Compute element and Voronoi region dilatations
        Returns
        -------

        """
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
        # self.lengths = l_i = |r^{i+1} - r^{i}|
        r_dot_v = np.einsum("ij,ij->j", self.position, self.velocity)
        r_plus_one_dot_v = np.einsum(
            "ij, ij->j", self.position[..., 1:], self.velocity[..., :-1]
        )
        r_dot_v_plus_one = np.einsum(
            "ij, ij->j", self.position[..., :-1], self.velocity[..., 1:]
        )
        self.dilatation_rate = (
            (r_dot_v[..., :-1] + r_dot_v[..., 1:] - r_dot_v_plus_one - r_plus_one_dot_v)
            / self.lengths
            / self.rest_lengths
        )

    def _compute_shear_stretch_strains(self):
        # Quick trick : Instead of evaliation Q(et-d^3), use property that Q*d3 = (0,0,1), a constant
        self._compute_all_dilatations()
        self.sigma = (
            self.dilatation * _batch_matvec(self.directors, self.tangents)
            - _get_z_vector()
        )

    def _compute_bending_twist_strains(self):
        self.kappa = _inv_rotate(self.directors) / self.rest_voronoi_lengths

    def _compute_damping_forces(self):
        # Internal damping foces.
        damping_forces = self.nu * self.velocity
        damping_forces[..., 0] *= 0.5  # first and last nodes have half mass
        damping_forces[..., -1] *= 0.5  # first and last nodes have half mass

        return damping_forces

    def _compute_internal_forces(self):
        # Compute n_l and cache it using internal_stress
        # Be careful about usage though
        self._compute_internal_shear_stretch_stresses_from_model()
        # Signifies Q^T n_L / e
        # Not using batch matvec as I don't want to take directors.T here
        cosserat_internal_stress = (
            np.einsum("jik, jk->ik", self.directors, self.internal_stress)
            / self.dilatation  # computed in comp_dilatation <- compute_strain <- compute_stress
        )
        return (
            difference_kernel(cosserat_internal_stress) - self._compute_damping_forces()
        )

    def _compute_damping_torques(self):
        # Internal damping torques
        damping_torques = self.nu * self.omega
        return damping_torques

    def _compute_internal_torques(self):
        # Compute \tau_l and cache it using internal_couple
        # Be careful about usage though
        self._compute_internal_bending_twist_stresses_from_model()
        # Compute dilatation rate when needed, dilatation itself is done before
        # in internal_stresses
        self._compute_dilatation_rate()

        voronoi_dilatation_cube_cached = 1.0 / self.voronoi_dilatation ** 3
        # Delta(\tau_L / \Epsilon^3)
        bend_twist_couple_2D = difference_kernel(
            self.internal_couple * voronoi_dilatation_cube_cached
        )
        # \mathcal{A}[ (\kappa x \tau_L ) * \hat{D} / \Epislon^3 ]
        bend_twist_couple_3D = quadrature_kernel(
            _batch_cross(self.kappa, self.internal_couple)
            * self.rest_voronoi_lengths
            * voronoi_dilatation_cube_cached
        )
        # (Qt x n_L) * \hat{l}
        shear_stretch_couple = (
            _batch_cross(
                _batch_matvec(self.directors, self.tangents), self.internal_stress
            )
            * self.rest_lengths
        )

        # I apply common sub expression elimination here, as J w / e is used in both the lagrangian and dilatation
        # terms
        # TODO : the _batch_matvec kernel needs to depend on the representation of J, and should be coded as such
        J_omega_upon_e = (
            _batch_matvec(self.mass_second_moment_of_inertia, self.omega)
            / self.dilatation
        )

        # (J \omega_L / e) x \omega_L
        # Warning : Do not do micro-optimization here : you can ignore dividing by dilatation as we later multiply by it
        # but this causes confusion and violates SRP
        lagrangian_transport = _batch_cross(J_omega_upon_e, self.omega)

        # Note : in the computation of dilatation_rate, there is an optimization opportunity as dilatation rate has
        # a dilatation-like term in the numerator, which we cancel here
        # (J \omega_L / e^2) . (de/dt)
        unsteady_dilatation = J_omega_upon_e * self.dilatation_rate / self.dilatation

        return (
            bend_twist_couple_2D
            + bend_twist_couple_3D
            + shear_stretch_couple
            + lagrangian_transport
            + unsteady_dilatation
            - self._compute_damping_torques()
        )


class CosseratRod(_LinearConstitutiveModel, _CosseratRodBase):
    def __init__(self, n_elements, shear_matrix, bend_matrix, rod, *args, **kwargs):
        _LinearConstitutiveModel.__init__(
            self,
            n_elements,
            shear_matrix,
            bend_matrix,
            rod.rest_lengths,
            *args,
            **kwargs
        )
        _CosseratRodBase.__init__(
            self,
            n_elements,
            rod.position,
            rod.directors,
            rod.rest_lengths,
            rod.mass,
            rod.density,
            rod.volume,
            rod.mass_second_moment_of_inertia,
            rod.inv_mass_second_moment_of_inertia,
            rod.nu,
            rod.rest_voronoi_lengths,
            *args,
            **kwargs
        )
        del rod

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

        # Shear Modulus
        shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

        # Second moment of inertia
        A0 = np.pi * base_radius * base_radius
        I0_1 = A0 * A0 / (4.0 * np.pi)
        I0_2 = I0_1
        I0_3 = 2.0 * I0_2
        I0 = np.array([I0_1, I0_2, I0_3])

        # Mass second moment of inertia for disk cross-section
        mass_second_moment_of_inertia = np.zeros(
            (MaxDimension.value(), MaxDimension.value()), np.float64
        )
        np.fill_diagonal(
            mass_second_moment_of_inertia, I0 * density * base_length / n_elements
        )

        # Shear/Stretch matrix
        shear_matrix = np.zeros(
            (MaxDimension.value(), MaxDimension.value()), np.float64
        )
        np.fill_diagonal(
            shear_matrix,
            [
                alpha_c * shear_modulus * A0,
                alpha_c * shear_modulus * A0,
                youngs_modulus * A0,
            ],
        )

        # Bend/Twist matrix
        bend_matrix = np.zeros((MaxDimension.value(), MaxDimension.value()), np.float64)
        np.fill_diagonal(
            bend_matrix,
            [youngs_modulus * I0_1, youngs_modulus * I0_2, shear_modulus * I0_3],
        )

        rod = _CosseratRodBase.straight_rod(
            n_elements,
            start,
            direction,
            normal,
            base_length,
            base_radius,
            density,
            nu,
            mass_second_moment_of_inertia,
            *args,
            **kwargs
        )
        return cls(n_elements, shear_matrix, bend_matrix, rod, *args, **kwargs)
