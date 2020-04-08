__doc__ = """ Module containing implementations of Cosserat Rod classes """
import numpy as np
import functools

from elastica._linalg import _batch_matvec, _batch_cross
from elastica._calculus import quadrature_kernel, difference_kernel
from elastica._rotations import _inv_rotate
from elastica.utils import MaxDimension, Tolerance

from elastica.rod import RodBase
from elastica.rod.constitutive_model import _LinearConstitutiveModelMixin
from elastica.rod.data_structures import _RodSymplecticStepperMixin

# TODO Add documentation for all functions


@functools.lru_cache(maxsize=1)
def _get_z_vector():
    """
    Generates and returns d3 vector.

    Returns
    -------
    numpy.ndarray
        2D (dim,1) array containing data with 'float' type.
    """
    return np.array([0.0, 0.0, 1.0]).reshape(3, -1)


class _CosseratRodBase(RodBase):
    """
    CosseratRodBase is a base class that contains all Cosserat rod equations.

    Attributes
    -----------
    n_elems: int
        The number of elements of the rod.
    _vector_states: numpy.ndarray
        2D (dim, :math:`*`) array containing data with 'float' type.
        Array containing position_collection, velocity_collection, omega_collection,
        alpha_collection.
    _matrix_states:
        3D array containing data with 'float' type.
        An array containing directors.
    rest_lengths: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod element lengths at rest configuration.
    density: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod elements densities.
    volume: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod element volumes.
    mass: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod node masses. Note that masses are stored on the nodes, not on elements.
    mass_second_moment_of_inertia: numpy.ndarray
        3D (dim, dim, blocksize) array containing data with 'float' type.
        Rod element mass second moment of interia.
    inv_mass_second_moment_of_inertia: numpy.ndarray
        3D (dim, dim, blocksize) array containing data with 'float' type.
        Rod element inverse mass moment of inertia.
    nu: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod element dissipation coefficient.
    rest_voronoi_lengths: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod lengths on the voronoi domain at the rest configuration.
    internal_forces: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
        Rod node internal forces. Note that internal forces are stored on the node, not on elements.
    internal_torques: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
        Rod element internal torques.
    external_forces: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
        External forces acting on rod nodes.
    external_torques: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
        External torques acting on rod elements.
    lengths: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod element lengths.
    tangents: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
        Rod element tangent vectors.
    radius: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod element radius.
    dilatation: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod element dilatation.
    voronoi_dilatation: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod dilatation on voronoi domain.
    dilatation_rate: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod element dilatation rates.


    """

    def __init__(
        self,
        n_elements,
        position,
        directors,
        rest_lengths,
        density,
        volume,
        mass_second_moment_of_inertia,
        nu,
        *args,
        **kwargs
    ):
        """
        Parameters
        ----------
        n_elements: int
        position: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type.
            Rod node position array.
        directors: numpy.ndarray
            3D (dim, dim, blocksize) array containing data with 'float' type.
            Rod element directors array.
        rest_lengths: numpy.ndarray
            1D (blocksize) array containing data with 'float' type.
            Rod element rest lengths.
        density: numpy.ndarray
            1D (blocksize) array containing data with 'float' type.
            Rod element density.
        volume: numpy.ndarray
            1D (blocksize) array containing data with 'float' type.
            Rod element volume.
        mass_second_moment_of_inertia: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type.
            Rod element mass second moment of inertia.
        nu: numpy.ndarray
           1D (blocksize) array containing data with 'float' type.
           Rod element dissipation constant.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        velocities = np.zeros((MaxDimension.value(), n_elements + 1))
        omegas = np.zeros((MaxDimension.value(), n_elements))  # + 1e-16
        accelerations = 0.0 * velocities
        angular_accelerations = 0.0 * omegas
        self.n_elems = n_elements
        self._vector_states = np.hstack(
            (position, velocities, omegas, accelerations, angular_accelerations)
        )
        self._matrix_states = directors.copy()
        # initial set to zero; if coming through kwargs then modify
        self.rest_lengths = rest_lengths
        self.density = density
        self.volume = volume

        self.mass = np.zeros(n_elements + 1)
        self.mass[:-1] += 0.5 * self.density * self.volume
        self.mass[1:] += 0.5 * self.density * self.volume

        self.mass_second_moment_of_inertia = mass_second_moment_of_inertia

        self.inv_mass_second_moment_of_inertia = np.zeros(
            (MaxDimension.value(), MaxDimension.value(), n_elements)
        )
        for i in range(n_elements):
            # Check rank of mass moment of inertia matrix to see if it is invertible
            assert (
                np.linalg.matrix_rank(mass_second_moment_of_inertia[..., i])
                == MaxDimension.value()
            )
            self.inv_mass_second_moment_of_inertia[..., i] = np.linalg.inv(
                mass_second_moment_of_inertia[..., i]
            )

        self.nu = nu
        self.rest_voronoi_lengths = 0.5 * (
            self.rest_lengths[1:] + self.rest_lengths[:-1]
        )
        # calculated in `_compute_internal_forces_and_torques`
        self.internal_forces = 0 * accelerations
        self.internal_torques = 0 * angular_accelerations

        # will apply external force and torques externally
        self.external_forces = 0 * accelerations
        self.external_torques = 0 * angular_accelerations

        # calculated in `compute_geometry_from_state`
        self.lengths = NotImplemented
        self.tangents = NotImplemented
        self.radius = NotImplemented

        # calculated in `compute_all_dilatatation`
        self.dilatation = NotImplemented
        self.voronoi_dilatation = NotImplemented
        self.dilatation_rate = NotImplemented

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

        # set directors
        # check this order once
        directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elements))
        normal_collection = np.repeat(normal[:, np.newaxis], n_elements, axis=1)
        directors[0, ...] = normal_collection
        directors[1, ...] = _batch_cross(tangents, normal_collection)
        directors[2, ...] = tangents

        volume = np.pi * base_radius ** 2 * rest_lengths

        inertia_collection = np.repeat(
            mass_second_moment_of_inertia[:, :, np.newaxis], n_elements, axis=2
        )

        # create rod
        return cls(
            n_elements,
            position,
            directors,
            rest_lengths,
            density,
            volume,
            inertia_collection,
            nu,
            *args,
            **kwargs
        )

    def _compute_geometry_from_state(self):
        """
        This method computes element length, tangent and radius.

        Returns
        -------

        """
        # Compute eq (3.3) from 2018 RSOS paper

        # Note : we can use the two-point difference kernel, but it needs unnecessary padding
        # and hence will always be slower
        position_diff = (
            self.position_collection[..., 1:] - self.position_collection[..., :-1]
        )
        self.lengths = np.sqrt(np.einsum("ij,ij->j", position_diff, position_diff))
        self.tangents = position_diff / self.lengths
        # resize based on volume conservation
        self.radius = np.sqrt(self.volume / self.lengths / np.pi)

    def _compute_all_dilatations(self):
        """
        This method compute element dilatation and voronoi region dilatations.

        Returns
        -------

        """
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
        This method computes element dilatation rate.

        Returns
        -------

        """
        # TODO Use the vector formula rather than separating it out
        # self.lengths = l_i = |r^{i+1} - r^{i}|
        r_dot_v = np.einsum(
            "ij,ij->j", self.position_collection, self.velocity_collection
        )
        r_plus_one_dot_v = np.einsum(
            "ij, ij->j",
            self.position_collection[..., 1:],
            self.velocity_collection[..., :-1],
        )
        r_dot_v_plus_one = np.einsum(
            "ij, ij->j",
            self.position_collection[..., :-1],
            self.velocity_collection[..., 1:],
        )
        self.dilatation_rate = (
            (r_dot_v[..., :-1] + r_dot_v[..., 1:] - r_dot_v_plus_one - r_plus_one_dot_v)
            / self.lengths
            / self.rest_lengths
        )

    def _compute_shear_stretch_strains(self):
        """
        This method computes shear and stretch strains of the elements.

        Returns
        -------

        """
        # Quick trick : Instead of evaliation Q(et-d^3), use property that Q*d3 = (0,0,1), a constant
        self._compute_all_dilatations()
        self.sigma = (
            self.dilatation * _batch_matvec(self.director_collection, self.tangents)
            - _get_z_vector()
        )

    def _compute_bending_twist_strains(self):
        """
        This method computes bending and twist strains of the elements.

        Returns
        -------

        """
        # Note: dilatations are computed previously inside ` _compute_all_dilatations `
        self.kappa = _inv_rotate(self.director_collection) / self.rest_voronoi_lengths

    def _compute_damping_forces(self):
        """
        This method computes internal damping forces acting on the elements.

        Returns
        -------

        """
        # Internal damping forces.
        elemental_velocities = 0.5 * (
            self.velocity_collection[..., :-1] + self.velocity_collection[..., 1:]
        )
        elemental_damping_forces = self.nu * elemental_velocities * self.lengths
        nodal_damping_forces = quadrature_kernel(elemental_damping_forces)
        return nodal_damping_forces

    def _compute_internal_forces(self):
        """
        This method computes internal forces acting on elements.

        Returns
        -------

        """
        # Compute n_l and cache it using internal_stress
        # Be careful about usage though
        self._compute_internal_shear_stretch_stresses_from_model()
        # Signifies Q^T n_L / e
        # Not using batch matvec as I don't want to take directors.T here
        cosserat_internal_stress = (
            np.einsum("jik, jk->ik", self.director_collection, self.internal_stress)
            / self.dilatation  # computed in comp_dilatation <- compute_strain <- compute_stress
        )
        return (
            difference_kernel(cosserat_internal_stress) - self._compute_damping_forces()
        )

    def _compute_damping_torques(self):
        """
        This method computes damping torques acting on elements.

        Returns
        -------

        """
        # Internal damping torques
        damping_torques = self.nu * self.omega_collection * self.lengths
        return damping_torques

    def _compute_internal_torques(self):
        """
        This method computes internal torques acting on elements.

        Returns
        -------

        """
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

        return (
            bend_twist_couple_2D
            + bend_twist_couple_3D
            + shear_stretch_couple
            + lagrangian_transport
            + unsteady_dilatation
            - self._compute_damping_torques()
        )

    def _compute_internal_forces_and_torques(self, time):
        """
        This method is a wrapper called by the time-stepper algorithm to compute internal forces and torques.

        Parameters
        ----------
        time: float
            Simulation time.

        Returns
        -------

        """
        self.internal_forces = self._compute_internal_forces()
        self.internal_torques = self._compute_internal_torques()

    # Interface to time-stepper mixins (Symplectic, Explicit), which calls this method
    def update_accelerations(self, time):
        """
        This method computes translational and angular acceleration and reset external forces and torques to zero.

        Parameters
        ----------
        time: float
            Simulation time.

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
        """
        This method computes the total translational energy of the rod.

        Returns
        -------

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
        This method computes the total rotational energy of the rod.

        Returns
        -------

        """
        J_omega_upon_e = (
            _batch_matvec(self.mass_second_moment_of_inertia, self.omega_collection)
            / self.dilatation
        )
        return 0.5 * np.einsum("ik,ik->k", self.omega_collection, J_omega_upon_e).sum()

    def compute_velocity_center_of_mass(self):
        """
        This method computes velocity of rod center of mass.

        Returns
        -------

        """
        mass_times_velocity = np.einsum("j,ij->ij", self.mass, self.velocity_collection)
        sum_mass_times_velocity = np.einsum("ij->i", mass_times_velocity)

        return sum_mass_times_velocity / self.mass.sum()

    def compute_position_center_of_mass(self):
        """
        This method computes position of the rod center of mass.

        Returns
        -------

        """
        mass_times_position = np.einsum("j,ij->ij", self.mass, self.position_collection)
        sum_mass_times_position = np.einsum("ij->i", mass_times_position)

        return sum_mass_times_position / self.mass.sum()


# TODO Fix this classmethod weirdness to a more scalable and maintainable solution
# TODO Fix the SymplecticStepperMixin interface class as it does not belong here
class CosseratRod(
    _LinearConstitutiveModelMixin, _CosseratRodBase, _RodSymplecticStepperMixin
):
    """
    Cosserat Rod public class. This is the preferred class for rods because it is derived from some
    of the essential base classes.
    Although the attributes of this CosseratRod class are inherited from its parent classes, for convenience and easy access, the variable
    names are given below.

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
        1D (blocksize) array containing data with 'float' type.
        Rod element lengths at rest configuration.
    density: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod elements densities.
    volume: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod element volumes.
    mass: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod node masses. Note that masses are stored on the nodes, not on elements.
    mass_second_moment_of_inertia: numpy.ndarray
        3D (dim, dim, blocksize) array containing data with 'float' type.
        Rod element mass second moment of interia.
    inv_mass_second_moment_of_inertia: numpy.ndarray
        3D (dim, dim, blocksize) array containing data with 'float' type.
        Rod element inverse mass moment of inertia.
    nu: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod element dissipation coefficient.
    rest_voronoi_lengths: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod lengths on the voronoi domain at the rest configuration.
    internal_forces: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
        Rod node internal forces. Note that internal forces are stored on the node, not on elements.
    internal_torques: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
        Rod element internal torques.
    external_forces: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
        External forces acting on rod nodes.
    external_torques: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
        External torques acting on rod elements.
    lengths: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod element lengths.
    tangents: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
        Rod element tangent vectors.
    radius: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod element radius.
    dilatation: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod element dilatation.
    voronoi_dilatation: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod dilatation on voronoi domain.
    dilatation_rate: numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
        Rod element dilatation rates.

    """

    def __init__(self, n_elements, shear_matrix, bend_matrix, rod, *args, **kwargs):
        _LinearConstitutiveModelMixin.__init__(
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
            rod._vector_states.copy()[..., : n_elements + 1],
            rod._matrix_states.copy(),
            rod.rest_lengths,
            rod.density,
            rod.volume,
            rod.mass_second_moment_of_inertia,
            rod.nu,
            *args,
            **kwargs
        )
        _RodSymplecticStepperMixin.__init__(self)
        del rod

        # This below two lines are for initializing sigma and kappa
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
        """
        Call this method to initialize and generate a Cosserat rod object that is a straight rod. Future versions will contain
        methods for curvilinear rods.

        Parameters
        ----------
        n_elements: float
            Rod number of elements.
        start: numpy.ndarray
            1D (dim) array containing data with 'float' type. Start position of the rod.
        direction: numpy.ndarray
            1D (dim) array containing data with 'float' type. Direction or tangent of the rod.
        normal: numpy.ndarray
            1D (dim) array containing data with 'float' type. Normal direction of the rod.
        base_length: float
            Initial length of the rod.
        base_radius: float
            Initial radius of the rod.
        density: float
            Density of the rod.
        nu: float
            Dissipation constant of the rod.
        youngs_modulus: float
            Youngs modulus of the rod.
        poisson_ratio: float
            Poisson ratio of the rod is used to compute shear modulus.
        alpha_c: float
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------

        """
        # FIXME: Make sure G=E/(poisson_ratio+1.0) in wikipedia it is different
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
