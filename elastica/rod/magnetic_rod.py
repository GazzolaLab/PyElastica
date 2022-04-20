__doc__ = """ Rod classes and implementation details """
__all__ = ["MagneticRod"]
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
from elastica.rod.cosserat_rod import (
    CosseratRod,
    _compute_shear_stretch_strains,
    _compute_bending_twist_strains,
)
from elastica._calculus import (
    quadrature_kernel_for_block_structure,
    difference_kernel_for_block_structure,
    _difference,
    _average,
)
from elastica.interaction import node_to_element_pos_or_vel

position_difference_kernel = _difference
position_average = _average


@functools.lru_cache(maxsize=1)
def _get_z_vector():
    return np.array([0.0, 0.0, 1.0]).reshape(3, -1)


class MagneticRod(CosseratRod):
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
        magnetization_collection: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
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
        args,
        kwargs,
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

        if kwargs.__contains__("magnetization_collection"):
            self.magnetization_collection = kwargs.get("magnetization_collection")
        else:
            raise AttributeError(
                "Did you forget to input magnetization_collection in kwargs ?"
            )

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
        **kwargs
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
        start : numpy.ndarray[3, float]
            Starting coordinate in 3D
        direction : numpy.ndarray[3, float]
            Direction of the rod in 3D
        normal : numpy.ndarray[3, float]
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
        MagneticRod

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
            **kwargs
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
            args,
            kwargs,
        )
