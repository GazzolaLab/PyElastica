"""
Energy computation mixin for Cosserat rods.

This mixin provides methods to compute various energy quantities
(translational, rotational, bending, shear) and center of mass properties.
"""

import numpy as np
from numpy.typing import NDArray

from elastica._linalg import _batch_matvec, _batch_dot


class RodEnergy:
    """
    Mixin class providing energy computation methods for rods.

    This mixin should be used with RodBase-derived classes that have
    the required attributes (mass, velocity, omega, etc.).

    Example usage::

        class MyRod(RodBase, RodEnergy):
            ...

        rod = MyRod(...)
        kinetic_energy = rod.compute_translational_energy()
        bending_energy = rod.compute_bending_energy()

    """

    # Required attributes (provided by RodBase-derived class)
    mass: NDArray[np.float64]
    velocity_collection: NDArray[np.float64]
    position_collection: NDArray[np.float64]
    omega_collection: NDArray[np.float64]
    mass_second_moment_of_inertia: NDArray[np.float64]
    dilatation: NDArray[np.float64]
    kappa: NDArray[np.float64]
    rest_kappa: NDArray[np.float64]
    bend_matrix: NDArray[np.float64]
    rest_voronoi_lengths: NDArray[np.float64]
    sigma: NDArray[np.float64]
    rest_sigma: NDArray[np.float64]
    shear_matrix: NDArray[np.float64]
    rest_lengths: NDArray[np.float64]

    def compute_translational_energy(self) -> NDArray[np.float64]:
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

    def compute_rotational_energy(self) -> NDArray[np.float64]:
        """
        Compute total rotational energy of the rod at the instance.
        """
        J_omega_upon_e = (
            _batch_matvec(self.mass_second_moment_of_inertia, self.omega_collection)
            / self.dilatation
        )
        return 0.5 * np.einsum("ik,ik->k", self.omega_collection, J_omega_upon_e).sum()

    def compute_velocity_center_of_mass(self) -> NDArray[np.float64]:
        """
        Compute velocity center of mass of the rod at the instance.
        """
        mass_times_velocity = np.einsum("j,ij->ij", self.mass, self.velocity_collection)
        sum_mass_times_velocity = np.einsum("ij->i", mass_times_velocity)

        return sum_mass_times_velocity / self.mass.sum()

    def compute_position_center_of_mass(self) -> NDArray[np.float64]:
        """
        Compute position center of mass of the rod at the instance.
        """
        mass_times_position = np.einsum("j,ij->ij", self.mass, self.position_collection)
        sum_mass_times_position = np.einsum("ij->i", mass_times_position)

        return sum_mass_times_position / self.mass.sum()

    def compute_bending_energy(self) -> NDArray[np.float64]:
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

    def compute_shear_energy(self) -> NDArray[np.float64]:
        """
        Compute total shear energy of the rod at the instance.
        """

        sigma_diff = self.sigma - self.rest_sigma
        shear_internal_forces = _batch_matvec(self.shear_matrix, sigma_diff)

        return (
            0.5
            * (_batch_dot(sigma_diff, shear_internal_forces) * self.rest_lengths).sum()
        )
