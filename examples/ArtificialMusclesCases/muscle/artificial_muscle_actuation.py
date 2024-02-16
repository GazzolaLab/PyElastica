__all__ = ["ArtficialMuscleActuation"]
import numpy as np
from numba import njit
from elastica import NoForces
from elastica._linalg import (
    _batch_cross,
    _batch_norm,
    _batch_dot,
    _batch_matvec,
)
from examples.ArtificialMusclesCases.muscle.muscle_utils import *


class ArtficialMuscleActuation(NoForces):
    """
    This class applies force to make a coiled artificial muscle contract.

        Attributes
        ----------


    """

    def __init__(
        self,
        start_density,
        start_kappa,
        start_sigma,
        start_radius,
        start_bend_matrix,
        start_shear_matrix,
        start_mass_second_moment_of_inertia,
        start_inv_mass_second_moment_of_inertia,
        contraction_time,
        start_time,
        kappa_change,
        thermal_expansion_coefficient,
        room_temperature,
        end_temperature,
        youngs_modulus_coefficients,
        **kwargs,
    ):
        """

        Parameters
        ----------
        start_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to first node of the rod-like object.
        end_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to last node of the rod-like object.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

        """
        super(NoForces, self).__init__()
        self.start_kappa = start_kappa
        self.start_sigma = start_sigma
        self.start_radius = start_radius
        self.start_density = start_density
        self.start_bend_matrix = start_bend_matrix
        self.start_shear_matrix = start_shear_matrix
        self.start_mass_second_moment_of_inertia = start_mass_second_moment_of_inertia
        self.start_inv_mass_second_moment_of_inertia = (
            start_inv_mass_second_moment_of_inertia
        )
        self.start_time = start_time

        assert contraction_time > 0.0
        self.contraction_time = contraction_time
        self.kappa_change = kappa_change
        self.thermal_expansion_coefficient = thermal_expansion_coefficient
        self.end_temperature = end_temperature
        self.youngs_modulus_coefficients = youngs_modulus_coefficients
        self.room_temperature = room_temperature

        self.end_radius = (
            (thermal_expansion_coefficient * (end_temperature - room_temperature)) + 1
        ) * start_radius
        self.end_beta = self.end_radius / start_radius
        self.end_gamma = gamma_func(
            end_temperature, youngs_modulus_coefficients, room_temperature
        )
        self.end_phi = 1 / (self.end_gamma * (self.end_beta ** 4))
        print(self.end_beta)

    def apply_forces(self, system, time=0.0):
        compute_untwist(
            system.rest_kappa,
            system.rest_sigma,
            system.density,
            system.volume,
            system.lengths,
            system.shear_matrix,
            system.mass_second_moment_of_inertia,
            system.bend_matrix,
            self.start_density,
            self.start_kappa,
            self.start_sigma,
            self.start_radius,
            self.start_shear_matrix,
            self.start_mass_second_moment_of_inertia,
            self.start_inv_mass_second_moment_of_inertia,
            self.start_bend_matrix,
            time,
            self.contraction_time,
            self.start_time,
            self.kappa_change,
            system.inv_mass_second_moment_of_inertia,
            self.thermal_expansion_coefficient,
            self.room_temperature,
            self.end_temperature,
            self.youngs_modulus_coefficients,
        )

    # @staticmethod


# @njit(cache=True)
def compute_untwist(
    rest_kappa,
    rest_sigma,
    density,
    volume,
    length,
    shear_matrix,
    mass_second_moment_of_inertia,
    bend_matrix,
    start_density,
    start_kappa,
    start_sigma,
    start_radius,
    start_shear_matrix,
    start_mass_second_moment_of_inertia,
    start_inv_mass_second_moment_of_inertia,
    start_bend_matrix,
    time,
    contraction_time,
    start_time,
    kappa_change,
    inv_mass_second_moment_of_inertia,
    thermal_expansion_coefficient,
    room_temperature,
    end_temperature,
    youngs_modulus_coefficients,
):
    """
    Compute end point forces that are applied on the rod using numba njit decorator.

    Parameters
    ----------
    external_forces: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type. External force vector.
    start_force: numpy.ndarray
        1D (dim) array containing data with 'float' type.
    end_force: numpy.ndarray
        1D (dim) array containing data with 'float' type.
        Force applied to last node of the rod-like object.
    time: float
    ramp_up_time: float
        Applied forces are ramped up until ramp up time.

    Returns
    -------

    """

    factor = min(1.0, max(time - start_time, 0) / contraction_time)
    current_temperature = room_temperature + factor * (
        end_temperature - room_temperature
    )
    # radius = ((thermal_expansion_coeficient*(current_temperature-room_temperature))+1)*start_radius
    # beta = radius/start_radius #change in radius
    # gamma = (gamma_func(current_temperature,youngs_modulus_coefficients,room_temperature)) #change in Young's Modulus

    phi = 1 + factor * kappa_change
    gamma = gamma_func(
        current_temperature, youngs_modulus_coefficients, room_temperature
    )
    beta = (gamma * phi) ** (-1 / 4)
    radius = beta * start_radius
    # rest_kappa[:] = start_kappa[:]/(gamma*(beta**4))
    # rest_sigma[:] = start_sigma[:]/(gamma*(beta**2))
    # print(end_phi)
    rest_kappa[:] = phi * start_kappa[:]

    volume[:] = np.pi * (radius ** 2) * length[:]
    density[:] = start_density * ((1 / beta) ** 2)
    shear_matrix[:] = start_shear_matrix * (beta ** 2) * gamma
    mass_second_moment_of_inertia[:] = start_mass_second_moment_of_inertia * (beta ** 2)
    inv_mass_second_moment_of_inertia[:] = start_inv_mass_second_moment_of_inertia / (
        beta ** 2
    )
    bend_matrix[:] = start_bend_matrix * (beta ** 4) * gamma


class ManualArtficialMuscleActuation(NoForces):
    """
    This class applies force to make a coiled artificial muscle contract.

        Attributes
        ----------


    """

    def __init__(
        self,
        start_kappa,
        contraction_time,
        start_time,
        kappa_change,
        **kwargs,
    ):
        """

        Parameters
        ----------
        start_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to first node of the rod-like object.
        end_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to last node of the rod-like object.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

        """
        super(NoForces, self).__init__()
        self.start_kappa = start_kappa
        self.start_time = start_time

        assert contraction_time > 0.0
        self.contraction_time = contraction_time
        self.kappa_change = kappa_change

    def apply_forces(self, system, time=0.0):
        compute_untwist_manual(
            system.rest_kappa,
            self.start_kappa,
            time,
            self.contraction_time,
            self.start_time,
            self.kappa_change,
        )


def compute_untwist_manual(
    rest_kappa,
    start_kappa,
    time,
    contraction_time,
    start_time,
    kappa_change,
):
    """
    Compute end point forces that are applied on the rod using numba njit decorator.

    Parameters
    ----------
    external_forces: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type. External force vector.
    start_force: numpy.ndarray
        1D (dim) array containing data with 'float' type.
    end_force: numpy.ndarray
        1D (dim) array containing data with 'float' type.
        Force applied to last node of the rod-like object.
    time: float
    ramp_up_time: float
        Applied forces are ramped up until ramp up time.

    Returns
    -------

    """

    factor = min(1.0, max(time - start_time, 0) / contraction_time)
    rest_kappa[:] = (1 + factor * kappa_change) * start_kappa[:]


class ArtficialMuscleActuationDecoupled(NoForces):
    """
    This class applies force to make a coiled artificial muscle contract.

        Attributes
        ----------


    """

    def __init__(
        self,
        start_density,
        start_kappa,
        start_sigma,
        start_radius,
        start_bend_matrix,
        start_shear_matrix,
        start_mass_second_moment_of_inertia,
        start_inv_mass_second_moment_of_inertia,
        contraction_time,
        start_time,
        kappa_change,
        room_temperature,
        end_temperature,
        youngs_modulus_coefficients,
        thermal_expansion_coefficient,
        **kwargs,
    ):
        """

        Parameters
        ----------
        start_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to first node of the rod-like object.
        end_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to last node of the rod-like object.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

        """
        super(NoForces, self).__init__()
        self.start_kappa = start_kappa
        self.start_sigma = start_sigma
        self.start_radius = start_radius
        self.start_density = start_density
        self.start_bend_matrix = start_bend_matrix
        self.start_shear_matrix = start_shear_matrix
        self.start_mass_second_moment_of_inertia = start_mass_second_moment_of_inertia
        self.start_inv_mass_second_moment_of_inertia = (
            start_inv_mass_second_moment_of_inertia
        )
        self.start_time = start_time

        assert contraction_time > 0.0
        self.contraction_time = contraction_time
        self.kappa_change = kappa_change
        self.end_temperature = end_temperature
        self.youngs_modulus_coefficients = youngs_modulus_coefficients
        self.room_temperature = room_temperature
        self.thermal_expansion_coefficient = thermal_expansion_coefficient

    def apply_forces(self, system, time=0.0):
        compute_untwist_decoupled(
            system.rest_kappa,
            system.rest_sigma,
            system.density,
            system.volume,
            system.lengths,
            system.shear_matrix,
            system.mass_second_moment_of_inertia,
            system.bend_matrix,
            self.start_density,
            self.start_kappa,
            self.start_sigma,
            self.start_radius,
            self.start_shear_matrix,
            self.start_mass_second_moment_of_inertia,
            self.start_inv_mass_second_moment_of_inertia,
            self.start_bend_matrix,
            time,
            self.contraction_time,
            self.start_time,
            self.kappa_change,
            system.inv_mass_second_moment_of_inertia,
            self.room_temperature,
            self.end_temperature,
            self.youngs_modulus_coefficients,
            self.thermal_expansion_coefficient,
        )

    # @staticmethod


@njit(cache=True)
def compute_untwist_decoupled(
    rest_kappa,
    rest_sigma,
    density,
    volume,
    length,
    shear_matrix,
    mass_second_moment_of_inertia,
    bend_matrix,
    start_density,
    start_kappa,
    start_sigma,
    start_radius,
    start_shear_matrix,
    start_mass_second_moment_of_inertia,
    start_inv_mass_second_moment_of_inertia,
    start_bend_matrix,
    time,
    contraction_time,
    start_time,
    kappa_change,
    inv_mass_second_moment_of_inertia,
    room_temperature,
    end_temperature,
    youngs_modulus_coefficients,
    thermal_expansion_coefficient,
):
    """
    Compute end point forces that are applied on the rod using numba njit decorator.

    Parameters
    ----------
    external_forces: numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type. External force vector.
    start_force: numpy.ndarray
        1D (dim) array containing data with 'float' type.
    end_force: numpy.ndarray
        1D (dim) array containing data with 'float' type.
        Force applied to last node of the rod-like object.
    time: float
    ramp_up_time: float
        Applied forces are ramped up until ramp up time.

    Returns
    -------

    """

    factor = min(1.0, max(time - start_time, 0) / contraction_time)
    phi = 1 + (factor) * kappa_change  # controls contraction amount
    current_temperature = room_temperature + factor * (
        end_temperature - room_temperature
    )
    gamma = gamma_func(
        current_temperature, youngs_modulus_coefficients, room_temperature
    )  # change in young's modulus
    radius = (
        (thermal_expansion_coefficient * (current_temperature - room_temperature)) + 1
    ) * start_radius
    beta = radius / start_radius
    rest_kappa[:] = phi * start_kappa[:]

    volume[:] = np.pi * (radius ** 2) * length[:]
    density[:] = start_density * ((1 / beta) ** 2)
    shear_matrix[:] = start_shear_matrix * (beta ** 2) * gamma
    mass_second_moment_of_inertia[:] = start_mass_second_moment_of_inertia * (beta ** 2)
    inv_mass_second_moment_of_inertia[:] = start_inv_mass_second_moment_of_inertia / (
        beta ** 2
    )
    bend_matrix[:] = start_bend_matrix * (beta ** 4) * gamma
