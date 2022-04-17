import numpy as np
import numba
from numba import njit
from elastica._linalg import _batch_matvec, _batch_cross
from elastica.external_forces import NoForces


class MagneticTorquesForUniformMagneticField(NoForces):
    def __init__(self, ramp_interval,start_time, end_time, magnetization_density, magnetic_field_vector):
        self.ramp_interval = ramp_interval
        self.start_time = start_time
        self.end_time = end_time

        self.magnetization_density = magnetization_density
        self.magnetic_field_vector = np.ones((magnetization_density.shape[-1])) * magnetic_field_vector.reshape(3, 1)


    def apply_forces(self, system, time: np.float64 = 0.0):
        # No forces are applied since magnetic field is constant
        pass

    def apply_torques(self, system, time: np.float64 = 0.0):

        factor = 0.0
        time = round(time, 5)

        if time > self.start_time:
            factor = min(1.0, (time - self.start_time) / self.ramp_interval)

        if time > self.end_time:
            factor = max(
                0.0, -1 / self.ramp_interval * (time - self.end_time) + 1.0
            )

        if factor > 0.0:
            magnetization_vector = self.magnetization_density*system.volume*system.tangents
            magnetic_torques = factor * _batch_cross(magnetization_vector, self.magnetic_field_vector)
            np.round_(magnetic_torques, 12, magnetic_torques)

            system.external_torques += _batch_matvec(
                system.director_collection, magnetic_torques
            )


class MagneticTorquesForOscillatingMagneticField(NoForces):
    def __init__(self, ramp_interval, start_time, end_time, start_idx, end_idx, magnetization_density, magnetic_field_vector, frequency):

        self.ramp_interval = ramp_interval
        self.start_time = start_time
        self.end_time = end_time

        self.start_idx = start_idx
        self.end_idx = end_idx
        n_elem = end_idx-start_idx

        self.magnetization_density = magnetization_density*np.ones((n_elem))
        self.magnetic_field_vector = np.ones((n_elem)) * magnetic_field_vector.reshape(3, 1)


        self.wave_number = 2*np.pi*frequency

    def apply_torques(self, system, time: np.float64 = 0.0):
        factor = 0.0
        time = round(time, 5)

        if time > self.start_time:
            factor = min(1.0, (time - self.start_time) / self.ramp_interval)

        if time > self.end_time:
            factor = max(
                0.0, -1 / self.ramp_interval * (time - self.end_time) + 1.0
            )

        if factor > 0.0:
            magnetization_vector = self.magnetization_density*system.volume[self.start_idx:self.end_idx]*system.tangents[:,self.start_idx:self.end_idx]
            magnetic_torques = factor * _batch_cross(magnetization_vector, self.magnetic_field_vector*np.sin(self.wave_number*time))
            np.round_(magnetic_torques, 12, magnetic_torques)

            system.external_torques[...,self.start_idx:self.end_idx] += _batch_matvec(
                system.director_collection[...,self.start_idx:self.end_idx], magnetic_torques
            )