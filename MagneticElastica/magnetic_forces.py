import numpy as np
from elastica._linalg import _batch_matvec, _batch_cross
from elastica.external_forces import NoForces


class MagneticTorquesForUniformMagneticField(NoForces):
    def __init__(
        self,
        ramp_interval,
        start_time,
        end_time,
        magnetization_density,
        magnetic_field_vector,
    ):
        self.ramp_interval = ramp_interval
        self.start_time = start_time
        self.end_time = end_time

        self.magnetization_density = magnetization_density
        self.magnetic_field_vector = np.ones(
            (magnetization_density.shape[-1])
        ) * magnetic_field_vector.reshape(3, 1)

    def apply_forces(self, system, time: np.float64 = 0.0):
        # No forces are applied since magnetic field is constant
        pass

    def apply_torques(self, system, time: np.float64 = 0.0):

        factor = 0.0
        time = round(time, 5)

        if time > self.start_time:
            factor = min(1.0, (time - self.start_time) / self.ramp_interval)

        if time > self.end_time:
            factor = max(0.0, -1 / self.ramp_interval * (time - self.end_time) + 1.0)

        if factor > 0.0:
            magnetization_vector = (
                self.magnetization_density * system.volume * system.tangents
            )
            magnetic_torques = factor * _batch_cross(
                magnetization_vector, self.magnetic_field_vector
            )
            np.round_(magnetic_torques, 12, magnetic_torques)

            system.external_torques += _batch_matvec(
                system.director_collection, magnetic_torques
            )


class MagneticTorquesForOscillatingMagneticField(NoForces):
    def __init__(
        self,
        ramp_interval,
        start_time,
        end_time,
        start_idx,
        end_idx,
        magnetization_density,
        magnetic_field_vector,
        frequency,
    ):

        self.ramp_interval = ramp_interval
        self.start_time = start_time
        self.end_time = end_time

        self.start_idx = start_idx
        self.end_idx = end_idx
        n_elem = end_idx - start_idx

        self.magnetization_density = magnetization_density * np.ones((n_elem))
        self.magnetic_field_vector = np.ones((n_elem)) * magnetic_field_vector.reshape(
            3, 1
        )

        self.wave_number = 2 * np.pi * frequency

    def apply_torques(self, system, time: np.float64 = 0.0):
        factor = 0.0
        time = round(time, 5)

        if time > self.start_time:
            factor = min(1.0, (time - self.start_time) / self.ramp_interval)

        if time > self.end_time:
            factor = max(0.0, -1 / self.ramp_interval * (time - self.end_time) + 1.0)

        if factor > 0.0:
            magnetization_vector = (
                self.magnetization_density
                * system.volume[self.start_idx : self.end_idx]
                * system.tangents[:, self.start_idx : self.end_idx]
            )
            magnetic_torques = factor * _batch_cross(
                magnetization_vector,
                self.magnetic_field_vector * np.sin(self.wave_number * time),
            )
            np.round_(magnetic_torques, 12, magnetic_torques)

            system.external_torques[
                ..., self.start_idx : self.end_idx
            ] += _batch_matvec(
                system.director_collection[..., self.start_idx : self.end_idx],
                magnetic_torques,
            )


class ConstantMagneticField:
    def __init__(self, magnetic_field_amplitude):
        # TODO documentation needed!
        self.magnetic_field_amplitude = magnetic_field_amplitude

    # assuming only time dependence as discussed
    def value(self, time: np.float64 = 0.0):
        return self.magnetic_field_amplitude


class SingleModeOscillatingMagneticField:
    def __init__(
        self,
        magnetic_field_amplitude,
        magnetic_field_angular_frequency,
        magnetic_field_phase_difference,
    ):
        # TODO documentation needed!
        self.magnetic_field_amplitude = magnetic_field_amplitude
        self.magnetic_field_angular_frequency = magnetic_field_angular_frequency
        self.magnetic_field_phase_difference = magnetic_field_phase_difference

    # assuming only time dependence as discussed
    def value(self, time: np.float64 = 0.0):
        return self.magnetic_field_amplitude * np.sin(
            self.magnetic_field_angular_frequency * time
            + self.magnetic_field_phase_difference
        )


class ExternalMagneticFieldForces(NoForces):
    def __init__(self, external_magnetic_field):
        # TODO documentation needed!
        # NOTE for different magnetic fields, this will be different
        # class with method .value().
        self.external_magnetic_field = external_magnetic_field

    def apply_torques(self, system, time: np.float64 = 0.0):
        system.external_torques += _batch_cross(
            system.magnetization_collection,
            # convert external_magnetic_field to local frame
            _batch_matvec(
                system.director_collection,
                # Arman a better way of doing the step below?
                np.tile(
                    self.external_magnetic_field.value(time=time).reshape(3, 1),
                    reps=(1, system.director_collection.shape[-1]),
                ),
            ),
        )
