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
    """
    This class represents a magnetic field constant in time.

        Attributes
        ----------
        magnetic_field_amplitude: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Amplitude of the constant magnetic field.
        ramp_interval : float
            ramping time for magnetic field.
        start_time : float
            Turning on time of magnetic field.
        end_time : float
            Turning off time of magnetic field.

    """

    def __init__(self, magnetic_field_amplitude, ramp_interval, start_time, end_time):
        """

        Parameters
        ----------
        magnetic_field_amplitude: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Amplitude of the constant magnetic field.
        ramp_interval : float
            ramping time for magnetic field.
        start_time : float
            Turning on time of magnetic field.
        end_time : float
            Turning off time of magnetic field.

        """
        self.magnetic_field_amplitude = magnetic_field_amplitude
        self.ramp_interval = ramp_interval
        self.start_time = start_time
        self.end_time = end_time

    def value(self, time: np.float64 = 0.0):
        """
        This function returns the value of the magnetic field vector based on the
        magnetic_field_amplitude.

        Parameters
        ----------
        time : float
            The time of simulation.

        Returns
        -------
        magnetic_field: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Value of the constant magnetic field.
        Notes
        -------
        Assumes only time dependence.

        """
        factor = 0.0
        time = round(time, 5)
        magnetic_field = np.zeros((3))
        if time > self.start_time:
            factor = min(1.0, (time - self.start_time) / self.ramp_interval)

        if time > self.end_time:
            factor = max(0.0, -1 / self.ramp_interval * (time - self.end_time) + 1.0)

        if factor > 0.0:
            magnetic_field[:] = self.magnetic_field_amplitude * factor

        return magnetic_field


class SingleModeOscillatingMagneticField:
    """
    This class represents a magnetic field oscillating sinusoidally in time
    with one mode.

        Attributes
        ----------
        magnetic_field_amplitude: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Amplitude of the oscillating magnetic field.
        magnetic_field_angular_frequency: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Angular frequency of the oscillating magnetic field.
        magnetic_field_phase_difference: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Phase difference of the oscillating magnetic field.

    """

    def __init__(
        self,
        magnetic_field_amplitude,
        magnetic_field_angular_frequency,
        magnetic_field_phase_difference,
    ):
        """

        Parameters
        ----------
        magnetic_field_amplitude: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Amplitude of the oscillating magnetic field.
        magnetic_field_angular_frequency: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Angular frequency of the oscillating magnetic field.
        magnetic_field_phase_difference: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Phase difference of the oscillating magnetic field.

        """
        self.magnetic_field_amplitude = magnetic_field_amplitude
        self.magnetic_field_angular_frequency = magnetic_field_angular_frequency
        self.magnetic_field_phase_difference = magnetic_field_phase_difference

    def value(self, time: np.float64 = 0.0):
        """
        This function returns the value of the sinusoidally oscillating magnetic field
        vector, based on amplitude, frequency and phase difference.

        Parameters
        ----------
        time : float
            The time of simulation.

        Returns
        -------
        magnetic_field: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Value of the oscillatory magnetic field.
        Notes
        -------
        Assumes only time dependence.

        """
        return self.magnetic_field_amplitude * np.sin(
            self.magnetic_field_angular_frequency * time
            + self.magnetic_field_phase_difference
        )


class ExternalMagneticFieldForces(NoForces):
    """
    This class applies magnetic forces on a magnetic Cosserat rod, based on an
    external magnetic field.

        Attributes
        ----------
        external_magnetic_field: object
            External magnetic field object, that returns the value of the magnetic field vector
            via a .value() method.

    """

    def __init__(self, external_magnetic_field):
        """
        Parameters
        ----------
        external_magnetic_field: object
            External magnetic field object, that returns the value of the magnetic field vector
            via a .value() method.

        """
        self.external_magnetic_field = external_magnetic_field

    def apply_torques(self, system, time: np.float64 = 0.0):
        system.external_torques += _batch_cross(
            system.magnetization_collection,
            # convert external_magnetic_field to local frame
            _batch_matvec(
                system.director_collection,
                self.external_magnetic_field.value(time=time).reshape(3, 1)
                * np.ones((system.n_elems,)),
            ),
        )
