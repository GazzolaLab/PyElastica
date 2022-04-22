import numpy as np
from elastica._linalg import _batch_matvec, _batch_cross
from elastica.external_forces import NoForces


def compute_ramp_factor(time, ramp_interval, start_time, end_time):
    """
    This function returns a linear ramping up factor based on time, ramp_interval,
    start_time and end_time.

    Parameters
    ----------
    time : float
        The time of simulation.
    ramp_interval : float
        ramping time for magnetic field.
    start_time : float
        Turning on time of magnetic field.
    end_time : float
        Turning off time of magnetic field.

    Returns
    -------
    factor : float
        Ramp up factor.

    """
    factor = (time > start_time) * (time < end_time) * min(
        1.0, (time - start_time) / ramp_interval
    ) + (time > end_time) * max(0.0, -1 / ramp_interval * (time - end_time) + 1.0)
    return factor


class BaseMagneticField:
    """
    This is the base class for external magnetic field objects.

    Notes
    -----
    Every new magnetic field class must be derived
    from BaseMagneticField class.

    """

    def __init__(self):
        """
        BaseMagneticField class does not need any input parameters.
        """
        pass

    def value(self, time: np.float64 = 0.0):
        """Returns the value of the magnetic field vector.

        In BaseMagneticField class, this routine simply passes.

        Parameters
        ----------
        time : float
            The time of simulation.

        Returns
        -------

        """
        pass


class ConstantMagneticField(BaseMagneticField):
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
        # to bypass division by timestep issues,
        # TODO Arman can word it better?
        time = round(time, 5)
        factor = compute_ramp_factor(
            time=time,
            ramp_interval=self.ramp_interval,
            start_time=self.start_time,
            end_time=self.end_time,
        )
        return self.magnetic_field_amplitude * factor


class SingleModeOscillatingMagneticField(BaseMagneticField):
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
        ramp_interval : float
            ramping time for magnetic field.
        start_time : float
            Turning on time of magnetic field.
        end_time : float
            Turning off time of magnetic field.

    """

    def __init__(
        self,
        magnetic_field_amplitude,
        magnetic_field_angular_frequency,
        magnetic_field_phase_difference,
        ramp_interval,
        start_time,
        end_time,
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
        ramp_interval : float
            ramping time for magnetic field.
        start_time : float
            Turning on time of magnetic field.
        end_time : float
            Turning off time of magnetic field.

        """
        self.magnetic_field_amplitude = magnetic_field_amplitude
        self.magnetic_field_angular_frequency = magnetic_field_angular_frequency
        self.magnetic_field_phase_difference = magnetic_field_phase_difference
        self.ramp_interval = ramp_interval
        self.start_time = start_time
        self.end_time = end_time

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
        # to bypass division by timestep issues,
        # TODO Arman can word it better?
        time = round(time, 5)
        factor = compute_ramp_factor(
            time=time,
            ramp_interval=self.ramp_interval,
            start_time=self.start_time,
            end_time=self.end_time,
        )
        return (
            factor
            * self.magnetic_field_amplitude
            * np.sin(
                self.magnetic_field_angular_frequency * time
                + self.magnetic_field_phase_difference
            )
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
